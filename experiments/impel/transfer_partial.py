import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import numpy as np
import os
import time
import argparse
import yaml
import pickle
import scipy.sparse as sp
from scipy.sparse import linalg

import torch.nn as nn
import torch

from src.utils.helper import get_dataloader, check_device, get_num_nodes, get_null_value
from src.utils.metrics import masked_mae
from src.models.impel import IMPEL
from src.trainers.impel_trainer import IMPEL_Trainer
from src.utils.graph_algo import load_graph_data
from src.utils.args import get_public_config

def get_config():
    parser = get_public_config()

    # get private config
    parser.add_argument('--model_name', type=str, default='impel',
                        help='which model to train')
    parser.add_argument('--enabled_experts', type=str, default='itsc,raft')
    parser.add_argument('--router_ckpt', type=str, default='')
    parser.add_argument('--expert_config', type=str, default='configs/rag_moe/experts.yaml')
    parser.add_argument('--router_config', type=str, default='configs/rag_moe/router.yaml')
    parser.add_argument('--rag_source_data', type=str, default=None)
    parser.add_argument('--transfer_protocol', type=str, default='partial', choices=['partial', 'strict_zero_shot'])
    parser.add_argument('--direct_expert', type=str, default='',
                        help='Eval-only: return one expert candidate directly instead of router fusion.')
    parser.add_argument('--allow_raw_prior_direct', action='store_true',
                        help='Debug-only: allow --direct_expert to return a raw retrieval prior.')
    parser.add_argument('--n_filters', type=int, default=0,
                        help='number of hidden units')
    parser.add_argument('--filter_type', type=str, default='doubletransition')

    parser.add_argument('--node_dim', type=int, default=32)
    parser.add_argument('--input_len', type=int, default=24)
    parser.add_argument('--output_len', type=int, default=24)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--num_layer', type=int, default=3)
    parser.add_argument('--mp_layers', type=int, default=1)

    parser.add_argument('--llm_enc_dim', type=int, default=4096)
    parser.add_argument('--source_data', type=str, default='Delivery_SH')
    parser.add_argument('--target_data', type=str, default='Delivery_HZ')
    parser.add_argument('--num_unknown_nodes', type=int, default=5)  # 5 for JL and 10 for others

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir_pretrained', type=str, default='./logs/Delivery_SH/impel/impel-32-1.0')  # update this according to the source city

    args = parser.parse_args()
    if args.direct_expert and args.mode == 'train':
        raise ValueError("--direct_expert is eval-only and cannot be used with --mode train")
    if args.direct_expert and args.model_name != 'rag_moe_impel':
        raise ValueError("--direct_expert requires --model_name rag_moe_impel")
    args.steps = [10, 20, 30, 40, 50]
    print(args)

    args.num_nodes = get_num_nodes(args.target_data)
    args.null_value = get_null_value(args.target_data)

    if args.filter_type == 'identity':
        args.support_len = 1
    else:
        args.support_len = 3

    args.datapath = os.path.join('./data', args.target_data)
    args.graph_pkl = 'data/sensor_graph/adj_mx_{}.pkl'.format(args.target_data.lower())
    if args.seed != 0:
        torch.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    return args


def maybe_wrap_transfer_rag_moe(args, backbone, device):
    if args.model_name != 'rag_moe_impel':
        return backbone

    from src.models.rag_moe_impel import RAGMoEIMPEL
    from src.rag_moe.config import load_rag_moe_configs
    from src.rag_moe.registry import build_experts

    expert_cfg, router_cfg = load_rag_moe_configs(args.expert_config, args.router_config)
    source_data = args.rag_source_data if args.rag_source_data is not None else args.source_data
    experts = build_experts(
        parse_enabled_experts_arg(args.enabled_experts),
        expert_cfg['experts'],
        {
            'dataset': args.target_data,
            'target_data': args.target_data,
            'source_data': source_data,
            'transfer_protocol': args.transfer_protocol,
            'input_len': args.input_len,
            'output_len': args.output_len,
            'output_dim': args.output_dim,
            'llm_enc_dim': args.llm_enc_dim,
        },
    )
    router_settings = router_cfg['router']
    model = RAGMoEIMPEL(
        backbone=backbone,
        experts=experts,
        output_len=args.output_len,
        output_dim=args.output_dim,
        router_hidden_dim=int(router_settings.get('hidden_dim', 128)),
        router_dropout=float(router_settings.get('dropout', 0.1)),
    ).to(device)
    model.return_dict = False
    if args.router_ckpt:
        model.router.load_state_dict(torch.load(args.router_ckpt, map_location=device), strict=False)
    return model


def parse_enabled_experts_arg(value):
    return [item.strip() for item in str(value).split(',') if item.strip()]


def main():
    args = get_config()
    device = check_device()
    _, _, adj_mat = load_graph_data(args.graph_pkl)

    args.llmencpath = f'./data/llmvec_llama3_{args.target_data}.npy'
    llm_encoding = np.load(args.llmencpath)
    llm_encoding = torch.from_numpy(llm_encoding).cuda()

    model = IMPEL(
                 node_dim=args.node_dim,
                 input_len=args.input_len,
                 in_dim=args.input_dim,
                 embed_dim=args.embed_dim,
                 output_len=args.output_len,
                 num_layer=args.num_layer,
                 name=args.model_name,
                 dataset=args.target_data,
                 device=device,
                 num_nodes=args.num_nodes,
                 seq_len=args.seq_len,
                 horizon=args.horizon,
                 input_dim=args.input_dim,
                 output_dim=args.output_dim,
                 llm_enc_dim=args.llm_enc_dim,
                 supports_len=args.support_len,
                 mp_layers=args.mp_layers,
                 )
    model = maybe_wrap_transfer_rag_moe(args, model, device)

    data = get_dataloader(args.datapath,
                          args.batch_size,
                          args.input_dim,
                          args.output_dim,
                          include_metadata=args.model_name == 'rag_moe_impel')

    
    result_path = args.result_path + '/' + args.target_data + '/{}_{}_{}_{}'.format(args.seq_len, args.horizon, args.input_dim, args.output_dim)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    #####Masked training####
    n_u = args.num_unknown_nodes
    rand = np.random.RandomState(42)  # Fixed random output, just an example when seed = 0.
    unknown_set = rand.choice(list(range(0, args.num_nodes)), n_u, replace=False)
    unknown_set = set(unknown_set)
    full_set = set(range(0, args.num_nodes))
    known_set = full_set - unknown_set
    #####Masked training####


    trainer = IMPEL_Trainer(model=model,
                            adj_mat=adj_mat,
                            filter_type=args.filter_type,
                            data=data,
                            aug=args.aug,
                            base_lr=args.base_lr,
                            steps=args.steps,
                            lr_decay_ratio=args.lr_decay_ratio,
                            log_dir=args.log_dir_pretrained,
                            n_exp=args.n_exp,
                            save_iter=args.save_iter,
                            clip_grad_value=args.max_grad_norm,
                            max_epochs=args.max_epochs,
                            patience=args.patience,
                            device=device,
                            model_name=args.model_name,
                            result_path=result_path,
                            null_value =args.null_value,
                            unknown_set=unknown_set,
                            known_set=known_set,
                            n_m=0,
                            llm_encoding=llm_encoding,
                            direct_expert=args.direct_expert,
                            allow_raw_prior_direct=args.allow_raw_prior_direct,
                            )


    trainer.test(-1, 'test')
    if args.save_preds:
        trainer.save_preds(-1)


if __name__ == "__main__":
    main()
