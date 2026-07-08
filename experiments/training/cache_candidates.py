import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch

from experiments.training.dpo_pairs import compute_candidate_node_errors

from experiments.training.multisource_data import build_city_contexts
from experiments.training.protocol import DEFAULT_CITIES
from src.rag_moe.features import build_router_features
from src.utils.helper import move_batch_meta, split_batch


def build_candidate_cache_payload(features, available, candidates, labels, candidate_names, metadata=None):
    errors = compute_candidate_node_errors(candidates, labels)
    if available.shape != errors.shape:
        raise ValueError("available must match computed candidate error shape")
    if len(candidate_names) != errors.shape[-1]:
        raise ValueError("candidate_names length must match candidate count")
    return {
        "features": features.detach().cpu(),
        "available": available.detach().cpu().bool(),
        "candidates": candidates.detach().cpu(),
        "labels": labels.detach().cpu(),
        "candidate_errors": errors.detach().cpu(),
        "candidate_names": list(candidate_names),
        "metadata": dict(metadata or {}),
    }


def save_candidate_cache(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return str(path)


def _pad_node_tensor(tensor, max_nodes, node_dim, pad_value=0):
    if tensor.shape[node_dim] == max_nodes:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[node_dim] = max_nodes - tensor.shape[node_dim]
    padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, padding], dim=node_dim)


def _sample_ids_from_meta(batch_meta, batch_size):
    sample_ids = batch_meta.get("sample_ids")
    if sample_ids is None:
        return list(range(batch_size))
    if hasattr(sample_ids, "detach"):
        sample_ids = sample_ids.detach().cpu()
    if getattr(sample_ids, "dim", lambda: 0)() > 1:
        sample_ids = sample_ids[:, 0]
    return [int(value) for value in sample_ids.reshape(-1).tolist()]


def _candidate_tensor_from_outputs(outputs):
    baseline = outputs["baseline_pred"]
    names = list(outputs.get("candidate_names") or outputs.get("correction_names"))
    residual_deltas = outputs["residual_deltas"]
    candidates = []
    for name in names:
        if name == "none":
            candidates.append(baseline)
        else:
            candidates.append(baseline + residual_deltas[name])
    return torch.stack(candidates, dim=2), names


def _available_from_outputs(outputs, names, device):
    baseline = outputs["baseline_pred"]
    masks = []
    for name in names:
        if name == "none":
            masks.append(torch.ones(baseline.shape[0], baseline.shape[2], dtype=torch.bool, device=device))
        else:
            masks.append(outputs["expert_outputs"][name].available.bool())
    return torch.stack(masks, dim=-1)


@torch.no_grad()
def collect_candidate_cache(model, contexts, split="train", device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.eval()
    max_nodes = max(context.num_nodes for context in contexts.values())
    feature_chunks = []
    available_chunks = []
    candidate_chunks = []
    label_chunks = []
    metadata = {
        "city": [],
        "city_id": [],
        "sample_ids": [],
        "split": [],
        "num_nodes": [],
    }
    candidate_names = None

    for context in contexts.values():
        loader = context.loaders["%s_loader" % split]
        supports = [support.to(device) for support in context.supports]
        llm_encoding = context.llm_encoding.to(device)
        for batch in loader:
            x, label, batch_meta = split_batch(batch, getattr(loader, "batch_meta_keys", context.loaders.get("batch_meta_keys", [])))
            x = x.to(device)
            label = label.to(device)
            batch_meta = move_batch_meta(batch_meta, device)
            batch_meta["city_id"] = torch.full((x.shape[0],), int(context.city_id), dtype=torch.long, device=device)
            batch_meta["city"] = context.city

            outputs = model(x, supports, llm_encoding, batch_meta=batch_meta)
            candidates, names = _candidate_tensor_from_outputs(outputs)
            if candidate_names is None:
                candidate_names = names
            elif candidate_names != names:
                raise ValueError("candidate_names changed across batches: %r vs %r" % (candidate_names, names))

            expert_deltas = {
                name: outputs["residual_deltas"][name]
                for name in names
                if name != "none"
            }
            features = build_router_features(x, outputs["baseline_pred"], expert_deltas)
            available = _available_from_outputs(outputs, names, device=device)

            feature_chunks.append(_pad_node_tensor(features.detach().cpu(), max_nodes, node_dim=1))
            available_chunks.append(_pad_node_tensor(available.detach().cpu(), max_nodes, node_dim=1, pad_value=False))
            candidate_chunks.append(_pad_node_tensor(candidates.detach().cpu(), max_nodes, node_dim=3))
            label_chunks.append(_pad_node_tensor(label.detach().cpu(), max_nodes, node_dim=2))

            batch_size = int(x.shape[0])
            metadata["city"].extend([context.city] * batch_size)
            metadata["city_id"].extend([int(context.city_id)] * batch_size)
            metadata["sample_ids"].extend(_sample_ids_from_meta(batch_meta, batch_size))
            metadata["split"].extend([split] * batch_size)
            metadata["num_nodes"].extend([int(context.num_nodes)] * batch_size)

    if not feature_chunks:
        raise ValueError("no batches found for split %r" % split)

    return build_candidate_cache_payload(
        features=torch.cat(feature_chunks, dim=0),
        available=torch.cat(available_chunks, dim=0),
        candidates=torch.cat(candidate_chunks, dim=0),
        labels=torch.cat(label_chunks, dim=0),
        candidate_names=candidate_names,
        metadata=metadata,
    )


def _parse_enabled_experts(value):
    experts = [item.strip() for item in value.split(",") if item.strip()]
    if not experts or experts == ["all"]:
        from experiments.training.train_experts_multisource import ALL_MULTISOURCE_EXPERTS

        return list(ALL_MULTISOURCE_EXPERTS)
    return experts


def build_rag_moe_for_cache(args, source_cities, device):
    from src.models.impel import IMPEL
    from src.models.rag_moe_impel import RAGMoEIMPEL
    from src.rag_moe.config import load_rag_moe_configs
    from src.rag_moe.registry import build_experts

    backbone = IMPEL(
        node_dim=args.node_dim,
        input_len=args.input_len,
        in_dim=args.input_dim,
        embed_dim=args.embed_dim,
        output_len=args.output_len,
        num_layer=args.num_layer,
        name="impel",
        dataset="multisource_to_%s" % args.target_city,
        device=device,
        num_nodes=0,
        seq_len=args.seq_len,
        horizon=args.horizon,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        llm_enc_dim=args.llm_enc_dim,
        supports_len=1,
        mp_layers=args.mp_layers,
    )
    state = torch.load(args.backbone_ckpt, map_location=device)
    backbone.load_state_dict(state, strict=False)

    expert_cfg, router_cfg = load_rag_moe_configs(args.expert_config, args.router_config)
    experts = build_experts(
        _parse_enabled_experts(args.enabled_experts),
        expert_cfg["experts"],
        {
            "dataset": "multisource_to_%s" % args.target_city,
            "source_data": ",".join(source_cities),
            "input_len": args.input_len,
            "output_len": args.output_len,
            "output_dim": args.output_dim,
            "llm_enc_dim": args.llm_enc_dim,
        },
    )
    router_settings = router_cfg["router"]
    return RAGMoEIMPEL(
        backbone=backbone,
        experts=experts,
        output_len=args.output_len,
        output_dim=args.output_dim,
        router_hidden_dim=int(router_settings.get("hidden_dim", 128)),
        router_dropout=float(router_settings.get("dropout", 0.1)),
    ).to(device)


def build_parser():
    parser = argparse.ArgumentParser(description="Cache multi-source RAG-MoE candidates for router training.")
    parser.add_argument("--cities", default=",".join(DEFAULT_CITIES))
    parser.add_argument("--target_city", required=True)
    parser.add_argument("--backbone_ckpt", required=True)
    parser.add_argument("--run_dir", default="results/training/candidate_cache")
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--enabled_experts", default="itsc,raft")
    parser.add_argument("--expert_config", default="configs/rag_moe/experts.yaml")
    parser.add_argument("--router_config", default="configs/rag_moe/router.yaml")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--input_dim", type=int, default=1)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--node_dim", type=int, default=32)
    parser.add_argument("--input_len", type=int, default=24)
    parser.add_argument("--output_len", type=int, default=24)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_layer", type=int, default=3)
    parser.add_argument("--mp_layers", type=int, default=1)
    parser.add_argument("--llm_enc_dim", type=int, default=4096)
    parser.add_argument("--num_unknown_nodes", type=int, default=10)
    parser.add_argument("--num_masked_nodes", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="")
    return parser


def main():
    args = build_parser().parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    protocol, contexts = build_city_contexts(
        args.cities.split(","),
        args.target_city,
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        num_unknown_nodes=args.num_unknown_nodes,
        num_masked_nodes=args.num_masked_nodes,
        seed=args.seed,
    )
    model = build_rag_moe_for_cache(args, protocol.source_cities, device)
    split_payloads = []
    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        split_payloads.append(collect_candidate_cache(model, contexts, split=split, device=device))

    if len(split_payloads) == 1:
        payload = split_payloads[0]
    else:
        payload = dict(split_payloads[0])
        for key in ("features", "available", "candidates", "labels", "candidate_errors"):
            payload[key] = torch.cat([item[key] for item in split_payloads], dim=0)
        metadata = {key: [] for key in split_payloads[0]["metadata"]}
        for item in split_payloads:
            for key, value in item["metadata"].items():
                metadata[key].extend(value)
        payload["metadata"] = metadata

    out_path = Path(args.run_dir) / "source_candidates.pt"
    save_candidate_cache(out_path, payload)
    print(str(out_path))


if __name__ == "__main__":
    main()
