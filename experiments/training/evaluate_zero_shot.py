import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch

from experiments.training.router_dpo import guarded_hard_selection

from experiments.training.cache_candidates import _available_from_outputs, _candidate_tensor_from_outputs, build_rag_moe_for_cache
from experiments.training.multisource_data import build_target_city_context
from experiments.training.protocol import DEFAULT_CITIES, split_source_target
from src.utils.helper import move_batch_meta, split_batch
from src.utils.metrics import masked_mae


def apply_guarded_hard_candidates(candidates, probabilities, none_index=0, min_best_prob=0.5, min_margin_over_none=0.0):
    if candidates.dim() != 5:
        raise ValueError("candidates must be [B, T, E, N, C]")
    selected = guarded_hard_selection(
        probabilities,
        none_index=none_index,
        min_best_prob=min_best_prob,
        min_margin_over_none=min_margin_over_none,
    )
    _, horizon, _, _, channels = candidates.shape
    gather_index = selected.unsqueeze(1).unsqueeze(2).unsqueeze(-1).expand(-1, horizon, 1, -1, channels)
    return candidates.gather(dim=2, index=gather_index).squeeze(2)


def fuse_soft_candidates(candidates, weights):
    if candidates.dim() != 5:
        raise ValueError("candidates must be [B, T, E, N, C]")
    if weights.dim() != 3:
        raise ValueError("weights must be [B, N, E]")
    expanded = weights.permute(0, 2, 1).unsqueeze(1).unsqueeze(-1)
    return (candidates * expanded).sum(dim=2)


def _selected_rates(selected, candidate_names):
    total = max(int(selected.numel()), 1)
    return {
        name: float((selected == idx).float().sum().item() / total)
        for idx, name in enumerate(candidate_names)
    }


def evaluate_cached_candidates(
    candidates,
    labels,
    weights,
    select_prob,
    available,
    candidate_names,
    null_value,
    run_dir,
    min_best_prob=0.5,
    min_margin_over_none=0.0,
):
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    soft_pred = fuse_soft_candidates(candidates, weights)
    hard_pred = apply_guarded_hard_candidates(
        candidates,
        select_prob,
        none_index=0,
        min_best_prob=min_best_prob,
        min_margin_over_none=min_margin_over_none,
    )
    selected = guarded_hard_selection(
        select_prob,
        none_index=0,
        min_best_prob=min_best_prob,
        min_margin_over_none=min_margin_over_none,
    )
    summary = {
        "soft_mae": float(masked_mae(soft_pred, labels, null_value).detach().cpu()),
        "guarded_hard_mae": float(masked_mae(hard_pred, labels, null_value).detach().cpu()),
        "candidate_names": list(candidate_names),
        "per_expert_selected_rate": _selected_rates(selected.detach().cpu(), candidate_names),
        "available_rate": float(available.float().mean().detach().cpu()),
        "min_best_prob": float(min_best_prob),
        "min_margin_over_none": float(min_margin_over_none),
    }
    (run_path / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


@torch.no_grad()
def evaluate_zero_shot_model(
    model,
    context,
    router_ckpt,
    run_dir,
    device=None,
    min_best_prob=0.5,
    min_margin_over_none=0.0,
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    state = torch.load(router_ckpt, map_location=device)
    model.router.load_state_dict(state, strict=False)
    model.eval()
    loader = context.loaders["test_loader"]
    supports = [support.to(device) for support in context.supports]
    llm_encoding = context.llm_encoding.to(device)
    candidate_chunks = []
    label_chunks = []
    weight_chunks = []
    prob_chunks = []
    available_chunks = []
    candidate_names = None
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
        available = _available_from_outputs(outputs, names, device=device)
        candidate_chunks.append(candidates.detach().cpu())
        label_chunks.append(label.detach().cpu())
        weight_chunks.append(outputs["weights"].detach().cpu())
        prob_chunks.append(outputs["select_prob"].detach().cpu())
        available_chunks.append(available.detach().cpu())

    if not candidate_chunks:
        raise ValueError("target test loader produced no batches")
    return evaluate_cached_candidates(
        candidates=torch.cat(candidate_chunks, dim=0),
        labels=torch.cat(label_chunks, dim=0),
        weights=torch.cat(weight_chunks, dim=0),
        select_prob=torch.cat(prob_chunks, dim=0),
        available=torch.cat(available_chunks, dim=0),
        candidate_names=candidate_names,
        null_value=context.null_value,
        run_dir=run_dir,
        min_best_prob=min_best_prob,
        min_margin_over_none=min_margin_over_none,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate multi-source RAG-MoE zero-shot on a held-out target city.")
    parser.add_argument("--cities", default=",".join(DEFAULT_CITIES))
    parser.add_argument("--target_city", required=True)
    parser.add_argument("--backbone_ckpt", required=True)
    parser.add_argument("--router_ckpt", required=True)
    parser.add_argument("--run_dir", default="results/training/zero_shot_eval")
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
    parser.add_argument("--min_best_prob", type=float, default=0.5)
    parser.add_argument("--min_margin_over_none", type=float, default=0.0)
    parser.add_argument("--device", default="")
    return parser


def main():
    args = build_parser().parse_args()
    source_cities, _ = split_source_target(args.cities, args.target_city)
    context = build_target_city_context(
        args.cities.split(","),
        args.target_city,
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        num_unknown_nodes=args.num_unknown_nodes,
        num_masked_nodes=args.num_masked_nodes,
        seed=args.seed,
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_rag_moe_for_cache(args, source_cities, device)
    summary = evaluate_zero_shot_model(
        model=model,
        context=context,
        router_ckpt=args.router_ckpt,
        run_dir=args.run_dir,
        device=device,
        min_best_prob=args.min_best_prob,
        min_margin_over_none=args.min_margin_over_none,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
