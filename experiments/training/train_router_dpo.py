import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.training.dpo_pairs import build_dpo_pairs
from experiments.training.router_dpo import (
    configure_router_dpo_tuning,
    dpo_preference_loss,
)
from src.rag_moe.router import TwoStageRAGRouter


def _safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _router_policy_logits(router, features, available):
    encoded = router.encoder(features)
    logits = router.selector(encoded) + router.weighter(encoded)
    return logits.masked_fill(~available.bool(), torch.finfo(logits.dtype).min)


def _flatten_pair_logits(logits, pairs):
    return logits[pairs.batch_index.to(logits.device), pairs.node_index.to(logits.device)]


def train_router_dpo_from_cache(
    cache_path,
    router_ckpt,
    run_dir,
    hidden_dim=128,
    beta=0.1,
    rel_margin=0.01,
    abs_margin=0.01,
    max_epochs=20,
    lr=1e-3,
    train_scope="heads",
    dropout=0.0,
    device=None,
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cache = _safe_torch_load(cache_path, map_location="cpu")
    features = cache["features"].float().to(device)
    available = cache["available"].bool().to(device)
    errors = cache["candidate_errors"].float()
    pairs = build_dpo_pairs(errors, cache["available"].bool(), rel_margin=rel_margin, abs_margin=abs_margin)
    if len(pairs) == 0:
        raise ValueError("DPO pair construction produced no pairs")

    router = TwoStageRAGRouter(
        num_candidates=int(cache.get("num_candidates", available.shape[-1])),
        input_dim=features.shape[-1],
        hidden_dim=int(cache.get("hidden_dim", hidden_dim)),
        dropout=float(cache.get("dropout", dropout)),
    ).to(device)
    state = _safe_torch_load(router_ckpt, map_location=device)
    router.load_state_dict(state, strict=False)
    trainable = configure_router_dpo_tuning(router, train_scope=train_scope)
    optimizer = torch.optim.Adam([param for _, param in trainable], lr=float(lr))

    router.eval()
    with torch.no_grad():
        reference_logits = _router_policy_logits(router, features, available).detach()

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    history_path = run_path / "history.csv"
    best_loss = float("inf")
    best_epoch = -1
    history_rows = []

    for epoch in range(int(max_epochs)):
        router.train()
        optimizer.zero_grad()
        policy_logits = _router_policy_logits(router, features, available)
        loss = dpo_preference_loss(
            _flatten_pair_logits(policy_logits, pairs),
            _flatten_pair_logits(reference_logits, pairs),
            pairs.chosen,
            pairs.rejected,
            beta=beta,
            weight=pairs.weight,
        )
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        row = {"epoch": epoch, "dpo_loss": value, "num_pairs": len(pairs)}
        history_rows.append(row)
        if value < best_loss:
            best_loss = value
            best_epoch = epoch
            torch.save(router.state_dict(), run_path / "best_router.pt")

    torch.save(router.state_dict(), run_path / "last_router.pt")
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "dpo_loss", "num_pairs"])
        writer.writeheader()
        writer.writerows(history_rows)

    torch.save(
        {
            "chosen": pairs.chosen,
            "rejected": pairs.rejected,
            "batch_index": pairs.batch_index,
            "node_index": pairs.node_index,
            "weight": pairs.weight,
        },
        run_path / "dpo_pairs.pt",
    )
    summary = {
        "best_epoch": best_epoch,
        "best_dpo_loss": best_loss,
        "num_pairs": len(pairs),
        "train_scope": train_scope,
        "beta": float(beta),
        "rel_margin": float(rel_margin),
        "abs_margin": float(abs_margin),
        "artifacts": {
            "best_router": str(run_path / "best_router.pt"),
            "last_router": str(run_path / "last_router.pt"),
            "history": str(history_path),
        },
    }
    (run_path / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="DPO-tune RAG-MoE router heads from cached source candidates.")
    parser.add_argument("--candidate_cache", required=True)
    parser.add_argument("--router_ckpt", required=True)
    parser.add_argument("--run_dir", default="results/training/dpo_router")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--rel_margin", type=float, default=0.01)
    parser.add_argument("--abs_margin", type=float, default=0.01)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_scope", choices=["heads", "all"], default="heads")
    return parser


def main():
    args = build_parser().parse_args()
    summary = train_router_dpo_from_cache(
        cache_path=args.candidate_cache,
        router_ckpt=args.router_ckpt,
        run_dir=args.run_dir,
        hidden_dim=args.hidden_dim,
        beta=args.beta,
        rel_margin=args.rel_margin,
        abs_margin=args.abs_margin,
        max_epochs=args.max_epochs,
        lr=args.lr,
        train_scope=args.train_scope,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
