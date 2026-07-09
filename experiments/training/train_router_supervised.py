import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn.functional as F

from src.rag_moe.router_targets import build_good_expert_targets
from src.rag_moe.router import TwoStageRAGRouter


def _safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

def _split_indices(cache, split_name):
    metadata = cache.get("metadata") or {}
    splits = metadata.get("split")
    if not splits:
        return None
    indices = [idx for idx, value in enumerate(splits) if value == split_name]
    if not indices:
        return None
    return torch.as_tensor(indices, dtype=torch.long)


def _router_epoch(router, features, available, targets, optimizer=None, lambda_sparse=0.01):
    if optimizer is not None:
        router.train()
        optimizer.zero_grad()
    else:
        router.eval()

    outputs = router(features, available)
    bce = F.binary_cross_entropy_with_logits(outputs["select_logits"], targets.float(), reduction="none")
    mask = available.float()
    select_loss = (bce * mask).sum() / mask.sum().clamp_min(1.0)
    sparse_loss = (outputs["select_prob"][..., 1:] * available[..., 1:].float()).sum()
    sparse_loss = sparse_loss / available[..., 1:].float().sum().clamp_min(1.0)
    loss = select_loss + float(lambda_sparse) * sparse_loss
    if optimizer is not None:
        loss.backward()
        optimizer.step()
    return {
        "loss": float(loss.detach().cpu()),
        "select_loss": float(select_loss.detach().cpu()),
        "sparse_loss": float(sparse_loss.detach().cpu()),
    }


def _subset(tensor, indices):
    if indices is None:
        return tensor
    return tensor.index_select(0, indices.to(tensor.device))


def train_router_supervised_from_cache(
    cache_path,
    run_dir,
    hidden_dim=128,
    max_epochs=20,
    lr=1e-3,
    oracle_margin=0.98,
    lambda_sparse=0.01,
    patience=10,
    dropout=0.1,
    device=None,
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cache = _safe_torch_load(cache_path, map_location="cpu")
    features = cache["features"].float().to(device)
    available = cache["available"].bool().to(device)
    targets = build_good_expert_targets(cache["candidate_errors"].float(), cache["available"].bool(), oracle_margin).to(device)

    train_idx = _split_indices(cache, "train")
    val_idx = _split_indices(cache, "val")
    if train_idx is None:
        train_idx = torch.arange(features.shape[0], dtype=torch.long)
    if val_idx is None:
        val_idx = train_idx

    router = TwoStageRAGRouter(
        num_candidates=available.shape[-1],
        input_dim=features.shape[-1],
        hidden_dim=int(cache.get("hidden_dim", hidden_dim)),
        dropout=float(cache.get("dropout", dropout)),
    ).to(device)
    optimizer = torch.optim.Adam(router.parameters(), lr=float(lr))

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    history_path = run_path / "history.csv"
    best_loss = float("inf")
    best_epoch = -1
    stale = 0
    rows = []
    for epoch in range(int(max_epochs)):
        train_metrics = _router_epoch(
            router,
            _subset(features, train_idx),
            _subset(available, train_idx),
            _subset(targets, train_idx),
            optimizer=optimizer,
            lambda_sparse=lambda_sparse,
        )
        with torch.no_grad():
            val_metrics = _router_epoch(
                router,
                _subset(features, val_idx),
                _subset(available, val_idx),
                _subset(targets, val_idx),
                optimizer=None,
                lambda_sparse=lambda_sparse,
            )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_select_loss": train_metrics["select_loss"],
            "train_sparse_loss": train_metrics["sparse_loss"],
            "val_loss": val_metrics["loss"],
            "val_select_loss": val_metrics["select_loss"],
            "val_sparse_loss": val_metrics["sparse_loss"],
        }
        rows.append(row)
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            stale = 0
            torch.save(router.state_dict(), run_path / "best_router.pt")
        else:
            stale += 1
        if stale >= int(patience):
            break

    torch.save(router.state_dict(), run_path / "last_router.pt")
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "candidate_names": list(cache.get("candidate_names", [])),
        "num_train_samples": int(len(train_idx)),
        "num_val_samples": int(len(val_idx)),
        "oracle_margin": float(oracle_margin),
        "lambda_sparse": float(lambda_sparse),
        "artifacts": {
            "best_router": str(run_path / "best_router.pt"),
            "last_router": str(run_path / "last_router.pt"),
            "history": str(history_path),
        },
    }
    (run_path / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="Train a supervised RAG-MoE router warmup from cached source candidates.")
    parser.add_argument("--candidate_cache", required=True)
    parser.add_argument("--run_dir", default="results/training/supervised_router")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--oracle_margin", type=float, default=0.98)
    parser.add_argument("--lambda_sparse", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="")
    return parser


def main():
    args = build_parser().parse_args()
    summary = train_router_supervised_from_cache(
        cache_path=args.candidate_cache,
        run_dir=args.run_dir,
        hidden_dim=args.hidden_dim,
        max_epochs=args.max_epochs,
        lr=args.lr,
        oracle_margin=args.oracle_margin,
        lambda_sparse=args.lambda_sparse,
        patience=args.patience,
        dropout=args.dropout,
        device=args.device or None,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
