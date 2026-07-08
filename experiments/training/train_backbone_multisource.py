import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.training.multisource_data import build_city_contexts
from experiments.training.protocol import DEFAULT_CITIES, split_source_target
from src.models.impel import IMPEL
from src.utils.helper import move_batch_meta, split_batch
from src.utils.metrics import masked_mae


def inverse_transform_with_scalers(tensor, scalers, node_indices=None):
    output = tensor.clone()
    if node_indices is None:
        node_indices = list(range(output.shape[2]))
    for local_idx, original_idx in enumerate(node_indices):
        if original_idx >= len(scalers):
            continue
        output[:, :, local_idx, :1] = scalers[original_idx].inverse_transform(output[:, :, local_idx, :1])
    return output


def _masked_training_batch(model, batch, context, optimizer, device, rng, clip_grad_value=5.0):
    model.train()
    x, label, batch_meta = split_batch(batch, getattr(context.loaders["train_loader"], "batch_meta_keys", context.loaders.get("batch_meta_keys", [])))
    x = x.to(device)
    label = label.to(device)
    batch_meta = move_batch_meta(batch_meta, device)
    train_nodes = sorted(context.known_set)
    x = x[:, :, train_nodes, :]
    label = label[:, :, train_nodes, :]
    supports = [support.to(device)[:, train_nodes][train_nodes, :] for support in context.supports]
    llm_encoding = context.llm_encoding.to(device)[train_nodes, :]
    missing_index = torch.ones_like(x)
    mask_count = min(int(context.num_masked_nodes), len(train_nodes))
    for row in range(x.shape[0]):
        masked = rng.choice(len(train_nodes), mask_count, replace=False)
        missing_index[row, :, masked, :] = 0.0
    x = x * missing_index

    optimizer.zero_grad()
    pred = model(x, supports, llm_encoding)
    pred = inverse_transform_with_scalers(pred, context.scalers, train_nodes)
    label = inverse_transform_with_scalers(label, context.scalers, train_nodes)
    loss = masked_mae(pred, label, context.null_value)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip_grad_value))
    optimizer.step()
    return float(loss.detach().cpu())


@torch.no_grad()
def _evaluate_city(model, context, device):
    model.eval()
    loader = context.loaders["val_loader"]
    preds = []
    labels = []
    supports = [support.to(device) for support in context.supports]
    llm_encoding = context.llm_encoding.to(device)
    for batch in loader:
        x, label, batch_meta = split_batch(batch, getattr(loader, "batch_meta_keys", context.loaders.get("batch_meta_keys", [])))
        x = x.to(device)
        label = label.to(device)
        _ = move_batch_meta(batch_meta, device)
        missing_index = torch.ones_like(x)
        if context.unknown_set:
            missing_index[:, :, list(context.unknown_set), :] = 0.0
        pred = model(x * missing_index, supports, llm_encoding)
        pred = inverse_transform_with_scalers(pred, context.scalers)
        label = inverse_transform_with_scalers(label, context.scalers)
        preds.append(pred.cpu())
        labels.append(label.cpu())
    if not preds:
        raise ValueError("validation loader for %s produced no batches" % context.city)
    return float(masked_mae(torch.cat(preds, dim=0), torch.cat(labels, dim=0), context.null_value).detach().cpu())


def build_impel_backbone(args, device):
    return IMPEL(
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
    ).to(device)


def train_backbone_multisource(args):
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
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    model = build_impel_backbone(args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.base_lr))
    run_path = Path(args.run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    history_path = run_path / "history.csv"
    best_loss = float("inf")
    best_epoch = -1
    stale = 0
    rows = []
    rng = np.random.RandomState(int(args.seed))

    for epoch in range(int(args.max_epochs)):
        started = time.time()
        train_losses = []
        for context in contexts.values():
            for batch in context.loaders["train_loader"]:
                train_losses.append(
                    _masked_training_batch(
                        model,
                        batch,
                        context,
                        optimizer,
                        device,
                        rng,
                        clip_grad_value=args.max_grad_norm,
                    )
                )
        val_by_city = {city: _evaluate_city(model, context, device) for city, context in contexts.items()}
        val_macro = float(sum(val_by_city.values()) / max(len(val_by_city), 1))
        row = {
            "epoch": epoch,
            "train_mae": float(sum(train_losses) / max(len(train_losses), 1)),
            "val_macro_mae": val_macro,
            "epoch_time": time.time() - started,
        }
        row.update({"val_%s_mae" % city: value for city, value in val_by_city.items()})
        rows.append(row)
        if val_macro < best_loss:
            best_loss = val_macro
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), run_path / "best_backbone.pt")
        else:
            stale += 1
        print("epoch=%s train_mae=%.6f val_macro_mae=%.6f best_epoch=%s" % (epoch, row["train_mae"], val_macro, best_epoch))
        if stale >= int(args.patience):
            break

    torch.save(model.state_dict(), run_path / "last_backbone.pt")
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "target_city": protocol.target_city,
        "source_cities": protocol.source_cities,
        "best_epoch": best_epoch,
        "best_val_macro_mae": best_loss,
        "artifacts": {
            "best_backbone": str(run_path / "best_backbone.pt"),
            "last_backbone": str(run_path / "last_backbone.pt"),
            "history": str(history_path),
        },
    }
    (run_path / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="Train one shared IMPEL backbone on multiple source cities.")
    parser.add_argument("--cities", default=",".join(DEFAULT_CITIES))
    parser.add_argument("--target_city", required=True)
    parser.add_argument("--run_dir", default="results/training/backbone")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=250)
    parser.add_argument("--base_lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--device", default="")
    return parser


def main():
    args = build_parser().parse_args()
    sources, target = split_source_target(args.cities, args.target_city)
    print("source_cities=%s target_city=%s" % (",".join(sources), target))
    summary = train_backbone_multisource(args)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
