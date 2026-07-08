from typing import Dict

import torch
import torch.nn.functional as F

from src.utils.metrics import masked_mae
from src.utils.helper import move_batch_meta, split_batch


def compute_good_experts(candidates: torch.Tensor, label: torch.Tensor, oracle_margin: float):
    if candidates.dim() != 5:
        raise ValueError("candidates must be [B, T, E, N, C]")
    label_expanded = label.unsqueeze(2)
    errors = (candidates - label_expanded).abs().mean(dim=(1, 4)).permute(0, 2, 1)
    baseline_error = errors[:, :, 0:1]
    good = errors <= baseline_error * float(oracle_margin)
    return good, errors


def train_router_epoch(
    model,
    loader,
    optimizer,
    supports,
    llm_encoding,
    null_value,
    lambda_select,
    lambda_sparse,
    oracle_margin,
    device,
) -> Dict[str, float]:
    model.train()
    model.backbone.eval()
    for expert in model.experts:
        expert.eval()

    return _router_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        supports=supports,
        llm_encoding=llm_encoding,
        null_value=null_value,
        lambda_select=lambda_select,
        lambda_sparse=lambda_sparse,
        oracle_margin=oracle_margin,
        device=device,
    )


@torch.no_grad()
def evaluate_router_epoch(
    model,
    loader,
    supports,
    llm_encoding,
    null_value,
    lambda_select,
    lambda_sparse,
    oracle_margin,
    device,
) -> Dict[str, float]:
    model.eval()
    return _router_epoch(
        model=model,
        loader=loader,
        optimizer=None,
        supports=supports,
        llm_encoding=llm_encoding,
        null_value=null_value,
        lambda_select=lambda_select,
        lambda_sparse=lambda_sparse,
        oracle_margin=oracle_margin,
        device=device,
    )


def _router_epoch(
    model,
    loader,
    optimizer,
    supports,
    llm_encoding,
    null_value,
    lambda_select,
    lambda_sparse,
    oracle_margin,
    device,
) -> Dict[str, float]:
    totals = {"loss": 0.0, "pred_loss": 0.0, "select_loss": 0.0, "sparse_loss": 0.0, "batches": 0}
    for batch in loader:
        x, label, batch_meta = split_batch(batch, getattr(loader, 'batch_meta_keys', []))
        x = x.to(device)
        label = label.to(device)
        batch_meta = move_batch_meta(batch_meta, device)
        if optimizer is not None:
            optimizer.zero_grad()

        outputs = model(x, supports, llm_encoding, batch_meta=batch_meta)
        virtual_candidates = [outputs["baseline_pred"]]
        for name in outputs["expert_outputs"]:
            virtual_candidates.append(outputs["baseline_pred"] + outputs["expert_outputs"][name].delta)
        candidate_tensor = torch.stack(virtual_candidates, dim=2)
        good, _ = compute_good_experts(candidate_tensor.detach(), label, oracle_margin)

        pred_loss = masked_mae(outputs["prediction"], label, null_value)
        select_loss = F.binary_cross_entropy_with_logits(outputs["select_logits"], good.float())
        sparse_loss = outputs["select_prob"][:, :, 1:].mean()
        loss = pred_loss + float(lambda_select) * select_loss + float(lambda_sparse) * sparse_loss
        if optimizer is not None:
            loss.backward()
            optimizer.step()

        totals["loss"] += float(loss.detach().cpu())
        totals["pred_loss"] += float(pred_loss.detach().cpu())
        totals["select_loss"] += float(select_loss.detach().cpu())
        totals["sparse_loss"] += float(sparse_loss.detach().cpu())
        totals["batches"] += 1

    batches = max(totals.pop("batches"), 1)
    return {key: value / batches for key, value in totals.items()}


def summarize_router_usage(outputs, source_city, target_city, seed):
    active = outputs["active_mask"].detach().float().cpu()
    weights = outputs["weights"].detach().float().cpu()
    rows = []
    names = outputs.get("correction_names") or outputs.get("candidate_names")
    if names is None:
        raise KeyError("outputs must contain correction_names or candidate_names")
    for idx, name in enumerate(names):
        rows.append(
            {
                "source_city": source_city,
                "target_city": target_city,
                "seed": int(seed),
                "expert": name,
                "avg_selected_rate": float(active[:, :, idx].mean().item()),
                "avg_weight": float(weights[:, :, idx].mean().item()),
            }
        )
    return rows
