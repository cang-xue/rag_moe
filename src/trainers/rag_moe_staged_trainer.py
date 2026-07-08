from typing import Dict

import torch
import torch.nn.functional as F

from src.utils.helper import move_batch_meta, split_batch
from src.utils.metrics import masked_mae


def compute_counterfactual_pseudo_dist(itsc_loss: torch.Tensor, raft_loss: torch.Tensor, temperature: float):
    if itsc_loss.shape != raft_loss.shape:
        raise ValueError("itsc_loss and raft_loss must have the same shape")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    losses = torch.stack([itsc_loss, raft_loss], dim=-1)
    return torch.softmax(-losses / float(temperature), dim=-1)


def train_staged_router_epoch(
    model,
    loader,
    optimizer,
    supports,
    llm_encoding,
    null_value,
    stage,
    lambda_itsc,
    lambda_raft,
    lambda_entropy,
    lambda_balance,
    pseudo_temperature,
    device,
    unknown_nodes=None,
    known_nodes=None,
    num_masked_nodes=0,
    mask_unknown_inputs=False,
) -> Dict[str, float]:
    model.train()
    model.backbone.eval()
    for expert in model.experts:
        expert.eval()
    return _staged_router_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        supports=supports,
        llm_encoding=llm_encoding,
        null_value=null_value,
        stage=stage,
        lambda_itsc=lambda_itsc,
        lambda_raft=lambda_raft,
        lambda_entropy=lambda_entropy,
        lambda_balance=lambda_balance,
        pseudo_temperature=pseudo_temperature,
        unknown_nodes=unknown_nodes,
        known_nodes=known_nodes,
        num_masked_nodes=num_masked_nodes,
        mask_unknown_inputs=mask_unknown_inputs,
        training=True,
        device=device,
    )


@torch.no_grad()
def evaluate_staged_router_epoch(
    model,
    loader,
    supports,
    llm_encoding,
    null_value,
    stage,
    lambda_itsc,
    lambda_raft,
    lambda_entropy,
    lambda_balance,
    pseudo_temperature,
    device,
    unknown_nodes=None,
    known_nodes=None,
    mask_unknown_inputs=False,
) -> Dict[str, float]:
    model.eval()
    return _staged_router_epoch(
        model=model,
        loader=loader,
        optimizer=None,
        supports=supports,
        llm_encoding=llm_encoding,
        null_value=null_value,
        stage=stage,
        lambda_itsc=lambda_itsc,
        lambda_raft=lambda_raft,
        lambda_entropy=lambda_entropy,
        lambda_balance=lambda_balance,
        pseudo_temperature=pseudo_temperature,
        unknown_nodes=unknown_nodes,
        known_nodes=known_nodes,
        num_masked_nodes=0,
        mask_unknown_inputs=mask_unknown_inputs,
        training=False,
        device=device,
    )


def _staged_router_epoch(
    model,
    loader,
    optimizer,
    supports,
    llm_encoding,
    null_value,
    stage,
    lambda_itsc,
    lambda_raft,
    lambda_entropy,
    lambda_balance,
    pseudo_temperature,
    unknown_nodes,
    known_nodes,
    num_masked_nodes,
    mask_unknown_inputs,
    training,
    device,
) -> Dict[str, float]:
    if stage not in {"stage1", "stage2", "stage3"}:
        raise ValueError("stage must be one of stage1, stage2, stage3")
    _require_itsc_raft(model)

    totals = {
        "loss": 0.0,
        "pred_loss": 0.0,
        "itsc_loss": 0.0,
        "raft_loss": 0.0,
        "ce_loss": 0.0,
        "entropy_loss": 0.0,
        "balance_loss": 0.0,
        "batches": 0,
    }
    for batch in loader:
        x, label, batch_meta = split_batch(batch, getattr(loader, "batch_meta_keys", []))
        x = x.to(device)
        label = label.to(device)
        batch_meta = move_batch_meta(batch_meta, device)
        x = apply_staged_input_mask(
            x,
            unknown_nodes=unknown_nodes,
            known_nodes=known_nodes,
            num_masked_nodes=num_masked_nodes,
            mask_unknown_inputs=mask_unknown_inputs,
            training=training,
        )
        if optimizer is not None:
            optimizer.zero_grad()

        outputs = model(x, supports, llm_encoding, batch_meta=batch_meta)
        candidates = _candidate_tensor(outputs)
        candidate_losses = _candidate_node_losses(candidates, label, null_value)
        itsc_loss = candidate_losses[:, :, 1].mean()
        raft_loss = candidate_losses[:, :, 2].mean()
        pred_loss = masked_mae(outputs["prediction"], label, null_value)
        entropy_loss = _expert_entropy_loss(outputs["select_logits"])
        balance_loss = _expert_load_balance_loss(outputs["select_logits"])
        ce_loss = torch.zeros((), device=pred_loss.device, dtype=pred_loss.dtype)

        if stage == "stage1":
            loss = (
                pred_loss
                + float(lambda_itsc) * itsc_loss
                + float(lambda_raft) * raft_loss
                + float(lambda_entropy) * entropy_loss
                + float(lambda_balance) * balance_loss
            )
        elif stage == "stage2":
            pseudo = compute_counterfactual_pseudo_dist(
                candidate_losses[:, :, 1].detach(),
                candidate_losses[:, :, 2].detach(),
                pseudo_temperature,
            )
            ce_loss = _soft_cross_entropy(outputs["select_logits"][:, :, 1:3], pseudo)
            loss = (
                pred_loss
                + ce_loss
                + float(lambda_entropy) * entropy_loss
                + float(lambda_balance) * balance_loss
            )
        else:
            loss = pred_loss + float(lambda_entropy) * entropy_loss + float(lambda_balance) * balance_loss

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        totals["loss"] += float(loss.detach().cpu())
        totals["pred_loss"] += float(pred_loss.detach().cpu())
        totals["itsc_loss"] += float(itsc_loss.detach().cpu())
        totals["raft_loss"] += float(raft_loss.detach().cpu())
        totals["ce_loss"] += float(ce_loss.detach().cpu())
        totals["entropy_loss"] += float(entropy_loss.detach().cpu())
        totals["balance_loss"] += float(balance_loss.detach().cpu())
        totals["batches"] += 1

    batches = max(totals.pop("batches"), 1)
    return {key: value / batches for key, value in totals.items()}


def _require_itsc_raft(model):
    names = list(getattr(model, "correction_names", []))
    if "itsc" not in names or "raft" not in names:
        raise ValueError("staged ITS-C/RAFT training requires enabled experts to include itsc and raft")
    if names.index("itsc") != 1 or names.index("raft") != 2:
        raise ValueError("staged training expects correction order ['none', 'itsc', 'raft']")


def _candidate_tensor(outputs):
    candidates = [outputs["baseline_pred"]]
    for name in outputs["expert_outputs"]:
        candidates.append(outputs["baseline_pred"] + outputs["expert_outputs"][name].delta)
    return torch.stack(candidates, dim=2)


def _candidate_node_losses(candidates, label, null_value):
    if candidates.dim() != 5:
        raise ValueError("candidates must be [B, T, E, N, C]")
    label_expanded = label.unsqueeze(2)
    abs_error = torch.abs(candidates - label_expanded)
    if torch.isnan(torch.as_tensor(null_value, device=label.device, dtype=label.dtype)):
        valid = ~torch.isnan(label_expanded)
    else:
        valid = label_expanded > float(null_value) + 0.1
    valid = valid.to(abs_error.dtype)
    numerator = (abs_error * valid).sum(dim=(1, 4)).permute(0, 2, 1)
    denominator = valid.sum(dim=(1, 4)).permute(0, 2, 1).clamp_min(1.0)
    return numerator / denominator


def _soft_cross_entropy(logits, target_dist):
    log_prob = F.log_softmax(logits, dim=-1)
    return -(target_dist * log_prob).sum(dim=-1).mean()


def _expert_entropy_loss(select_logits):
    expert_prob = torch.softmax(select_logits[:, :, 1:3], dim=-1)
    entropy = -(expert_prob * torch.log(expert_prob.clamp_min(1e-8))).sum(dim=-1).mean()
    return -entropy


def _expert_load_balance_loss(select_logits):
    expert_prob = torch.softmax(select_logits[:, :, 1:3], dim=-1)
    usage = expert_prob.mean(dim=(0, 1))
    target = torch.full_like(usage, 1.0 / usage.numel())
    return torch.mean((usage - target) ** 2)


def apply_staged_input_mask(
    x,
    unknown_nodes=None,
    known_nodes=None,
    num_masked_nodes=0,
    mask_unknown_inputs=False,
    training=False,
):
    if not mask_unknown_inputs and (not training or int(num_masked_nodes) <= 0):
        return x
    masked = x.clone()
    if mask_unknown_inputs and unknown_nodes:
        masked[:, :, list(unknown_nodes), :] = 0
    if training and int(num_masked_nodes) > 0 and known_nodes:
        known = torch.as_tensor(list(known_nodes), device=masked.device, dtype=torch.long)
        if known.numel() > 0:
            count = min(int(num_masked_nodes), int(known.numel()))
            for batch_index in range(masked.shape[0]):
                selected = known[torch.randperm(known.numel(), device=masked.device)[:count]]
                masked[batch_index, :, selected, :] = 0
    return masked
