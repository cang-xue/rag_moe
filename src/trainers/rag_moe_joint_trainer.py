from collections import OrderedDict
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from src.rag_moe.features import build_router_features
from src.rag_moe.fusion import fuse_residuals
from src.trainers.rag_moe_staged_trainer import (
    _candidate_node_losses,
    _expert_entropy_loss,
    _expert_load_balance_loss,
    _soft_cross_entropy,
    compute_counterfactual_pseudo_dist,
)
from src.utils.helper import move_batch_meta, split_batch
from src.utils.metrics import masked_mae


class TrainableRAFTResidual(nn.Module):
    def __init__(self, initial_alpha=0.0, min_alpha=None, max_alpha=None):
        super().__init__()
        self.prior_alpha = nn.Parameter(torch.tensor(float(initial_alpha)))
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

    def forward(self, raw_prior, baseline_pred):
        return self.prior_alpha.to(device=baseline_pred.device, dtype=baseline_pred.dtype) * (
            raw_prior - baseline_pred
        )

    def clamp_(self):
        min_alpha = None if self.min_alpha is None else float(self.min_alpha)
        max_alpha = None if self.max_alpha is None else float(self.max_alpha)
        if min_alpha is not None or max_alpha is not None:
            with torch.no_grad():
                self.prior_alpha.clamp_(min=min_alpha, max=max_alpha)


def collect_joint_trainable_parameters(
    backbone,
    router,
    itsc_extractor,
    raft_residual,
    router_lr,
    itsc_lr,
    raft_lr,
    backbone_lr=0.0,
    train_raft_alpha=True,
    train_backbone=False,
):
    for parameter in backbone.parameters():
        parameter.requires_grad_(bool(train_backbone))
    for parameter in itsc_extractor.parameters():
        parameter.requires_grad_(False)
    for parameter in router.parameters():
        parameter.requires_grad_(True)
    for parameter in raft_residual.parameters():
        parameter.requires_grad_(bool(train_raft_alpha))

    itsc_params = []
    for name, parameter in itsc_extractor.model.named_parameters():
        if name == "prior_alpha" or name.startswith(("prior_out_proj.", "prior_out_gate.")):
            parameter.requires_grad_(True)
            itsc_params.append(parameter)
    if not itsc_params:
        raise ValueError("ITSC extractor has no trainable residual branch parameters")

    groups = [
        {"params": list(router.parameters()), "lr": float(router_lr)},
        {"params": itsc_params, "lr": float(itsc_lr)},
    ]
    if train_backbone:
        groups.insert(0, {"params": list(backbone.parameters()), "lr": float(backbone_lr)})
    if train_raft_alpha:
        groups.append({"params": list(raft_residual.parameters()), "lr": float(raft_lr)})
    return groups


def train_joint_epoch(
    backbone,
    itsc_extractor,
    raft_expert,
    raft_residual,
    router,
    loader,
    optimizer,
    supports,
    llm_encoding,
    null_value,
    lambda_itsc,
    lambda_raft,
    lambda_ce,
    lambda_entropy,
    lambda_balance,
    lambda_select_bce,
    lambda_delta_l1,
    pseudo_temperature,
    baseline_margin,
    expert_improvement_margin,
    unknown_nodes,
    known_nodes,
    num_masked_nodes,
    mask_unknown_inputs,
    train_backbone,
    device,
) -> Dict[str, float]:
    if train_backbone:
        backbone.train()
    else:
        backbone.eval()
    itsc_extractor.train()
    raft_expert.eval()
    raft_residual.train()
    router.train()
    return _joint_epoch(
        backbone=backbone,
        itsc_extractor=itsc_extractor,
        raft_expert=raft_expert,
        raft_residual=raft_residual,
        router=router,
        loader=loader,
        optimizer=optimizer,
        supports=supports,
        llm_encoding=llm_encoding,
        null_value=null_value,
        lambda_itsc=lambda_itsc,
        lambda_raft=lambda_raft,
        lambda_ce=lambda_ce,
        lambda_entropy=lambda_entropy,
        lambda_balance=lambda_balance,
        lambda_select_bce=lambda_select_bce,
        lambda_delta_l1=lambda_delta_l1,
        pseudo_temperature=pseudo_temperature,
        baseline_margin=baseline_margin,
        expert_improvement_margin=expert_improvement_margin,
        unknown_nodes=unknown_nodes,
        known_nodes=known_nodes,
        num_masked_nodes=num_masked_nodes,
        mask_unknown_inputs=mask_unknown_inputs,
        training=True,
        train_backbone=train_backbone,
        device=device,
    )


@torch.no_grad()
def evaluate_joint_epoch(
    backbone,
    itsc_extractor,
    raft_expert,
    raft_residual,
    router,
    loader,
    supports,
    llm_encoding,
    null_value,
    lambda_itsc,
    lambda_raft,
    lambda_ce,
    lambda_entropy,
    lambda_balance,
    lambda_select_bce,
    lambda_delta_l1,
    pseudo_temperature,
    baseline_margin,
    expert_improvement_margin,
    unknown_nodes,
    known_nodes,
    mask_unknown_inputs,
    device,
) -> Dict[str, float]:
    backbone.eval()
    itsc_extractor.eval()
    raft_expert.eval()
    raft_residual.eval()
    router.eval()
    return _joint_epoch(
        backbone=backbone,
        itsc_extractor=itsc_extractor,
        raft_expert=raft_expert,
        raft_residual=raft_residual,
        router=router,
        loader=loader,
        optimizer=None,
        supports=supports,
        llm_encoding=llm_encoding,
        null_value=null_value,
        lambda_itsc=lambda_itsc,
        lambda_raft=lambda_raft,
        lambda_ce=lambda_ce,
        lambda_entropy=lambda_entropy,
        lambda_balance=lambda_balance,
        lambda_select_bce=lambda_select_bce,
        lambda_delta_l1=lambda_delta_l1,
        pseudo_temperature=pseudo_temperature,
        baseline_margin=baseline_margin,
        expert_improvement_margin=expert_improvement_margin,
        unknown_nodes=unknown_nodes,
        known_nodes=known_nodes,
        num_masked_nodes=0,
        mask_unknown_inputs=mask_unknown_inputs,
        training=False,
        train_backbone=False,
        device=device,
    )


def _joint_epoch(
    backbone,
    itsc_extractor,
    raft_expert,
    raft_residual,
    router,
    loader,
    optimizer,
    supports,
    llm_encoding,
    null_value,
    lambda_itsc,
    lambda_raft,
    lambda_ce,
    lambda_entropy,
    lambda_balance,
    lambda_select_bce,
    lambda_delta_l1,
    pseudo_temperature,
    baseline_margin,
    expert_improvement_margin,
    unknown_nodes,
    known_nodes,
    num_masked_nodes,
    mask_unknown_inputs,
    training,
    train_backbone,
    device,
):
    totals = {
        "loss": 0.0,
        "pred_loss": 0.0,
        "itsc_loss": 0.0,
        "raft_loss": 0.0,
        "ce_loss": 0.0,
        "select_bce_loss": 0.0,
        "entropy_loss": 0.0,
        "balance_loss": 0.0,
        "delta_l1_loss": 0.0,
        "raft_alpha": 0.0,
        "baseline_weight": 0.0,
        "itsc_weight": 0.0,
        "raft_weight": 0.0,
        "batches": 0,
    }
    for batch in loader:
        x, label, batch_meta = split_batch(batch, getattr(loader, "batch_meta_keys", []))
        x = x.to(device)
        label = label.to(device)
        batch_meta = move_batch_meta(batch_meta, device)
        batch_meta.setdefault("output_len", label.shape[1])
        batch_meta.setdefault("output_dim", label.shape[-1])
        x = apply_joint_input_mask(
            x,
            unknown_nodes=unknown_nodes,
            known_nodes=known_nodes,
            num_masked_nodes=num_masked_nodes,
            mask_unknown_inputs=mask_unknown_inputs,
            training=training,
        )

        if optimizer is not None:
            optimizer.zero_grad()

        if train_backbone and optimizer is not None:
            baseline_pred = backbone(x, supports, llm_encoding)
        else:
            with torch.no_grad():
                baseline_pred = backbone(x, supports, llm_encoding)
        itsc_output = itsc_extractor.forward_train_correction(
            history_data=x,
            baseline_pred=baseline_pred.detach(),
            llm_encoding=llm_encoding,
            batch_meta=batch_meta,
        )
        raft_prior = raft_expert.forward_prior(
            x,
            supports=supports,
            llm=llm_encoding,
            batch_meta=batch_meta,
        ).prior.to(device=baseline_pred.device, dtype=baseline_pred.dtype)
        raft_delta = raft_residual(raft_prior, baseline_pred.detach())

        expert_deltas = OrderedDict()
        expert_deltas["itsc"] = itsc_output.delta
        expert_deltas["raft"] = raft_delta
        features = build_router_features(x, baseline_pred, expert_deltas)
        available = torch.ones(
            baseline_pred.shape[0],
            baseline_pred.shape[2],
            3,
            dtype=torch.bool,
            device=baseline_pred.device,
        )
        router_outputs = router(features, available)
        delta_tensor = torch.stack(
            [torch.zeros_like(baseline_pred), itsc_output.delta, raft_delta],
            dim=2,
        )
        routed_delta = fuse_residuals(delta_tensor, router_outputs["weights"])
        prediction = baseline_pred + routed_delta

        candidates = torch.stack(
            [
                baseline_pred,
                baseline_pred + itsc_output.delta,
                baseline_pred + raft_delta,
            ],
            dim=2,
        )
        candidate_losses = _candidate_node_losses(candidates, label, null_value)
        pseudo = compute_baseline_inclusive_pseudo_dist(
            candidate_losses.detach(),
            pseudo_temperature,
            baseline_margin,
        )
        select_targets = compute_expert_improvement_targets(
            candidate_losses.detach(),
            expert_improvement_margin,
        )
        ce_loss = _soft_cross_entropy(router_outputs["select_logits"], pseudo)
        select_bce_loss = F.binary_cross_entropy_with_logits(
            router_outputs["select_logits"],
            select_targets.float(),
        )
        pred_loss = masked_mae(prediction, label, null_value)
        itsc_loss = masked_mae(baseline_pred + itsc_output.delta, label, null_value)
        raft_loss = masked_mae(baseline_pred + raft_delta, label, null_value)
        entropy_loss = _expert_entropy_loss(router_outputs["select_logits"])
        balance_loss = _expert_load_balance_loss(router_outputs["select_logits"])
        delta_l1_loss = routed_delta.abs().mean()
        loss = (
            pred_loss
            + float(lambda_itsc) * itsc_loss
            + float(lambda_raft) * raft_loss
            + float(lambda_ce) * ce_loss
            + float(lambda_select_bce) * select_bce_loss
            + float(lambda_entropy) * entropy_loss
            + float(lambda_balance) * balance_loss
            + float(lambda_delta_l1) * delta_l1_loss
        )

        if optimizer is not None:
            loss.backward()
            optimizer.step()
            raft_residual.clamp_()

        totals["loss"] += float(loss.detach().cpu())
        totals["pred_loss"] += float(pred_loss.detach().cpu())
        totals["itsc_loss"] += float(itsc_loss.detach().cpu())
        totals["raft_loss"] += float(raft_loss.detach().cpu())
        totals["ce_loss"] += float(ce_loss.detach().cpu())
        totals["select_bce_loss"] += float(select_bce_loss.detach().cpu())
        totals["entropy_loss"] += float(entropy_loss.detach().cpu())
        totals["balance_loss"] += float(balance_loss.detach().cpu())
        totals["delta_l1_loss"] += float(delta_l1_loss.detach().cpu())
        totals["raft_alpha"] += float(raft_residual.prior_alpha.detach().cpu())
        weights = router_outputs["weights"].detach().mean(dim=(0, 1)).cpu()
        totals["baseline_weight"] += float(weights[0])
        totals["itsc_weight"] += float(weights[1])
        totals["raft_weight"] += float(weights[2])
        totals["batches"] += 1

    batches = max(totals.pop("batches"), 1)
    return {key: value / batches for key, value in totals.items()}


def compute_baseline_inclusive_pseudo_dist(candidate_losses, temperature, baseline_margin=0.0):
    if candidate_losses.dim() != 3:
        raise ValueError("candidate_losses must be [B, N, E]")
    if candidate_losses.shape[-1] != 3:
        raise ValueError("candidate_losses must include [baseline, itsc, raft]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    adjusted = candidate_losses.clone()
    adjusted[:, :, 1:] = adjusted[:, :, 1:] + float(baseline_margin)
    return torch.softmax(-adjusted / float(temperature), dim=-1)


def compute_expert_improvement_targets(candidate_losses, expert_improvement_margin=0.0):
    if candidate_losses.dim() != 3:
        raise ValueError("candidate_losses must be [B, N, E]")
    if candidate_losses.shape[-1] != 3:
        raise ValueError("candidate_losses must include [baseline, itsc, raft]")
    targets = torch.zeros_like(candidate_losses, dtype=torch.bool)
    targets[:, :, 0] = True
    baseline_loss = candidate_losses[:, :, 0:1]
    targets[:, :, 1:] = candidate_losses[:, :, 1:] + float(expert_improvement_margin) < baseline_loss
    return targets


def apply_joint_input_mask(
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
