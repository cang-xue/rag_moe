from typing import Iterable, Tuple

import torch
import torch.nn.functional as F


def configure_router_dpo_tuning(router, train_scope="heads"):
    if train_scope not in {"heads", "all"}:
        raise ValueError("train_scope must be 'heads' or 'all'")

    for parameter in router.parameters():
        parameter.requires_grad_(train_scope == "all")

    if train_scope == "heads":
        for module in (router.selector, router.weighter):
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    return [
        (name, parameter)
        for name, parameter in router.named_parameters()
        if parameter.requires_grad
    ]


def dpo_preference_loss(
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    beta: float = 0.1,
    weight: torch.Tensor = None,
) -> torch.Tensor:
    if policy_logits.shape != reference_logits.shape:
        raise ValueError("policy_logits and reference_logits must have the same shape")
    if policy_logits.dim() != 2:
        raise ValueError("logits must be [P, E]")
    chosen = chosen.to(device=policy_logits.device, dtype=torch.long)
    rejected = rejected.to(device=policy_logits.device, dtype=torch.long)
    policy_logp = F.log_softmax(policy_logits, dim=-1)
    reference_logp = F.log_softmax(reference_logits.to(policy_logits.device), dim=-1)
    row_index = torch.arange(policy_logits.shape[0], device=policy_logits.device)
    policy_margin = policy_logp[row_index, chosen] - policy_logp[row_index, rejected]
    reference_margin = reference_logp[row_index, chosen] - reference_logp[row_index, rejected]
    losses = -F.logsigmoid(float(beta) * (policy_margin - reference_margin))
    if weight is not None:
        pair_weight = weight.to(device=policy_logits.device, dtype=losses.dtype)
        losses = losses * pair_weight
        return losses.sum() / pair_weight.sum().clamp_min(1e-6)
    return losses.mean()


def guarded_hard_selection(
    probabilities: torch.Tensor,
    none_index: int = 0,
    min_best_prob: float = 0.5,
    min_margin_over_none: float = 0.0,
) -> torch.Tensor:
    if probabilities.dim() != 3:
        raise ValueError("probabilities must be [B, N, E]")
    if not 0 <= int(none_index) < probabilities.shape[-1]:
        raise ValueError("none_index is out of range")
    best_prob, best_index = probabilities.max(dim=-1)
    none_prob = probabilities[..., int(none_index)]
    confident = best_prob >= float(min_best_prob)
    beats_none = (best_prob - none_prob) >= float(min_margin_over_none)
    use_best = confident & beats_none
    return torch.where(use_best, best_index, torch.full_like(best_index, int(none_index)))

