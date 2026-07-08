from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class DPOPairBatch:
    chosen: torch.Tensor
    rejected: torch.Tensor
    batch_index: torch.Tensor
    node_index: torch.Tensor
    weight: torch.Tensor

    def __len__(self):
        return int(self.chosen.numel())


def compute_candidate_node_errors(candidates: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if candidates.dim() != 5:
        raise ValueError("candidates must be [B, T, E, N, C]")
    if labels.dim() != 4:
        raise ValueError("labels must be [B, T, N, C]")
    if candidates.shape[0] != labels.shape[0] or candidates.shape[1] != labels.shape[1]:
        raise ValueError("candidate batch/horizon dimensions must match labels")
    if candidates.shape[3] != labels.shape[2] or candidates.shape[4] != labels.shape[3]:
        raise ValueError("candidate node/channel dimensions must match labels")
    label_expanded = labels.unsqueeze(2)
    return (candidates - label_expanded).abs().mean(dim=(1, 4)).permute(0, 2, 1)


def build_dpo_pairs(
    candidate_errors: torch.Tensor,
    available: torch.Tensor,
    rel_margin: float = 0.01,
    abs_margin: float = 0.01,
    none_index: int = 0,
    eps: float = 1e-6,
) -> DPOPairBatch:
    if candidate_errors.dim() != 3:
        raise ValueError("candidate_errors must be [B, N, E]")
    if available.shape != candidate_errors.shape:
        raise ValueError("available must match candidate_errors shape")
    if not 0 <= int(none_index) < candidate_errors.shape[-1]:
        raise ValueError("none_index is out of range")

    errors = candidate_errors.detach().cpu()
    mask = available.detach().cpu().bool()
    chosen = []
    rejected = []
    batch_index = []
    node_index = []
    weights = []

    batch_count, node_count, expert_count = errors.shape
    for batch_id in range(batch_count):
        for node_id in range(node_count):
            if not bool(mask[batch_id, node_id, none_index]):
                continue
            valid = [idx for idx in range(expert_count) if bool(mask[batch_id, node_id, idx])]
            non_none = [idx for idx in valid if idx != none_index]
            if not non_none:
                continue

            baseline_error = errors[batch_id, node_id, none_index]
            non_none_errors = errors[batch_id, node_id, non_none]
            best_offset = int(torch.argmin(non_none_errors).item())
            best_index = int(non_none[best_offset])
            best_error = errors[batch_id, node_id, best_index]
            absolute_gain = baseline_error - best_error
            relative_gain = absolute_gain / (baseline_error.abs() + float(eps))

            if float(relative_gain) >= float(rel_margin) and float(absolute_gain) >= float(abs_margin):
                chosen_index = best_index
                rejected_index = none_index
                pair_weight = max(float(relative_gain), float(eps))
            else:
                chosen_index = none_index
                rejected_index = best_index
                pair_weight = 1.0

            chosen.append(chosen_index)
            rejected.append(rejected_index)
            batch_index.append(batch_id)
            node_index.append(node_id)
            weights.append(pair_weight)

    return DPOPairBatch(
        chosen=torch.as_tensor(chosen, dtype=torch.long),
        rejected=torch.as_tensor(rejected, dtype=torch.long),
        batch_index=torch.as_tensor(batch_index, dtype=torch.long),
        node_index=torch.as_tensor(node_index, dtype=torch.long),
        weight=torch.as_tensor(weights, dtype=torch.float32),
    )

