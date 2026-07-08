from typing import Dict

import torch


def _node_summary(tensor: torch.Tensor) -> torch.Tensor:
    mean = tensor.mean(dim=1).squeeze(-1)
    std = tensor.std(dim=1, unbiased=False).squeeze(-1)
    last = tensor[:, -1].squeeze(-1)
    return torch.stack([mean, std, last], dim=-1)


def build_router_features(
    history_data: torch.Tensor,
    baseline_pred: torch.Tensor,
    expert_deltas: Dict[str, torch.Tensor],
) -> torch.Tensor:
    history_value = history_data[..., :1]
    features = [_node_summary(history_value)]

    baseline_mean = baseline_pred.mean(dim=1).squeeze(-1).unsqueeze(-1)
    baseline_std = baseline_pred.std(dim=1, unbiased=False).squeeze(-1).unsqueeze(-1)
    features.extend([baseline_mean, baseline_std])

    for name in expert_deltas:
        delta = expert_deltas[name].abs().mean(dim=1).squeeze(-1).unsqueeze(-1)
        features.append(delta)

    return torch.cat(features, dim=-1)
