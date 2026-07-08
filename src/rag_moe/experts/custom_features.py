import math

import torch


def require_baseline_pred(expert_name, baseline_pred):
    if baseline_pred is None:
        raise ValueError(f"{expert_name} requires baseline_pred for residual correction")
    if baseline_pred.dim() != 4:
        raise ValueError(f"{expert_name} baseline_pred must be [B, T, N, C]")
    return baseline_pred


def output_len_from(reference, batch_meta, configured_output_len=None):
    batch_meta = batch_meta or {}
    return int(batch_meta.get("output_len") or configured_output_len or reference.shape[1])


def output_dim_from(reference, batch_meta, configured_output_dim=None):
    batch_meta = batch_meta or {}
    return int(batch_meta.get("output_dim") or configured_output_dim or reference.shape[-1])


def all_available(reference):
    return torch.ones(reference.shape[0], reference.shape[2], dtype=torch.bool, device=reference.device)


def none_available(reference):
    return torch.zeros(reference.shape[0], reference.shape[2], dtype=torch.bool, device=reference.device)


def horizon_parameter(value, reference, name):
    tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if tensor.numel() == 1:
        return tensor.reshape(1, 1, 1, 1)
    if tensor.numel() != reference.shape[1]:
        raise ValueError(f"{name} length {tensor.numel()} does not match horizon {reference.shape[1]}")
    return tensor.reshape(1, reference.shape[1], 1, 1)


def history_node_features(history_data, eps=1e-6):
    values = history_data[:, :, :, :1]
    mean = values.mean(dim=1)
    std = values.std(dim=1, unbiased=False)
    max_value = values.max(dim=1).values
    last = values[:, -1]
    half = max(int(values.shape[1] // 2), 1)
    recent = values[:, -half:].mean(dim=1)
    early = values[:, :half].mean(dim=1)
    slope = recent - early
    zero_ratio = (values.abs() <= eps).to(values.dtype).mean(dim=1)
    return torch.cat([mean, std, max_value, last, slope, zero_ratio], dim=-1)


def horizon_fraction(reference):
    horizon = reference.shape[1]
    if horizon == 0:
        return torch.zeros(1, 0, 1, 1, dtype=reference.dtype, device=reference.device)
    if horizon == 1:
        return torch.zeros(1, 1, 1, 1, dtype=reference.dtype, device=reference.device)
    return torch.linspace(0.0, 1.0, horizon, dtype=reference.dtype, device=reference.device).view(1, horizon, 1, 1)


def normalized_entropy(weights):
    if torch.any(weights < 0):
        raise ValueError("normalized_entropy weights must be non-negative")
    if weights.shape[-1] <= 1:
        return torch.zeros(weights.shape[:-1], dtype=weights.dtype, device=weights.device)
    entropy_dtype = torch.float32 if weights.dtype in (torch.float16, torch.bfloat16) else weights.dtype
    working = weights.to(entropy_dtype)
    total = working.sum(dim=-1, keepdim=True)
    probabilities = torch.where(total > 0, working / total.clamp_min(torch.finfo(entropy_dtype).tiny), torch.zeros_like(working))
    entropy_terms = torch.zeros_like(probabilities)
    positive = probabilities > 0
    entropy_terms[positive] = probabilities[positive] * probabilities[positive].log()
    entropy = -entropy_terms.sum(dim=-1) / math.log(weights.shape[-1])
    return entropy.to(dtype=weights.dtype)
