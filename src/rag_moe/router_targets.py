import torch


def build_good_expert_targets(
    candidate_errors: torch.Tensor,
    available: torch.Tensor | None = None,
    oracle_margin: float = 0.98,
    none_index: int = 0,
) -> torch.Tensor:
    if candidate_errors.dim() != 3:
        raise ValueError("candidate_errors must be [B, N, E]")
    if not 0 <= int(none_index) < candidate_errors.shape[-1]:
        raise ValueError("none_index is out of range")

    if available is None:
        available_mask = torch.ones_like(candidate_errors, dtype=torch.bool)
    else:
        if available.shape != candidate_errors.shape:
            raise ValueError("available must match candidate_errors shape")
        available_mask = available.bool()

    baseline_error = candidate_errors[..., none_index:none_index + 1]
    good = candidate_errors <= baseline_error * float(oracle_margin)
    good = good & available_mask

    # Keep the baseline fallback trainable even when oracle_margin < 1.
    good[..., none_index] = available_mask[..., none_index]
    return good
