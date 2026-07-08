import torch
from torch import nn

from src.utils.metrics import masked_mae


class HorizonAffineCalibrator(nn.Module):
    def __init__(self, horizon, output_dim=1):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, int(horizon), 1, int(output_dim)))
        self.bias = nn.Parameter(torch.zeros(1, int(horizon), 1, int(output_dim)))

    def forward(self, preds):
        return preds * self.scale + self.bias


class SharedAffineCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, preds):
        return preds * self.scale + self.bias


class GlobalSoftmaxMixer(nn.Module):
    def __init__(self, num_candidates):
        super().__init__()
        if int(num_candidates) <= 0:
            raise ValueError("num_candidates must be positive")
        self.logits = nn.Parameter(torch.zeros(int(num_candidates)))

    def forward(self, candidates):
        weights = torch.softmax(self.logits, dim=0).view(1, 1, -1, 1, 1)
        return (candidates * weights).sum(dim=2)

    def weights(self):
        return torch.softmax(self.logits.detach(), dim=0)


def add_negative_residual_candidate(candidates, names, source_name, negative_name):
    if "none" not in names:
        raise ValueError("names must include 'none' baseline candidate")
    if source_name not in names:
        raise ValueError("source candidate %r is missing" % (source_name,))
    baseline_idx = names.index("none")
    source_idx = names.index(source_name)
    baseline = candidates[:, :, baseline_idx:baseline_idx + 1]
    source = candidates[:, :, source_idx:source_idx + 1]
    negative = 2.0 * baseline - source
    return torch.cat([candidates, negative], dim=2), list(names) + [negative_name]


def apply_source_robust_baseline_gate(mixer, splits, null_value, margin=0.0):
    kept = True
    with torch.no_grad():
        for candidates, labels in splits:
            mixed_loss = masked_mae(mixer(candidates), labels, null_value).item()
            baseline_loss = masked_mae(candidates[:, :, 0], labels, null_value).item()
            if mixed_loss > baseline_loss - float(margin):
                kept = False
                break
        if not kept:
            mixer.logits.fill_(-20.0)
            mixer.logits[0] = 20.0
    return kept


def fit_horizon_source_consistent_selector(splits, names, null_value, margin=0.0):
    if not splits:
        raise ValueError("splits must not be empty")
    if "none" not in names:
        raise ValueError("names must include 'none' baseline candidate")
    baseline_idx = names.index("none")
    num_horizons = int(splits[0][0].shape[1])
    num_candidates = int(splits[0][0].shape[2])
    if num_candidates != len(names):
        raise ValueError("candidate axis does not match names")

    selected_indices = []
    horizon_summary = []
    with torch.no_grad():
        for horizon_idx in range(num_horizons):
            best_idx = baseline_idx
            best_mean_gain = 0.0
            candidate_rows = []
            for candidate_idx, candidate_name in enumerate(names):
                gains = []
                split_losses = []
                for candidates, labels in splits:
                    baseline_loss = masked_mae(
                        candidates[:, horizon_idx, baseline_idx],
                        labels[:, horizon_idx],
                        null_value,
                    ).item()
                    candidate_loss = masked_mae(
                        candidates[:, horizon_idx, candidate_idx],
                        labels[:, horizon_idx],
                        null_value,
                    ).item()
                    gains.append(baseline_loss - candidate_loss)
                    split_losses.append(candidate_loss)
                min_gain = min(gains)
                mean_gain = sum(gains) / len(gains)
                candidate_rows.append(
                    {
                        "name": candidate_name,
                        "min_gain": float(min_gain),
                        "mean_gain": float(mean_gain),
                        "split_losses": [float(value) for value in split_losses],
                    }
                )
                if candidate_idx != baseline_idx and min_gain >= float(margin) and mean_gain > best_mean_gain:
                    best_idx = candidate_idx
                    best_mean_gain = mean_gain
            selected_indices.append(best_idx)
            horizon_summary.append(
                {
                    "horizon": horizon_idx + 1,
                    "selected_index": int(best_idx),
                    "selected_name": names[best_idx],
                    "selected_mean_gain": float(best_mean_gain),
                    "candidates": candidate_rows,
                }
            )
    return selected_indices, horizon_summary


def make_temporal_source_splits(splits, num_blocks):
    num_blocks = int(num_blocks)
    if num_blocks <= 1:
        return list(splits)
    if not splits:
        raise ValueError("splits must not be empty")

    candidates = torch.cat([split_candidates for split_candidates, _ in splits], dim=0)
    labels = torch.cat([split_labels for _, split_labels in splits], dim=0)
    if candidates.shape[0] != labels.shape[0]:
        raise ValueError("candidate and label batch dimensions must match")
    if candidates.shape[0] < num_blocks:
        raise ValueError("num_blocks cannot exceed the number of source samples")

    candidate_blocks = torch.tensor_split(candidates, num_blocks, dim=0)
    label_blocks = torch.tensor_split(labels, num_blocks, dim=0)
    return [
        (candidate_block, label_block)
        for candidate_block, label_block in zip(candidate_blocks, label_blocks)
        if int(candidate_block.shape[0]) > 0
    ]


def apply_residual_magnitude_gate(baseline, candidate, max_abs_delta):
    delta = candidate - baseline
    use_candidate = delta.abs() <= float(max_abs_delta)
    return torch.where(use_candidate, candidate, baseline)


def fit_residual_magnitude_gate(splits, candidate_index, null_value, thresholds, margin=0.0):
    if not splits:
        raise ValueError("splits must not be empty")
    candidate_index = int(candidate_index)
    best_threshold = 0.0
    best_mean_gain = 0.0
    rows = []

    with torch.no_grad():
        for threshold in thresholds:
            threshold = float(threshold)
            gains = []
            split_losses = []
            for candidates, labels in splits:
                baseline = candidates[:, :, 0]
                candidate = candidates[:, :, candidate_index]
                gated = apply_residual_magnitude_gate(baseline, candidate, threshold)
                baseline_loss = masked_mae(baseline, labels, null_value).item()
                gated_loss = masked_mae(gated, labels, null_value).item()
                gains.append(baseline_loss - gated_loss)
                split_losses.append(gated_loss)
            min_gain = min(gains)
            mean_gain = sum(gains) / len(gains)
            rows.append(
                {
                    "threshold": threshold,
                    "min_gain": float(min_gain),
                    "mean_gain": float(mean_gain),
                    "split_losses": [float(value) for value in split_losses],
                }
            )
            if min_gain >= float(margin) and mean_gain > best_mean_gain:
                best_threshold = threshold
                best_mean_gain = mean_gain

    return best_threshold, {
        "kept": bool(best_threshold > 0.0),
        "threshold": float(best_threshold),
        "selected_mean_gain": float(best_mean_gain),
        "candidates": rows,
    }


def apply_horizon_candidate_selector(candidates, selected_indices):
    if candidates.dim() != 5:
        raise ValueError("candidates must have shape [B, H, C, N, D]")
    if len(selected_indices) != int(candidates.shape[1]):
        raise ValueError("selected_indices length must match horizon dimension")
    selected = []
    for horizon_idx, candidate_idx in enumerate(selected_indices):
        selected.append(candidates[:, horizon_idx, int(candidate_idx)])
    return torch.stack(selected, dim=1)
