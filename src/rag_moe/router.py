import torch
from torch import nn


class TwoStageRAGRouter(nn.Module):
    def __init__(self, num_candidates, input_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.num_candidates = num_candidates
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.selector = nn.Linear(hidden_dim, num_candidates)
        self.weighter = nn.Linear(hidden_dim, num_candidates)

    def forward(self, features, available):
        self._validate_inputs(features, available)

        encoded = self.encoder(features)
        select_logits = self.selector(encoded)
        select_prob = torch.sigmoid(select_logits)
        available_mask = available.to(dtype=torch.bool)

        active_mask = (select_prob >= 0.5) & available_mask
        if self.num_candidates > 0:
            active_mask = active_mask.clone()
            active_mask[:, :, 0] = active_mask[:, :, 0] | available_mask[:, :, 0]
            no_active = ~active_mask.any(dim=-1)
            baseline_fallback = no_active & available_mask[:, :, 0]
            active_mask[:, :, 0] = active_mask[:, :, 0] | baseline_fallback

        weight_logits = self.weighter(encoded)
        has_active = active_mask.any(dim=-1, keepdim=True)
        masked_logits = weight_logits.masked_fill(
            ~active_mask,
            torch.finfo(weight_logits.dtype).min,
        )
        masked_logits = torch.where(has_active, masked_logits, torch.zeros_like(masked_logits))
        weights = torch.softmax(masked_logits, dim=-1)
        weights = weights.masked_fill(~active_mask, 0.0)

        return {
            "select_logits": select_logits,
            "select_prob": select_prob,
            "weights": weights,
            "active_mask": active_mask,
        }

    def _validate_inputs(self, features, available):
        if not isinstance(features, torch.Tensor):
            raise ValueError("features must be a torch.Tensor")
        if not isinstance(available, torch.Tensor):
            raise ValueError("available must be a torch.Tensor")
        if features.dim() != 3:
            raise ValueError(
                "features shape %r must be [B, N, F]" % (tuple(features.shape),)
            )
        if available.dim() != 3:
            raise ValueError(
                "available shape %r must be [B, N, E]" % (tuple(available.shape),)
            )

        batch_size, node_count, feature_dim = features.shape
        available_batch, available_nodes, available_experts = available.shape
        if feature_dim != self.input_dim:
            raise ValueError(
                "feature dimension %r does not match input_dim %r"
                % (feature_dim, self.input_dim)
            )
        if available_batch != batch_size:
            raise ValueError(
                "available batch dimension %r does not match features batch %r"
                % (available_batch, batch_size)
            )
        if available_nodes != node_count:
            raise ValueError(
                "available node dimension %r does not match features nodes %r"
                % (available_nodes, node_count)
            )
        if available_experts != self.num_candidates:
            raise ValueError(
                "available expert dimension %r does not match num_candidates %r"
                % (available_experts, self.num_candidates)
            )
