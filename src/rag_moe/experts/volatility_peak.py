import os
from collections.abc import Mapping

import torch

from src.rag_moe.experts.base import RAGCorrectionOutput, RAGExpertAdapter
from src.rag_moe.experts.custom_features import (
    all_available,
    history_node_features,
    horizon_fraction,
    output_dim_from,
    output_len_from,
    require_baseline_pred,
)


class VolatilityPeakResidual(torch.nn.Module):
    def __init__(self, feature_dim=6, hidden_dim=8, output_len=24):
        super().__init__()
        self.output_len = int(output_len)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim + 1, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 1),
        )
        torch.nn.init.xavier_uniform_(self.net[0].weight, gain=0.05)
        torch.nn.init.zeros_(self.net[0].bias)
        torch.nn.init.normal_(self.net[2].weight, mean=0.0, std=1.0e-3)
        torch.nn.init.zeros_(self.net[2].bias)

    def forward(self, features, horizon_frac):
        batch_size, node_count, feature_dim = features.shape
        horizon = horizon_frac.shape[1]
        expanded_features = features.unsqueeze(1).expand(
            batch_size,
            horizon,
            node_count,
            feature_dim,
        )
        expanded_horizon = horizon_frac.expand(batch_size, horizon, node_count, 1)
        return self.net(torch.cat([expanded_features, expanded_horizon], dim=-1))


class VolatilityPeakExpert(RAGExpertAdapter):
    name = "volatility_peak"

    def prepare(self, config, data_context):
        config = config or {}
        self.mode = str(config.get("mode", "identity"))
        self.output_len = config.get("output_len")
        self.output_dim = config.get("output_dim")
        self.hidden_dim = int(config.get("hidden_dim", 8))
        self.max_abs_delta = float(config.get("max_abs_delta", 1.0))
        self.min_history_std = float(config.get("min_history_std", 0.0))
        self.min_history_max = float(config.get("min_history_max", 0.0))
        self.data_context = dict(data_context or {})
        self.legacy_affine_noop = False
        module_output_len = int(self.output_len or 24)
        self.residual = VolatilityPeakResidual(
            feature_dim=6,
            hidden_dim=self.hidden_dim,
            output_len=module_output_len,
        )
        checkpoint_path = config.get("checkpoint_path", "")
        self.checkpoint_load_info = None
        if checkpoint_path:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    "VolatilityPeakExpert checkpoint_path %s not found"
                    % (checkpoint_path,)
                )
            state = _load_residual_state(checkpoint_path)
            self.checkpoint_load_info = _load_state_with_prefix_cleanup(
                self.residual,
                state,
                checkpoint_path,
            )
            self.legacy_affine_noop = bool(
                self.checkpoint_load_info.get("legacy_affine_artifact", False)
            )
        return self.freeze()

    @torch.no_grad()
    def forward_correction(
        self,
        history_data,
        supports=None,
        llm_encoding=None,
        batch_meta=None,
        baseline_pred=None,
    ):
        if (
            baseline_pred is None
            and isinstance(batch_meta, Mapping)
            and "baseline_pred" in batch_meta
        ):
            baseline_pred = batch_meta["baseline_pred"]
        baseline_pred = require_baseline_pred(self.name, baseline_pred)
        batch_meta = batch_meta or {}
        output_len = output_len_from(baseline_pred, batch_meta, self.output_len)
        output_dim = output_dim_from(baseline_pred, batch_meta, self.output_dim)
        if output_len != baseline_pred.shape[1]:
            raise ValueError(
                "VolatilityPeakExpert output_len %r does not match baseline horizon %r"
                % (output_len, baseline_pred.shape[1])
            )
        if output_dim != baseline_pred.shape[-1]:
            raise ValueError(
                "VolatilityPeakExpert output_dim %r does not match baseline output_dim %r"
                % (output_dim, baseline_pred.shape[-1])
            )

        features = history_node_features(history_data).to(
            dtype=baseline_pred.dtype,
            device=baseline_pred.device,
        )
        available = self._available(features, baseline_pred)
        if self.mode == "identity" or self.legacy_affine_noop:
            delta = torch.zeros_like(baseline_pred)
        else:
            residual_param = next(self.residual.parameters())
            residual_dtype = residual_param.dtype
            residual_device = residual_param.device
            model_features = features.to(dtype=residual_dtype, device=residual_device)
            model_horizon = horizon_fraction(baseline_pred).to(
                dtype=residual_dtype,
                device=residual_device,
            )
            raw_delta = self.residual(model_features, model_horizon)
            delta = self.max_abs_delta * torch.tanh(raw_delta)
            delta = delta.to(dtype=baseline_pred.dtype, device=baseline_pred.device)
            delta = delta.expand_as(baseline_pred)
        delta = delta * available.view(
            baseline_pred.shape[0],
            1,
            baseline_pred.shape[2],
            1,
        ).to(dtype=baseline_pred.dtype)
        raw_prior = baseline_pred + delta
        confidence = available.to(dtype=baseline_pred.dtype, device=baseline_pred.device)
        aux = {
            "correction_type": "volatility_peak_residual",
            "mode": self.mode,
            "max_abs_delta": self.max_abs_delta,
            "min_history_std": self.min_history_std,
            "min_history_max": self.min_history_max,
        }
        if self.checkpoint_load_info is not None:
            aux["checkpoint_load_info"] = self.checkpoint_load_info
        return RAGCorrectionOutput(
            name=self.name,
            delta=delta.to(dtype=baseline_pred.dtype, device=baseline_pred.device),
            available=available,
            raw_prior=raw_prior.to(dtype=baseline_pred.dtype, device=baseline_pred.device),
            confidence=confidence,
            aux=aux,
        )

    def _available(self, features, reference):
        if self.min_history_std <= 0.0 and self.min_history_max <= 0.0:
            return all_available(reference)
        available = torch.ones(
            features.shape[0],
            features.shape[1],
            dtype=torch.bool,
            device=reference.device,
        )
        if self.min_history_std > 0.0:
            available = available & (features[:, :, 1] >= self.min_history_std)
        if self.min_history_max > 0.0:
            available = available & (features[:, :, 2] >= self.min_history_max)
        return available


def _load_residual_state(checkpoint_path):
    try:
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        try:
            loaded = torch.load(checkpoint_path, map_location="cpu")
        except Exception as exc:
            raise RuntimeError(
                "VolatilityPeakExpert checkpoint_path %s failed to load"
                % (checkpoint_path,)
            ) from exc
    except Exception as exc:
        raise RuntimeError(
            "VolatilityPeakExpert checkpoint_path %s failed to load"
            % (checkpoint_path,)
        ) from exc

    if isinstance(loaded, Mapping) and "state_dict" in loaded:
        if not isinstance(loaded["state_dict"], Mapping):
            raise ValueError(
                "VolatilityPeakExpert checkpoint_path %s key state_dict must be a mapping"
                % (checkpoint_path,)
            )
        loaded = loaded["state_dict"]
    elif isinstance(loaded, Mapping) and "model_state_dict" in loaded:
        if not isinstance(loaded["model_state_dict"], Mapping):
            raise ValueError(
                "VolatilityPeakExpert checkpoint_path %s key model_state_dict must be a mapping"
                % (checkpoint_path,)
            )
        loaded = loaded["model_state_dict"]
    if not isinstance(loaded, Mapping):
        raise ValueError(
            "VolatilityPeakExpert checkpoint_path %s must be a mapping state dict"
            % (checkpoint_path,)
        )
    return loaded


def _load_state_with_prefix_cleanup(module, state, checkpoint_path):
    module_keys = set(module.state_dict().keys())
    cleaned = _exact_state_candidate(dict(state), module_keys)
    if cleaned is None and _is_legacy_affine_artifact(state):
        return {
            "missing_keys": [],
            "unexpected_keys": [],
            "legacy_affine_artifact": True,
            "ignored_keys": sorted(str(key) for key in state.keys()),
        }
    if cleaned is None:
        state_keys = set(state.keys())
        missing_keys = sorted(module_keys - state_keys, key=str)
        unexpected_keys = sorted(state_keys - module_keys, key=str)
        raise ValueError(
            "VolatilityPeakExpert checkpoint_path %s residual keys must match exactly; "
            "missing_keys=%r unexpected_keys=%r"
            % (checkpoint_path, missing_keys, unexpected_keys)
        )
    try:
        module.load_state_dict(cleaned, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "VolatilityPeakExpert checkpoint_path %s has incompatible residual state"
            % (checkpoint_path,)
        ) from exc
    return {"missing_keys": [], "unexpected_keys": []}


def _is_legacy_affine_artifact(state):
    keys = set(state.keys())
    if not keys or not keys.issubset({"scale", "bias"}):
        return False
    for key in keys:
        try:
            value = torch.as_tensor(state[key])
        except Exception:
            return False
        if value.numel() != 1:
            return False
    return True


def _exact_state_candidate(state, expected_keys):
    if set(state.keys()) == expected_keys:
        return state
    for prefix in ("module.", "model."):
        if not all(isinstance(key, str) and key.startswith(prefix) for key in state):
            continue
        stripped = {key[len(prefix):]: value for key, value in state.items()}
        if set(stripped.keys()) == expected_keys:
            return stripped
    return None
