import os
from collections.abc import Mapping

import torch

from src.rag_moe.config import load_yaml
from src.rag_moe.experts.base import (
    RAGCorrectionOutput,
    RAGExpertAdapter,
    validate_correction_output,
)
from src.rag_moe.experts.custom_features import (
    history_node_features,
    horizon_fraction,
    require_baseline_pred,
)
from src.rag_moe.experts.itsc import ITSCExpert


def _load_base_itsc_expert(config, data_context):
    if "base_expert" in config:
        return config["base_expert"]
    base_config_path = config.get("base_itsc_config", "")
    if not base_config_path:
        raise ValueError("ITSCSegmentGateExpert requires base_itsc_config or base_expert")
    expert_config = load_yaml(base_config_path)
    itsc_config = expert_config.get("experts", {}).get("itsc", {})
    if not itsc_config:
        raise ValueError("base_itsc_config must contain experts.itsc")
    return ITSCExpert().prepare(itsc_config, data_context)


class ITSCSegmentGate(torch.nn.Module):
    def __init__(self, feature_dim=6, hidden_dim=8):
        super().__init__()
        hidden_dim = int(hidden_dim)
        if hidden_dim > 0:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(feature_dim + 1, hidden_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_dim, 1),
            )
        else:
            self.net = torch.nn.Linear(feature_dim + 1, 1)
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                module.weight.data.mul_(1e-3)
                torch.nn.init.zeros_(module.bias)

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


class ITSCSegmentGateExpert(RAGExpertAdapter):
    name = "itsc_segment_gate"

    def prepare(self, config, data_context):
        config = config or {}
        self.output_len = config.get("output_len")
        self.output_dim = config.get("output_dim")
        self.hidden_dim = int(config.get("hidden_dim", 8))
        self.min_gate_available = float(config.get("min_gate_available", 0.0))
        self.data_context = dict(data_context or {})
        self.base_expert = _load_base_itsc_expert(config, data_context)
        self.gate = ITSCSegmentGate(feature_dim=6, hidden_dim=self.hidden_dim)

        checkpoint_path = config.get("checkpoint_path", "")
        has_checkpoint = bool(checkpoint_path)
        has_explicit_gate = "gate_scale" in config or "gate_bias" in config
        self.gate_mode = str(
            config.get(
                "gate_mode",
                "learned" if has_checkpoint or has_explicit_gate else "zero",
            )
        )
        if self.gate_mode not in ("zero", "learned"):
            raise ValueError(
                "ITSCSegmentGateExpert gate_mode must be 'zero' or 'learned', got %r"
                % (self.gate_mode,)
            )
        if self.gate_mode == "learned" and not has_checkpoint and not has_explicit_gate:
            raise ValueError(
                "ITSCSegmentGateExpert gate_mode='learned' requires checkpoint_path, gate_scale, or gate_bias"
            )
        self.gate_scale = float(config.get("gate_scale", 1.0 if has_checkpoint else 0.0))
        self.gate_bias = float(config.get("gate_bias", 0.0))
        self.checkpoint_load_info = None
        self.scalar_gate_checkpoint = False
        if checkpoint_path:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    "ITSCSegmentGateExpert checkpoint_path %s not found"
                    % (checkpoint_path,)
                )
            state = _load_gate_state(checkpoint_path)
            self.checkpoint_load_info = _load_exact_gate_state(
                self.gate,
                state,
                checkpoint_path,
            )
            if self.checkpoint_load_info.get("scalar_gate_state"):
                self.gate_scale = self.checkpoint_load_info["gate_scale"]
                self.gate_bias = self.checkpoint_load_info["gate_bias"]
                self.scalar_gate_checkpoint = True
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
        self._validate_requested_shape(batch_meta, baseline_pred)

        base_output = self.base_expert.forward_correction(
            history_data,
            supports,
            llm_encoding,
            batch_meta,
            baseline_pred,
        )
        validate_correction_output(base_output, baseline_pred, expected_name=base_output.name)

        gate = self._gate(history_data, baseline_pred)
        base_available = _normalize_available(
            base_output.available,
            baseline_pred,
            "base ITSC",
        )
        gate_mean = gate.mean(dim=1).squeeze(-1)
        if self.min_gate_available > 0.0:
            available = base_available & (gate_mean >= self.min_gate_available)
        else:
            available = base_available

        available_mask = available.view(
            available.shape[0],
            1,
            available.shape[1],
            1,
        )
        if available_mask.shape[2] == 1:
            available_mask = available_mask.expand(
                baseline_pred.shape[0],
                baseline_pred.shape[1],
                baseline_pred.shape[2],
                baseline_pred.shape[3],
            )
        else:
            available_mask = available_mask.expand(
                baseline_pred.shape[0],
                baseline_pred.shape[1],
                baseline_pred.shape[2],
                1,
            )

        delta = gate * base_output.delta.to(
            dtype=baseline_pred.dtype,
            device=baseline_pred.device,
        )
        delta = delta * available_mask.to(dtype=baseline_pred.dtype)
        raw_prior = baseline_pred + delta

        aux = {
            "correction_type": "itsc_segment_gate_residual",
            "gate_mode": self.gate_mode,
            "gate_scale": self.gate_scale,
            "gate_bias": self.gate_bias,
            "min_gate_available": self.min_gate_available,
        }
        base_aux = dict(base_output.aux or {})
        if "correction_type" in base_aux:
            aux["base_correction_type"] = base_aux["correction_type"]
        if self.checkpoint_load_info is not None:
            aux["checkpoint_load_info"] = self.checkpoint_load_info

        return RAGCorrectionOutput(
            name=self.name,
            delta=delta.to(dtype=baseline_pred.dtype, device=baseline_pred.device),
            available=available,
            raw_prior=raw_prior.to(dtype=baseline_pred.dtype, device=baseline_pred.device),
            confidence=gate_mean.to(dtype=baseline_pred.dtype, device=baseline_pred.device),
            aux=aux,
        )

    def _validate_requested_shape(self, batch_meta, baseline_pred):
        output_len = int(batch_meta.get("output_len") or self.output_len or baseline_pred.shape[1])
        output_dim = int(batch_meta.get("output_dim") or self.output_dim or baseline_pred.shape[-1])
        if output_len != baseline_pred.shape[1]:
            raise ValueError(
                "ITSCSegmentGateExpert output_len %r does not match baseline horizon %r"
                % (output_len, baseline_pred.shape[1])
            )
        if output_dim != baseline_pred.shape[-1]:
            raise ValueError(
                "ITSCSegmentGateExpert output_dim %r does not match baseline output_dim %r"
                % (output_dim, baseline_pred.shape[-1])
            )

    def _gate(self, history_data, baseline_pred):
        if self.gate_mode == "zero":
            return torch.zeros_like(baseline_pred)
        if self.scalar_gate_checkpoint:
            gate = torch.sigmoid(
                torch.as_tensor(
                    self.gate_bias,
                    dtype=baseline_pred.dtype,
                    device=baseline_pred.device,
                )
            )
            return torch.full_like(baseline_pred, gate.item())
        features = history_node_features(history_data).to(
            dtype=baseline_pred.dtype,
            device=baseline_pred.device,
        )
        gate_param = next(self.gate.parameters())
        model_features = features.to(dtype=gate_param.dtype, device=gate_param.device)
        model_horizon = horizon_fraction(baseline_pred).to(
            dtype=gate_param.dtype,
            device=gate_param.device,
        )
        logits = self.gate(model_features, model_horizon)
        logits = self.gate_scale * logits + self.gate_bias
        gate = torch.sigmoid(logits).to(dtype=baseline_pred.dtype, device=baseline_pred.device)
        return gate.expand_as(baseline_pred)


def _load_gate_state(checkpoint_path):
    try:
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        try:
            loaded = torch.load(checkpoint_path, map_location="cpu")
        except Exception as exc:
            raise RuntimeError(
                "ITSCSegmentGateExpert checkpoint_path %s failed to load"
                % (checkpoint_path,)
            ) from exc
    except Exception as exc:
        raise RuntimeError(
            "ITSCSegmentGateExpert checkpoint_path %s failed to load"
            % (checkpoint_path,)
        ) from exc

    if isinstance(loaded, Mapping) and "state_dict" in loaded:
        if not isinstance(loaded["state_dict"], Mapping):
            raise ValueError(
                "ITSCSegmentGateExpert checkpoint_path %s key state_dict must be a mapping"
                % (checkpoint_path,)
            )
        loaded = loaded["state_dict"]
    elif isinstance(loaded, Mapping) and "model_state_dict" in loaded:
        if not isinstance(loaded["model_state_dict"], Mapping):
            raise ValueError(
                "ITSCSegmentGateExpert checkpoint_path %s key model_state_dict must be a mapping"
                % (checkpoint_path,)
            )
        loaded = loaded["model_state_dict"]
    if not isinstance(loaded, Mapping):
        raise ValueError(
            "ITSCSegmentGateExpert checkpoint_path %s must be a mapping state dict"
            % (checkpoint_path,)
        )
    return loaded


def _normalize_available(available, reference, source_name):
    available = available.to(device=reference.device, dtype=torch.bool)
    if available.shape[1] == reference.shape[2]:
        return available
    if available.shape[1] == 1:
        return available.expand(reference.shape[0], reference.shape[2])
    raise ValueError(
        "ITSCSegmentGateExpert %s available node dimension %r must match baseline nodes %r or be 1"
        % (source_name, available.shape[1], reference.shape[2])
    )


def _load_exact_gate_state(module, state, checkpoint_path):
    module_keys = set(module.state_dict().keys())
    cleaned = _exact_state_candidate(dict(state), module_keys)
    if cleaned is None:
        scalar_gate = _scalar_gate_state_candidate(state, checkpoint_path)
        if scalar_gate is not None:
            return scalar_gate
        state_keys = set(state.keys())
        missing_keys = sorted(module_keys - state_keys, key=str)
        unexpected_keys = sorted(state_keys - module_keys, key=str)
        raise ValueError(
            "ITSCSegmentGateExpert checkpoint_path %s gate keys must match exactly; "
            "missing_keys=%r unexpected_keys=%r"
            % (checkpoint_path, missing_keys, unexpected_keys)
        )
    try:
        module.load_state_dict(cleaned, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "ITSCSegmentGateExpert checkpoint_path %s has incompatible gate state"
            % (checkpoint_path,)
        ) from exc
    return {"missing_keys": [], "unexpected_keys": []}


def _exact_state_candidate(state, expected_keys):
    if set(state.keys()) == expected_keys:
        return state
    return None


def _scalar_gate_state_candidate(state, checkpoint_path):
    allowed = {"gate_scale", "gate_bias"}
    keys = set(state.keys())
    if not keys <= allowed or "gate_scale" not in keys:
        return None
    gate_scale = _scalar_tensor_value(state["gate_scale"], checkpoint_path, "gate_scale")
    gate_bias = _scalar_tensor_value(state.get("gate_bias", torch.tensor([0.0])), checkpoint_path, "gate_bias")
    return {
        "missing_keys": [],
        "unexpected_keys": [],
        "scalar_gate_state": True,
        "gate_scale": gate_scale,
        "gate_bias": gate_bias,
    }


def _scalar_tensor_value(value, checkpoint_path, key):
    tensor = torch.as_tensor(value)
    if tensor.numel() != 1:
        raise ValueError(
            "ITSCSegmentGateExpert checkpoint_path %s key %s must be scalar, got shape %r"
            % (checkpoint_path, key, tuple(tensor.shape))
        )
    return float(tensor.reshape(-1)[0].item())
