import os
from collections.abc import Mapping

import torch

from src.rag_moe.experts.base import RAGCorrectionOutput, RAGExpertAdapter
from src.rag_moe.experts.custom_features import (
    all_available,
    horizon_parameter,
    require_baseline_pred,
)


def _load_calibration_state(checkpoint_path):
    try:
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        try:
            loaded = torch.load(checkpoint_path, map_location="cpu")
        except Exception as exc:
            raise RuntimeError(
                "CalibrationResidualExpert checkpoint_path %s failed to load"
                % (checkpoint_path,)
            ) from exc
    except Exception as exc:
        raise RuntimeError(
            "CalibrationResidualExpert checkpoint_path %s failed to load"
            % (checkpoint_path,)
        ) from exc

    if isinstance(loaded, Mapping) and "state_dict" in loaded:
        if not isinstance(loaded["state_dict"], Mapping):
            raise ValueError(
                "CalibrationResidualExpert checkpoint_path %s key state_dict "
                "must be a mapping"
                % (checkpoint_path,)
            )
        loaded = loaded["state_dict"]
    elif isinstance(loaded, Mapping) and "model_state_dict" in loaded:
        if not isinstance(loaded["model_state_dict"], Mapping):
            raise ValueError(
                "CalibrationResidualExpert checkpoint_path %s key "
                "model_state_dict must be a mapping"
                % (checkpoint_path,)
            )
        loaded = loaded["model_state_dict"]
    if not isinstance(loaded, Mapping):
        raise ValueError(
            "CalibrationResidualExpert checkpoint_path %s must be a mapping "
            "with optional scale/bias"
            % (checkpoint_path,)
        )
    if "scale" not in loaded and "bias" not in loaded:
        raise ValueError(
            "CalibrationResidualExpert checkpoint_path %s expected at least "
            "one calibration key: scale or bias"
            % (checkpoint_path,)
        )
    return loaded


def _set_parameter(module, name, value, checkpoint_path=None):
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32).clone()
    except Exception as exc:
        if checkpoint_path:
            raise ValueError(
                "CalibrationResidualExpert checkpoint_path %s invalid %s value"
                % (checkpoint_path, name)
            ) from exc
        raise ValueError(
            "CalibrationResidualExpert invalid %s value" % (name,)
        ) from exc
    if name in module._buffers:
        del module._buffers[name]
    setattr(module, name, torch.nn.Parameter(tensor))


class CalibrationResidualExpert(RAGExpertAdapter):
    name = "calibration"

    def prepare(self, config, data_context):
        config = config or {}
        self.data_context = dict(data_context or {})
        self.mode = str(config.get("mode", "identity"))
        self.output_len = config.get("output_len")
        self.output_dim = config.get("output_dim")
        scale = config.get("scale", 1.0)
        bias = config.get("bias", 0.0)
        checkpoint_path = config.get("checkpoint_path", "")
        if checkpoint_path:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    "CalibrationResidualExpert checkpoint_path %s not found"
                    % (checkpoint_path,)
                )
            state = _load_calibration_state(checkpoint_path)
            scale = state.get("scale", scale)
            bias = state.get("bias", bias)
        _set_parameter(self, "scale", scale, checkpoint_path or None)
        _set_parameter(self, "bias", bias, checkpoint_path or None)
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
        scale = horizon_parameter(self.scale, baseline_pred, "calibration scale")
        bias = horizon_parameter(self.bias, baseline_pred, "calibration bias")
        calibrated = baseline_pred * scale + bias
        delta = calibrated - baseline_pred
        return RAGCorrectionOutput(
            name=self.name,
            delta=delta,
            available=all_available(baseline_pred),
            raw_prior=calibrated,
            confidence=torch.ones(
                baseline_pred.shape[0],
                baseline_pred.shape[2],
                dtype=baseline_pred.dtype,
                device=baseline_pred.device,
            ),
            aux={"correction_type": "calibration_affine_residual"},
        )
