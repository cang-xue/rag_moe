import json
import os
import sys
from types import SimpleNamespace

import torch
from torch import nn

from src.rag_moe.full_model_utils import (
    extract_state_dict,
    load_torch_artifact,
    require_artifact,
)


class TPBCompatModel(nn.Module):
    def __init__(self, pred_num, his_num, output_dim=1, **_):
        super().__init__()
        self.pred_num = int(pred_num)
        self.his_num = int(his_num)
        self.output_dim = int(output_dim)
        self.bias = nn.Parameter(torch.zeros(self.output_dim))

    def forward(self, data_i, A=None, stage="test"):
        _ = A, stage
        x = data_i.x
        if x.dim() != 4:
            raise ValueError("TPBCompatModel expected data_i.x [B,N,T,C], got %s" % (tuple(x.shape),))
        if x.shape[2] != self.his_num:
            raise ValueError(
                "TPBCompatModel expected history length %d, got %d"
                % (self.his_num, x.shape[2])
            )
        bsz, num_nodes, _, channels = x.shape
        if channels < self.output_dim:
            raise ValueError(
                "TPBCompatModel expected at least %d channels, got %d"
                % (self.output_dim, channels)
            )
        base = x[:, :, -1:, : self.output_dim].repeat(1, 1, self.pred_num, 1)
        return base + self.bias.view(1, 1, 1, self.output_dim)


class TPBFullModelPredictor(nn.Module):
    def __init__(
        self,
        checkpoint_path,
        pattern_path,
        config_path,
        original_code_path,
        model_config,
        model_factory=None,
        map_location="cpu",
    ):
        super().__init__()
        self.checkpoint_path = require_artifact(
            "TPBExpert",
            "checkpoint_path",
            checkpoint_path,
        )
        self.pattern_path = require_artifact("TPBExpert", "pattern_path", pattern_path)
        self.config_path = require_artifact("TPBExpert", "config_path", config_path)
        self.original_code_path = self._require_code_root(original_code_path)
        self.model_config = dict(model_config or {})
        self.map_location = map_location
        self.pattern_artifact = None
        self.config = self._load_config(self.config_path)
        if model_factory is None:
            _require_patchfsl_config(self.config, self.config_path)
            model_factory = _import_patchfsl(self.original_code_path)
            self.model = model_factory(
                self.config["data_args"],
                self.config["model_args"],
                self.config["task_args"],
                self.config["PatchFSL_cfg"],
                self.config.get("STmodel", "GWN"),
            )
        else:
            self.model = model_factory(**self.model_config)
        self._load_artifacts()

    def _require_code_root(self, original_code_path):
        if not original_code_path:
            raise FileNotFoundError("TPBExpert original_code_path is required but empty")
        if not os.path.isdir(original_code_path):
            raise FileNotFoundError(
                "TPBExpert original_code_path does not exist: %s" % original_code_path
            )
        return original_code_path

    def _load_config(self, config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            content = handle.read()
        config = _parse_config_content(content, config_path)
        if not isinstance(config, dict):
            raise RuntimeError(
                "TPBExpert config_path must contain a mapping: %s" % config_path
            )
        return config

    def _load_artifacts(self):
        self.pattern_artifact = load_torch_artifact(
            "TPBExpert",
            "pattern_path",
            self.pattern_path,
            self.map_location,
        )
        state = extract_state_dict(
            load_torch_artifact(
                "TPBExpert",
                "checkpoint_path",
                self.checkpoint_path,
                self.map_location,
            )
        )
        state = self._validate_checkpoint_state(state)
        incompatible = self.model.load_state_dict(state, strict=False)
        self._validate_loaded_coverage(state, incompatible)
        self.model.eval()

    def _validate_checkpoint_state(self, state):
        if not isinstance(state, dict) or not state:
            raise RuntimeError(
                "TPBExpert checkpoint_path must contain a non-empty state_dict: %s"
                % self.checkpoint_path
            )
        state = _strip_module_prefix(state, self.checkpoint_path)
        expected_keys = set(self.model.state_dict().keys())
        if not expected_keys:
            return state
        missing_keys = sorted(expected_keys - set(state.keys()))
        if missing_keys:
            preview = ", ".join(missing_keys[:8])
            if len(missing_keys) > 8:
                preview = "%s, ..." % preview
            raise RuntimeError(
                "TPBExpert checkpoint_path missing %d/%d expected tensors: %s; path: %s"
                % (len(missing_keys), len(expected_keys), preview, self.checkpoint_path)
            )
        return state

    def _validate_loaded_coverage(self, state, incompatible):
        expected_keys = set(self.model.state_dict().keys())
        if not expected_keys:
            return
        loaded_keys = set(state.keys()) & expected_keys
        missing_keys = sorted(expected_keys - loaded_keys)
        if not missing_keys:
            return
        preview = ", ".join(missing_keys[:8])
        if len(missing_keys) > 8:
            preview = "%s, ..." % preview
        unexpected_count = len(getattr(incompatible, "unexpected_keys", []) or [])
        raise RuntimeError(
            "TPBExpert checkpoint_path missing %d/%d expected tensors: %s; "
            "unexpected tensors ignored: %d; path: %s"
            % (
                len(missing_keys),
                len(expected_keys),
                preview,
                unexpected_count,
                self.checkpoint_path,
            )
        )

    @torch.no_grad()
    def forward(self, history_data, supports=None, batch_meta=None):
        batch_meta = dict(batch_meta or {})
        output_len = int(batch_meta.get("output_len") or self.model_config.get("pred_num") or 1)
        output_dim = int(batch_meta.get("output_dim") or self.model_config.get("output_dim") or history_data.shape[-1])
        bsz, _, num_nodes, _ = history_data.shape
        self.model.to(history_data.device)
        model_input = history_data.permute(0, 2, 1, 3).contiguous()
        if output_dim == 1:
            y = torch.zeros(bsz, num_nodes, output_len, dtype=history_data.dtype, device=history_data.device)
        else:
            y = torch.zeros(
                bsz,
                num_nodes,
                output_len,
                output_dim,
                dtype=history_data.dtype,
                device=history_data.device,
            )
        data_i = SimpleNamespace(
            x=model_input,
            y=y,
            means=torch.zeros(1, dtype=history_data.dtype, device=history_data.device),
            stds=torch.ones(1, dtype=history_data.dtype, device=history_data.device),
        )
        adjacency = _adjacency_for(supports, num_nodes, history_data)
        prediction = self.model(data_i, adjacency, stage=batch_meta.get("stage", "test"))
        if isinstance(prediction, dict):
            prediction = prediction.get("pred", prediction.get("prediction"))
        if isinstance(prediction, (tuple, list)):
            prediction = prediction[0]
        prediction = torch.as_tensor(prediction, dtype=history_data.dtype, device=history_data.device)
        if prediction.dim() == 3:
            prediction = prediction.unsqueeze(-1)
        if prediction.dim() != 4:
            raise ValueError("TPBExpert full_model output must be [B,N,T] or [B,N,T,C], got %s" % (tuple(prediction.shape),))
        if prediction.shape[0] != bsz or prediction.shape[1] != num_nodes:
            raise ValueError(
                "TPBExpert full_model output batch/node mismatch: expected (%d,%d), got %s"
                % (bsz, num_nodes, tuple(prediction.shape))
            )
        prediction = prediction[:, :, :output_len, :output_dim]
        if prediction.shape[2] != output_len or prediction.shape[3] != output_dim:
            raise ValueError(
                "TPBExpert full_model output cannot satisfy output_len/output_dim: %s"
                % (tuple(prediction.shape),)
            )
        return prediction.permute(0, 2, 1, 3).contiguous()


def _parse_config_content(content, config_path):
    if not content.strip():
        return {}
    errors = []
    try:
        import yaml
    except Exception as exc:
        errors.append("yaml unavailable: %s" % exc)
    else:
        try:
            return yaml.safe_load(content)
        except Exception as exc:
            errors.append("yaml: %s" % exc)

    try:
        return json.loads(content)
    except Exception as exc:
        errors.append("json: %s" % exc)

    raise RuntimeError(
        "TPBExpert config_path failed to parse as YAML or JSON: %s; %s"
        % (config_path, "; ".join(errors))
    )


def _require_patchfsl_config(config, config_path):
    required_keys = ("data_args", "model_args", "task_args", "PatchFSL_cfg")
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise RuntimeError(
            "TPBExpert config_path missing required PatchFSL sections %s: %s"
            % (", ".join(missing), config_path)
        )
    non_mapping = [key for key in required_keys if not isinstance(config[key], dict)]
    if non_mapping:
        raise RuntimeError(
            "TPBExpert config_path PatchFSL sections must be mappings %s: %s"
            % (", ".join(non_mapping), config_path)
        )


def _import_patchfsl(original_code_path):
    paths = [
        original_code_path,
        os.path.join(original_code_path, "model"),
        os.path.join(original_code_path, "model", "Meta_Models"),
        os.path.join(original_code_path, "model", "TSFormer"),
    ]
    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)
    from rep_model_final import PatchFSL

    return PatchFSL


def _strip_module_prefix(state, checkpoint_path):
    prefixed = [key.startswith("module.") for key in state]
    if not any(prefixed):
        return state
    if not all(prefixed):
        raise RuntimeError(
            "TPBExpert checkpoint_path has mixed module. prefixes, which are unsupported: %s"
            % checkpoint_path
        )
    return {key[len("module.") :]: value for key, value in state.items()}


def _adjacency_for(supports, num_nodes, history_data):
    if supports:
        return torch.as_tensor(supports[0], dtype=history_data.dtype, device=history_data.device)
    return torch.eye(num_nodes, dtype=history_data.dtype, device=history_data.device)
