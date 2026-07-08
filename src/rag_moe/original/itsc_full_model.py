import pickle

import torch
from torch import nn

from src.rag_moe.full_model_utils import (
    extract_state_dict,
    load_torch_artifact,
    require_artifact,
)


class ITSCFullModelPredictor(nn.Module):
    def __init__(
        self,
        checkpoint_path,
        bank_path,
        model_config,
        model_factory=None,
        map_location="cpu",
    ):
        super().__init__()
        self.checkpoint_path = require_artifact(
            "ITSCExpert",
            "checkpoint_path",
            checkpoint_path,
        )
        self.bank_path = require_artifact("ITSCExpert", "bank_path", bank_path)
        self.model_config = dict(model_config or {})
        if model_factory is None:
            from src.rag_moe.original.itsc_ragimpel import RAGIMPEL

            model_factory = RAGIMPEL
        self.model = model_factory(**self.model_config)
        self.map_location = map_location
        self.retrieval_bank = None
        self._load_artifacts()

    def _load_bank(self, bank_path):
        with open(bank_path, "rb") as handle:
            return pickle.load(handle)

    def _load_artifacts(self):
        state = extract_state_dict(
            load_torch_artifact(
                "ITSCExpert",
                "checkpoint_path",
                self.checkpoint_path,
                self.map_location,
            )
        )
        state = self._validate_checkpoint_state(state)
        incompatible = self.model.load_state_dict(state, strict=False)
        self._validate_loaded_coverage(state, incompatible)
        self.model.eval()
        self.retrieval_bank = self._load_bank(self.bank_path)

    def _validate_checkpoint_state(self, state):
        if not isinstance(state, dict) or not state:
            raise RuntimeError(
                "ITSCExpert checkpoint_path must contain a non-empty state_dict: %s"
                % self.checkpoint_path
            )
        state = _strip_module_prefix(state)
        model_keys = set(self.model.state_dict().keys())
        checkpoint_keys = set(state.keys())
        required_roots = [
            "time_series_emb_layer",
            "encoder",
            "regression_layer",
            "llm_adapter",
            "retriever_encoder",
            "rag_memory",
            "prior_alpha",
            "prior_out_proj",
            "prior_out_gate",
        ]
        if any(_root_keys(model_keys, "gconv")):
            required_roots.append("gconv")
        missing_roots = [
            root
            for root in required_roots
            if not (_root_keys(model_keys, root) & _root_keys(checkpoint_keys, root))
        ]
        if missing_roots:
            raise RuntimeError(
                "ITSCExpert checkpoint_path missing required roots %s: %s"
                % (", ".join(missing_roots), self.checkpoint_path)
            )
        return state

    def _validate_loaded_coverage(self, state, incompatible):
        expected_keys = set(self.model.state_dict().keys())
        loaded_keys = set(state.keys()) & expected_keys
        missing_keys = expected_keys - loaded_keys
        allowed_missing = _allowed_missing_keys(expected_keys)
        missing_keys = sorted(missing_keys - allowed_missing)
        if not missing_keys:
            return
        preview = ", ".join(missing_keys[:8])
        if len(missing_keys) > 8:
            preview = "%s, ..." % preview
        unexpected_count = len(getattr(incompatible, "unexpected_keys", []) or [])
        raise RuntimeError(
            "ITSCExpert checkpoint_path missing %d/%d expected tensors: %s; "
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
    def forward(self, history_data, supports=None, llm_encoding=None, batch_meta=None):
        _ = supports
        batch_meta = dict(batch_meta or {})
        prediction = self.model(
            history_data,
            llm_encoding,
            x_hour=batch_meta.get("x_hour"),
            x_weekday=batch_meta.get("x_weekday"),
            sample_idx=batch_meta.get("sample_ids", batch_meta.get("rag_index")),
            retrieval_bank=self.retrieval_bank,
            teacher_forcing=False,
            return_aux=False,
        )
        if isinstance(prediction, dict):
            prediction = prediction["pred"]
        return prediction


def _strip_module_prefix(state):
    if not any(key.startswith("module.") for key in state):
        return state
    return {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in state.items()
    }


def _root_keys(keys, root):
    return {
        key
        for key in keys
        if key == root or key.startswith(root + ".")
    }


def _allowed_missing_keys(expected_keys):
    if any(key == "gconv" or key.startswith("gconv.") for key in expected_keys):
        return set()
    return {
        key
        for key in expected_keys
        if key == "gconv" or key.startswith("gconv.")
    }
