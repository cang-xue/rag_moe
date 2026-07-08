import pickle

import torch
from torch import nn

from src.rag_moe.experts.base import RAGCorrectionOutput
from src.rag_moe.full_model_utils import (
    extract_state_dict,
    load_torch_artifact,
    require_artifact,
)
from src.rag_moe.original.itsc_full_model import _strip_module_prefix


class ITSCResidualCorrection(nn.Module):
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
        state = _strip_module_prefix(state)
        self._validate_checkpoint_state(state)
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
        required_roots = [
            "rag_memory",
            "prior_alpha",
            "prior_out_proj",
            "prior_out_gate",
        ]
        checkpoint_keys = set(state.keys())
        missing_roots = [
            root
            for root in required_roots
            if not any(
                key == root or key.startswith(root + ".")
                for key in checkpoint_keys
            )
        ]
        if missing_roots:
            raise RuntimeError(
                "ITSCExpert checkpoint_path missing residual roots %s: %s"
                % (", ".join(missing_roots), self.checkpoint_path)
            )

    def _validate_loaded_coverage(self, state, incompatible):
        expected_keys = set(self.model.state_dict().keys())
        loaded_keys = set(state.keys()) & expected_keys
        required_roots = [
            "rag_memory",
            "prior_alpha",
            "prior_out_proj",
            "prior_out_gate",
        ]
        missing_keys = []
        for root in required_roots:
            root_expected = _root_keys(expected_keys, root)
            if root_expected:
                missing_keys.extend(sorted(root_expected - loaded_keys))
        if not missing_keys:
            return
        preview = ", ".join(missing_keys[:8])
        if len(missing_keys) > 8:
            preview = "%s, ..." % preview
        unexpected_count = len(getattr(incompatible, "unexpected_keys", []) or [])
        raise RuntimeError(
            "ITSCExpert checkpoint_path missing %d residual tensors: %s; "
            "unexpected tensors ignored: %d; path: %s"
            % (
                len(missing_keys),
                preview,
                unexpected_count,
                self.checkpoint_path,
            )
        )

    def forward_correction(
        self,
        history_data,
        baseline_pred,
        llm_encoding=None,
        batch_meta=None,
    ):
        with torch.no_grad():
            return self.forward_train_correction(
                history_data=history_data,
                baseline_pred=baseline_pred,
                llm_encoding=llm_encoding,
                batch_meta=batch_meta,
            )

    def forward_train_correction(
        self,
        history_data,
        baseline_pred,
        llm_encoding=None,
        batch_meta=None,
    ):
        batch_meta = dict(batch_meta or {})
        with torch.no_grad():
            _, bank_prior, retriever_aux_loss = self.model.rag_memory.retrieve_from_bank(
                query_emb=llm_encoding,
                query_history=history_data[..., :1],
                x_hour=batch_meta.get("x_hour"),
                x_minute=batch_meta.get("x_minute"),
                x_weekday=batch_meta.get("x_weekday"),
                query_sample_idx=batch_meta.get(
                    "sample_ids",
                    batch_meta.get("rag_index"),
                ),
                retrieval_bank=self.retrieval_bank,
                input_len=self.model.input_len,
                output_len=self.model.output_len,
                query_future=batch_meta.get("query_future"),
                exclude_self=False,
            )
        if bank_prior is None:
            raise ValueError("ITSC residual correction requires a retrieved bank_prior")
        bank_prior = torch.nan_to_num(
            bank_prior.to(device=baseline_pred.device, dtype=baseline_pred.dtype),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if tuple(bank_prior.shape) != tuple(baseline_pred.shape):
            raise ValueError(
                "ITSC bank_prior shape %r does not match baseline prediction shape %r"
                % (tuple(bank_prior.shape), tuple(baseline_pred.shape))
            )
        prior_feat = self.model.prior_out_proj(bank_prior)
        gate_input = torch.cat([baseline_pred, prior_feat], dim=1)
        gate = self.model.prior_out_gate(gate_input)
        prior_alpha = self.model.prior_alpha.to(
            device=baseline_pred.device,
            dtype=baseline_pred.dtype,
        )
        delta = prior_alpha * gate * prior_feat
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        available = torch.ones(
            baseline_pred.shape[0],
            baseline_pred.shape[2],
            dtype=torch.bool,
            device=baseline_pred.device,
        )
        return RAGCorrectionOutput(
            name="itsc",
            delta=delta,
            available=available,
            raw_prior=bank_prior,
            aux={
                "bank_used": True,
                "correction_type": "itsc_prior_gate",
                "prior_alpha": float(prior_alpha.detach().cpu()),
                "retriever_aux_loss": retriever_aux_loss,
            },
        )


def _root_keys(keys, root):
    return {
        key
        for key in keys
        if key == root or key.startswith(root + ".")
    }
