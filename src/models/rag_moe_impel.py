from collections import OrderedDict
from typing import Iterable

import torch
from torch import nn

from src.rag_moe.experts.base import validate_correction_output
from src.rag_moe.features import build_router_features
from src.rag_moe.fusion import fuse_residuals
from src.rag_moe.router import TwoStageRAGRouter


class RAGMoEIMPEL(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        experts: Iterable[nn.Module],
        output_len: int,
        output_dim: int,
        router_hidden_dim: int = 128,
        router_dropout: float = 0.1,
        allow_raw_prior_candidates: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.experts = nn.ModuleList(list(experts))
        self.output_len = int(output_len)
        self.output_dim = int(output_dim)
        self.name = "rag_moe_impel"
        self.horizon = self.output_len
        self.return_dict = True
        self.allow_raw_prior_candidates = bool(allow_raw_prior_candidates)

        for attr in ("dataset", "device", "num_nodes", "seq_len", "input_dim"):
            if hasattr(backbone, attr):
                setattr(self, attr, getattr(backbone, attr))

        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for expert in self.experts:
            expert.freeze()

        self.correction_names = ["none"] + [expert.name for expert in self.experts]
        self.candidate_names = self.correction_names
        feature_dim = 5 + len(self.experts)
        self.router = TwoStageRAGRouter(
            num_candidates=len(self.correction_names),
            input_dim=feature_dim,
            hidden_dim=router_hidden_dim,
            dropout=router_dropout,
        )

    def param_num(self, name):
        return sum(parameter.nelement() for parameter in self.parameters())

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if self._looks_like_unprefixed_backbone_state(state_dict):
            return self.backbone.load_state_dict(state_dict, strict=False)
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:
            return super().load_state_dict(state_dict, strict=strict)

    def _looks_like_unprefixed_backbone_state(self, state_dict):
        if not state_dict:
            return False
        if any(key.startswith(("backbone.", "router.", "experts.")) for key in state_dict):
            return False
        backbone_keys = set(self.backbone.state_dict().keys())
        return any(key in backbone_keys for key in state_dict)

    def forward(self, history_data, supports, llm_encoding, batch_meta=None):
        batch_meta = dict(batch_meta or {})
        batch_meta.setdefault("output_len", self.output_len)
        batch_meta.setdefault("output_dim", self.output_dim)

        with torch.no_grad():
            baseline_pred = self.backbone(history_data, supports, llm_encoding)

        expert_outputs = OrderedDict()
        expert_deltas = OrderedDict()
        available_masks = [
            torch.ones(
                baseline_pred.shape[0],
                baseline_pred.shape[2],
                device=baseline_pred.device,
                dtype=torch.bool,
            )
        ]

        for expert in self.experts:
            output = expert.forward_correction(
                history_data,
                supports,
                llm_encoding,
                batch_meta,
                baseline_pred,
            )
            validate_correction_output(output, baseline_pred, expected_name=expert.name)
            expert_outputs[expert.name] = output
            expert_deltas[expert.name] = output.delta
            available_masks.append(output.available.bool())

        features = build_router_features(history_data, baseline_pred, expert_deltas)
        available = torch.stack(available_masks, dim=-1)
        router_outputs = self.router(features, available)

        residual_deltas = OrderedDict()
        residual_deltas["none"] = torch.zeros_like(baseline_pred)
        for name, delta in expert_deltas.items():
            residual_deltas[name] = delta

        deltas = [residual_deltas[name] for name in self.correction_names]
        delta_tensor = torch.stack(deltas, dim=2)
        routed_delta = fuse_residuals(delta_tensor, router_outputs["weights"])
        prediction = baseline_pred + routed_delta
        if not self.return_dict:
            return prediction

        return {
            "prediction": prediction,
            "baseline_pred": baseline_pred,
            "routed_delta": routed_delta,
            "residual_deltas": residual_deltas,
            "expert_outputs": expert_outputs,
            "correction_names": self.correction_names,
            "candidate_names": self.candidate_names,
            "select_logits": router_outputs["select_logits"],
            "select_prob": router_outputs["select_prob"],
            "weights": router_outputs["weights"],
            "active_mask": router_outputs["active_mask"],
        }

    @torch.no_grad()
    def forward_direct_expert(
        self,
        expert_name,
        history_data,
        supports,
        llm_encoding,
        batch_meta=None,
        allow_raw_prior=False,
        return_delta=False,
    ):
        _ = allow_raw_prior
        batch_meta = dict(batch_meta or {})
        batch_meta.setdefault("output_len", self.output_len)
        batch_meta.setdefault("output_dim", self.output_dim)

        baseline_pred = self.backbone(history_data, supports, llm_encoding)
        for expert in self.experts:
            if expert.name != expert_name:
                continue
            output = expert.forward_correction(
                history_data,
                supports,
                llm_encoding,
                batch_meta,
                baseline_pred,
            )
            validate_correction_output(output, baseline_pred, expected_name=expert.name)
            if return_delta:
                return output.delta
            return baseline_pred + output.delta
        raise KeyError("unknown direct expert %r; available experts are %r" % (
            expert_name,
            [expert.name for expert in self.experts],
        ))
