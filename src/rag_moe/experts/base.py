from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn


@dataclass
class ExpertOutput:
    name: str
    prior: torch.Tensor
    available: torch.Tensor
    confidence: Optional[torch.Tensor] = None
    aux: Optional[Dict[str, Any]] = None


@dataclass
class RAGCorrectionOutput:
    name: str
    delta: torch.Tensor
    available: torch.Tensor
    raw_prior: Optional[torch.Tensor] = None
    confidence: Optional[torch.Tensor] = None
    aux: Optional[Dict[str, Any]] = None


class RAGExpertAdapter(nn.Module):
    name = "base"

    def prepare(self, *args, **kwargs):
        return None

    @torch.no_grad()
    def forward_prior(self, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def forward_candidate(self, history_data, supports, llm_encoding, batch_meta, baseline_pred):
        output = self.forward_prior(history_data, supports, llm_encoding, batch_meta)
        aux = dict(output.aux or {})
        aux.setdefault("candidate_type", "raw_prior")
        output.aux = aux
        return output

    @torch.no_grad()
    def forward_correction(
        self,
        history_data,
        supports=None,
        llm_encoding=None,
        batch_meta=None,
        baseline_pred=None,
    ):
        output = self.forward_candidate(
            history_data,
            supports,
            llm_encoding,
            batch_meta,
            baseline_pred,
        )
        aux = dict(output.aux or {})
        candidate_type = aux.setdefault("candidate_type", "raw_prior")
        if candidate_type == "final_prediction":
            correction_type = "final_prediction_minus_baseline"
        else:
            correction_type = "raw_prior_minus_baseline"
        aux["correction_type"] = correction_type
        return RAGCorrectionOutput(
            name=output.name,
            delta=output.prior - baseline_pred,
            available=output.available,
            raw_prior=output.prior,
            confidence=output.confidence,
            aux=aux,
        )

    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)
        return self


def validate_expert_output(output, baseline_pred, expected_name):
    if output.name != expected_name:
        raise ValueError(
            "expected expert %r, got %r" % (expected_name, output.name)
        )

    if tuple(output.prior.shape) != tuple(baseline_pred.shape):
        raise ValueError(
            "prior shape %r does not match baseline prediction shape %r"
            % (tuple(output.prior.shape), tuple(baseline_pred.shape))
        )

    if len(output.prior.shape) != 4:
        raise ValueError("prior shape must be [B, T, N, C]")

    if not isinstance(output.available, torch.Tensor):
        raise ValueError("available must be a torch.Tensor")

    if output.available.dim() != 2:
        raise ValueError(
            "available shape %r must be [B, N] or [B, 1]"
            % (tuple(output.available.shape),)
        )

    batch_size = baseline_pred.shape[0]
    node_count = baseline_pred.shape[2]
    available_batch, available_nodes = output.available.shape
    if available_batch != batch_size:
        raise ValueError(
            "available batch dimension %r does not match baseline batch %r"
            % (available_batch, batch_size)
        )

    if available_nodes not in (node_count, 1):
        raise ValueError(
            "available node dimension %r must match baseline nodes %r or be 1"
            % (available_nodes, node_count)
        )

    return output


def validate_correction_output(output, baseline_pred, expected_name):
    if output.name != expected_name:
        raise ValueError(
            "expected expert %r, got %r" % (expected_name, output.name)
        )

    if tuple(output.delta.shape) != tuple(baseline_pred.shape):
        raise ValueError(
            "delta shape %r does not match baseline prediction shape %r"
            % (tuple(output.delta.shape), tuple(baseline_pred.shape))
        )

    if len(output.delta.shape) != 4:
        raise ValueError("delta shape must be [B, T, N, C]")

    if output.raw_prior is not None and tuple(output.raw_prior.shape) != tuple(baseline_pred.shape):
        raise ValueError(
            "raw_prior shape %r does not match baseline prediction shape %r"
            % (tuple(output.raw_prior.shape), tuple(baseline_pred.shape))
        )

    if not isinstance(output.available, torch.Tensor):
        raise ValueError("available must be a torch.Tensor")

    if output.available.dim() != 2:
        raise ValueError(
            "available shape %r must be [B, N] or [B, 1]"
            % (tuple(output.available.shape),)
        )

    batch_size = baseline_pred.shape[0]
    node_count = baseline_pred.shape[2]
    available_batch, available_nodes = output.available.shape
    if available_batch != batch_size:
        raise ValueError(
            "available batch dimension %r does not match baseline batch %r"
            % (available_batch, batch_size)
        )

    if available_nodes not in (node_count, 1):
        raise ValueError(
            "available node dimension %r must match baseline nodes %r or be 1"
            % (available_nodes, node_count)
        )

    return output
