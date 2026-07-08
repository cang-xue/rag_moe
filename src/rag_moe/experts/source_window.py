import os
from collections.abc import Mapping

import torch
import torch.nn.functional as F

from src.rag_moe.experts.base import RAGCorrectionOutput, RAGExpertAdapter
from src.rag_moe.experts.custom_features import (
    horizon_parameter,
    none_available,
    normalized_entropy,
    output_dim_from,
    output_len_from,
    require_baseline_pred,
)


def confidence_from_retrieval(top_scores, top_weights, min_top1, min_margin, max_entropy):
    top1 = top_scores[:, 0]
    if top_scores.shape[1] > 1:
        margin = top_scores[:, 0] - top_scores[:, 1]
    else:
        margin = top_scores[:, 0]
    entropy = normalized_entropy(top_weights)
    passed = (top1 >= min_top1) & (margin >= min_margin) & (entropy <= max_entropy)
    raw = 0.5 * top1 + 0.3 * margin + 0.2 * (1.0 - entropy)
    return raw, passed, {"top1": top1, "margin": margin, "entropy": entropy}


class SourceWindowExpert(RAGExpertAdapter):
    name = "source_window"

    def prepare(self, config, data_context):
        config = config or {}
        self.mode = str(config.get("mode", "source_window"))
        self.output_len = config.get("output_len")
        self.output_dim = config.get("output_dim")
        self.alpha = config.get("alpha", 0.0)
        self.top_k = int(config.get("top_k", config.get("rag_top_k", 5)))
        self.temperature = float(config.get("temperature", config.get("rag_temp", 0.1)))
        self.min_top1 = float(config.get("min_top1", -1.0))
        self.min_margin = float(config.get("min_margin", 0.0))
        self.max_entropy = float(config.get("max_entropy", 1.0))
        self.confidence_threshold = float(config.get("confidence_threshold", 0.0))
        self.score_chunk_size = int(config.get("score_chunk_size") or 0)
        self.data_context = dict(data_context or {})
        self.bank = config.get("bank")
        if self.bank is None:
            self.bank = _load_bank_path(config.get("bank_path"))
        self._bank_tensor_cache = {}
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

        if self.bank is None:
            return RAGCorrectionOutput(
                name=self.name,
                delta=torch.zeros_like(baseline_pred),
                available=none_available(baseline_pred),
                raw_prior=baseline_pred,
                confidence=torch.zeros(
                    baseline_pred.shape[0],
                    baseline_pred.shape[2],
                    dtype=baseline_pred.dtype,
                    device=baseline_pred.device,
                ),
                aux={"bank_used": False, "alpha": self.alpha},
            )

        output_len = output_len_from(baseline_pred, batch_meta, self.output_len)
        output_dim = output_dim_from(baseline_pred, batch_meta, self.output_dim)
        (
            prior,
            top_indices,
            top_weights,
            confidence_scores,
            confidence_weights,
        ) = self._retrieve_prior(
            history_data,
            output_len,
            output_dim,
            baseline_pred,
        )

        raw_confidence, passed, confidence_aux = confidence_from_retrieval(
            confidence_scores,
            confidence_weights,
            self.min_top1,
            self.min_margin,
            self.max_entropy,
        )
        available_flat = passed & (raw_confidence >= self.confidence_threshold)
        batch_size = baseline_pred.shape[0]
        node_count = baseline_pred.shape[2]
        available = available_flat.view(batch_size, node_count)
        confidence = raw_confidence.view(batch_size, node_count).to(
            dtype=baseline_pred.dtype,
            device=baseline_pred.device,
        )

        alpha = horizon_parameter(self.alpha, baseline_pred, "source_window alpha")
        delta = alpha * (prior - baseline_pred)
        delta = delta * available.view(batch_size, 1, node_count, 1).to(delta.dtype)
        delivered = baseline_pred + delta

        aux = {
            "bank_used": True,
            "top_indices": top_indices.view(batch_size, node_count, -1),
            "top_weights": top_weights.view(batch_size, node_count, -1).to(
                dtype=baseline_pred.dtype,
                device=baseline_pred.device,
            ),
            "top1": confidence_aux["top1"].view(batch_size, node_count).to(
                dtype=baseline_pred.dtype,
                device=baseline_pred.device,
            ),
            "margin": confidence_aux["margin"].view(batch_size, node_count).to(
                dtype=baseline_pred.dtype,
                device=baseline_pred.device,
            ),
            "entropy": confidence_aux["entropy"].view(batch_size, node_count).to(
                dtype=baseline_pred.dtype,
                device=baseline_pred.device,
            ),
            "confidence_threshold": self.confidence_threshold,
            "min_top1": self.min_top1,
            "min_margin": self.min_margin,
            "max_entropy": self.max_entropy,
            "alpha": self.alpha,
        }
        return RAGCorrectionOutput(
            name=self.name,
            delta=delta.to(dtype=baseline_pred.dtype, device=baseline_pred.device),
            available=available,
            raw_prior=delivered.to(dtype=baseline_pred.dtype, device=baseline_pred.device),
            confidence=confidence,
            aux=aux,
        )

    def _retrieve_prior(self, history_data, output_len, output_dim, reference):
        normalized_keys, values = self._bank_tensors(reference, output_len, output_dim)
        query = history_data[:, :, :, :output_dim].permute(0, 2, 1, 3).reshape(
            history_data.shape[0] * history_data.shape[2],
            history_data.shape[1] * output_dim,
        )
        query = query.to(dtype=torch.float32, device=reference.device)
        if normalized_keys.shape[1] != query.shape[1]:
            raise ValueError(
                "SourceWindowExpert bank key width %r does not match query width %r"
                % (normalized_keys.shape[1], query.shape[1])
            )
        normalized_query = F.normalize(query, dim=-1, eps=1e-8)
        (
            prior,
            top_indices,
            top_weights,
            confidence_scores,
            confidence_weights,
        ) = _retrieve_from_scores(
            normalized_query,
            normalized_keys,
            values,
            self.top_k,
            self.temperature,
            self.score_chunk_size,
        )
        prior = prior.view(history_data.shape[0], history_data.shape[2], output_len, output_dim)
        prior = prior.transpose(1, 2).contiguous()
        expected = (
            reference.shape[0],
            output_len,
            reference.shape[2],
            output_dim,
        )
        if tuple(prior.shape) != expected:
            raise ValueError(
                "SourceWindowExpert retrieved prior has invalid shape %r, expected %r"
                % (tuple(prior.shape), expected)
            )
        return (
            prior.to(dtype=reference.dtype),
            top_indices,
            top_weights,
            confidence_scores,
            confidence_weights,
        )

    def _bank_tensors(self, reference, output_len, output_dim):
        cache_key = (
            str(reference.device),
            str(reference.dtype),
            int(output_len),
            int(output_dim),
        )
        cached = self._bank_tensor_cache.get(cache_key)
        if cached is not None:
            return cached

        bank = _validate_bank(self.bank)
        keys = torch.as_tensor(bank["keys"], dtype=torch.float32, device=reference.device)
        values = torch.as_tensor(bank["values"], dtype=reference.dtype, device=reference.device)
        if keys.dim() != 2:
            raise ValueError(
                "SourceWindowExpert bank keys must be [M, input_len * output_dim], got %r"
                % (tuple(keys.shape),)
            )
        if keys.shape[0] <= 0:
            raise ValueError("SourceWindowExpert bank must contain at least one key")
        values = _format_values(values, keys.shape[0], output_len, output_dim)
        cached = (F.normalize(keys, dim=-1, eps=1e-8), values)
        self._bank_tensor_cache[cache_key] = cached
        return cached


def _load_bank_path(bank_path):
    if not bank_path:
        return None
    if not os.path.exists(bank_path):
        raise FileNotFoundError(
            "SourceWindowExpert bank_path %s not found" % (bank_path,)
        )
    try:
        return torch.load(bank_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(bank_path, map_location="cpu")


def _validate_bank(bank):
    if not isinstance(bank, Mapping):
        raise ValueError("SourceWindowExpert bank must be a mapping with keys and values")
    if "keys" not in bank or "values" not in bank:
        raise ValueError("SourceWindowExpert bank must contain keys and values")
    return bank


def _format_values(values, bank_size, output_len, output_dim):
    if values.dim() == 2:
        values = values.unsqueeze(-1)
    if values.dim() != 3:
        raise ValueError(
            "SourceWindowExpert bank values must be [M, T] or [M, T, C], got %r"
            % (tuple(values.shape),)
        )
    if values.shape[0] != bank_size:
        raise ValueError(
            "SourceWindowExpert bank values length %r does not match keys length %r"
            % (values.shape[0], bank_size)
        )
    if values.shape[1] != output_len:
        raise ValueError(
            "SourceWindowExpert bank values horizon %r does not match output_len %r"
            % (values.shape[1], output_len)
        )
    if values.shape[2] != output_dim:
        raise ValueError(
            "SourceWindowExpert bank values output_dim %r does not match expected %r"
            % (values.shape[2], output_dim)
        )
    return values


def _weighted_values(scores, values, top_k, temperature):
    k = min(int(top_k), scores.shape[1])
    if k <= 0:
        raise ValueError("SourceWindowExpert bank top_k must be positive")
    top_scores, top_indices = torch.topk(scores, k=k, dim=1)
    weights = _safe_softmax_weights(top_scores, temperature)
    gathered = values[top_indices]
    prior = (weights.to(values.dtype).unsqueeze(-1).unsqueeze(-1) * gathered).sum(dim=1)
    return prior, top_indices, weights


def _retrieve_from_scores(
    normalized_query,
    normalized_keys,
    values,
    top_k,
    temperature,
    score_chunk_size,
):
    if score_chunk_size and int(score_chunk_size) > 0:
        chunk_size = int(score_chunk_size)
        prior_chunks = []
        top_index_chunks = []
        top_weight_chunks = []
        confidence_score_chunks = []
        confidence_weight_chunks = []
        for start in range(0, normalized_query.shape[0], chunk_size):
            chunk = normalized_query[start:start + chunk_size]
            chunk_output = _retrieve_score_chunk(
                chunk,
                normalized_keys,
                values,
                top_k,
                temperature,
            )
            prior_chunks.append(chunk_output[0])
            top_index_chunks.append(chunk_output[1])
            top_weight_chunks.append(chunk_output[2])
            confidence_score_chunks.append(chunk_output[3])
            confidence_weight_chunks.append(chunk_output[4])
        return (
            torch.cat(prior_chunks, dim=0),
            torch.cat(top_index_chunks, dim=0),
            torch.cat(top_weight_chunks, dim=0),
            torch.cat(confidence_score_chunks, dim=0),
            torch.cat(confidence_weight_chunks, dim=0),
        )
    return _retrieve_score_chunk(
        normalized_query,
        normalized_keys,
        values,
        top_k,
        temperature,
    )


def _retrieve_score_chunk(normalized_query, normalized_keys, values, top_k, temperature):
    scores = normalized_query @ normalized_keys.t()
    prior, top_indices, top_weights = _weighted_values(
        scores,
        values,
        top_k,
        temperature,
    )
    confidence_k = 1 if scores.shape[1] == 1 else max(int(top_k), 2)
    confidence_scores, confidence_weights = _top_scores_and_weights(
        scores,
        confidence_k,
        temperature,
    )
    return prior, top_indices, top_weights, confidence_scores, confidence_weights


def _top_scores_and_weights(scores, top_k, temperature):
    k = min(int(top_k), scores.shape[1])
    if k <= 0:
        raise ValueError("SourceWindowExpert bank top_k must be positive")
    top_scores, _ = torch.topk(scores, k=k, dim=1)
    return top_scores, _safe_softmax_weights(top_scores, temperature)


def _safe_softmax_weights(top_scores, temperature):
    valid = torch.isfinite(top_scores)
    safe_scores = torch.where(valid, top_scores, torch.full_like(top_scores, -1e9))
    weights = torch.softmax(safe_scores / max(float(temperature), 1e-6), dim=1)
    weights = weights * valid.to(weights.dtype)
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
