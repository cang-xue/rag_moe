import os

import torch
import torch.nn.functional as F

from src.rag_moe.experts.base import ExpertOutput, RAGCorrectionOutput, RAGExpertAdapter
from src.rag_moe.full_model_utils import require_config_keys
from src.rag_moe.original.raft_full_model import RAFTFullModelPredictor


class RAFTExpert(RAGExpertAdapter):
    name = "raft"

    def prepare(self, config, data_context):
        config = config or {}
        self.mode = str(config.get("mode", "prior"))
        self.full_model = None
        self.scale = float(config.get("scale", 1.0))
        self.output_len = config.get("output_len")
        self.output_dim = config.get("output_dim")
        self.top_k = int(config.get("top_k", config.get("rag_top_k", 5)))
        self.temperature = float(config.get("temperature", config.get("rag_temp", 0.1)))
        self.prior_alpha = _parse_prior_alpha(config.get("prior_alpha", 0.0))
        self.data_context = dict(data_context or {})
        if self.mode == "full_model":
            require_config_keys(
                "RAFTExpert",
                config,
                ["checkpoint_path", "retrieval_cache_path", "model_config"],
            )
            self.full_model = RAFTFullModelPredictor(
                checkpoint_path=config["checkpoint_path"],
                retrieval_cache_path=config["retrieval_cache_path"],
                model_config=config["model_config"],
                map_location=config.get("map_location", "cpu"),
            )
            return self.freeze()
        self.bank = config.get("bank") or self._load_bank(config)
        if self.mode in {"final_prediction", "final", "residual"} and self.bank is None:
            raise FileNotFoundError(
                "RAFTExpert final_prediction mode requires a temporal-shape bank; "
                "set bank_path or bank_dir/source_data to an existing original-consistent bank"
            )
        return self.freeze()

    @torch.no_grad()
    def forward_prior(self, history_data, supports=None, llm=None, batch_meta=None):
        batch_meta = batch_meta or {}
        if getattr(self, "mode", "prior") == "full_model":
            if self.full_model is None:
                raise ValueError("RAFTExpert full_model was not prepared")
            sample_ids = batch_meta.get("sample_ids", batch_meta.get("rag_index"))
            if sample_ids is None:
                raise ValueError("RAFTExpert full_model requires batch_meta sample_ids or rag_index")
            sample_ids = torch.as_tensor(sample_ids, dtype=torch.long, device=history_data.device)
            if sample_ids.dim() > 1:
                sample_ids = sample_ids[:, -1]
            prior = self.full_model(history_data, sample_ids, batch_meta=batch_meta)
            available = torch.ones(
                history_data.shape[0],
                history_data.shape[2],
                dtype=torch.bool,
                device=history_data.device,
            )
            return ExpertOutput(
                name=self.name,
                prior=prior,
                available=available,
                aux={"full_model_used": True, "candidate_type": "final_prediction"},
            )
        output_len = self._output_len(history_data, batch_meta)
        output_dim = self._output_dim(history_data, batch_meta)
        if self.bank is not None:
            prior, aux = self._retrieve_bank_prior(history_data, batch_meta, output_len, output_dim)
        else:
            prior = history_data[:, :, :, :output_dim].mean(dim=1, keepdim=True).repeat(1, output_len, 1, 1)
            aux = {"bank_used": False}
        available = torch.ones(
            history_data.shape[0],
            history_data.shape[2],
            dtype=torch.bool,
            device=history_data.device,
        )
        return ExpertOutput(
            name=self.name,
            prior=prior * self.scale,
            available=available,
            aux=_with_candidate_type(aux, "raw_prior"),
        )

    @torch.no_grad()
    def forward_correction(self, history_data, supports, llm_encoding, batch_meta, baseline_pred):
        output = self.forward_prior(history_data, supports, llm_encoding, batch_meta)
        aux = dict(output.aux or {})
        if aux.get("candidate_type") == "final_prediction":
            delta = output.prior - baseline_pred
            raw_prior = aux.get("raw_prior", output.prior)
            aux["correction_type"] = "final_prediction_minus_baseline"
        else:
            raw_prior = output.prior
            alpha = _prior_alpha_tensor(self.prior_alpha, baseline_pred)
            delta = alpha * (raw_prior - baseline_pred)
            aux["correction_type"] = "raft_prior_alpha_residual"
            aux["raw_prior"] = raw_prior
            aux["prior_alpha"] = self.prior_alpha
        return RAGCorrectionOutput(
            name=self.name,
            delta=delta,
            available=output.available,
            raw_prior=raw_prior,
            confidence=output.confidence,
            aux=aux,
        )

    @torch.no_grad()
    def forward_candidate(self, history_data, supports, llm, batch_meta, baseline_pred):
        correction = self.forward_correction(history_data, supports, llm, batch_meta, baseline_pred)
        aux = dict(correction.aux or {})
        aux["candidate_type"] = "final_prediction"
        return ExpertOutput(
            name=self.name,
            prior=baseline_pred + correction.delta,
            available=correction.available,
            confidence=correction.confidence,
            aux=aux,
        )

    def _output_len(self, history_data, batch_meta):
        return int(
            batch_meta.get("output_len")
            or self.output_len
            or self.data_context.get("output_len")
            or history_data.shape[1]
        )

    def _output_dim(self, history_data, batch_meta):
        return int(
            batch_meta.get("output_dim")
            or self.output_dim
            or self.data_context.get("output_dim")
            or min(1, history_data.shape[-1])
        )

    def _load_bank(self, config):
        bank_path = config.get("bank_path")
        if not bank_path and config.get("bank_dir"):
            source_data = self.data_context.get("source_data") or self.data_context.get("dataset")
            if source_data:
                source_key = str(source_data).replace(",", "__").replace("/", "_").replace("\\", "_")
                bank_path = os.path.join(config["bank_dir"], f"{source_key}_temporal_shape_bank.pt")
        if not bank_path or not os.path.exists(bank_path):
            return None
        return torch.load(bank_path, map_location="cpu")

    def _retrieve_bank_prior(self, history_data, batch_meta, output_len, output_dim):
        keys = torch.as_tensor(self.bank["keys"], dtype=torch.float32, device=history_data.device)
        values = torch.as_tensor(self.bank["values"], dtype=torch.float32, device=history_data.device)
        query = history_data[:, :, :, :output_dim].permute(0, 2, 1, 3).reshape(
            history_data.shape[0] * history_data.shape[2],
            history_data.shape[1] * output_dim,
        )
        if keys.dim() != 2:
            raise ValueError(f"RAFT bank keys must be [M, T], got {tuple(keys.shape)}")
        if keys.shape[1] != query.shape[1]:
            raise ValueError(f"RAFT bank key width {keys.shape[1]} does not match query width {query.shape[1]}")
        values = _format_values(values, output_len, output_dim)

        scores = F.normalize(query, dim=-1, eps=1e-8) @ F.normalize(keys, dim=-1, eps=1e-8).t()
        scores = self._apply_self_exclusion(scores, batch_meta, history_data.device)
        prior, top_index, top_weight = _weighted_values(scores, values, self.top_k, self.temperature)
        prior = prior.view(history_data.shape[0], history_data.shape[2], output_len, output_dim)
        prior = prior.transpose(1, 2).contiguous()
        if prior.shape != (history_data.shape[0], output_len, history_data.shape[2], output_dim):
            raise ValueError(f"RAFT bank prior has invalid shape {tuple(prior.shape)}")
        return prior, {"bank_used": True, "top_indices": top_index, "top_weights": top_weight}

    def _apply_self_exclusion(self, scores, batch_meta, device):
        sample_indices = self.bank.get("sample_indices")
        query_ids = batch_meta.get("sample_ids", batch_meta.get("rag_index"))
        if sample_indices is None or query_ids is None:
            return scores
        sample_indices = torch.as_tensor(sample_indices, dtype=torch.long, device=device)
        query_ids = torch.as_tensor(query_ids, dtype=torch.long, device=device)
        if query_ids.dim() > 1:
            query_ids = query_ids[:, -1]
        batch_size = query_ids.numel()
        nodes = scores.shape[0] // batch_size
        mask = sample_indices.view(1, -1).eq(query_ids.view(-1, 1))

        source_ids = self.bank.get("source_ids")
        source_to_id = self.bank.get("source_to_id", {})
        source_data = self.data_context.get("source_data")
        if source_ids is not None and source_data in source_to_id:
            source_ids = torch.as_tensor(source_ids, dtype=torch.long, device=device)
            mask = mask & source_ids.view(1, -1).eq(int(source_to_id[source_data]))
        return scores.masked_fill(mask.repeat_interleave(nodes, dim=0), float("-inf"))


def _format_values(values, output_len, output_dim):
    if values.dim() == 2:
        values = values.unsqueeze(-1)
    if values.dim() != 3:
        raise ValueError(f"RAFT bank values must be [M, T] or [M, T, C], got {tuple(values.shape)}")
    if values.shape[1] < output_len:
        repeat = (output_len + values.shape[1] - 1) // values.shape[1]
        values = values.repeat(1, repeat, 1)
    values = values[:, :output_len, :]
    if values.shape[2] < output_dim:
        repeat = (output_dim + values.shape[2] - 1) // values.shape[2]
        values = values.repeat(1, 1, repeat)
    return values[:, :, :output_dim]


def _weighted_values(scores, values, top_k, temperature):
    k = min(int(top_k), scores.shape[1])
    if k <= 0:
        raise ValueError("RAFT bank is empty")
    top_score, top_index = torch.topk(scores, k=k, dim=1)
    valid = torch.isfinite(top_score)
    safe_score = torch.where(valid, top_score, torch.full_like(top_score, -1e9))
    weights = torch.softmax(safe_score / max(float(temperature), 1e-6), dim=1) * valid.float()
    weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-8)
    gathered = values[top_index]
    prior = (weights.unsqueeze(-1).unsqueeze(-1) * gathered).sum(dim=1)
    return prior, top_index, weights


def _with_candidate_type(aux, candidate_type):
    aux = dict(aux or {})
    aux.setdefault("candidate_type", candidate_type)
    return aux


def _parse_prior_alpha(value):
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return [float(item) for item in value.detach().cpu().view(-1).tolist()]
    return float(value)


def _prior_alpha_tensor(value, reference):
    if isinstance(value, list):
        alpha = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
        if alpha.numel() != reference.shape[1]:
            raise ValueError(
                "RAFT horizon prior_alpha length {} does not match output horizon {}".format(
                    alpha.numel(),
                    reference.shape[1],
                )
            )
        return alpha.view(1, -1, 1, 1)
    return torch.as_tensor(float(value), dtype=reference.dtype, device=reference.device)
