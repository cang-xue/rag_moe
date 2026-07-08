import os
import pickle

import torch
from torch import nn
import torch.nn.functional as F

from src.rag_moe.experts.base import ExpertOutput, RAGExpertAdapter
from src.rag_moe.full_model_utils import require_config_keys
from src.rag_moe.original.itsc_correction import ITSCResidualCorrection
from src.rag_moe.original.itsc_full_model import ITSCFullModelPredictor


class RetrieverEncoder(nn.Module):
    def __init__(self, llm_dim, ts_len, retriever_dim=128):
        super().__init__()
        self.retriever_dim = int(retriever_dim)
        self.llm_proj = nn.Sequential(
            nn.Linear(int(llm_dim), self.retriever_dim),
            nn.GELU(),
            nn.Linear(self.retriever_dim, self.retriever_dim),
        )
        self.ts_proj = nn.Sequential(
            nn.Linear(int(ts_len), self.retriever_dim),
            nn.GELU(),
            nn.Linear(self.retriever_dim, self.retriever_dim),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.retriever_dim * 2),
            nn.Linear(self.retriever_dim * 2, self.retriever_dim),
            nn.GELU(),
            nn.Linear(self.retriever_dim, self.retriever_dim),
        )

    @staticmethod
    def _safe_normalize(value):
        return F.normalize(value, dim=-1, eps=1e-8)

    def _encode(self, llm_emb, ts_emb):
        encoded = self.fusion(torch.cat([llm_emb, ts_emb], dim=-1))
        encoded = torch.nan_to_num(encoded, nan=0.0, posinf=0.0, neginf=0.0)
        return self._safe_normalize(encoded)

    def encode_query(self, llm_query, ts_query, query_hour=None, query_weekday=None):
        batch_size = ts_query.shape[0]
        llm = self.llm_proj(llm_query).unsqueeze(0).expand(batch_size, -1, -1)
        ts = self.ts_proj(ts_query)
        return self._encode(llm, ts)

    def encode_key(self, llm_keys, ts_keys, key_hour=None, key_weekday=None):
        llm = self.llm_proj(llm_keys)
        ts = self.ts_proj(ts_keys)
        return self._encode(llm, ts)


class ITSCExpert(RAGExpertAdapter):
    name = "itsc"

    def prepare(self, config, data_context):
        config = config or {}
        self.mode = str(config.get("mode", "prior"))
        self.full_model = None
        self.correction_model = None
        self.scale = float(config.get("scale", 1.0))
        self.output_len = config.get("output_len")
        self.output_dim = config.get("output_dim")
        self.top_k = int(config.get("top_k", config.get("rag_top_k", 5)))
        self.temperature = float(config.get("temperature", config.get("rag_temp", 0.2)))
        self.use_hour_bucket = bool(config.get("use_hour_bucket", True))
        self.use_weekday_filter = bool(config.get("use_weekday_filter", True))
        self.data_context = dict(data_context or {})
        if self.mode == "full_model":
            require_config_keys(
                "ITSCExpert",
                config,
                ["checkpoint_path", "bank_path", "model_config"],
            )
            self.full_model = ITSCFullModelPredictor(
                checkpoint_path=config["checkpoint_path"],
                bank_path=config["bank_path"],
                model_config=config["model_config"],
                map_location=config.get("map_location", "cpu"),
            )
            return self.freeze()
        if self.mode in {"residual", "correction"}:
            require_config_keys(
                "ITSCExpert",
                config,
                ["checkpoint_path", "bank_path", "model_config"],
            )
            self.correction_model = ITSCResidualCorrection(
                checkpoint_path=config["checkpoint_path"],
                bank_path=config["bank_path"],
                model_config=config["model_config"],
                map_location=config.get("map_location", "cpu"),
            )
            return self.freeze()
        self.use_retriever_encoder = bool(config.get("use_retriever_encoder", False))
        self.require_retriever_pretrained = bool(config.get("require_retriever_pretrained", True))
        self.retriever_encoder = None
        if self.use_retriever_encoder:
            self.retriever_encoder = self._build_retriever_encoder(config)
        self.bank = config.get("bank") or self._load_bank(config)
        return self.freeze()

    @torch.no_grad()
    def forward_prior(self, history_data, supports=None, llm=None, batch_meta=None):
        batch_meta = batch_meta or {}
        if getattr(self, "mode", "prior") == "full_model":
            if self.full_model is None:
                raise ValueError("ITSCExpert full_model was not prepared")
            prior = self.full_model(
                history_data,
                supports=supports,
                llm_encoding=llm,
                batch_meta=batch_meta,
            )
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
            prior, aux = self._retrieve_bank_prior(history_data, llm, batch_meta, output_len, output_dim)
        else:
            prior = history_data[:, -1:, :, :output_dim].repeat(1, output_len, 1, 1)
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
    def forward_correction(
        self,
        history_data,
        supports,
        llm_encoding,
        batch_meta,
        baseline_pred,
    ):
        if getattr(self, "mode", "prior") in {"residual", "correction"}:
            if self.correction_model is None:
                raise ValueError("ITSCExpert residual mode was not prepared")
            return self.correction_model.forward_correction(
                history_data=history_data,
                baseline_pred=baseline_pred,
                llm_encoding=llm_encoding,
                batch_meta=batch_meta,
            )
        return super().forward_correction(
            history_data,
            supports,
            llm_encoding,
            batch_meta,
            baseline_pred,
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
            samples_per_hour = config.get("top_samples_per_hour", 5000)
            if source_data:
                bank_path = os.path.join(
                    config["bank_dir"],
                    f"{source_data}_source_bank_{samples_per_hour}_v5.pkl",
                )
        if not bank_path or not os.path.exists(bank_path):
            return None
        with open(bank_path, "rb") as handle:
            return pickle.load(handle)

    def _retrieve_bank_prior(self, history_data, llm, batch_meta, output_len, output_dim):
        bucket = self.bank.get("global") if isinstance(self.bank, dict) else None
        if bucket is None:
            raise ValueError("ITSC bank must contain a 'global' block")

        query = _flatten_history(history_data, output_dim)
        keys = bucket.get("dyn_keys")
        if keys is None:
            keys = bucket.get("hist")
        llm_keys = bucket.get("llm_keys")
        future = bucket.get("future")
        if keys is None or future is None:
            raise ValueError("ITSC bank global block must contain dyn_keys/hist and future")
        keys = _to_device_float(keys, history_data.device)
        future = _to_device_float(future, history_data.device)
        future = _format_future(future, output_len, output_dim)

        retriever_used = False
        if self.use_retriever_encoder:
            if llm_keys is None:
                raise ValueError("ITSC retriever encoder requires bank global block to contain llm_keys")
            scores = self._retriever_scores(
                history_data,
                output_dim,
                llm,
                _to_device_float(llm_keys, history_data.device),
                keys,
                bucket,
                batch_meta,
            )
            retriever_used = True
        else:
            scores = _cosine_scores(query, keys)
            llm_scores = _maybe_llm_scores(llm, llm_keys, history_data.shape[0], history_data.shape[2], history_data.device)
            if llm_scores is not None:
                scores = 0.5 * scores + 0.5 * llm_scores
        scores = self._apply_time_masks(
            scores,
            bucket,
            batch_meta,
            history_data.device,
            history_data.shape[0],
            history_data.shape[2],
        )
        scores = self._apply_self_exclusion(scores, bucket, batch_meta, history_data.device)
        prior, top_index, top_weight = _weighted_future(scores, future, self.top_k, self.temperature)
        prior = prior.view(history_data.shape[0], history_data.shape[2], output_len, output_dim)
        prior = prior.transpose(1, 2).contiguous()
        if prior.shape != (history_data.shape[0], output_len, history_data.shape[2], output_dim):
            raise ValueError(f"ITSC bank prior has invalid shape {tuple(prior.shape)}")
        return prior, {
            "bank_used": True,
            "retriever_used": retriever_used,
            "top_indices": top_index,
            "top_weights": top_weight,
        }

    def _build_retriever_encoder(self, config):
        input_len = int(config.get("input_len") or self.data_context.get("input_len") or 0)
        llm_dim = int(config.get("llm_enc_dim") or self.data_context.get("llm_enc_dim") or 0)
        if input_len <= 0 or llm_dim <= 0:
            message = (
                "ITSCExpert retriever encoder requires input_len and llm_enc_dim "
                "in config or data_context"
            )
            print(message)
            raise ValueError(message)
        retriever = RetrieverEncoder(
            llm_dim=llm_dim,
            ts_len=input_len,
            retriever_dim=int(config.get("retriever_dim", 128)),
        )
        checkpoint_path = config.get("retriever_pretrained_path", "")
        if not checkpoint_path:
            message = (
                "ITSCExpert use_retriever_encoder=True requires "
                "retriever_pretrained_path when require_retriever_pretrained=True"
            )
            print(message)
            if self.require_retriever_pretrained:
                raise ValueError(message)
            return retriever.eval()
        if not os.path.exists(checkpoint_path):
            message = f"ITSCExpert retriever_pretrained_path does not exist: {checkpoint_path}"
            print(message)
            raise ValueError(message)
        try:
            try:
                state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            state = _extract_retriever_state(state, retriever.state_dict())
            retriever.load_state_dict(state, strict=False)
        except Exception as exc:
            message = f"ITSCExpert failed to load retriever_pretrained_path={checkpoint_path}: {exc}"
            print(message)
            raise ValueError(message) from exc
        return retriever.eval()

    def _retriever_scores(self, history_data, output_dim, llm, llm_keys, dyn_keys, bucket, batch_meta):
        if self.retriever_encoder is None:
            raise ValueError("ITSC retriever encoder is not initialized")
        llm_query = _format_llm_query(llm, history_data.shape[0], history_data.shape[2], history_data.device)
        hist_bn_t = history_data[:, :, :, :output_dim].squeeze(-1).transpose(1, 2)
        if hist_bn_t.dim() != 3:
            raise ValueError(f"ITSC retriever query history must be [B, N, T], got {tuple(hist_bn_t.shape)}")
        if dyn_keys.shape[1] != hist_bn_t.shape[-1]:
            raise ValueError(
                f"ITSC retriever key width {dyn_keys.shape[1]} does not match query width {hist_bn_t.shape[-1]}"
            )
        q_enc = self.retriever_encoder.encode_query(
            llm_query=llm_query,
            ts_query=hist_bn_t,
            query_hour=_last_meta_value(batch_meta["x_hour"], history_data.device) if "x_hour" in batch_meta else None,
            query_weekday=_last_meta_value(batch_meta["x_weekday"], history_data.device) if "x_weekday" in batch_meta else None,
        )
        k_enc = self.retriever_encoder.encode_key(
            llm_keys=llm_keys,
            ts_keys=dyn_keys,
            key_hour=_to_device_long(bucket["hour_ids"], history_data.device) if "hour_ids" in bucket else None,
            key_weekday=_to_device_long(bucket["weekday_ids"], history_data.device) if "weekday_ids" in bucket else None,
        )
        scores = torch.einsum("bnd,md->bnm", q_enc, k_enc)
        return scores.reshape(history_data.shape[0] * history_data.shape[2], k_enc.shape[0])

    def _apply_time_masks(self, scores, bucket, batch_meta, device, batch_size, num_nodes):
        candidate_count = scores.shape[-1]
        sample_mask = torch.ones(batch_size, candidate_count, dtype=torch.bool, device=device)
        hour_ids = bucket.get("hour_ids")
        if self.use_hour_bucket and "x_hour" in batch_meta and hour_ids is not None:
            hour_ids = _to_device_long(hour_ids, device)
            hour = _last_meta_value(batch_meta["x_hour"], device).view(batch_size, 1)
            hour_mask = hour_ids.view(1, -1).eq(hour)
            sample_mask = hour_mask
            weekday_ids = bucket.get("weekday_ids")
            if self.use_weekday_filter and "x_weekday" in batch_meta and weekday_ids is not None:
                weekday_ids = _to_device_long(weekday_ids, device)
                weekday = _last_meta_value(batch_meta["x_weekday"], device).view(batch_size, 1)
                weekday_mask = hour_mask & weekday_ids.view(1, -1).eq(weekday)
                sample_mask = torch.where(weekday_mask.any(dim=1, keepdim=True), weekday_mask, hour_mask)
            sample_mask = torch.where(hour_mask.any(dim=1, keepdim=True), sample_mask, torch.ones_like(sample_mask))
        return scores.masked_fill(~sample_mask.repeat_interleave(num_nodes, dim=0), float("-inf"))

    def _apply_self_exclusion(self, scores, bucket, batch_meta, device):
        sample_ids = bucket.get("sample_ids")
        query_ids = batch_meta.get("sample_ids", batch_meta.get("sample_idx", batch_meta.get("rag_index")))
        if sample_ids is None or query_ids is None:
            return scores
        sample_ids = _to_device_long(sample_ids, device)
        query_ids = _last_meta_value(query_ids, device).long()
        batch_size = query_ids.numel()
        nodes = scores.shape[0] // batch_size
        mask = sample_ids.view(1, -1).eq(query_ids.view(-1, 1))
        return scores.masked_fill(mask.repeat_interleave(nodes, dim=0), float("-inf"))


def _flatten_history(history_data, output_dim):
    return history_data[:, :, :, :output_dim].permute(0, 2, 1, 3).reshape(
        history_data.shape[0] * history_data.shape[2],
        history_data.shape[1] * output_dim,
    )


def _to_device_float(value, device):
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _to_device_long(value, device):
    return torch.as_tensor(value, dtype=torch.long, device=device)


def _last_meta_value(value, device):
    tensor = torch.as_tensor(value, device=device)
    if tensor.dim() > 1:
        tensor = tensor[:, -1]
    return tensor


def _format_llm_query(llm, batch_size, num_nodes, device):
    if llm is None:
        raise ValueError("ITSC retriever encoder requires llm query embeddings")
    tensor = _to_device_float(llm, device)
    if tensor.dim() == 3:
        if tensor.shape[0] != batch_size or tensor.shape[1] != num_nodes:
            raise ValueError(
                f"ITSC llm query shape {tuple(tensor.shape)} must be [B, N, D] with B={batch_size}, N={num_nodes}"
            )
        tensor = tensor[0]
    if tensor.dim() != 2 or tensor.shape[0] != num_nodes:
        raise ValueError(f"ITSC llm query shape {tuple(tensor.shape)} must be [N, D]")
    return tensor


def _extract_retriever_state(state, expected_state):
    if not isinstance(state, dict):
        raise ValueError("retriever checkpoint must be a state dict")
    expected_keys = set(expected_state.keys())
    direct = {key: value for key, value in state.items() if key in expected_keys}
    if direct:
        return direct
    prefix = "retriever_encoder."
    stripped = {
        key[len(prefix):]: value
        for key, value in state.items()
        if key.startswith(prefix) and key[len(prefix):] in expected_keys
    }
    if stripped:
        return stripped
    raise ValueError("retriever checkpoint contains no matching RetrieverEncoder keys")


def _with_candidate_type(aux, candidate_type):
    aux = dict(aux or {})
    aux.setdefault("candidate_type", candidate_type)
    return aux


def _cosine_scores(query, keys):
    if keys.dim() != 2:
        raise ValueError(f"ITSC bank keys must have rank 2, got {tuple(keys.shape)}")
    if keys.shape[1] != query.shape[1]:
        raise ValueError(f"ITSC bank key width {keys.shape[1]} does not match query width {query.shape[1]}")
    return F.normalize(query, dim=-1, eps=1e-8) @ F.normalize(keys, dim=-1, eps=1e-8).t()


def _maybe_llm_scores(llm, llm_keys, batch_size, num_nodes, device):
    if llm is None or llm_keys is None:
        return None
    llm_keys = _to_device_float(llm_keys, device)
    llm_query = _to_device_float(llm, device)
    if llm_query.dim() == 2:
        llm_query = llm_query.unsqueeze(0).expand(batch_size, -1, -1)
    if llm_query.shape[0] != batch_size or llm_query.shape[1] != num_nodes:
        return None
    llm_query = llm_query.reshape(batch_size * num_nodes, llm_query.shape[-1])
    if llm_keys.dim() != 2 or llm_keys.shape[1] != llm_query.shape[1]:
        return None
    return F.normalize(llm_query, dim=-1, eps=1e-8) @ F.normalize(llm_keys, dim=-1, eps=1e-8).t()


def _format_future(future, output_len, output_dim):
    if future.dim() == 2:
        future = future.unsqueeze(-1)
    if future.dim() != 3:
        raise ValueError(f"ITSC bank future must be [M, T] or [M, T, C], got {tuple(future.shape)}")
    if future.shape[1] < output_len:
        repeat = (output_len + future.shape[1] - 1) // future.shape[1]
        future = future.repeat(1, repeat, 1)
    future = future[:, :output_len, :]
    if future.shape[2] < output_dim:
        repeat = (output_dim + future.shape[2] - 1) // future.shape[2]
        future = future.repeat(1, 1, repeat)
    return future[:, :, :output_dim]


def _weighted_future(scores, future, top_k, temperature):
    candidate_count = scores.shape[1]
    k = min(int(top_k), candidate_count)
    if k <= 0:
        raise ValueError("ITSC bank is empty")
    top_score, top_index = torch.topk(scores, k=k, dim=1)
    valid = torch.isfinite(top_score)
    safe_score = torch.where(valid, top_score, torch.full_like(top_score, -1e9))
    weights = torch.softmax(safe_score / max(float(temperature), 1e-6), dim=1) * valid.float()
    weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-8)
    gathered = future[top_index]
    prior = (weights.unsqueeze(-1).unsqueeze(-1) * gathered).sum(dim=1)
    return prior, top_index, weights
