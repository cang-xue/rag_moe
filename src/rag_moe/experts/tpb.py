import os

import torch
from torch import nn
import torch.nn.functional as F

from src.rag_moe.experts.base import ExpertOutput, RAGExpertAdapter
from src.rag_moe.full_model_utils import require_config_keys
from src.rag_moe.original.tpb_full_model import TPBFullModelPredictor
from src.layers.gcn import GCN
from src.models.mlp import MultiLayerPerceptron


class TPBExpert(RAGExpertAdapter):
    name = "tpb"

    def prepare(self, config, data_context):
        config = config or {}
        self.mode = str(config.get("mode", "prior"))
        self.full_model = None
        self.scale = float(config.get("scale", 1.0))
        self.output_len = config.get("output_len")
        self.output_dim = config.get("output_dim")
        self.top_k = int(config.get("top_k", config.get("rag_top_k", 8)))
        self.temperature = float(config.get("temperature", config.get("rag_temp", 1.0)))
        self.patch_len = int(config.get("patch_len", config.get("rag_patch_len", 12)))
        self.source_weighting = config.get("source_weighting", "uniform")
        self.prior_alpha = float(config.get("prior_alpha", 0.0))
        self.data_context = dict(data_context or {})
        self.final_model = None
        if self.mode == "full_model":
            require_config_keys(
                "TPBExpert",
                config,
                [
                    "checkpoint_path",
                    "pattern_path",
                    "config_path",
                    "original_code_path",
                    "model_config",
                ],
            )
            predictor_kwargs = {
                "checkpoint_path": config["checkpoint_path"],
                "pattern_path": config["pattern_path"],
                "config_path": config["config_path"],
                "original_code_path": config["original_code_path"],
                "model_config": config["model_config"],
                "map_location": config.get("map_location", "cpu"),
            }
            if "model_factory" in config:
                predictor_kwargs["model_factory"] = config["model_factory"]
            self.full_model = TPBFullModelPredictor(**predictor_kwargs)
            return self.freeze()
        self.bank = config.get("bank") or self._load_bank(config)
        if self.mode in {"final_prediction", "final"}:
            require_config_keys("TPBExpert", config, ["model_config"])
            if self.bank is None:
                raise FileNotFoundError(
                    "TPBExpert final_prediction mode requires a pattern bank; "
                    "set bank_path or bank_dir/bank_name to an existing original-consistent bank"
                )
            self.final_model = TPBIMPELFinalCandidate(
                model_config=config["model_config"],
                patch_len=self.patch_len,
                top_k=self.top_k,
                temperature=self.temperature,
                prior_alpha=self.prior_alpha,
            )
            self.final_model.prepare_rag_bank(self.bank)
            checkpoint_path = config.get("checkpoint_path")
            if checkpoint_path:
                if not os.path.exists(checkpoint_path):
                    raise FileNotFoundError(f"TPBExpert checkpoint_path does not exist: {checkpoint_path}")
                state = torch.load(checkpoint_path, map_location="cpu")
                self.final_model.load_state_dict(state, strict=False)
        return self.freeze()

    @torch.no_grad()
    def forward_prior(self, history_data, supports=None, llm=None, batch_meta=None):
        batch_meta = batch_meta or {}
        if getattr(self, "mode", "prior") == "full_model":
            if self.full_model is None:
                raise ValueError("TPBExpert full_model was not prepared")
            prior = self.full_model(history_data, supports=supports, batch_meta=batch_meta)
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
        input_len = history_data.shape[1]
        if self.patch_len > input_len:
            raise ValueError(f"patch_len={self.patch_len} is larger than input_len={input_len}")
        if self.bank is not None:
            prior, aux = self._retrieve_pattern_prior(history_data, output_len, output_dim)
        else:
            window_len = min(self.patch_len, input_len)
            prior = history_data[:, -window_len:, :, :output_dim].mean(dim=1, keepdim=True)
            prior = prior.repeat(1, output_len, 1, 1)
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
    def forward_candidate(self, history_data, supports, llm, batch_meta, baseline_pred):
        if self.mode in {"final_prediction", "final"}:
            if self.final_model is None:
                raise ValueError("TPBExpert final_prediction mode was not prepared")
            model_device = next(self.final_model.parameters()).device
            if model_device != history_data.device:
                self.final_model.to(history_data.device)
            prior = self.final_model(history_data, supports, llm)
            output_len = self._output_len(history_data, batch_meta)
            output_dim = self._output_dim(history_data, batch_meta)
            prior = prior[:, :output_len, :, :output_dim] * self.scale
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
                aux={"candidate_type": "final_prediction", "bank_used": self.bank is not None},
            )
        return super().forward_candidate(history_data, supports, llm, batch_meta, baseline_pred)

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
            metadata_name = config.get("bank_name")
            if metadata_name:
                bank_path = os.path.join(config["bank_dir"], metadata_name)
        if not bank_path or not os.path.exists(bank_path):
            return None
        return torch.load(bank_path, map_location="cpu")

    def _retrieve_pattern_prior(self, history_data, output_len, output_dim):
        if "patterns" not in self.bank:
            raise ValueError("TPB bank must contain 'patterns'")
        patterns = torch.as_tensor(self.bank["patterns"], dtype=torch.float32, device=history_data.device)
        pattern_dim = self.patch_len * output_dim
        if patterns.dim() != 2 or patterns.shape[1] != pattern_dim:
            raise ValueError(
                f"TPB pattern dimension mismatch: expected {pattern_dim}, got {tuple(patterns.shape)}"
            )

        patches = self._history_patches(history_data, output_dim)
        flat_patches = patches.reshape(-1, pattern_dim)
        scores = F.normalize(flat_patches, dim=-1, eps=1e-8) @ F.normalize(patterns, dim=-1, eps=1e-8).t()
        k = min(self.top_k, patterns.shape[0])
        if k <= 0:
            raise ValueError("TPB bank is empty")
        top_score, top_index = torch.topk(scores, k=k, dim=1)
        weights = torch.softmax(top_score / max(self.temperature, 1e-6), dim=1)
        retrieved = (weights.unsqueeze(-1) * patterns[top_index]).sum(dim=1)
        retrieved = retrieved.view(
            history_data.shape[0],
            history_data.shape[2],
            patches.shape[2],
            self.patch_len,
            output_dim,
        )
        pattern_prior = retrieved.mean(dim=2)
        prior = _tile_pattern(pattern_prior, output_len)
        return prior, {"bank_used": True, "top_indices": top_index, "top_weights": weights}

    def _history_patches(self, history_data, output_dim):
        input_data = history_data[:, :, :, :output_dim]
        usable_len = (input_data.shape[1] // self.patch_len) * self.patch_len
        if usable_len == 0:
            raise ValueError(f"patch_len={self.patch_len} is larger than input_len={input_data.shape[1]}")
        patches = input_data[:, :usable_len].reshape(
            input_data.shape[0],
            usable_len // self.patch_len,
            self.patch_len,
            input_data.shape[2],
            output_dim,
        )
        patches = patches.permute(0, 3, 1, 2, 4).contiguous()
        return patches.reshape(
            input_data.shape[0],
            input_data.shape[2],
            usable_len // self.patch_len,
            self.patch_len * output_dim,
        )


def _tile_pattern(pattern_prior, output_len):
    batch_size, num_nodes, patch_len, output_dim = pattern_prior.shape
    repeat = (output_len + patch_len - 1) // patch_len
    tiled = pattern_prior.repeat(1, 1, repeat, 1)[:, :, :output_len, :]
    return tiled.permute(0, 2, 1, 3).contiguous()


def _with_candidate_type(aux, candidate_type):
    aux = dict(aux or {})
    aux.setdefault("candidate_type", candidate_type)
    return aux


class TPBIMPELFinalCandidate(nn.Module):
    def __init__(self, model_config, patch_len, top_k, temperature, prior_alpha):
        super().__init__()
        cfg = dict(model_config or {})
        self.node_dim = int(cfg["node_dim"])
        self.input_len = int(cfg["input_len"])
        self.input_dim = int(cfg.get("in_dim", cfg.get("input_dim", 1)))
        self.embed_dim = int(cfg["embed_dim"])
        self.output_len = int(cfg["output_len"])
        self.num_layer = int(cfg["num_layer"])
        self.llm_enc_dim = int(cfg["llm_enc_dim"])
        self.mp_layers = int(cfg.get("mp_layers", 0))
        self.supports_len = 1
        self.rag_top_k = int(top_k)
        self.rag_temp = float(temperature)
        self.rag_patch_len = int(patch_len)
        self.rag_pattern_dim = self.rag_patch_len * self.input_dim
        self.prior_alpha = nn.Parameter(torch.tensor(float(prior_alpha)))

        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len,
            out_channels=self.embed_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.hidden_dim = self.embed_dim + self.node_dim
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)]
        )
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.output_len,
            kernel_size=(1, 1),
            bias=True,
        )
        self.llm_adapter = nn.Linear(self.llm_enc_dim, self.node_dim)
        self.gconv = nn.ModuleList(
            [GCN(self.hidden_dim, self.hidden_dim, dropout=0.0, support_len=self.supports_len)
             for _ in range(self.mp_layers)]
        )
        self.pattern_query_proj = nn.Linear(self.rag_pattern_dim, self.rag_pattern_dim)
        self.pattern_key_proj = nn.Linear(self.rag_pattern_dim, self.rag_pattern_dim)
        self.pattern_value_proj = nn.Linear(self.rag_pattern_dim, self.rag_pattern_dim)
        self.pattern_fusion_proj = nn.Linear(self.rag_pattern_dim, self.hidden_dim)
        self.register_buffer("rag_patterns", torch.empty(0, self.rag_pattern_dim), persistent=False)

    def prepare_rag_bank(self, bank):
        if bank is None:
            self.rag_patterns = torch.empty(0, self.rag_pattern_dim)
            return
        if "patterns" not in bank:
            raise ValueError("TPB final candidate bank must contain clustered 'patterns'")
        patterns = torch.as_tensor(bank["patterns"], dtype=torch.float32)
        if patterns.dim() != 2 or patterns.shape[1] != self.rag_pattern_dim:
            raise ValueError(
                "TPB final candidate patterns must be [K,%d], got %s"
                % (self.rag_pattern_dim, tuple(patterns.shape))
            )
        self.rag_patterns = patterns.detach()

    def retrieve_pattern_representation(self, history_data):
        if self.rag_patterns.numel() == 0:
            return None
        input_data = history_data[..., : self.input_dim]
        batch_size, _, num_nodes, _ = input_data.shape
        usable_len = (input_data.shape[1] // self.rag_patch_len) * self.rag_patch_len
        if usable_len == 0:
            return None
        patches = input_data[:, :usable_len].reshape(
            batch_size,
            usable_len // self.rag_patch_len,
            self.rag_patch_len,
            num_nodes,
            self.input_dim,
        )
        patches = patches.permute(0, 3, 1, 2, 4).contiguous()
        patches = patches.view(batch_size * num_nodes, usable_len // self.rag_patch_len, self.rag_pattern_dim)

        patterns = self.rag_patterns.to(device=history_data.device, dtype=history_data.dtype)
        query = self.pattern_query_proj(patches)
        key = self.pattern_key_proj(patterns)
        value = self.pattern_value_proj(patterns)

        scores = torch.einsum(
            "bpd,kd->bpk",
            F.normalize(query, dim=-1, eps=1e-8),
            F.normalize(key, dim=-1, eps=1e-8),
        ) / max(float(self.rag_temp), 1e-6)
        top_k = min(int(self.rag_top_k), patterns.shape[0])
        top_scores, top_indices = torch.topk(scores, k=top_k, dim=-1)
        weights = F.softmax(top_scores, dim=-1)
        top_patterns = value[top_indices]
        pattern_context = torch.einsum("bpk,bpkd->bpd", weights, top_patterns).mean(dim=1)
        return pattern_context.view(batch_size, num_nodes, -1)

    def forward(self, history_data, supports, llm_encoding):
        input_data = history_data[..., : self.input_dim]
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)

        llm_enc = self.llm_adapter(llm_encoding)
        node_emb = llm_enc.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)
        hidden = torch.cat([time_series_emb, node_emb], dim=1)

        adp = F.softmax(F.gelu(torch.mm(llm_enc, llm_enc.T)), dim=1)
        for layer in self.gconv:
            hidden = layer(hidden, [adp]) + hidden

        bank_pattern = self.retrieve_pattern_representation(history_data)
        if bank_pattern is not None:
            pattern_delta = self.pattern_fusion_proj(bank_pattern).transpose(1, 2).unsqueeze(-1)
            hidden = hidden + self.prior_alpha * pattern_delta

        hidden = self.encoder(hidden)
        return self.regression_layer(hidden)
