import torch
from torch import nn

from src.rag_moe.full_model_utils import (
    extract_state_dict,
    load_torch_artifact,
    require_artifact,
)


class RAFTCompatModel(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        n_period=3,
        topm=20,
        task_name="short_term_forecast",
        **_,
    ):
        super().__init__()
        self.task_name = task_name
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.channels = int(enc_in)
        self.n_period = int(n_period)
        self.topm = int(topm)

        self.linear_x = nn.Linear(self.seq_len, self.pred_len)
        period_num = [16, 8, 4, 2, 1]
        self.period_num = sorted(period_num[-1 * self.n_period :], reverse=True)
        self.retrieval_pred = nn.ModuleList(
            [nn.Linear(self.pred_len // period, self.pred_len) for period in self.period_num]
        )
        self.linear_pred = nn.Linear(2 * self.pred_len, self.pred_len)
        self.retrieval_dict = {}

    def encoder(self, x, index, mode):
        bsz, seq_len, channels = x.shape
        if seq_len != self.seq_len or channels != self.channels:
            raise ValueError(
                "RAFTCompatModel expected input [B,%d,%d], got %s"
                % (self.seq_len, self.channels, tuple(x.shape))
            )
        if mode not in self.retrieval_dict:
            raise ValueError("RAFT retrieval cache does not contain mode: %s" % mode)

        index = index.to(dtype=torch.long, device=x.device)
        x_offset = x[:, -1:, :].detach()
        x_norm = x - x_offset
        x_pred_from_x = self.linear_x(x_norm.permute(0, 2, 1)).permute(0, 2, 1)

        retrieval = self.retrieval_dict[mode].to(device=x.device, dtype=x.dtype)
        pred_from_retrieval = retrieval[:, index]
        retrieval_pred_list = []
        for i, pr in enumerate(pred_from_retrieval):
            if tuple(pr.shape) != (bsz, self.pred_len, channels):
                raise ValueError(
                    "RAFT retrieval mode %s period %d has invalid indexed shape %s"
                    % (mode, i, tuple(pr.shape))
                )
            period = self.period_num[i]
            pr = pr.reshape(bsz, self.pred_len // period, period, channels)
            pr = pr[:, :, 0, :]
            pr = self.retrieval_pred[i](pr.permute(0, 2, 1)).permute(0, 2, 1)
            retrieval_pred_list.append(pr.reshape(bsz, self.pred_len, self.channels))

        retrieval_pred = torch.stack(retrieval_pred_list, dim=1).sum(dim=1)
        pred = torch.cat([x_pred_from_x, retrieval_pred], dim=1)
        pred = self.linear_pred(pred.permute(0, 2, 1)).permute(0, 2, 1)
        return pred.reshape(bsz, self.pred_len, self.channels) + x_offset

    def forward(self, x_enc, index, mode="train"):
        if self.task_name in {
            "long_term_forecast",
            "short_term_forecast",
            "imputation",
            "anomaly_detection",
        }:
            return self.encoder(x_enc, index, mode)[:, -self.pred_len :, :]
        if self.task_name == "classification":
            output = self.encoder(x_enc, index, mode).reshape(x_enc.shape[0], -1)
            return self.projection(output)
        return None


class RAFTFullModelPredictor(nn.Module):
    def __init__(
        self,
        checkpoint_path,
        retrieval_cache_path,
        model_config,
        model_factory=None,
        map_location="cpu",
    ):
        super().__init__()
        self.checkpoint_path = require_artifact(
            "RAFTExpert",
            "checkpoint_path",
            checkpoint_path,
        )
        self.retrieval_cache_path = require_artifact(
            "RAFTExpert",
            "retrieval_cache_path",
            retrieval_cache_path,
        )
        self.model_config = dict(model_config or {})
        self.map_location = map_location
        if model_factory is None:
            model_factory = RAFTCompatModel
        self.model = model_factory(**self.model_config)
        self._load_artifacts()

    def _load_artifacts(self):
        retrieval_dict = self._load_retrieval_cache()
        state = extract_state_dict(
            load_torch_artifact(
                "RAFTExpert",
                "checkpoint_path",
                self.checkpoint_path,
                self.map_location,
            )
        )
        state = self._validate_checkpoint_state(state)
        incompatible = self.model.load_state_dict(state, strict=False)
        self._validate_loaded_coverage(state, incompatible)
        self.model.retrieval_dict = retrieval_dict
        self.model.eval()

    def _load_retrieval_cache(self):
        artifact = load_torch_artifact(
            "RAFTExpert",
            "retrieval_cache_path",
            self.retrieval_cache_path,
            self.map_location,
        )
        retrieval_dict = artifact.get("retrieval_dict", artifact) if isinstance(artifact, dict) else artifact
        if not isinstance(retrieval_dict, dict) or "test" not in retrieval_dict:
            raise RuntimeError(
                "RAFTExpert retrieval_cache_path must contain retrieval_dict with test mode: %s"
                % self.retrieval_cache_path
            )
        allowed_modes = {"train", "valid", "test"}
        loaded = {}
        for mode, value in retrieval_dict.items():
            if mode not in allowed_modes:
                continue
            tensor = torch.as_tensor(value)
            if tensor.dim() != 4:
                raise RuntimeError(
                    "RAFTExpert retrieval_cache_path mode %s must be [G,S,P,C], got %s"
                    % (mode, tuple(tensor.shape))
                )
            if tensor.shape[0] != self.model.n_period:
                raise RuntimeError(
                    "RAFTExpert retrieval_cache_path mode %s period count %d != %d"
                    % (mode, tensor.shape[0], self.model.n_period)
                )
            if tensor.shape[2] != self.model.pred_len or tensor.shape[3] != self.model.channels:
                raise RuntimeError(
                    "RAFTExpert retrieval_cache_path mode %s shape %s does not match pred_len/channels"
                    % (mode, tuple(tensor.shape))
                )
            loaded[mode] = tensor.detach()
        if "test" not in loaded:
            raise RuntimeError(
                "RAFTExpert retrieval_cache_path must contain test retrieval tensor: %s"
                % self.retrieval_cache_path
            )
        return loaded

    def _validate_checkpoint_state(self, state):
        if not isinstance(state, dict) or not state:
            raise RuntimeError(
                "RAFTExpert checkpoint_path must contain a non-empty state_dict: %s"
                % self.checkpoint_path
            )
        state = _strip_module_prefix(state, self.checkpoint_path)
        expected_keys = set(self.model.state_dict().keys())
        missing_keys = sorted(expected_keys - set(state.keys()))
        if missing_keys:
            preview = ", ".join(missing_keys[:8])
            if len(missing_keys) > 8:
                preview = "%s, ..." % preview
            raise RuntimeError(
                "RAFTExpert checkpoint_path missing %d/%d expected tensors: %s; path: %s"
                % (len(missing_keys), len(expected_keys), preview, self.checkpoint_path)
            )
        return state

    def _validate_loaded_coverage(self, state, incompatible):
        expected_keys = set(self.model.state_dict().keys())
        loaded_keys = set(state.keys()) & expected_keys
        missing_keys = sorted(expected_keys - loaded_keys)
        if not missing_keys:
            return
        preview = ", ".join(missing_keys[:8])
        if len(missing_keys) > 8:
            preview = "%s, ..." % preview
        unexpected_count = len(getattr(incompatible, "unexpected_keys", []) or [])
        raise RuntimeError(
            "RAFTExpert checkpoint_path missing %d/%d expected tensors: %s; "
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
    def forward(self, history_data, sample_ids, batch_meta=None):
        batch_meta = dict(batch_meta or {})
        mode = batch_meta.get("raft_mode", "test")
        bsz, _, num_nodes, _ = history_data.shape
        model_device = next(self.model.parameters()).device
        if model_device != history_data.device:
            self.model.to(history_data.device)
        x = history_data[..., : self.model.channels].permute(0, 2, 1, 3).reshape(
            bsz * num_nodes,
            history_data.shape[1],
            self.model.channels,
        )
        sample_ids = torch.as_tensor(sample_ids, dtype=torch.long, device=history_data.device)
        if sample_ids.dim() > 1:
            sample_ids = sample_ids[:, -1]
        index = sample_ids.repeat_interleave(num_nodes)
        pred = self.model(x, index, mode=mode)
        return pred.reshape(
            bsz,
            num_nodes,
            self.model.pred_len,
            self.model.channels,
        ).permute(0, 2, 1, 3).contiguous()


def _strip_module_prefix(state, checkpoint_path):
    prefixed = [key.startswith("module.") for key in state]
    if not any(prefixed):
        return state
    if not all(prefixed):
        raise RuntimeError(
            "RAFTExpert checkpoint_path has mixed module. prefixes, which are unsupported: %s"
            % checkpoint_path
        )
    return {
        key[len("module.") :]: value
        for key, value in state.items()
    }
