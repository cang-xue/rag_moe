import torch
from torch import nn
import torch.nn.functional as F


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data):
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))
        return hidden + input_data


class nconv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, A):
        x = torch.einsum("ncvl,vw->ncwl", (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.mlp = nn.Conv2d(
            c_in,
            c_out,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
            bias=True,
        )

    def forward(self, x):
        return self.mlp(x)


class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super().__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, supports):
        out = [x]
        for a in supports:
            x1 = self.nconv(x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        return F.dropout(h, self.dropout, training=self.training)


class RetrieverEncoder(nn.Module):
    """Joint TS+LLM retrieval encoder."""

    def __init__(self, llm_dim, ts_len, retriever_dim=128):
        super().__init__()
        self.retriever_dim = retriever_dim
        self.llm_proj = nn.Sequential(
            nn.Linear(llm_dim, retriever_dim),
            nn.GELU(),
            nn.Linear(retriever_dim, retriever_dim),
        )
        self.ts_proj = nn.Sequential(
            nn.Linear(ts_len, retriever_dim),
            nn.GELU(),
            nn.Linear(retriever_dim, retriever_dim),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(retriever_dim * 2),
            nn.Linear(retriever_dim * 2, retriever_dim),
            nn.GELU(),
            nn.Linear(retriever_dim, retriever_dim),
        )

    @staticmethod
    def _safe_normalize(x):
        return F.normalize(x, dim=-1, eps=1e-8)

    def _encode(self, llm_emb, ts_emb):
        z = torch.cat([llm_emb, ts_emb], dim=-1)
        z = self.fusion(z)
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        return self._safe_normalize(z)

    def encode_query(self, llm_query, ts_query, query_hour=None, query_weekday=None):
        _ = query_hour
        _ = query_weekday
        batch_size = ts_query.shape[0]
        llm = self.llm_proj(llm_query).unsqueeze(0).expand(batch_size, -1, -1)
        ts = self.ts_proj(ts_query)
        return self._encode(llm, ts)

    def encode_key(self, llm_keys, ts_keys, key_hour=None, key_weekday=None):
        _ = key_hour
        _ = key_weekday
        llm = self.llm_proj(llm_keys)
        ts = self.ts_proj(ts_keys)
        return self._encode(llm, ts)


class RAGMemory(nn.Module):
    """Retrieval module with temporal filtering and future-aware auxiliary training."""

    def __init__(
        self,
        top_k=5,
        temperature=0.2,
        use_hour_bucket=True,
        use_weekday_filter=True,
        retriever_encoder=None,
        use_retriever_encoder=True,
        retriever_aux_weight=0.0,
        aux_top_r=64,
        print_candidate_stats=False,
        print_candidate_every=200,
    ):
        super().__init__()
        self.top_k = top_k
        self.temperature = temperature
        self.use_hour_bucket = use_hour_bucket
        self.use_weekday_filter = use_weekday_filter
        self.retriever_encoder = retriever_encoder
        self.use_retriever_encoder = use_retriever_encoder
        self.retriever_aux_weight = retriever_aux_weight
        self.aux_top_r = aux_top_r
        self.print_candidate_stats = print_candidate_stats
        self.print_candidate_every = print_candidate_every
        self._stat_step = 0

    @staticmethod
    def _safe_normalize(x):
        return F.normalize(x, dim=-1, eps=1e-8)

    @staticmethod
    def _safe_topk_softmax(score, k, temperature):
        k = min(k, score.shape[-1])
        topv, topi = torch.topk(score, k=k, dim=-1)
        valid = torch.isfinite(topv)
        topv = torch.where(valid, topv, torch.full_like(topv, -1e9))
        weight = torch.softmax(topv / max(float(temperature), 1e-6), dim=-1)
        weight = weight * valid.float()
        denom = torch.clamp(weight.sum(dim=-1, keepdim=True), min=1e-8)
        weight = weight / denom
        return topi, weight

    @staticmethod
    def _to_last_step(x):
        if x is None:
            return None
        return x[:, -1] if x.dim() > 1 else x

    def _maybe_print_candidate_stats(self, mask_1d):
        if (not self.print_candidate_stats) or (not self.training):
            return
        if self.print_candidate_every <= 0:
            return
        self._stat_step += 1
        if self._stat_step % self.print_candidate_every != 0:
            return
        cnt = mask_1d.sum(dim=1).float()
        print(
            "[RAG][candidate_stats] step=%d avg_per_time_series=%.1f, min=%.0f, max=%.0f"
            % (self._stat_step, cnt.mean().item(), cnt.min().item(), cnt.max().item())
        )

    def _future_aware_aux_loss(self, sim_enc, topi, cand_future, shortlist_score, query_future):
        if (query_future is None) or (query_future.numel() == 0):
            return None
        qf = query_future[..., 0].transpose(1, 2)
        qf = torch.nan_to_num(qf, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(qf).all():
            return None

        valid = torch.isfinite(shortlist_score)
        dist = torch.norm(cand_future - qf.unsqueeze(2), dim=-1)
        dist = dist.masked_fill(~valid, float("inf"))
        valid_row = torch.isfinite(dist).any(dim=-1)
        if not valid_row.any():
            return None
        pos_idx = torch.argmin(dist, dim=-1)
        pos_global = torch.gather(topi, -1, pos_idx.unsqueeze(-1)).squeeze(-1)
        pos_sim = torch.gather(sim_enc, -1, pos_global.unsqueeze(-1)).squeeze(-1)
        pos_sim = pos_sim[valid_row]
        if pos_sim.numel() == 0:
            return None
        return torch.mean(1.0 - pos_sim)

    def _retrieve_from_bucket(
        self,
        query_emb,
        query_history,
        query_future,
        query_sample_idx,
        bucket,
        input_len,
        output_len,
        exclude_self=False,
        candidate_mask=None,
    ):
        device = query_emb.device
        batch_size, _, num_nodes, _ = query_history.shape

        llm_keys = _as_float_tensor(bucket["llm_keys"], device)
        dyn_keys = _as_float_tensor(bucket["dyn_keys"], device)
        hist = _as_float_tensor(bucket["hist"], device)
        future = _as_float_tensor(bucket["future"], device)
        sample_ids = _as_long_tensor(bucket.get("sample_ids"), device)

        if llm_keys.numel() == 0:
            empty_hist = torch.zeros((batch_size, input_len, num_nodes, 1), device=device)
            empty_prior = torch.zeros((batch_size, output_len, num_nodes, 1), device=device)
            return empty_hist, empty_prior, None

        mask_1d = torch.ones((batch_size, llm_keys.shape[0]), dtype=torch.bool, device=device)
        if candidate_mask is not None:
            mask_1d = mask_1d & candidate_mask
        if exclude_self and query_sample_idx is not None and sample_ids is not None:
            query_sample_idx = torch.as_tensor(query_sample_idx, dtype=torch.long, device=device)
            query_sample_idx = self._to_last_step(query_sample_idx).view(batch_size, 1)
            self_mask = query_sample_idx.eq(sample_ids.view(1, -1))
            mask_1d = mask_1d & (~self_mask)
        self._maybe_print_candidate_stats(mask_1d)
        mask_3d = mask_1d.unsqueeze(1).expand(-1, num_nodes, -1)

        hist_bn_t = query_history[..., 0].transpose(1, 2)
        q_llm = self._safe_normalize(query_emb)
        k_llm = self._safe_normalize(llm_keys)
        score = torch.einsum("nd,md->nm", q_llm, k_llm).unsqueeze(0).expand(batch_size, -1, -1)

        sim_enc = None
        if self.use_retriever_encoder and (self.retriever_encoder is not None):
            q_enc = self.retriever_encoder.encode_query(
                llm_query=query_emb,
                ts_query=hist_bn_t,
            )
            k_enc = self.retriever_encoder.encode_key(
                llm_keys=llm_keys,
                ts_keys=dyn_keys,
            )
            sim_enc = torch.einsum("bnd,md->bnm", q_enc, k_enc)
            score = sim_enc

        score = score.masked_fill(~mask_3d, float("-inf"))
        topi, weight = self._safe_topk_softmax(score, self.top_k, self.temperature)

        cand_hist = hist[topi]
        cand_future = future[topi]
        ref_hist = torch.sum(weight.unsqueeze(-1) * cand_hist, dim=2)
        ref_future = torch.sum(weight.unsqueeze(-1) * cand_future, dim=2)
        bank_hist = ref_hist.transpose(1, 2).unsqueeze(-1)
        bank_prior = ref_future.transpose(1, 2).unsqueeze(-1)

        aux_loss = None
        if (
            (sim_enc is not None)
            and self.training
            and (self.retriever_aux_weight > 0)
            and (query_future is not None)
        ):
            aux_r = min(self.aux_top_r, score.shape[-1])
            shortlist_i = torch.topk(score, k=aux_r, dim=-1).indices
            shortlist_score = torch.gather(score, -1, shortlist_i)
            shortlist_future = future[shortlist_i]
            aux_loss = self._future_aware_aux_loss(
                sim_enc,
                shortlist_i,
                shortlist_future,
                shortlist_score,
                query_future,
            )

        return bank_hist, bank_prior, aux_loss

    def retrieve_from_bank(
        self,
        query_emb,
        query_history,
        x_hour,
        x_minute,
        x_weekday,
        query_sample_idx,
        retrieval_bank,
        input_len,
        output_len,
        query_future=None,
        exclude_self=False,
    ):
        _ = x_minute
        if retrieval_bank is None:
            return None, None, None
        bucket = retrieval_bank.get("global")
        if bucket is None:
            return None, None, None

        device = query_emb.device
        hour_ids = _as_long_tensor(bucket.get("hour_ids"), device)
        weekday_ids = _as_long_tensor(bucket.get("weekday_ids"), device)

        candidate_mask = None
        if self.use_hour_bucket and (x_hour is not None) and (hour_ids is not None):
            hour_vec = self._to_last_step(torch.as_tensor(x_hour, dtype=torch.long, device=device))
            hour_mask = hour_ids.view(1, -1).eq(hour_vec.view(-1, 1))
            candidate_mask = hour_mask

            if (
                self.use_weekday_filter
                and x_weekday is not None
                and weekday_ids is not None
                and torch.any(weekday_ids >= 0)
            ):
                weekday_vec = self._to_last_step(torch.as_tensor(x_weekday, dtype=torch.long, device=device))
                wh_mask = hour_mask & weekday_ids.view(1, -1).eq(weekday_vec.view(-1, 1))
                has_wh = wh_mask.any(dim=-1, keepdim=True)
                candidate_mask = torch.where(has_wh, wh_mask, hour_mask)

            has_hour = hour_mask.any(dim=-1, keepdim=True)
            candidate_mask = torch.where(has_hour, candidate_mask, torch.ones_like(hour_mask))

        bank_hist, bank_prior, aux_loss = self._retrieve_from_bucket(
            query_emb=query_emb,
            query_history=query_history,
            query_future=query_future,
            query_sample_idx=query_sample_idx,
            bucket=bucket,
            input_len=input_len,
            output_len=output_len,
            exclude_self=exclude_self,
            candidate_mask=candidate_mask,
        )
        return (
            torch.nan_to_num(bank_hist, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(bank_prior, nan=0.0, posinf=0.0, neginf=0.0),
            aux_loss,
        )


class RAGIMPEL(nn.Module):
    def __init__(
        self,
        node_dim,
        input_len,
        in_dim,
        embed_dim,
        output_len,
        num_layer,
        llm_enc_dim,
        mp_layers,
        enable_rag=True,
        rag_top_k=5,
        rag_temp=0.2,
        rag_use_hour_bucket=True,
        rag_use_weekday_filter=True,
        rag_use_bank_prior=False,
        rag_exclude_self=True,
        rag_use_retriever_encoder=True,
        retriever_dim=128,
        retriever_aux_weight=0.0,
        retriever_aux_top_r=64,
        rag_print_candidate_stats=False,
        rag_print_candidate_every=200,
        retriever_pretrained_path="",
        retriever_freeze=False,
        **args
    ):
        super().__init__()
        _ = args
        self.node_dim = node_dim
        self.input_len = input_len
        self.input_dim = in_dim
        self.embed_dim = embed_dim
        self.output_len = output_len
        self.num_layer = num_layer
        self.llm_enc_dim = llm_enc_dim
        self.mp_layers = mp_layers

        self.enable_rag = enable_rag
        self.rag_use_bank_prior = rag_use_bank_prior
        self.rag_exclude_self = rag_exclude_self

        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len,
            out_channels=self.embed_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        self.hidden_dim = self.embed_dim + self.node_dim
        self.encoder = nn.Sequential(
            *[
                MultiLayerPerceptron(self.hidden_dim, self.hidden_dim)
                for _ in range(self.num_layer)
            ]
        )
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.output_len,
            kernel_size=(1, 1),
            bias=True,
        )
        self.llm_adapter = nn.Linear(self.llm_enc_dim, self.node_dim)

        self.gconv = nn.ModuleList()
        for _ in range(mp_layers):
            self.gconv.append(GCN(self.hidden_dim, self.hidden_dim, dropout=0.0, support_len=1))

        self.retriever_encoder = RetrieverEncoder(
            llm_dim=self.llm_enc_dim,
            ts_len=self.input_len,
            retriever_dim=retriever_dim,
        )
        self.rag_memory = RAGMemory(
            top_k=rag_top_k,
            temperature=rag_temp,
            use_hour_bucket=rag_use_hour_bucket,
            use_weekday_filter=rag_use_weekday_filter,
            retriever_encoder=self.retriever_encoder,
            use_retriever_encoder=rag_use_retriever_encoder,
            retriever_aux_weight=retriever_aux_weight,
            aux_top_r=retriever_aux_top_r,
            print_candidate_stats=rag_print_candidate_stats,
            print_candidate_every=rag_print_candidate_every,
        )
        self.prior_alpha = nn.Parameter(torch.tensor(0.0))
        self.prior_out_proj = nn.Conv2d(
            in_channels=self.output_len,
            out_channels=self.output_len,
            kernel_size=(1, 1),
            bias=True,
        )
        self.prior_out_gate = nn.Sequential(
            nn.Conv2d(self.output_len * 2, self.output_len, kernel_size=(1, 1), bias=True),
            nn.ReLU(),
            nn.Conv2d(self.output_len, self.output_len, kernel_size=(1, 1), bias=True),
            nn.Sigmoid(),
        )
        if retriever_pretrained_path:
            state = torch.load(retriever_pretrained_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.retriever_encoder.load_state_dict(state, strict=False)
        if retriever_freeze:
            for param in self.retriever_encoder.parameters():
                param.requires_grad = False

    def forward(
        self,
        history_data,
        llm_encoding,
        obs_mask=None,
        x_hour=None,
        x_minute=None,
        x_weekday=None,
        sample_idx=None,
        retrieval_bank=None,
        query_future=None,
        teacher_forcing=False,
        return_aux=False,
    ):
        _ = obs_mask
        input_data = history_data[..., range(self.input_dim)]

        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)

        llm_enc = self.llm_adapter(llm_encoding)
        llm_enc = torch.nan_to_num(llm_enc, nan=0.0, posinf=0.0, neginf=0.0)
        node_emb = llm_enc.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)

        hidden = torch.cat([time_series_emb, node_emb], dim=1)
        adp = F.softmax(F.gelu(torch.mm(llm_enc, llm_enc.T)), dim=1)
        new_supports = [adp]

        for i in range(self.mp_layers):
            hidden = self.gconv[i](hidden, new_supports) + hidden

        bank_prior = None
        retriever_aux_loss = None
        if self.enable_rag and (retrieval_bank is not None):
            _, bank_prior, retriever_aux_loss = self.rag_memory.retrieve_from_bank(
                query_emb=llm_encoding,
                query_history=history_data[..., :1],
                x_hour=x_hour,
                x_minute=x_minute,
                x_weekday=x_weekday,
                query_sample_idx=sample_idx,
                retrieval_bank=retrieval_bank,
                input_len=self.input_len,
                output_len=self.output_len,
                query_future=query_future,
                exclude_self=(teacher_forcing and self.rag_exclude_self),
            )

        hidden = self.encoder(hidden)
        prediction = self.regression_layer(hidden)

        if self.enable_rag and self.rag_use_bank_prior and (bank_prior is not None):
            prior_feat = self.prior_out_proj(
                torch.nan_to_num(bank_prior, nan=0.0, posinf=0.0, neginf=0.0)
            )
            gate = self.prior_out_gate(torch.cat([prediction, prior_feat], dim=1))
            prediction = prediction + self.prior_alpha * gate * prior_feat

        prediction = torch.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
        if return_aux:
            return {
                "pred": prediction,
                "retriever_aux_loss": retriever_aux_loss,
            }
        return prediction


def _as_float_tensor(value, device):
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _as_long_tensor(value, device):
    if value is None:
        return None
    return torch.as_tensor(value, dtype=torch.long, device=device)
