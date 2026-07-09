import unittest
from unittest import mock
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.rag_moe_impel import RAGMoEIMPEL
from src.rag_moe.config import load_rag_moe_configs
from src.rag_moe.experts.base import ExpertOutput, RAGCorrectionOutput, RAGExpertAdapter
from src.trainers.impel_trainer import IMPEL_Trainer
from src.trainers.rag_moe_router_trainer import (
    compute_good_experts,
    evaluate_router_epoch,
    summarize_router_usage,
    train_router_epoch,
)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, history_data, supports, llm_encoding):
        return history_data[:, -1:, :, :1].repeat(1, 3, 1, 1) + self.bias


class OneExpert(RAGExpertAdapter):
    name = "one"

    def prepare(self, config, data_context):
        self.freeze()

    @torch.no_grad()
    def forward_prior(self, history_data, supports, llm_encoding, batch_meta):
        prior = torch.ones(history_data.shape[0], 3, history_data.shape[2], 1)
        available = torch.ones(history_data.shape[0], history_data.shape[2], dtype=torch.bool)
        return ExpertOutput(
            name=self.name,
            prior=prior,
            available=available,
            aux={"candidate_type": "final_prediction"},
        )

    @torch.no_grad()
    def forward_correction(self, history_data, supports, llm_encoding, batch_meta, baseline_pred):
        delta = torch.ones_like(baseline_pred)
        available = torch.ones(
            history_data.shape[0],
            history_data.shape[2],
            dtype=torch.bool,
            device=history_data.device,
        )
        return RAGCorrectionOutput(
            name=self.name,
            delta=delta,
            available=available,
            raw_prior=baseline_pred + delta,
            aux={"correction_type": "unit_delta"},
        )

class RecordingExpert(OneExpert):
    name = "recording"

    def __init__(self):
        super().__init__()
        self.seen_sample_ids = []
        self.last_batch_meta = None

    @torch.no_grad()
    def forward_prior(self, history_data, supports, llm_encoding, batch_meta):
        self.seen_sample_ids.append(batch_meta["sample_ids"].detach().cpu().clone())
        self.last_batch_meta = dict(batch_meta)
        return super().forward_prior(history_data, supports, llm_encoding, batch_meta)

    @torch.no_grad()
    def forward_correction(self, history_data, supports, llm_encoding, batch_meta, baseline_pred):
        self.seen_sample_ids.append(batch_meta["sample_ids"].detach().cpu().clone())
        self.last_batch_meta = dict(batch_meta)
        return super().forward_correction(history_data, supports, llm_encoding, batch_meta, baseline_pred)


class RawPriorMismatchExpert(RAGExpertAdapter):
    name = "mismatch"

    def prepare(self, config, data_context):
        self.freeze()

    @torch.no_grad()
    def forward_correction(self, history_data, supports, llm_encoding, batch_meta, baseline_pred):
        delta = torch.full_like(baseline_pred, 1.0)
        available = torch.ones(
            baseline_pred.shape[0],
            baseline_pred.shape[2],
            dtype=torch.bool,
            device=baseline_pred.device,
        )
        return RAGCorrectionOutput(
            name=self.name,
            delta=delta,
            available=available,
            raw_prior=torch.full_like(baseline_pred, 100.0),
            aux={"correction_type": "test_mismatch"},
        )


class RAGMoERouterTrainingTest(unittest.TestCase):
    def test_impel_trainer_consumes_direct_expert_before_base_init(self):
        recorded_kwargs = {}

        def fake_base_init(trainer, **kwargs):
            recorded_kwargs.update(kwargs)

        with mock.patch("src.trainers.impel_trainer.BaseTrainer.__init__", fake_base_init), \
                mock.patch.object(IMPEL_Trainer, "_calculate_supports", return_value=[torch.eye(1)]):
            trainer = IMPEL_Trainer(
                unknown_set=set(),
                known_set={0},
                n_m=0,
                llm_encoding=torch.zeros(1, 1),
                model=nn.Linear(1, 1),
                adj_mat=torch.eye(1).numpy(),
                filter_type="identity",
                direct_expert="",
            )

        self.assertNotIn("direct_expert", recorded_kwargs)
        self.assertEqual(trainer.direct_expert, "")

    def test_impel_trainer_samples_random_training_unknown_nodes(self):
        def fake_base_init(trainer, **kwargs):
            trainer._aug = 1.0

        with mock.patch("src.trainers.impel_trainer.BaseTrainer.__init__", fake_base_init), \
                mock.patch.object(IMPEL_Trainer, "_calculate_supports", return_value=[torch.eye(4)]), \
                mock.patch("src.trainers.impel_trainer.np.random.choice", side_effect=[
                    torch.tensor([0, 2]).numpy(),
                    torch.tensor([1, 3]).numpy(),
                ]):
            trainer = IMPEL_Trainer(
                unknown_set={0, 1},
                known_set={2, 3},
                n_m=0,
                llm_encoding=torch.zeros(4, 1),
                random_unknown_nodes_each_batch=True,
                num_random_unknown_nodes=2,
                model=nn.Linear(1, 1),
                adj_mat=torch.eye(4).numpy(),
                filter_type="identity",
                direct_expert="",
            )

            first_known = trainer._training_known_set()
            second_known = trainer._training_known_set()

        self.assertEqual(first_known, {1, 3})
        self.assertEqual(second_known, {0, 2})

    def test_compute_good_experts_keeps_baseline_fallback_trainable(self):
        candidates = torch.tensor(
            [
                [
                    [[[2.0]], [[1.0]]],
                    [[[1.0]], [[1.0]]],
                    [[[0.5]], [[1.0]]],
                ],
            ]
        ).permute(0, 2, 1, 3, 4)
        label = torch.ones(1, 2, 1, 1)

        good, errors = compute_good_experts(candidates, label, oracle_margin=0.98)

        self.assertEqual(tuple(good.shape), (1, 1, 3))
        self.assertEqual(tuple(errors.shape), (1, 1, 3))
        self.assertTrue(good[0, 0, 0].item())
        self.assertTrue(good[0, 0, 1].item())
        self.assertTrue(good[0, 0, 2].item())

    def test_train_router_epoch_updates_router_only(self):
        model = RAGMoEIMPEL(
            TinyBackbone(),
            [OneExpert()],
            output_len=3,
            output_dim=1,
            router_hidden_dim=8,
            router_dropout=0.0,
        )
        before_backbone = model.backbone.bias.detach().clone()
        before_router = [p.detach().clone() for p in model.router.parameters()]

        x = torch.zeros(4, 6, 2, 1)
        y = torch.ones(4, 3, 2, 1)
        loader = DataLoader(TensorDataset(x, y), batch_size=2)
        optimizer = torch.optim.Adam(model.router.parameters(), lr=0.01)

        metrics = train_router_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            supports=[torch.eye(2)],
            llm_encoding=torch.zeros(2, 8),
            null_value=-1.0,
            lambda_select=0.2,
            lambda_sparse=0.01,
            oracle_margin=0.98,
            device=torch.device("cpu"),
        )

        self.assertGreater(metrics["loss"], 0)
        self.assertTrue(torch.equal(model.backbone.bias.detach(), before_backbone))
        self.assertTrue(any(not torch.equal(p.detach(), before_router[i]) for i, p in enumerate(model.router.parameters())))

    def test_evaluate_router_epoch_does_not_update_router(self):
        model = RAGMoEIMPEL(
            TinyBackbone(),
            [OneExpert()],
            output_len=3,
            output_dim=1,
            router_hidden_dim=8,
            router_dropout=0.0,
        )
        before_router = [p.detach().clone() for p in model.router.parameters()]

        x = torch.zeros(4, 6, 2, 1)
        y = torch.ones(4, 3, 2, 1)
        loader = DataLoader(TensorDataset(x, y), batch_size=2)

        metrics = evaluate_router_epoch(
            model=model,
            loader=loader,
            supports=[torch.eye(2)],
            llm_encoding=torch.zeros(2, 8),
            null_value=-1.0,
            lambda_select=0.2,
            lambda_sparse=0.01,
            oracle_margin=0.98,
            device=torch.device("cpu"),
        )

        self.assertGreater(metrics["loss"], 0)
        self.assertTrue(all(torch.equal(p.detach(), before_router[i]) for i, p in enumerate(model.router.parameters())))

    def test_train_router_epoch_oracle_uses_baseline_plus_delta_candidates(self):
        model = RAGMoEIMPEL(
            TinyBackbone(),
            [RawPriorMismatchExpert()],
            output_len=3,
            output_dim=1,
            router_hidden_dim=8,
            router_dropout=0.0,
        )
        x = torch.full((2, 6, 1, 1), 3.0)
        y = torch.full((2, 3, 1, 1), 4.0)
        loader = DataLoader(TensorDataset(x, y), batch_size=2)
        optimizer = torch.optim.Adam(model.router.parameters(), lr=0.01)
        captured = {}
        original_compute_good = compute_good_experts

        def recording_compute_good(candidates, label, oracle_margin):
            captured["candidates"] = candidates.detach().clone()
            return original_compute_good(candidates, label, oracle_margin)

        with mock.patch("src.trainers.rag_moe_router_trainer.compute_good_experts", recording_compute_good):
            train_router_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                supports=[torch.eye(1)],
                llm_encoding=torch.zeros(1, 8),
                null_value=-1.0,
                lambda_select=0.2,
                lambda_sparse=0.01,
                oracle_margin=0.98,
                device=torch.device("cpu"),
            )

        self.assertTrue(torch.equal(captured["candidates"][:, :, 0], torch.full((2, 3, 1, 1), 3.0)))
        self.assertTrue(torch.equal(captured["candidates"][:, :, 1], torch.full((2, 3, 1, 1), 4.0)))

    def test_train_router_epoch_passes_batch_metadata_to_experts(self):
        expert = RecordingExpert()
        model = RAGMoEIMPEL(
            TinyBackbone(),
            [expert],
            output_len=3,
            output_dim=1,
            router_hidden_dim=8,
            router_dropout=0.0,
        )

        x = torch.zeros(4, 6, 2, 1)
        y = torch.ones(4, 3, 2, 1)
        sample_ids = torch.arange(4)
        x_hour = torch.arange(24).view(4, 6)
        loader = DataLoader(TensorDataset(x, y, sample_ids, x_hour), batch_size=2)
        loader.batch_meta_keys = ["sample_ids", "x_hour"]
        optimizer = torch.optim.Adam(model.router.parameters(), lr=0.01)

        train_router_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            supports=[torch.eye(2)],
            llm_encoding=torch.zeros(2, 8),
            null_value=-1.0,
            lambda_select=0.2,
            lambda_sparse=0.01,
            oracle_margin=0.98,
            device=torch.device("cpu"),
        )

        self.assertEqual([ids.tolist() for ids in expert.seen_sample_ids], [[0, 1], [2, 3]])

    def test_train_router_epoch_infers_legacy_time_metadata_tuple(self):
        expert = RecordingExpert()
        model = RAGMoEIMPEL(
            TinyBackbone(),
            [expert],
            output_len=3,
            output_dim=1,
            router_hidden_dim=8,
            router_dropout=0.0,
        )
        x = torch.zeros(2, 6, 2, 1)
        y = torch.ones(2, 3, 2, 1)
        x_hour = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
        x_minute = torch.zeros(2, 6, dtype=torch.long)
        x_weekday = torch.tensor([[0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 2]])
        sample_ids = torch.tensor([10, 11])
        loader = DataLoader(TensorDataset(x, y, x_hour, x_minute, x_weekday, sample_ids), batch_size=2)
        optimizer = torch.optim.Adam(model.router.parameters(), lr=0.01)

        train_router_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            supports=[torch.eye(2)],
            llm_encoding=torch.zeros(2, 8),
            null_value=-1.0,
            lambda_select=0.2,
            lambda_sparse=0.01,
            oracle_margin=0.98,
            device=torch.device("cpu"),
        )

        self.assertTrue(torch.equal(expert.last_batch_meta["x_hour"], x_hour))
        self.assertTrue(torch.equal(expert.last_batch_meta["x_minute"], x_minute))
        self.assertTrue(torch.equal(expert.last_batch_meta["x_weekday"], x_weekday))
        self.assertTrue(torch.equal(expert.last_batch_meta["sample_ids"], sample_ids))



class RAGMoEConfigTest(unittest.TestCase):
    def test_load_rag_moe_configs_reads_experts_and_router(self):
        root = Path(__file__).resolve().parents[1]
        experts, router = load_rag_moe_configs(
            root / "configs" / "rag_moe" / "experts.yaml",
            root / "configs" / "rag_moe" / "router.yaml",
        )

        self.assertGreaterEqual(set(experts["experts"]), {"itsc", "raft", "tpb"})
        self.assertEqual(experts["experts"]["itsc"]["mode"], "residual")
        self.assertEqual(experts["experts"]["raft"]["mode"], "residual")
        self.assertEqual(router["router"]["stage1"]["type"], "multilabel_selector")


class RAGMoEDiagnosticsTest(unittest.TestCase):
    def test_summarize_router_usage_reports_selected_rate_and_weight(self):
        outputs = {
            "candidate_names": ["baseline", "itsc"],
            "active_mask": torch.tensor([[[True, True], [True, False]]]),
            "weights": torch.tensor([[[0.25, 0.75], [1.0, 0.0]]]),
        }

        rows = summarize_router_usage(outputs, source_city="Delivery_SH", target_city="Delivery_HZ", seed=0)

        self.assertEqual(
            rows,
            [
                {
                    "source_city": "Delivery_SH",
                    "target_city": "Delivery_HZ",
                    "seed": 0,
                    "expert": "baseline",
                    "avg_selected_rate": 1.0,
                    "avg_weight": 0.625,
                },
                {
                    "source_city": "Delivery_SH",
                    "target_city": "Delivery_HZ",
                    "seed": 0,
                    "expert": "itsc",
                    "avg_selected_rate": 0.5,
                    "avg_weight": 0.375,
                },
            ],
        )

    def test_summarize_router_usage_prefers_correction_names(self):
        outputs = {
            "candidate_names": ["baseline", "itsc"],
            "correction_names": ["none", "itsc_delta"],
            "active_mask": torch.tensor([[[True, True], [True, False]]]),
            "weights": torch.tensor([[[0.25, 0.75], [1.0, 0.0]]]),
        }

        rows = summarize_router_usage(outputs, source_city="Delivery_SH", target_city="Delivery_HZ", seed=0)

        self.assertEqual([row["expert"] for row in rows], ["none", "itsc_delta"])

    def test_summarize_router_usage_accepts_correction_names_without_candidate_names(self):
        outputs = {
            "correction_names": ["none", "itsc"],
            "active_mask": torch.tensor([[[True, False]]]),
            "weights": torch.tensor([[[1.0, 0.0]]]),
        }

        rows = summarize_router_usage(outputs, source_city="Delivery_SH", target_city="Delivery_HZ", seed=0)

        self.assertEqual([row["expert"] for row in rows], ["none", "itsc"])


if __name__ == "__main__":
    unittest.main()
