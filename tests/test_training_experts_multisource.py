import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from experiments.training.multisource_data import CityDataContext
from src.trainers.impel_trainer import IMPEL_Trainer


class Loader(list):
    pass


class SyntheticBackbone(torch.nn.Module):
    def forward(self, history_data, supports=None, llm_encoding=None):
        return history_data[:, :2, :, :1] + 0.5


class TinyRagIMPELLike(torch.nn.Module):
    name = "ragimpel"
    rag_memory = True

    def __init__(self):
        super().__init__()
        self.last_kwargs = None

    def forward(self, history_data, supports, llm_encoding, **kwargs):
        self.last_kwargs = kwargs
        return {
            "pred": history_data[:, :2, :, :1],
            "retriever_aux_loss": torch.tensor(2.5, device=history_data.device),
        }


class TinyPlainModel(torch.nn.Module):
    name = "impel"

    def __init__(self):
        super().__init__()
        self.called = False

    def forward(self, history_data, supports, llm_encoding):
        self.called = True
        return history_data[:, :2, :, :1]


def make_context(city, city_id, num_nodes, train_value, val_value):
    train = Loader([
        (
            torch.full((1, 2, num_nodes, 1), train_value),
            torch.full((1, 2, num_nodes, 1), train_value + 1.0),
            torch.tensor([city_id * 10 + 1]),
        )
    ])
    val = Loader([
        (
            torch.full((1, 2, num_nodes, 1), val_value),
            torch.full((1, 2, num_nodes, 1), val_value + 1.0),
            torch.tensor([city_id * 10 + 2]),
        )
    ])
    train.batch_meta_keys = ["sample_ids"]
    val.batch_meta_keys = ["sample_ids"]
    return CityDataContext(
        city=city,
        city_id=city_id,
        loaders={
            "train_loader": train,
            "val_loader": val,
            "test_loader": val,
            "scalers": [],
            "batch_meta_keys": ["sample_ids"],
        },
        llm_encoding=torch.zeros(num_nodes, 4),
        num_nodes=num_nodes,
        null_value=-1.0,
        scalers=[],
        supports=[torch.eye(num_nodes)],
        unknown_set={0},
        known_set=set(range(1, num_nodes)),
        num_masked_nodes=1,
    )


class TrainingExpertsMultiSourceTest(unittest.TestCase):
    def test_itsc_aux_forward_passes_future_and_adds_weighted_aux_loss(self):
        trainer = IMPEL_Trainer.__new__(IMPEL_Trainer)
        trainer._retriever_aux_weight = 0.4
        trainer._model = TinyRagIMPELLike()
        x = torch.ones(1, 2, 2, 1)
        label = torch.full((1, 2, 2, 1), 3.0)

        output = trainer._model_forward_train(
            x,
            supports=[torch.eye(2)],
            llm_encoding=torch.zeros(2, 4),
            label=label,
            batch_meta={"sample_ids": torch.tensor([7])},
        )
        pred, aux_loss = trainer._split_prediction_and_aux(output)
        loss = trainer._add_retriever_aux_loss(torch.tensor(1.0), aux_loss)

        self.assertTrue(torch.equal(pred, x))
        self.assertTrue(torch.equal(trainer.model.last_kwargs["query_future"], label))
        self.assertTrue(trainer.model.last_kwargs["teacher_forcing"])
        self.assertTrue(trainer.model.last_kwargs["return_aux"])
        self.assertEqual(trainer.model.last_kwargs["sample_idx"].tolist(), [7])
        self.assertAlmostEqual(float(loss), 2.0)

    def test_plain_model_training_forward_keeps_legacy_call_signature(self):
        trainer = IMPEL_Trainer.__new__(IMPEL_Trainer)
        trainer._retriever_aux_weight = 0.4
        trainer._model = TinyPlainModel()

        output = trainer._model_forward_train(
            torch.ones(1, 2, 2, 1),
            supports=[torch.eye(2)],
            llm_encoding=torch.zeros(2, 4),
            label=torch.ones(1, 2, 2, 1),
            batch_meta={},
        )
        pred, aux_loss = trainer._split_prediction_and_aux(output)

        self.assertTrue(trainer.model.called)
        self.assertIsNone(aux_loss)
        self.assertEqual(tuple(pred.shape), (1, 2, 2, 1))

    def test_enabled_experts_all_expands_to_full_multisource_set(self):
        from experiments.training.train_experts_multisource import ALL_MULTISOURCE_EXPERTS, _enabled_experts

        self.assertEqual(_enabled_experts("all"), list(ALL_MULTISOURCE_EXPERTS))
        self.assertNotIn("tpb", _enabled_experts("all"))

    def test_candidate_cache_default_experts_omit_tpb(self):
        from experiments.training.cache_candidates import _parse_enabled_experts, build_parser
        from experiments.training.evaluate_zero_shot import build_parser as build_eval_parser

        args = build_parser().parse_args(["--target_city", "T", "--backbone_ckpt", "backbone.pt"])
        eval_args = build_eval_parser().parse_args(
            ["--target_city", "T", "--backbone_ckpt", "backbone.pt", "--router_ckpt", "router.pt"]
        )

        self.assertEqual(args.enabled_experts, "itsc,raft")
        self.assertEqual(eval_args.enabled_experts, "itsc,raft")
        self.assertNotIn("tpb", _parse_enabled_experts("all"))

    def test_build_source_window_bank_merges_cities_and_preserves_metadata(self):
        from experiments.training.train_experts_multisource import collect_expert_payloads, build_source_window_bank

        contexts = {
            "A": make_context("A", 0, 3, 1.0, 2.0),
            "B": make_context("B", 1, 2, 3.0, 4.0),
        }
        payloads = collect_expert_payloads(SyntheticBackbone(), contexts, torch.device("cpu"), output_len=2, output_dim=1)

        bank = build_source_window_bank(payloads["train"], output_len=2, output_dim=1)

        self.assertEqual(bank["keys"].shape, torch.Size([5, 2]))
        self.assertEqual(bank["values"].shape, torch.Size([5, 2, 1]))
        self.assertEqual(bank["source_city_names"], ["A", "A", "A", "B", "B"])
        self.assertEqual(bank["source_city_ids"].tolist(), [0, 0, 0, 1, 1])
        self.assertEqual(bank["sample_ids"].tolist(), [1, 1, 1, 11, 11])

    def test_train_experts_multisource_from_contexts_writes_merged_artifacts(self):
        from experiments.training.train_experts_multisource import train_experts_multisource_from_contexts

        with tempfile.TemporaryDirectory() as tmpdir:
            contexts = {
                "A": make_context("A", 0, 3, 1.0, 2.0),
                "B": make_context("B", 1, 2, 3.0, 4.0),
            }
            args = SimpleNamespace(
                run_dir=tmpdir,
                enabled_experts="calibration,source_window,volatility_peak",
                output_len=2,
                output_dim=1,
                input_dim=1,
                input_len=2,
                max_epochs=2,
                base_lr=0.01,
                patience=2,
                residual_l1=0.0,
                early_stop_min_delta=0.0,
                calibration_scope="shared",
                cal_lr=0.01,
                identity_l1=0.0,
                source_top_k=1,
                source_temperature=0.1,
                source_confidence_threshold=0.0,
                source_alpha_lr=0.01,
                source_alpha_mode="scalar",
                min_top1=0.0,
                min_margin=0.0,
                max_entropy=1.0,
                score_chunk_size=128,
                volatility_hidden_dim=4,
                max_abs_delta=0.2,
                identity_l2=0.0,
                max_delta_penalty=0.0,
                min_history_std=0.0,
                min_history_max=0.0,
                seed=7,
            )

            summary = train_experts_multisource_from_contexts(
                args=args,
                backbone=SyntheticBackbone(),
                contexts=contexts,
                device=torch.device("cpu"),
                source_cities=["A", "B"],
                target_city="T",
            )

            root = Path(tmpdir)
            config = yaml.safe_load((root / "experts_multisource.yaml").read_text(encoding="utf-8"))
            self.assertEqual(set(config["experts"]), {"calibration", "source_window", "volatility_peak"})
            self.assertTrue((root / "calibration" / "calibration.pt").exists())
            self.assertTrue((root / "source_window" / "source_window_bank.pt").exists())
            self.assertTrue((root / "volatility_peak" / "volatility_peak.pt").exists())
            self.assertTrue((root / "history.csv").exists())
            self.assertTrue((root / "summary.json").exists())
            self.assertEqual(summary["target_city"], "T")
            self.assertEqual(summary["source_cities"], ["A", "B"])

    def test_train_experts_multisource_all_writes_full_expert_config(self):
        from experiments.training.train_experts_multisource import ALL_MULTISOURCE_EXPERTS, train_experts_multisource_from_contexts

        with tempfile.TemporaryDirectory() as tmpdir:
            contexts = {
                "A": make_context("A", 0, 3, 1.0, 2.0),
                "B": make_context("B", 1, 2, 3.0, 4.0),
            }
            args = SimpleNamespace(
                run_dir=tmpdir,
                enabled_experts="all",
                output_len=2,
                output_dim=1,
                input_dim=1,
                input_len=2,
                max_epochs=2,
                base_lr=0.01,
                patience=2,
                residual_l1=0.0,
                early_stop_min_delta=0.0,
                calibration_scope="shared",
                cal_lr=0.01,
                identity_l1=0.0,
                source_top_k=1,
                source_temperature=0.1,
                source_confidence_threshold=0.0,
                source_alpha_lr=0.01,
                source_alpha_mode="scalar",
                min_top1=0.0,
                min_margin=0.0,
                max_entropy=1.0,
                score_chunk_size=128,
                volatility_hidden_dim=4,
                max_abs_delta=0.2,
                identity_l2=0.0,
                max_delta_penalty=0.0,
                min_history_std=0.0,
                min_history_max=0.0,
                seed=7,
                itsc_top_k=1,
                itsc_temperature=0.2,
                raft_top_k=1,
                raft_temperature=0.1,
                raft_alpha_lr=0.01,
                raft_alpha_mode="scalar",
                tpb_top_k=1,
                tpb_temperature=1.0,
                tpb_patch_len=1,
                tpb_max_patterns=100,
                itsc_gate_lr=0.01,
                itsc_gate_l1=0.0,
                itsc_gate_hidden_dim=4,
                min_gate_available=0.0,
            )

            summary = train_experts_multisource_from_contexts(
                args=args,
                backbone=SyntheticBackbone(),
                contexts=contexts,
                device=torch.device("cpu"),
                source_cities=["A", "B"],
                target_city="T",
            )

            root = Path(tmpdir)
            config = yaml.safe_load((root / "experts_multisource.yaml").read_text(encoding="utf-8"))
            self.assertEqual(list(config["experts"]), list(ALL_MULTISOURCE_EXPERTS))
            self.assertTrue((root / "itsc" / "itsc_bank.pkl").exists())
            self.assertTrue((root / "raft" / "raft_bank.pt").exists())
            self.assertFalse((root / "tpb" / "tpb_bank.pt").exists())
            self.assertTrue((root / "itsc_segment_gate" / "itsc_segment_gate.pt").exists())
            self.assertEqual(summary["enabled_experts"], list(ALL_MULTISOURCE_EXPERTS))


if __name__ == "__main__":
    unittest.main()
