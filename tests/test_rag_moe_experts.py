
import os
import pickle

import io

import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch

from src.rag_moe.experts.base import (
    ExpertOutput,
    RAGCorrectionOutput,
    RAGExpertAdapter,
    validate_correction_output,
    validate_expert_output,
)

from src.rag_moe.original.itsc_full_model import ITSCFullModelPredictor
from src.rag_moe.original.itsc_correction import ITSCResidualCorrection
from src.rag_moe.original.itsc_ragimpel import RAGIMPEL
from src.rag_moe.original.raft_full_model import RAFTCompatModel, RAFTFullModelPredictor
from src.rag_moe.original.tpb_full_model import TPBCompatModel, TPBFullModelPredictor

from src.rag_moe.experts.itsc import RetrieverEncoder
from src.rag_moe.experts.tpb import TPBExpert

from src.rag_moe.registry import build_experts, get_expert_class


ITSC_TINY_MODEL_CONFIG = {
    "node_dim": 2,
    "input_len": 2,
    "in_dim": 1,
    "embed_dim": 4,
    "output_len": 3,
    "num_layer": 1,
    "llm_enc_dim": 4,
    "mp_layers": 0,
    "enable_rag": False,
}

RAFT_TINY_MODEL_CONFIG = {
    "seq_len": 2,
    "pred_len": 3,
    "enc_in": 1,
    "n_period": 1,
    "topm": 1,
}

TPB_TINY_MODEL_CONFIG = {
    "pred_num": 3,
    "his_num": 4,
    "output_dim": 1,
}


class DummyAdapter(RAGExpertAdapter):
    name = "dummy"

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))


class RAGExpertAdapterContractTest(unittest.TestCase):
    def test_validation_accepts_matching_prior_shape(self):
        baseline_pred = torch.zeros(2, 3, 4, 1)
        output = ExpertOutput(
            name="temporal",
            prior=torch.ones_like(baseline_pred),
            available=torch.ones(2, 4, dtype=torch.bool),
        )

        self.assertIs(
            validate_expert_output(output, baseline_pred, expected_name="temporal"),
            output,
        )

    def test_validation_rejects_wrong_prior_shape(self):
        baseline_pred = torch.zeros(2, 3, 4, 1)
        output = ExpertOutput(
            name="temporal",
            prior=torch.zeros(2, 3, 4),
            available=torch.ones(2, 4, dtype=torch.bool),
        )

        with self.assertRaisesRegex(ValueError, "prior shape"):
            validate_expert_output(output, baseline_pred, expected_name="temporal")

    def test_validation_rejects_wrong_expert_name(self):
        baseline_pred = torch.zeros(2, 3, 4, 1)
        output = ExpertOutput(
            name="spatial",
            prior=torch.ones_like(baseline_pred),
            available=torch.ones(2, 4, dtype=torch.bool),
        )

        with self.assertRaisesRegex(ValueError, "expected expert"):
            validate_expert_output(output, baseline_pred, expected_name="temporal")

    def test_validation_accepts_batch_node_availability_mask(self):
        baseline_pred = torch.zeros(2, 3, 4, 1)
        output = ExpertOutput(
            name="temporal",
            prior=torch.ones_like(baseline_pred),
            available=torch.ones(2, 4, dtype=torch.bool),
        )

        self.assertIs(
            validate_expert_output(output, baseline_pred, expected_name="temporal"),
            output,
        )

    def test_validation_accepts_batch_broadcast_availability_mask(self):
        baseline_pred = torch.zeros(2, 3, 4, 1)
        output = ExpertOutput(
            name="temporal",
            prior=torch.ones_like(baseline_pred),
            available=torch.ones(2, 1, dtype=torch.bool),
        )

        self.assertIs(
            validate_expert_output(output, baseline_pred, expected_name="temporal"),
            output,
        )

    def test_validation_rejects_wrong_availability_rank(self):
        baseline_pred = torch.zeros(2, 3, 4, 1)
        output = ExpertOutput(
            name="temporal",
            prior=torch.ones_like(baseline_pred),
            available=torch.ones(2, 3, 4, dtype=torch.bool),
        )

        with self.assertRaisesRegex(ValueError, "available"):
            validate_expert_output(output, baseline_pred, expected_name="temporal")

    def test_validation_rejects_wrong_availability_batch(self):
        baseline_pred = torch.zeros(2, 3, 4, 1)
        output = ExpertOutput(
            name="temporal",
            prior=torch.ones_like(baseline_pred),
            available=torch.ones(3, 4, dtype=torch.bool),
        )

        with self.assertRaisesRegex(ValueError, "available"):
            validate_expert_output(output, baseline_pred, expected_name="temporal")

    def test_validation_rejects_wrong_availability_nodes(self):
        baseline_pred = torch.zeros(2, 3, 4, 1)
        output = ExpertOutput(
            name="temporal",
            prior=torch.ones_like(baseline_pred),
            available=torch.ones(2, 5, dtype=torch.bool),
        )

        with self.assertRaisesRegex(ValueError, "available"):
            validate_expert_output(output, baseline_pred, expected_name="temporal")

    def test_validation_rejects_non_tensor_availability(self):
        baseline_pred = torch.zeros(2, 3, 4, 1)
        output = ExpertOutput(
            name="temporal",
            prior=torch.ones_like(baseline_pred),
            available=True,
        )

        with self.assertRaisesRegex(ValueError, "available"):
            validate_expert_output(output, baseline_pred, expected_name="temporal")

    def test_correction_validation_accepts_delta_and_raw_prior(self):
        baseline_pred = torch.zeros(2, 3, 4, 1)
        output = RAGCorrectionOutput(
            name="temporal",
            delta=torch.ones_like(baseline_pred),
            available=torch.ones(2, 4, dtype=torch.bool),
            raw_prior=torch.full_like(baseline_pred, 2.0),
        )

        self.assertIs(
            validate_correction_output(output, baseline_pred, expected_name="temporal"),
            output,
        )

    def test_correction_validation_rejects_wrong_delta_shape(self):
        baseline_pred = torch.zeros(2, 3, 4, 1)
        output = RAGCorrectionOutput(
            name="temporal",
            delta=torch.ones(2, 3, 5, 1),
            available=torch.ones(2, 4, dtype=torch.bool),
        )

        with self.assertRaisesRegex(ValueError, "delta shape"):
            validate_correction_output(output, baseline_pred, expected_name="temporal")

    def test_base_forward_correction_converts_raw_prior_to_residual(self):
        class PriorAdapter(RAGExpertAdapter):
            name = "prior"

            @torch.no_grad()
            def forward_prior(self, history_data, supports=None, llm=None, batch_meta=None):
                prior = torch.full((1, 3, 2, 1), 5.0)
                return ExpertOutput(
                    name=self.name,
                    prior=prior,
                    available=torch.ones(1, 2, dtype=torch.bool),
                    aux={"candidate_type": "raw_prior"},
                )

        baseline_pred = torch.full((1, 3, 2, 1), 3.0)
        output = PriorAdapter().forward_correction(
            torch.zeros(1, 6, 2, 1),
            supports=[torch.eye(2)],
            llm_encoding=torch.zeros(2, 8),
            batch_meta={"output_len": 3, "output_dim": 1},
            baseline_pred=baseline_pred,
        )

        self.assertTrue(torch.equal(output.delta, torch.full_like(baseline_pred, 2.0)))
        self.assertTrue(torch.equal(output.raw_prior, torch.full_like(baseline_pred, 5.0)))
        self.assertEqual(output.aux["correction_type"], "raw_prior_minus_baseline")

    def test_base_forward_correction_preserves_final_candidate_semantics(self):
        class FinalCandidateAdapter(RAGExpertAdapter):
            name = "final"

            @torch.no_grad()
            def forward_prior(self, history_data, supports=None, llm=None, batch_meta=None):
                prior = torch.full((1, 3, 2, 1), 5.0)
                return ExpertOutput(
                    name=self.name,
                    prior=prior,
                    available=torch.ones(1, 2, dtype=torch.bool),
                    aux={"candidate_type": "raw_prior"},
                )

            @torch.no_grad()
            def forward_candidate(self, history_data, supports, llm_encoding, batch_meta, baseline_pred):
                raw = self.forward_prior(history_data, supports, llm_encoding, batch_meta)
                aux = dict(raw.aux or {})
                aux["candidate_type"] = "final_prediction"
                aux["raw_prior"] = raw.prior
                return ExpertOutput(
                    name=self.name,
                    prior=baseline_pred + 0.25 * (raw.prior - baseline_pred),
                    available=raw.available,
                    aux=aux,
                )

        baseline_pred = torch.full((1, 3, 2, 1), 3.0)
        output = FinalCandidateAdapter().forward_correction(
            torch.zeros(1, 6, 2, 1),
            supports=[torch.eye(2)],
            llm_encoding=torch.zeros(2, 8),
            batch_meta={"output_len": 3, "output_dim": 1},
            baseline_pred=baseline_pred,
        )

        self.assertTrue(torch.equal(output.delta, torch.full_like(baseline_pred, 0.5)))
        self.assertTrue(torch.equal(output.raw_prior, torch.full_like(baseline_pred, 3.5)))
        self.assertEqual(output.aux["raw_prior"].shape, baseline_pred.shape)
        self.assertEqual(output.aux["correction_type"], "final_prediction_minus_baseline")

    def test_freeze_disables_adapter_parameter_gradients(self):
        adapter = DummyAdapter()

        adapter.freeze()

        self.assertTrue(list(adapter.parameters()))
        self.assertTrue(all(not param.requires_grad for param in adapter.parameters()))


class InitialExpertRegistryTest(unittest.TestCase):
    def test_registry_returns_initial_expert_classes(self):
        for name in ("itsc", "raft"):
            self.assertEqual(get_expert_class(name).name, name)

    def test_registry_rejects_unknown_expert(self):
        for name in ("missing", "tpb"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(KeyError, "unknown expert"):
                    get_expert_class(name)

    def test_initial_experts_return_valid_priors_and_availability_masks(self):
        baseline_pred = torch.zeros(2, 24, 5, 1)
        history = torch.arange(2 * 24 * 5, dtype=torch.float32).reshape(2, 24, 5, 1)
        supports = [torch.eye(5)]
        llm = torch.zeros(2, 5, 3)
        batch_meta = {"city": ["sh", "hz"]}

        for name in ("itsc", "raft"):
            expert = get_expert_class(name)()
            expert.prepare({"scale": 0.5}, {"dataset": "Delivery_SH"})

            output = expert.forward_prior(history, supports, llm, batch_meta)

            self.assertIs(
                validate_expert_output(output, baseline_pred, expected_name=name),
                output,
            )
            self.assertEqual(tuple(output.available.shape), (2, 5))

    def test_build_experts_preserves_order_and_freezes_parameters(self):
        configs = {
            "itsc": {"scale": 0.25},
            "raft": {"scale": 0.5},
        }
        experts = build_experts(
            ["itsc", "raft"], configs, {"dataset": "Delivery_SH"}
        )

        self.assertEqual([expert.name for expert in experts], ["itsc", "raft"])
        for expert in experts:
            self.assertTrue(
                all(not param.requires_grad for param in expert.parameters())
            )

    def test_expert_prepare_stores_source_and_target_context(self):
        expert = get_expert_class("raft")()

        expert.prepare(
            {"scale": 1.0},
            {
                "dataset": "Delivery_HZ",
                "source_data": "Delivery_SH",
                "target_data": "Delivery_HZ",
            },
        )

        self.assertEqual(expert.data_context["source_data"], "Delivery_SH")
        self.assertEqual(expert.data_context["target_data"], "Delivery_HZ")

    def test_itsc_adapter_uses_loaded_bank_prior(self):
        bank = {
            "global": {
                "llm_keys": torch.zeros(1, 4),
                "dyn_keys": torch.zeros(1, 2),
                "hist": torch.zeros(1, 2),
                "future": torch.full((1, 3), 7.0),
            },
        }
        expert = get_expert_class("itsc")()
        expert.prepare({"bank": bank, "top_k": 1}, {"dataset": "Delivery_SH"})
        history = torch.zeros(1, 2, 2, 1)

        output = expert.forward_prior(
            history,
            supports=[torch.eye(2)],
            llm=torch.zeros(2, 4),
            batch_meta={"output_len": 3, "output_dim": 1},
        )

        self.assertTrue(torch.allclose(output.prior, torch.full((1, 3, 2, 1), 7.0)))
        self.assertTrue(output.aux["bank_used"])


    def test_itsc_full_model_requires_checkpoint_and_bank_path(self):
        expert = get_expert_class("itsc")()
        with self.assertRaisesRegex(ValueError, "ITSCExpert full_model requires"):
            expert.prepare({"mode": "full_model"}, {"dataset": "Delivery_SH"})

    def test_itsc_full_model_uses_predictor_candidate(self):
        class FakeITSCFullModel(torch.nn.Module):
            def forward(self, history_data, supports=None, llm_encoding=None, batch_meta=None):
                return torch.full(
                    (
                        history_data.shape[0],
                        batch_meta["output_len"],
                        history_data.shape[2],
                        batch_meta["output_dim"],
                    ),
                    13.0,
                    device=history_data.device,
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path, bank_path = _write_itsc_full_model_artifacts(tmpdir)
            expert = get_expert_class("itsc")()
            expert.prepare(
                {
                    "mode": "full_model",
                    "checkpoint_path": checkpoint_path,
                    "bank_path": bank_path,
                    "model_config": ITSC_TINY_MODEL_CONFIG,
                },
                {"dataset": "Delivery_SH", "input_len": 2, "llm_enc_dim": 4},
            )
            expert.full_model = FakeITSCFullModel()
            history = torch.zeros(1, 2, 2, 1)

            output = expert.forward_prior(
                history,
                supports=[torch.eye(2)],
                llm=torch.zeros(2, 4),
                batch_meta={"output_len": 3, "output_dim": 1},
            )

        self.assertTrue(torch.equal(output.prior, torch.full((1, 3, 2, 1), 13.0)))
        self.assertTrue(output.aux["full_model_used"])

    def test_itsc_residual_mode_uses_prior_branch_without_full_prediction_candidate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path, bank_path = _write_itsc_full_model_artifacts(tmpdir)
            expert = get_expert_class("itsc")()
            expert.prepare(
                {
                    "mode": "residual",
                    "checkpoint_path": checkpoint_path,
                    "bank_path": bank_path,
                    "model_config": ITSC_TINY_MODEL_CONFIG,
                },
                {"dataset": "Delivery_SH", "input_len": 2, "llm_enc_dim": 4},
            )
            baseline = torch.zeros(1, 3, 2, 1)
            output = expert.forward_correction(
                torch.zeros(1, 2, 2, 1),
                supports=[torch.eye(2)],
                llm_encoding=torch.zeros(2, 4),
                batch_meta={"output_len": 3, "output_dim": 1},
                baseline_pred=baseline,
            )

        self.assertEqual(output.name, "itsc")
        self.assertEqual(tuple(output.delta.shape), (1, 3, 2, 1))
        self.assertEqual(tuple(output.raw_prior.shape), (1, 3, 2, 1))
        self.assertFalse(
            torch.equal(output.raw_prior, torch.zeros_like(output.raw_prior))
        )
        self.assertEqual(output.aux["correction_type"], "itsc_prior_gate")
        self.assertIn("prior_alpha", output.aux)

    def test_itsc_residual_rejects_sparse_checkpoint_during_prepare(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sparse_checkpoint_path = f"{tmpdir}/sparse_residual_checkpoint.pt"
            bank_path = _write_itsc_bank_artifact(tmpdir)
            full_state = RAGIMPEL(**ITSC_TINY_MODEL_CONFIG).state_dict()
            sparse_state = {}
            for root in [
                "rag_memory",
                "prior_alpha",
                "prior_out_proj",
                "prior_out_gate",
            ]:
                for key, value in full_state.items():
                    if key == root or key.startswith(root + "."):
                        sparse_state[key] = value
                        break
            torch.save(sparse_state, sparse_checkpoint_path)

            expert = get_expert_class("itsc")()
            with self.assertRaisesRegex(RuntimeError, "ITSCExpert checkpoint_path"):
                expert.prepare(
                    {
                        "mode": "residual",
                        "checkpoint_path": sparse_checkpoint_path,
                        "bank_path": bank_path,
                        "model_config": ITSC_TINY_MODEL_CONFIG,
                    },
                    {"dataset": "Delivery_SH"},
                )

    def test_itsc_residual_correction_formula_uses_baseline_in_gate(self):
        class FakeMemory(torch.nn.Module):
            def retrieve_from_bank(self, **kwargs):
                bank_prior = torch.full((1, 3, 2, 1), 4.0)
                return None, bank_prior, None

        class FixedGate(torch.nn.Module):
            def forward(self, value):
                self.last_input = value.detach().clone()
                return torch.full(
                    (1, 3, 2, 1),
                    0.5,
                    dtype=value.dtype,
                    device=value.device,
                )

        class FakeModel(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.rag_memory = FakeMemory()
                self.prior_out_proj = torch.nn.Identity()
                self.prior_out_gate = FixedGate()
                self.prior_alpha = torch.nn.Parameter(torch.tensor(0.25))
                self.enable_rag = True
                self.rag_use_bank_prior = True
                self.input_len = 2
                self.output_len = 3
                self.rag_exclude_self = True

            def state_dict(self):
                return {
                    "rag_memory.marker": torch.tensor(1.0),
                    "prior_out_proj.marker": torch.tensor(1.0),
                    "prior_out_gate.marker": torch.tensor(1.0),
                    "prior_alpha": torch.tensor(0.25),
                }

            def load_state_dict(self, state, strict=False):
                return torch.nn.modules.module._IncompatibleKeys([], [])

            def eval(self):
                return self

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = f"{tmpdir}/model.pt"
            bank_path = _write_itsc_bank_artifact(tmpdir)
            torch.save(
                {
                    "rag_memory.marker": torch.tensor(1.0),
                    "prior_out_proj.marker": torch.tensor(1.0),
                    "prior_out_gate.marker": torch.tensor(1.0),
                    "prior_alpha": torch.tensor(0.25),
                },
                checkpoint_path,
            )
            extractor = ITSCResidualCorrection(
                checkpoint_path=checkpoint_path,
                bank_path=bank_path,
                model_config=ITSC_TINY_MODEL_CONFIG,
                model_factory=FakeModel,
            )
            output = extractor.forward_correction(
                history_data=torch.zeros(1, 2, 2, 1),
                baseline_pred=torch.ones(1, 3, 2, 1),
                llm_encoding=torch.zeros(2, 4),
                batch_meta={"sample_ids": torch.tensor([0])},
            )

        self.assertTrue(torch.equal(output.raw_prior, torch.full((1, 3, 2, 1), 4.0)))
        self.assertTrue(torch.equal(output.delta, torch.full((1, 3, 2, 1), 0.5)))
        gate_input = extractor.model.prior_out_gate.last_input
        self.assertTrue(torch.equal(gate_input[:, :3], torch.ones(1, 3, 2, 1)))
        self.assertTrue(torch.equal(gate_input[:, 3:], torch.full((1, 3, 2, 1), 4.0)))

    def test_itsc_full_model_rejects_corrupt_artifacts_during_prepare(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path, bank_path = _write_itsc_full_model_artifacts(tmpdir)
            corrupt_checkpoint_path = f"{tmpdir}/corrupt_checkpoint.txt"
            with open(corrupt_checkpoint_path, "w") as handle:
                handle.write("not a torch checkpoint")

            expert = get_expert_class("itsc")()
            with self.assertRaises(RuntimeError):
                expert.prepare(
                    {
                        "mode": "full_model",
                        "checkpoint_path": corrupt_checkpoint_path,
                        "bank_path": bank_path,
                        "model_config": ITSC_TINY_MODEL_CONFIG,
                    },
                    {"dataset": "Delivery_SH"},
                )

            corrupt_bank_path = f"{tmpdir}/corrupt_bank.pkl"
            with open(corrupt_bank_path, "w") as handle:
                handle.write("not a pickle bank")

            expert = get_expert_class("itsc")()
            with self.assertRaises((pickle.UnpicklingError, EOFError)):
                expert.prepare(
                    {
                        "mode": "full_model",
                        "checkpoint_path": checkpoint_path,
                        "bank_path": corrupt_bank_path,
                        "model_config": ITSC_TINY_MODEL_CONFIG,
                    },
                    {"dataset": "Delivery_SH"},
                )

    def test_itsc_full_model_rejects_empty_checkpoint_during_prepare(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_checkpoint_path = f"{tmpdir}/empty_checkpoint.pt"
            bank_path = _write_itsc_bank_artifact(tmpdir)
            torch.save({}, empty_checkpoint_path)

            expert = get_expert_class("itsc")()
            with self.assertRaisesRegex(RuntimeError, "ITSCExpert checkpoint_path"):
                expert.prepare(
                    {
                        "mode": "full_model",
                        "checkpoint_path": empty_checkpoint_path,
                        "bank_path": bank_path,
                        "model_config": ITSC_TINY_MODEL_CONFIG,
                    },
                    {"dataset": "Delivery_SH"},
                )

    def test_itsc_full_model_rejects_sparse_checkpoint_during_prepare(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sparse_checkpoint_path = f"{tmpdir}/sparse_checkpoint.pt"
            bank_path = _write_itsc_bank_artifact(tmpdir)
            full_state = RAGIMPEL(**ITSC_TINY_MODEL_CONFIG).state_dict()
            sparse_roots = [
                "time_series_emb_layer",
                "encoder",
                "regression_layer",
                "llm_adapter",
                "retriever_encoder",
                "rag_memory",
                "prior_alpha",
                "prior_out_proj",
                "prior_out_gate",
            ]
            sparse_state = {}
            for root in sparse_roots:
                for key, value in full_state.items():
                    if key == root or key.startswith(root + "."):
                        sparse_state[key] = value
                        break
            torch.save(sparse_state, sparse_checkpoint_path)

            expert = get_expert_class("itsc")()
            with self.assertRaisesRegex(RuntimeError, "ITSCExpert checkpoint_path"):
                expert.prepare(
                    {
                        "mode": "full_model",
                        "checkpoint_path": sparse_checkpoint_path,
                        "bank_path": bank_path,
                        "model_config": ITSC_TINY_MODEL_CONFIG,
                    },
                    {"dataset": "Delivery_SH"},
                )

    def test_itsc_full_model_predictor_smoke_uses_compatible_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path, bank_path = _write_itsc_full_model_artifacts(tmpdir)
            predictor = ITSCFullModelPredictor(
                checkpoint_path=checkpoint_path,
                bank_path=bank_path,
                model_config=ITSC_TINY_MODEL_CONFIG,
            )

            prediction = predictor(
                torch.zeros(1, 2, 2, 1),
                llm_encoding=torch.zeros(2, 4),
                batch_meta={"output_len": 3, "output_dim": 1},
            )

        self.assertEqual(tuple(prediction.shape), (1, 3, 2, 1))

    def test_itsc_retriever_encoder_requires_checkpoint_when_configured(self):
        expert = get_expert_class("itsc")()
        stream = io.StringIO()

        with self.assertRaisesRegex(ValueError, "retriever_pretrained_path"):
            with redirect_stdout(stream):
                expert.prepare(
                    {
                        "use_retriever_encoder": True,
                        "require_retriever_pretrained": True,
                    },
                    {"dataset": "Delivery_SH", "input_len": 2, "llm_enc_dim": 4},
                )

        self.assertIn("ITSCExpert", stream.getvalue())
        self.assertIn("retriever_pretrained_path", stream.getvalue())

    def test_itsc_retriever_encoder_score_and_time_masks_are_used(self):
        class FixedEncoder(torch.nn.Module):
            def encode_query(self, llm_query, ts_query, query_hour=None, query_weekday=None):
                return torch.tensor(
                    [[[1.0, 0.0], [1.0, 0.0]]],
                    dtype=ts_query.dtype,
                    device=ts_query.device,
                )

            def encode_key(self, llm_keys, ts_keys, key_hour=None, key_weekday=None):
                return torch.tensor(
                    [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
                    dtype=ts_keys.dtype,
                    device=ts_keys.device,
                )

        bank = {
            "global": {
                "llm_keys": torch.zeros(3, 4),
                "dyn_keys": torch.zeros(3, 2),
                "hist": torch.zeros(3, 2),
                "future": torch.tensor([[3.0, 3.0], [7.0, 7.0], [11.0, 11.0]]),
                "sample_ids": torch.tensor([0, 1, 2]),
                "hour_ids": torch.tensor([5, 5, 6]),
                "weekday_ids": torch.tensor([2, 3, 2]),
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "retriever.pt"
            torch.save(RetrieverEncoder(llm_dim=4, ts_len=2, retriever_dim=2).state_dict(), checkpoint)
            expert = get_expert_class("itsc")()
            expert.prepare(
                {
                    "bank": bank,
                    "top_k": 1,
                    "temperature": 0.2,
                    "use_retriever_encoder": True,
                    "require_retriever_pretrained": True,
                    "retriever_pretrained_path": str(checkpoint),
                    "retriever_dim": 2,
                },
                {"dataset": "Delivery_SH", "input_len": 2, "llm_enc_dim": 4},
            )
        expert.retriever_encoder = FixedEncoder()
        history = torch.zeros(1, 2, 2, 1)

        output = expert.forward_prior(
            history,
            supports=[torch.eye(2)],
            llm=torch.zeros(2, 4),
            batch_meta={
                "output_len": 2,
                "output_dim": 1,
                "x_hour": torch.tensor([[4, 5]]),
                "x_weekday": torch.tensor([[1, 2]]),
                "sample_ids": torch.tensor([1]),
            },
        )

        self.assertTrue(output.aux["bank_used"])
        self.assertTrue(output.aux["retriever_used"])
        self.assertTrue(torch.allclose(output.prior, torch.full((1, 2, 2, 1), 3.0)))

    def test_itsc_retriever_encoder_loads_prefixed_full_model_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "full_model.pt"
            encoder = RetrieverEncoder(llm_dim=4, ts_len=2, retriever_dim=2)
            state = {
                f"retriever_encoder.{key}": value
                for key, value in encoder.state_dict().items()
            }
            torch.save(state, checkpoint)

            expert = get_expert_class("itsc")()
            expert.prepare(
                {
                    "use_retriever_encoder": True,
                    "require_retriever_pretrained": True,
                    "retriever_pretrained_path": str(checkpoint),
                    "retriever_dim": 2,
                },
                {"dataset": "Delivery_SH", "input_len": 2, "llm_enc_dim": 4},
            )

        self.assertTrue(
            torch.allclose(
                expert.retriever_encoder.llm_proj[0].weight,
                encoder.llm_proj[0].weight,
            )
        )


    def test_raft_adapter_uses_loaded_temporal_shape_bank(self):
        bank = {
            "keys": torch.zeros(1, 2),
            "values": torch.full((1, 3), 5.0),
            "sample_indices": torch.tensor([0]),
            "source_ids": torch.tensor([0]),
            "source_to_id": {"Delivery_SH": 0},
        }
        expert = get_expert_class("raft")()
        expert.prepare({"bank": bank, "top_k": 1}, {"dataset": "Delivery_HZ", "source_data": "Delivery_SH"})
        history = torch.zeros(1, 2, 2, 1)

        output = expert.forward_prior(
            history,
            supports=[torch.eye(2)],
            llm=torch.zeros(2, 4),
            batch_meta={"output_len": 3, "output_dim": 1},
        )

        self.assertTrue(torch.allclose(output.prior, torch.full((1, 3, 2, 1), 5.0)))
        self.assertTrue(output.aux["bank_used"])

    def test_raft_residual_correction_applies_prior_alpha(self):
        bank = {
            "keys": torch.zeros(1, 2),
            "values": torch.full((1, 3), 9.0),
        }
        expert = get_expert_class("raft")()
        expert.prepare(
            {"mode": "residual", "bank": bank, "top_k": 1, "prior_alpha": 0.25},
            {"dataset": "Delivery_SH"},
        )
        baseline = torch.full((1, 3, 2, 1), 5.0)

        output = expert.forward_correction(
            torch.zeros(1, 2, 2, 1),
            supports=[torch.eye(2)],
            llm_encoding=torch.zeros(2, 4),
            batch_meta={"output_len": 3, "output_dim": 1},
            baseline_pred=baseline,
        )

        self.assertTrue(torch.equal(output.raw_prior, torch.full((1, 3, 2, 1), 9.0)))
        self.assertTrue(torch.equal(output.aux["raw_prior"], torch.full((1, 3, 2, 1), 9.0)))
        self.assertTrue(torch.equal(output.delta, torch.full((1, 3, 2, 1), 1.0)))
        self.assertEqual(output.aux["correction_type"], "raft_prior_alpha_residual")
        self.assertEqual(output.aux["prior_alpha"], 0.25)

    def test_raft_residual_correction_accepts_horizon_prior_alpha(self):
        bank = {
            "keys": torch.zeros(1, 2),
            "values": torch.full((1, 3), 9.0),
        }
        expert = get_expert_class("raft")()
        expert.prepare(
            {"mode": "residual", "bank": bank, "top_k": 1, "prior_alpha": [-0.1, 0.0, 0.1]},
            {"dataset": "Delivery_SH"},
        )
        baseline = torch.full((1, 3, 2, 1), 5.0)

        output = expert.forward_correction(
            torch.zeros(1, 2, 2, 1),
            supports=[torch.eye(2)],
            llm_encoding=torch.zeros(2, 4),
            batch_meta={"output_len": 3, "output_dim": 1},
            baseline_pred=baseline,
        )

        expected = torch.tensor([[[[-0.4], [-0.4]], [[0.0], [0.0]], [[0.4], [0.4]]]])
        self.assertTrue(torch.allclose(output.delta, expected))
        self.assertEqual(output.aux["prior_alpha"], [-0.1, 0.0, 0.1])

    def test_raft_forward_candidate_wraps_residual_for_legacy_diagnostics(self):
        bank = {"keys": torch.zeros(1, 2), "values": torch.full((1, 3), 9.0)}
        expert = get_expert_class("raft")()
        expert.prepare(
            {"mode": "residual", "bank": bank, "top_k": 1, "prior_alpha": 0.25},
            {"dataset": "Delivery_SH"},
        )
        baseline = torch.full((1, 3, 2, 1), 5.0)

        output = expert.forward_candidate(
            torch.zeros(1, 2, 2, 1),
            supports=[torch.eye(2)],
            llm=torch.zeros(2, 4),
            batch_meta={"output_len": 3, "output_dim": 1},
            baseline_pred=baseline,
        )

        self.assertTrue(torch.equal(output.prior, torch.full((1, 3, 2, 1), 6.0)))
        self.assertEqual(output.aux["candidate_type"], "final_prediction")

    def test_raft_full_model_requires_checkpoint_retrieval_and_config(self):
        expert = get_expert_class("raft")()
        with self.assertRaisesRegex(ValueError, "RAFTExpert full_model requires"):
            expert.prepare({"mode": "full_model"}, {"dataset": "Delivery_HZ"})

    def test_raft_full_model_requires_sample_ids_for_forward(self):
        class FakeRAFTFullModel(torch.nn.Module):
            def forward(self, history_data, sample_ids, batch_meta=None):
                return torch.full(
                    (history_data.shape[0], 3, history_data.shape[2], 1),
                    17.0,
                    device=history_data.device,
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path, retrieval_cache_path = _write_raft_full_model_artifacts(tmpdir)
            expert = get_expert_class("raft")()
            expert.prepare(
                {
                    "mode": "full_model",
                    "checkpoint_path": checkpoint_path,
                    "retrieval_cache_path": retrieval_cache_path,
                    "model_config": RAFT_TINY_MODEL_CONFIG,
                },
                {"dataset": "Delivery_HZ"},
            )
            expert.full_model = FakeRAFTFullModel()

            with self.assertRaisesRegex(ValueError, "sample_ids"):
                expert.forward_prior(
                    torch.zeros(1, 2, 2, 1),
                    supports=[torch.eye(2)],
                    llm=torch.zeros(2, 4),
                    batch_meta={"output_len": 3, "output_dim": 1},
                )

    def test_raft_full_model_rejects_empty_checkpoint_during_prepare(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_checkpoint_path = f"{tmpdir}/empty_checkpoint.pt"
            retrieval_cache_path = _write_raft_retrieval_cache_artifact(tmpdir)
            torch.save({}, empty_checkpoint_path)

            expert = get_expert_class("raft")()
            with self.assertRaisesRegex(RuntimeError, "RAFTExpert checkpoint_path"):
                expert.prepare(
                    {
                        "mode": "full_model",
                        "checkpoint_path": empty_checkpoint_path,
                        "retrieval_cache_path": retrieval_cache_path,
                        "model_config": RAFT_TINY_MODEL_CONFIG,
                    },
                    {"dataset": "Delivery_HZ"},
                )

    def test_raft_full_model_rejects_missing_or_corrupt_retrieval_cache_during_prepare(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = f"{tmpdir}/checkpoint.pt"
            torch.save(RAFTCompatModel(**RAFT_TINY_MODEL_CONFIG).state_dict(), checkpoint_path)

            expert = get_expert_class("raft")()
            with self.assertRaises(FileNotFoundError):
                expert.prepare(
                    {
                        "mode": "full_model",
                        "checkpoint_path": checkpoint_path,
                        "retrieval_cache_path": f"{tmpdir}/missing_cache.pt",
                        "model_config": RAFT_TINY_MODEL_CONFIG,
                    },
                    {"dataset": "Delivery_HZ"},
                )

            corrupt_cache_path = f"{tmpdir}/corrupt_cache.txt"
            with open(corrupt_cache_path, "w") as handle:
                handle.write("not a torch cache")

            expert = get_expert_class("raft")()
            with self.assertRaises(RuntimeError):
                expert.prepare(
                    {
                        "mode": "full_model",
                        "checkpoint_path": checkpoint_path,
                        "retrieval_cache_path": corrupt_cache_path,
                        "model_config": RAFT_TINY_MODEL_CONFIG,
                    },
                    {"dataset": "Delivery_HZ"},
                )

    def test_raft_full_model_rejects_sparse_checkpoint_during_prepare(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sparse_checkpoint_path = f"{tmpdir}/sparse_checkpoint.pt"
            retrieval_cache_path = _write_raft_retrieval_cache_artifact(tmpdir)
            full_state = RAFTCompatModel(**RAFT_TINY_MODEL_CONFIG).state_dict()
            sparse_state = {
                key: value
                for key, value in full_state.items()
                if key.startswith("linear_x.")
            }
            torch.save(sparse_state, sparse_checkpoint_path)

            expert = get_expert_class("raft")()
            with self.assertRaisesRegex(RuntimeError, "RAFTExpert checkpoint_path"):
                expert.prepare(
                    {
                        "mode": "full_model",
                        "checkpoint_path": sparse_checkpoint_path,
                        "retrieval_cache_path": retrieval_cache_path,
                        "model_config": RAFT_TINY_MODEL_CONFIG,
                    },
                    {"dataset": "Delivery_HZ"},
                )

    def test_raft_full_model_rejects_mixed_module_prefix_checkpoint_during_prepare(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mixed_checkpoint_path = f"{tmpdir}/mixed_checkpoint.pt"
            retrieval_cache_path = _write_raft_retrieval_cache_artifact(tmpdir)
            state = dict(RAFTCompatModel(**RAFT_TINY_MODEL_CONFIG).state_dict())
            state["module.linear_x.weight"] = torch.full_like(
                state["linear_x.weight"],
                99.0,
            )
            torch.save(state, mixed_checkpoint_path)

            expert = get_expert_class("raft")()
            with self.assertRaisesRegex(
                RuntimeError,
                "RAFTExpert checkpoint_path.*mixed module",
            ):
                expert.prepare(
                    {
                        "mode": "full_model",
                        "checkpoint_path": mixed_checkpoint_path,
                        "retrieval_cache_path": retrieval_cache_path,
                        "model_config": RAFT_TINY_MODEL_CONFIG,
                    },
                    {"dataset": "Delivery_HZ"},
                )

    def test_raft_full_model_accepts_all_module_prefixed_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prefixed_checkpoint_path = f"{tmpdir}/prefixed_checkpoint.pt"
            retrieval_cache_path = _write_raft_retrieval_cache_artifact(tmpdir)
            state = {
                f"module.{key}": value
                for key, value in RAFTCompatModel(**RAFT_TINY_MODEL_CONFIG).state_dict().items()
            }
            torch.save(state, prefixed_checkpoint_path)

            expert = get_expert_class("raft")()
            expert.prepare(
                {
                    "mode": "full_model",
                    "checkpoint_path": prefixed_checkpoint_path,
                    "retrieval_cache_path": retrieval_cache_path,
                    "model_config": RAFT_TINY_MODEL_CONFIG,
                },
                {"dataset": "Delivery_HZ"},
            )

        self.assertIsNotNone(expert.full_model)

    def test_raft_full_model_predictor_forward_returns_moe_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path, retrieval_cache_path = _write_raft_full_model_artifacts(tmpdir)
            predictor = RAFTFullModelPredictor(
                checkpoint_path=checkpoint_path,
                retrieval_cache_path=retrieval_cache_path,
                model_config=RAFT_TINY_MODEL_CONFIG,
            )

            prediction = predictor(
                torch.zeros(2, 2, 3, 1),
                sample_ids=torch.tensor([0, 1]),
                batch_meta={"raft_mode": "test"},
            )

        self.assertEqual(tuple(prediction.shape), (2, 3, 3, 1))

    def test_tpb_full_model_requires_checkpoint_pattern_and_config(self):
        expert = TPBExpert()
        with self.assertRaisesRegex(ValueError, "TPBExpert full_model requires"):
            expert.prepare({"mode": "full_model"}, {"dataset": "Delivery_HZ"})

    def test_tpb_full_model_uses_predictor_candidate(self):
        class FakeTPBFullModel(torch.nn.Module):
            def forward(self, history_data, supports=None, batch_meta=None):
                return torch.full(
                    (
                        history_data.shape[0],
                        batch_meta["output_len"],
                        history_data.shape[2],
                        batch_meta["output_dim"],
                    ),
                    19.0,
                    device=history_data.device,
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path, pattern_path, config_path, original_code_path = _write_tpb_full_model_artifacts(tmpdir)
            expert = TPBExpert()
            expert.prepare(
                {
                    "mode": "full_model",
                    "checkpoint_path": checkpoint_path,
                    "pattern_path": pattern_path,
                    "config_path": config_path,
                    "original_code_path": original_code_path,
                    "model_config": TPB_TINY_MODEL_CONFIG,
                },
                {"dataset": "Delivery_HZ"},
            )
            expert.full_model = FakeTPBFullModel()

            output = expert.forward_prior(
                torch.zeros(1, 4, 2, 1),
                supports=[torch.eye(2)],
                llm=torch.zeros(2, 4),
                batch_meta={"output_len": 3, "output_dim": 1},
            )

        self.assertTrue(torch.equal(output.prior, torch.full((1, 3, 2, 1), 19.0)))
        self.assertTrue(output.aux["full_model_used"])

    def test_tpb_full_model_rejects_empty_checkpoint_during_prepare(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, pattern_path, config_path, original_code_path = _write_tpb_full_model_artifacts(tmpdir)
            empty_checkpoint_path = f"{tmpdir}/empty_checkpoint.pt"
            torch.save({}, empty_checkpoint_path)

            expert = TPBExpert()
            with self.assertRaisesRegex(RuntimeError, "TPBExpert checkpoint_path"):
                expert.prepare(
                    {
                        "mode": "full_model",
                        "checkpoint_path": empty_checkpoint_path,
                        "pattern_path": pattern_path,
                        "config_path": config_path,
                        "original_code_path": original_code_path,
                        "model_config": TPB_TINY_MODEL_CONFIG,
                    },
                    {"dataset": "Delivery_HZ"},
                )

    def test_tpb_full_model_rejects_corrupt_pattern_and_config_paths_during_prepare(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path, pattern_path, config_path, original_code_path = _write_tpb_full_model_artifacts(tmpdir)
            corrupt_pattern_path = f"{tmpdir}/corrupt_pattern.txt"
            with open(corrupt_pattern_path, "w") as handle:
                handle.write("not a torch pattern")

            expert = TPBExpert()
            with self.assertRaises(RuntimeError):
                expert.prepare(
                    {
                        "mode": "full_model",
                        "checkpoint_path": checkpoint_path,
                        "pattern_path": corrupt_pattern_path,
                        "config_path": config_path,
                        "original_code_path": original_code_path,
                        "model_config": TPB_TINY_MODEL_CONFIG,
                    },
                    {"dataset": "Delivery_HZ"},
                )

            expert = TPBExpert()
            with self.assertRaises(FileNotFoundError):
                expert.prepare(
                    {
                        "mode": "full_model",
                        "checkpoint_path": checkpoint_path,
                        "pattern_path": pattern_path,
                        "config_path": f"{tmpdir}/missing_config.yaml",
                        "original_code_path": original_code_path,
                        "model_config": TPB_TINY_MODEL_CONFIG,
                    },
                    {"dataset": "Delivery_HZ"},
                )

    def test_tpb_full_model_rejects_malformed_config_during_prepare(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path, pattern_path, config_path, original_code_path = _write_tpb_full_model_artifacts(tmpdir)
            with open(config_path, "w") as handle:
                handle.write("data_args: [unterminated")

            expert = TPBExpert()
            with self.assertRaisesRegex(RuntimeError, "TPBExpert config_path"):
                expert.prepare(
                    {
                        "mode": "full_model",
                        "checkpoint_path": checkpoint_path,
                        "pattern_path": pattern_path,
                        "config_path": config_path,
                        "original_code_path": original_code_path,
                        "model_config": TPB_TINY_MODEL_CONFIG,
                    },
                    {"dataset": "Delivery_HZ"},
                )

    def test_tpb_full_model_rejects_mixed_module_prefix_checkpoint_during_prepare(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path, pattern_path, config_path, original_code_path = _write_tpb_full_model_artifacts(tmpdir)
            state = dict(TPBCompatModel(**TPB_TINY_MODEL_CONFIG).state_dict())
            state["module.bias"] = torch.full_like(state["bias"], 99.0)
            torch.save(state, checkpoint_path)

            expert = TPBExpert()
            with self.assertRaisesRegex(
                RuntimeError,
                "TPBExpert checkpoint_path.*mixed module",
            ):
                expert.prepare(
                    {
                        "mode": "full_model",
                        "checkpoint_path": checkpoint_path,
                        "pattern_path": pattern_path,
                        "config_path": config_path,
                        "original_code_path": original_code_path,
                        "model_config": TPB_TINY_MODEL_CONFIG,
                    },
                    {"dataset": "Delivery_HZ"},
                )

    def test_tpb_full_model_predictor_forward_returns_moe_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path, pattern_path, config_path, original_code_path = _write_tpb_full_model_artifacts(tmpdir)
            predictor = TPBFullModelPredictor(
                checkpoint_path=checkpoint_path,
                pattern_path=pattern_path,
                config_path=config_path,
                original_code_path=original_code_path,
                model_config=TPB_TINY_MODEL_CONFIG,
                model_factory=TPBCompatModel,
            )

            prediction = predictor(
                torch.zeros(2, 4, 3, 1),
                supports=[torch.eye(3)],
                batch_meta={"output_len": 3, "output_dim": 1},
            )

        self.assertEqual(tuple(prediction.shape), (2, 3, 3, 1))

    def test_tpb_adapter_converts_retrieved_pattern_to_output_prior(self):
        bank = {"patterns": torch.full((1, 2), 9.0), "metadata": {"patch_len": 2}}
        expert = TPBExpert()
        expert.prepare(
            {"bank": bank, "top_k": 1, "patch_len": 2},
            {"dataset": "Delivery_HZ", "source_data": "Delivery_SH,Delivery_CQ"},
        )
        history = torch.zeros(1, 4, 2, 1)

        output = expert.forward_prior(
            history,
            supports=[torch.eye(2)],
            llm=torch.zeros(2, 4),
            batch_meta={"output_len": 4, "output_dim": 1},
        )

        self.assertTrue(torch.allclose(output.prior, torch.full((1, 4, 2, 1), 9.0)))
        self.assertTrue(output.aux["bank_used"])

    def test_tpb_final_candidate_returns_model_prediction_not_raw_pattern_prior(self):
        bank = {"patterns": torch.full((1, 2), 9.0), "metadata": {"patch_len": 2}}
        model_config = {
            "node_dim": 2,
            "input_len": 4,
            "in_dim": 1,
            "embed_dim": 2,
            "output_len": 3,
            "num_layer": 1,
            "llm_enc_dim": 4,
            "supports_len": 1,
            "mp_layers": 0,
        }
        expert = TPBExpert()
        expert.prepare(
            {
                "mode": "final_prediction",
                "bank": bank,
                "top_k": 1,
                "patch_len": 2,
                "prior_alpha": 0.0,
                "model_config": model_config,
            },
            {"dataset": "Delivery_HZ", "source_data": "Delivery_SH"},
        )
        history = torch.zeros(1, 4, 2, 1)
        baseline = torch.full((1, 3, 2, 1), -4.0)

        output = expert.forward_candidate(
            history,
            [torch.eye(2)],
            torch.zeros(2, 4),
            {"output_len": 3, "output_dim": 1},
            baseline,
        )

        self.assertEqual(tuple(output.prior.shape), (1, 3, 2, 1))
        self.assertEqual(output.aux["candidate_type"], "final_prediction")
        self.assertFalse(torch.allclose(output.prior, torch.full((1, 3, 2, 1), 9.0)))


def _write_itsc_full_model_artifacts(tmpdir):
    checkpoint_path = f"{tmpdir}/checkpoint.pt"
    bank_path = _write_itsc_bank_artifact(tmpdir)
    torch.save(RAGIMPEL(**ITSC_TINY_MODEL_CONFIG).state_dict(), checkpoint_path)
    return checkpoint_path, bank_path


def _write_itsc_bank_artifact(tmpdir):
    bank_path = f"{tmpdir}/bank.pkl"
    with open(bank_path, "wb") as handle:
        pickle.dump(
            {
                "global": {
                    "llm_keys": torch.zeros(1, 4),
                    "dyn_keys": torch.zeros(1, 2),
                    "hist": torch.zeros(1, 2),
                    "future": torch.full((1, 3), 4.0),
                    "sample_ids": torch.tensor([0]),
                },
            },
            handle,
        )
    return bank_path


def _write_raft_full_model_artifacts(tmpdir):
    checkpoint_path = f"{tmpdir}/checkpoint.pt"
    retrieval_cache_path = _write_raft_retrieval_cache_artifact(tmpdir)
    torch.save(RAFTCompatModel(**RAFT_TINY_MODEL_CONFIG).state_dict(), checkpoint_path)
    return checkpoint_path, retrieval_cache_path


def _write_raft_retrieval_cache_artifact(tmpdir):
    retrieval_cache_path = f"{tmpdir}/retrieval_cache.pt"
    retrieval_dict = {
        "train": torch.zeros(1, 4, 3, 1),
        "valid": torch.zeros(1, 4, 3, 1),
        "test": torch.zeros(1, 4, 3, 1),
    }
    torch.save({"retrieval_dict": retrieval_dict}, retrieval_cache_path)
    return retrieval_cache_path


def _write_tpb_full_model_artifacts(tmpdir):
    checkpoint_path = f"{tmpdir}/checkpoint.pt"
    pattern_path = f"{tmpdir}/pattern.pt"
    config_path = f"{tmpdir}/config.yaml"
    original_code_path = f"{tmpdir}/original_code"
    torch.save(TPBCompatModel(**TPB_TINY_MODEL_CONFIG).state_dict(), checkpoint_path)
    torch.save({"patterns": torch.zeros(1, 4)}, pattern_path)
    with open(config_path, "w") as handle:
        handle.write(
            "data_args: {}\n"
            "model_args: {}\n"
            "task_args: {}\n"
            "PatchFSL_cfg: {}\n"
            "STmodel: GWN\n"
        )
    model_root = os.path.join(original_code_path, "model", "Meta_Models")
    os.makedirs(model_root, exist_ok=True)
    with open(os.path.join(model_root, "rep_model_final.py"), "w") as handle:
        handle.write(
            "import torch\n"
            "class PatchFSL(torch.nn.Module):\n"
            "    def __init__(self, data_args, model_args, task_args, PatchFSL_cfg, model='GWN'):\n"
            "        super().__init__()\n"
            "        self.bias = torch.nn.Parameter(torch.zeros(1))\n"
            "    def forward(self, data_i, A, stage='test'):\n"
            "        y = data_i.y\n"
            "        if y.dim() == 4:\n"
            "            shape = (data_i.x.shape[0], data_i.x.shape[1], y.shape[2], y.shape[3])\n"
            "        else:\n"
            "            shape = (data_i.x.shape[0], data_i.x.shape[1], y.shape[-1])\n"
            "        return torch.zeros(shape, dtype=data_i.x.dtype, device=data_i.x.device) + self.bias\n"
        )
    return checkpoint_path, pattern_path, config_path, original_code_path


if __name__ == "__main__":
    unittest.main()
