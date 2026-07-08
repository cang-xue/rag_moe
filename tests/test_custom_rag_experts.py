import os
import tempfile
import unittest

import torch
import yaml

from src.rag_moe.experts.base import (
    RAGCorrectionOutput,
    RAGExpertAdapter,
    validate_correction_output,
)
from src.rag_moe.registry import build_experts, get_expert_class
from src.rag_moe.experts.itsc_segment_gate import ITSCSegmentGate


class ConstantDeltaExpert(RAGExpertAdapter):
    name = "itsc"

    def __init__(self, value=2.0):
        super().__init__()
        self.value = float(value)

    def prepare(self, config, data_context):
        return self

    @torch.no_grad()
    def forward_correction(self, history_data, supports, llm_encoding, batch_meta, baseline_pred):
        return RAGCorrectionOutput(
            name=self.name,
            delta=torch.full_like(baseline_pred, self.value),
            available=torch.ones(
                baseline_pred.shape[0],
                baseline_pred.shape[2],
                dtype=torch.bool,
                device=baseline_pred.device,
            ),
            raw_prior=baseline_pred + self.value,
            confidence=torch.ones(
                baseline_pred.shape[0],
                baseline_pred.shape[2],
                dtype=baseline_pred.dtype,
                device=baseline_pred.device,
            ),
            aux={"correction_type": "constant_delta_for_test"},
        )


class CustomRAGExpertContractTest(unittest.TestCase):
    def sample(self):
        history = torch.arange(2 * 24 * 5, dtype=torch.float32).reshape(2, 24, 5, 1)
        baseline = torch.full((2, 24, 5, 1), 3.0)
        supports = [torch.eye(5)]
        llm = torch.zeros(5, 4)
        meta = {"output_len": 24, "output_dim": 1}
        return history, baseline, supports, llm, meta

    def assert_valid_correction(self, expert, output, baseline):
        validate_correction_output(output, baseline, expected_name=expert.name)
        self.assertEqual(tuple(output.delta.shape), tuple(baseline.shape))
        self.assertIn(
            tuple(output.available.shape),
            [
                (baseline.shape[0], baseline.shape[2]),
                (baseline.shape[0], 1),
            ],
        )

    def test_registry_returns_custom_experts(self):
        for name in (
            "calibration",
            "source_window",
            "volatility_peak",
            "itsc_segment_gate",
        ):
            expert_class = get_expert_class(name)
            self.assertTrue(issubclass(expert_class, RAGExpertAdapter))
            self.assertEqual(expert_class.name, name)

    def test_build_experts_freezes_custom_parameters(self):
        experts = build_experts(
            ["calibration", "volatility_peak"],
            {
                "calibration": {
                    "mode": "learned_bias",
                    "output_len": 24,
                    "output_dim": 1,
                    "init_bias": 0.0,
                },
                "volatility_peak": {
                    "mode": "learned_gate",
                    "output_len": 24,
                    "output_dim": 1,
                    "init_scale": 1.0,
                },
            },
            {"dataset": "Delivery_HZ", "source_data": "Delivery_SH"},
        )
        self.assertEqual([expert.name for expert in experts], ["calibration", "volatility_peak"])
        for expert in experts:
            self.assertGreater(
                sum(parameter.numel() for parameter in expert.parameters()),
                0,
            )
            self.assertTrue(all(not parameter.requires_grad for parameter in expert.parameters()))

    def test_calibration_identity_delta_is_zero(self):
        history, baseline, supports, llm, meta = self.sample()
        expert = get_expert_class("calibration")()
        expert.prepare(
            {"mode": "identity", "output_len": 24, "output_dim": 1},
            {"dataset": "Delivery_HZ", "source_data": "Delivery_SH"},
        )

        output = expert.forward_correction(history, supports, llm, meta, baseline)

        self.assert_valid_correction(expert, output, baseline)
        self.assertTrue(torch.equal(output.delta, torch.zeros_like(baseline)))

    def test_calibration_bias_delta_is_applied(self):
        history, baseline, supports, llm, meta = self.sample()
        expert = get_expert_class("calibration")()
        expert.prepare(
            {"scale": 1.0, "bias": 0.25, "output_len": 24, "output_dim": 1},
            {"dataset": "Delivery_HZ", "source_data": "Delivery_SH"},
        )

        output = expert.forward_correction(history, supports, llm, meta, baseline)

        self.assert_valid_correction(expert, output, baseline)
        self.assertTrue(torch.equal(output.delta, torch.full_like(baseline, 0.25)))
        self.assertTrue(torch.equal(output.raw_prior, baseline + 0.25))

    def test_source_window_without_bank_is_unavailable_and_zero_delta(self):
        history, baseline, supports, llm, meta = self.sample()
        expert = get_expert_class("source_window")()
        expert.prepare(
            {
                "mode": "source_window",
                "output_len": 24,
                "output_dim": 1,
                "bank": None,
            },
            {"dataset": "Delivery_HZ", "source_data": "Delivery_SH"},
        )

        output = expert.forward_correction(history, supports, llm, meta, baseline)

        self.assert_valid_correction(expert, output, baseline)
        self.assertTrue(torch.equal(output.delta, torch.zeros_like(baseline)))
        self.assertFalse(output.available.any())

    def test_source_window_high_confidence_match_returns_residual(self):
        history, baseline, supports, llm, meta = self.sample()
        query = torch.linspace(1.0, 24.0, steps=24)
        history = query.view(1, 24, 1, 1).expand_as(history).clone()
        retrieved_value = torch.full((24, 1), 7.0)
        bank = {
            "keys": torch.stack([query, -query], dim=0),
            "values": torch.stack(
                [
                    retrieved_value,
                    torch.full((24, 1), -7.0),
                ],
                dim=0,
            ),
            "metadata": {"source_data": "Delivery_SH"},
        }
        expert = get_expert_class("source_window")()
        expert.prepare(
            {
                "mode": "source_window",
                "bank": bank,
                "alpha": 0.5,
                "top_k": 1,
                "temperature": 0.1,
                "min_margin": 0.0,
                "confidence_threshold": 0.0,
                "output_len": 24,
                "output_dim": 1,
            },
            {"dataset": "Delivery_HZ", "source_data": "Delivery_SH"},
        )

        output = expert.forward_correction(history, supports, llm, meta, baseline)

        expected_delta = 0.5 * (retrieved_value.view(1, 24, 1, 1) - baseline)
        self.assert_valid_correction(expert, output, baseline)
        self.assertTrue(output.available.all())
        self.assertTrue(torch.allclose(output.delta, expected_delta))
        self.assertTrue(torch.allclose(output.raw_prior, baseline + expected_delta))

    def test_source_window_confidence_gate_blocks_low_margin_match(self):
        history, baseline, supports, llm, meta = self.sample()
        query = history[0, :, 0, 0].clone()
        query = query / query.norm()
        almost_same_query = query.clone()
        almost_same_query[-1] = almost_same_query[-1] + 1e-4
        almost_same_query = almost_same_query / almost_same_query.norm()
        bank = {
            "keys": torch.stack([query, almost_same_query], dim=0),
            "values": torch.stack(
                [
                    torch.full((24, 1), 4.0),
                    torch.full((24, 1), 6.0),
                ],
                dim=0,
            ),
            "metadata": {"source_data": "Delivery_SH"},
        }
        expert = get_expert_class("source_window")()
        expert.prepare(
            {
                "mode": "source_window",
                "bank": bank,
                "output_len": 24,
                "output_dim": 1,
                "min_margin": 0.25,
                "top_k": 2,
            },
            {"dataset": "Delivery_HZ", "source_data": "Delivery_SH"},
        )

        output = expert.forward_correction(history, supports, llm, meta, baseline)

        self.assert_valid_correction(expert, output, baseline)
        self.assertTrue(torch.equal(output.delta, torch.zeros_like(baseline)))
        self.assertFalse(output.available.any())

    def test_volatility_peak_identity_delta_is_zero(self):
        history, baseline, supports, llm, meta = self.sample()
        expert = get_expert_class("volatility_peak")()
        expert.prepare(
            {"mode": "identity", "output_len": 24, "output_dim": 1},
            {"dataset": "Delivery_HZ", "source_data": "Delivery_SH"},
        )

        output = expert.forward_correction(history, supports, llm, meta, baseline)

        self.assert_valid_correction(expert, output, baseline)
        self.assertTrue(torch.equal(output.delta, torch.zeros_like(baseline)))

    def test_itsc_segment_gate_requires_base_expert_or_config(self):
        expert = get_expert_class("itsc_segment_gate")()

        with self.assertRaisesRegex(ValueError, "base"):
            expert.prepare(
                {"mode": "segment_gate", "output_len": 24, "output_dim": 1},
                {"dataset": "Delivery_HZ", "source_data": "Delivery_SH"},
            )

    def test_itsc_segment_gate_zero_gate_blocks_constant_base_delta(self):
        history, baseline, supports, llm, meta = self.sample()
        expert = get_expert_class("itsc_segment_gate")()
        expert.prepare(
            {
                "mode": "segment_gate",
                "base_expert": ConstantDeltaExpert(2.0),
                "gate_mode": "zero",
                "output_len": 24,
                "output_dim": 1,
            },
            {"dataset": "Delivery_HZ", "source_data": "Delivery_SH"},
        )

        output = expert.forward_correction(history, supports, llm, meta, baseline)

        self.assert_valid_correction(expert, output, baseline)
        self.assertTrue(torch.equal(output.delta, torch.zeros_like(baseline)))
        self.assertTrue(torch.equal(output.raw_prior, baseline))
        self.assertTrue(output.available.all())

    def test_itsc_segment_gate_positive_gate_applies_base_delta(self):
        history, baseline, supports, llm, meta = self.sample()
        expert = get_expert_class("itsc_segment_gate")()
        expert.prepare(
            {
                "mode": "segment_gate",
                "base_expert": ConstantDeltaExpert(2.0),
                "gate_scale": 1.0,
                "gate_bias": 10.0,
                "output_len": 24,
                "output_dim": 1,
            },
            {"dataset": "Delivery_HZ", "source_data": "Delivery_SH"},
        )

        output = expert.forward_correction(history, supports, llm, meta, baseline)

        expected_delta = torch.full_like(
            baseline,
            2.0 * torch.sigmoid(torch.tensor(10.0)).item(),
        )
        self.assert_valid_correction(expert, output, baseline)
        self.assertTrue(torch.allclose(output.delta, expected_delta))
        self.assertTrue(torch.allclose(output.raw_prior, baseline + expected_delta))
        self.assertTrue(output.available.all())

    def test_itsc_segment_gate_initialization_can_learn_from_features(self):
        gate = ITSCSegmentGate(feature_dim=6, hidden_dim=8)

        weight_tensors = [
            parameter.detach()
            for name, parameter in gate.named_parameters()
            if "weight" in name
        ]

        self.assertTrue(any(torch.any(weight != 0.0) for weight in weight_tensors))

    def test_itsc_segment_gate_rejects_malformed_scalar_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "gate.pt")
            torch.save({"gate_scale": torch.ones(2), "gate_bias": torch.zeros(1)}, checkpoint_path)
            expert = get_expert_class("itsc_segment_gate")()

            with self.assertRaisesRegex(ValueError, "gate_scale.*scalar"):
                expert.prepare(
                    {
                        "mode": "segment_gate",
                        "base_expert": ConstantDeltaExpert(2.0),
                        "checkpoint_path": checkpoint_path,
                        "output_len": 24,
                        "output_dim": 1,
                    },
                    {"dataset": "Delivery_HZ", "source_data": "Delivery_SH"},
                )

    def test_itsc_segment_gate_scalar_checkpoint_is_constant(self):
        history, baseline, supports, llm, meta = self.sample()
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "gate.pt")
            torch.save({"gate_scale": torch.ones(1), "gate_bias": torch.tensor([-0.5])}, checkpoint_path)
            expert = get_expert_class("itsc_segment_gate")()
            expert.prepare(
                {
                    "mode": "segment_gate",
                    "base_expert": ConstantDeltaExpert(2.0),
                    "checkpoint_path": checkpoint_path,
                    "output_len": 24,
                    "output_dim": 1,
                },
                {"dataset": "Delivery_HZ", "source_data": "Delivery_SH"},
            )

            output = expert.forward_correction(history, supports, llm, meta, baseline)

            expected_gate = torch.sigmoid(torch.tensor(-0.5)).item()
            self.assertLess(
                torch.max(torch.abs(output.delta - torch.full_like(baseline, 2.0 * expected_gate))).item(),
                1e-7,
            )
            self.assertLess(
                torch.max(torch.abs(output.confidence - torch.full_like(output.confidence, expected_gate))).item(),
                1e-7,
            )

    def test_experts_custom_yaml_loads_with_build_experts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_itsc_config_path = os.path.join(tmpdir, "base_itsc.yaml")
            with open(base_itsc_config_path, "w") as handle:
                yaml.safe_dump(
                    {
                        "experts": {
                            "itsc": {
                                "mode": "prior",
                                "output_len": 24,
                                "output_dim": 1,
                            }
                        }
                    },
                    handle,
                )

            config_path = os.path.join(tmpdir, "experts_custom.yaml")
            with open(config_path, "w") as handle:
                yaml.safe_dump(
                    {
                        "experts": {
                            "calibration": {
                                "mode": "identity",
                                "output_len": 24,
                                "output_dim": 1,
                            },
                            "source_window": {
                                "mode": "source_window",
                                "output_len": 24,
                                "output_dim": 1,
                            },
                            "volatility_peak": {
                                "mode": "identity",
                                "output_len": 24,
                                "output_dim": 1,
                            },
                            "itsc_segment_gate": {
                                "mode": "segment_gate",
                                "base_itsc_config": base_itsc_config_path,
                                "output_len": 24,
                                "output_dim": 1,
                            },
                        }
                    },
                    handle,
                )

            with open(config_path) as handle:
                config = yaml.safe_load(handle)

            experts = build_experts(
                [
                    "calibration",
                    "source_window",
                    "volatility_peak",
                    "itsc_segment_gate",
                ],
                config["experts"],
                {"dataset": "Delivery_HZ", "source_data": "Delivery_SH"},
            )

            self.assertEqual(
                [expert.name for expert in experts],
                [
                    "calibration",
                    "source_window",
                    "volatility_peak",
                    "itsc_segment_gate",
                ],
            )

            history, baseline, supports, llm, meta = self.sample()
            experts_by_name = {expert.name: expert for expert in experts}
            for name in (
                "calibration",
                "source_window",
                "volatility_peak",
                "itsc_segment_gate",
            ):
                output = experts_by_name[name].forward_correction(
                    history,
                    supports,
                    llm,
                    meta,
                    baseline,
                )
                self.assert_valid_correction(experts_by_name[name], output, baseline)

            segment_gate = experts_by_name["itsc_segment_gate"]
            self.assertTrue(
                any(
                    getattr(segment_gate, attribute, None) is not None
                    for attribute in (
                        "base_expert",
                        "base_config",
                        "base_itsc_expert",
                        "base_itsc_config_data",
                    )
                ),
            )


if __name__ == "__main__":
    unittest.main()
