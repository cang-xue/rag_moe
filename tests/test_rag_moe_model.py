import unittest

import torch
from torch import nn

from src.models.rag_moe_impel import RAGMoEIMPEL
from src.rag_moe.experts.base import RAGCorrectionOutput, RAGExpertAdapter
from src.rag_moe.features import build_router_features


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1, 1)

    def forward(self, history_data, supports, llm_encoding):
        return history_data[:, -1:, :, :1].repeat(1, 3, 1, 1)


class FixedCorrectionExpert(RAGExpertAdapter):
    def __init__(self, name, value):
        super().__init__()
        self.name = name
        self.value = float(value)

    def prepare(self, config, data_context):
        return self.freeze()

    @torch.no_grad()
    def forward_correction(self, history_data, supports, llm_encoding, batch_meta, baseline_pred):
        delta = torch.full_like(baseline_pred, self.value)
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
            raw_prior=baseline_pred + delta,
            aux={"correction_type": "fixed_delta"},
        )


class RAGMoEIMPELTest(unittest.TestCase):
    def test_router_features_preserve_expert_insertion_order(self):
        history = torch.zeros(1, 2, 1, 1)
        baseline = torch.zeros(1, 3, 1, 1)
        expert_deltas = {
            "zeta": torch.full_like(baseline, 2.0),
            "alpha": torch.full_like(baseline, 5.0),
        }

        features = build_router_features(history, baseline, expert_deltas)

        self.assertEqual(features.shape[-1], 7)
        self.assertTrue(torch.equal(features[..., 5], torch.full((1, 1), 2.0)))
        self.assertTrue(torch.equal(features[..., 6], torch.full((1, 1), 5.0)))

    def test_rag_moe_impel_returns_prediction_and_diagnostics(self):
        model = RAGMoEIMPEL(
            backbone=TinyBackbone(),
            experts=[FixedCorrectionExpert("candidate", value=2.0)],
            output_len=3,
            output_dim=1,
            router_hidden_dim=8,
            router_dropout=0.0,
        )
        history = torch.randn(2, 6, 4, 1)
        outputs = model(history, [torch.eye(4)], torch.zeros(4, 8))
        self.assertEqual(tuple(outputs["prediction"].shape), (2, 3, 4, 1))
        self.assertEqual(tuple(outputs["baseline_pred"].shape), (2, 3, 4, 1))
        self.assertEqual(tuple(outputs["weights"].shape), (2, 4, 2))
        self.assertIn("candidate", outputs["expert_outputs"])

    def test_rag_moe_impel_can_return_prediction_tensor_for_existing_trainer(self):
        model = RAGMoEIMPEL(
            backbone=TinyBackbone(),
            experts=[FixedCorrectionExpert("candidate", value=2.0)],
            output_len=3,
            output_dim=1,
            router_hidden_dim=8,
            router_dropout=0.0,
        )
        model.return_dict = False
        history = torch.randn(2, 6, 4, 1)

        prediction = model(history, [torch.eye(4)], torch.zeros(4, 8))

        self.assertEqual(tuple(prediction.shape), (2, 3, 4, 1))

    def test_rag_moe_impel_freezes_backbone_and_experts_but_not_router(self):
        model = RAGMoEIMPEL(
            backbone=TinyBackbone(),
            experts=[FixedCorrectionExpert("candidate", value=2.0)],
            output_len=3,
            output_dim=1,
            router_hidden_dim=8,
            router_dropout=0.0,
        )
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
        self.assertTrue(all(not p.requires_grad for expert in model.experts for p in expert.parameters()))
        self.assertTrue(any(p.requires_grad for p in model.router.parameters()))

    def test_rag_moe_impel_exposes_trainer_compatibility_attributes(self):
        model = RAGMoEIMPEL(
            backbone=TinyBackbone(),
            experts=[FixedCorrectionExpert("candidate", value=2.0)],
            output_len=3,
            output_dim=1,
            router_hidden_dim=8,
            router_dropout=0.0,
        )

        self.assertEqual(model.name, "rag_moe_impel")
        self.assertEqual(model.horizon, 3)
        self.assertGreater(model.param_num(model.name), 0)

    def test_rag_moe_impel_loads_unprefixed_backbone_checkpoint(self):
        model = RAGMoEIMPEL(
            backbone=TinyBackbone(),
            experts=[FixedCorrectionExpert("tiny", value=1.0)],
            output_len=3,
            output_dim=1,
            router_hidden_dim=8,
            router_dropout=0.0,
        )
        backbone = TinyBackbone()
        with torch.no_grad():
            backbone.proj.weight.fill_(2.0)
            backbone.proj.bias.fill_(3.0)

        model.load_state_dict(backbone.state_dict())

        self.assertTrue(torch.equal(model.backbone.proj.weight, backbone.proj.weight))
        self.assertTrue(torch.equal(model.backbone.proj.bias, backbone.proj.bias))

    def test_forward_adds_routed_delta_to_baseline(self):
        model = RAGMoEIMPEL(
            backbone=TinyBackbone(),
            experts=[
                FixedCorrectionExpert("zeta", value=2.0),
                FixedCorrectionExpert("alpha", value=5.0),
            ],
            output_len=3,
            output_dim=1,
            router_hidden_dim=8,
            router_dropout=0.0,
        )
        x = torch.ones(1, 6, 2, 1) * 3.0

        outputs = model(x, supports=[torch.eye(2)], llm_encoding=torch.zeros(2, 8))

        none_weight = outputs["weights"][:, :, 0]
        zeta_weight = outputs["weights"][:, :, 1]
        alpha_weight = outputs["weights"][:, :, 2]
        expected = outputs["baseline_pred"]
        expected = expected + zeta_weight.unsqueeze(1).unsqueeze(-1) * 2.0
        expected = expected + alpha_weight.unsqueeze(1).unsqueeze(-1) * 5.0
        self.assertTrue(torch.allclose(outputs["prediction"], expected))
        self.assertTrue(torch.equal(outputs["residual_deltas"]["none"], torch.zeros_like(outputs["baseline_pred"])))
        self.assertTrue(torch.equal(outputs["residual_deltas"]["zeta"], torch.full_like(outputs["baseline_pred"], 2.0)))
        self.assertTrue(torch.equal(outputs["residual_deltas"]["alpha"], torch.full_like(outputs["baseline_pred"], 5.0)))
        self.assertEqual(outputs["correction_names"], ["none", "zeta", "alpha"])
        self.assertEqual(outputs["candidate_names"], ["none", "zeta", "alpha"])
        self.assertTrue(torch.all(none_weight >= 0.0))
        self.assertTrue(torch.all(zeta_weight >= 0.0))
        self.assertTrue(torch.all(alpha_weight >= 0.0))

    def test_forward_direct_expert_returns_baseline_plus_delta(self):
        model = RAGMoEIMPEL(
            TinyBackbone(),
            [
                FixedCorrectionExpert("zeta", value=2.0),
                FixedCorrectionExpert("alpha", value=5.0),
            ],
            output_len=3,
            output_dim=1,
            router_hidden_dim=8,
            router_dropout=0.0,
        )
        x = torch.ones(1, 6, 2, 1) * 3.0

        direct = model.forward_direct_expert(
            "zeta",
            x,
            supports=[torch.eye(2)],
            llm_encoding=torch.zeros(2, 8),
        )
        delta = model.forward_direct_expert(
            "zeta",
            x,
            supports=[torch.eye(2)],
            llm_encoding=torch.zeros(2, 8),
            return_delta=True,
        )

        self.assertTrue(torch.equal(delta, torch.full_like(delta, 2.0)))
        self.assertTrue(torch.equal(direct, torch.full_like(direct, 5.0)))

    def test_forward_direct_expert_rejects_unknown_expert(self):
        model = RAGMoEIMPEL(
            backbone=TinyBackbone(),
            experts=[FixedCorrectionExpert("tiny", value=1.0)],
            output_len=3,
            output_dim=1,
            router_hidden_dim=8,
            router_dropout=0.0,
        )
        history = torch.randn(2, 6, 4, 1)

        with self.assertRaisesRegex(KeyError, "unknown direct expert"):
            model.forward_direct_expert(
                "missing",
                history,
                supports=[torch.eye(4)],
                llm_encoding=torch.zeros(4, 8),
                batch_meta={"output_len": 3, "output_dim": 1},
            )


if __name__ == "__main__":
    unittest.main()
