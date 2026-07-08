import unittest

import torch

from src.rag_moe.fusion import fuse_candidates, fuse_residuals
from src.rag_moe.features import build_router_features
from src.rag_moe.router import TwoStageRAGRouter


class RAGMoEFusionTest(unittest.TestCase):
    def test_fuse_candidates_weighted_sum_by_node(self):
        baseline = torch.zeros(2, 3, 4, 1)
        itsc = torch.ones(2, 3, 4, 1)
        raft = torch.full((2, 3, 4, 1), 2.0)
        candidates = torch.stack([baseline, itsc, raft], dim=2)
        weights = torch.zeros(2, 4, 3)
        weights[:, :, 1] = 0.25
        weights[:, :, 2] = 0.75

        fused = fuse_candidates(candidates, weights)

        self.assertEqual(tuple(fused.shape), (2, 3, 4, 1))
        self.assertTrue(torch.allclose(fused, torch.full_like(fused, 1.75)))

    def test_fuse_candidates_rejects_mismatched_experts(self):
        candidates = torch.zeros(2, 3, 3, 4, 1)
        weights = torch.zeros(2, 4, 4)

        with self.assertRaisesRegex(ValueError, "expert"):
            fuse_candidates(candidates, weights)

    def test_fuse_residuals_returns_weighted_delta(self):
        deltas = torch.tensor(
            [
                [
                    [
                        [[0.0], [0.0]],
                        [[2.0], [4.0]],
                        [[10.0], [20.0]],
                    ]
                ]
            ],
            dtype=torch.float32,
        )
        weights = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.25, 0.75]]])

        fused = fuse_residuals(deltas, weights)

        self.assertEqual(tuple(fused.shape), (1, 1, 2, 1))
        self.assertTrue(torch.allclose(fused[0, 0, :, 0], torch.tensor([0.0, 16.0])))

    def test_fuse_residuals_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "deltas"):
            fuse_residuals(torch.zeros(1, 2, 3, 1), torch.zeros(1, 1, 3))


class TwoStageRAGRouterTest(unittest.TestCase):
    def test_router_returns_expected_shapes_and_normalized_weights(self):
        router = TwoStageRAGRouter(
            num_candidates=4,
            input_dim=9,
            hidden_dim=16,
            dropout=0.0,
        )
        features = torch.randn(2, 5, 9)
        available = torch.ones(2, 5, 4, dtype=torch.bool)

        output = router(features, available)

        self.assertEqual(tuple(output["select_logits"].shape), (2, 5, 4))
        self.assertEqual(tuple(output["select_prob"].shape), (2, 5, 4))
        self.assertEqual(tuple(output["weights"].shape), (2, 5, 4))
        self.assertTrue(
            torch.allclose(
                output["weights"].sum(dim=-1),
                torch.ones(2, 5),
            )
        )

    def test_router_falls_back_to_baseline_when_only_baseline_available(self):
        router = TwoStageRAGRouter(
            num_candidates=3,
            input_dim=9,
            hidden_dim=16,
            dropout=0.0,
        )
        features = torch.randn(2, 5, 9)
        available = torch.zeros(2, 5, 3, dtype=torch.bool)
        available[:, :, 0] = True

        output = router(features, available)

        self.assertTrue(torch.allclose(output["weights"][:, :, 0], torch.ones(2, 5)))
        self.assertTrue(torch.allclose(output["weights"][:, :, 1:], torch.zeros(2, 5, 2)))

    def test_router_keeps_all_unavailable_candidates_inactive(self):
        router = TwoStageRAGRouter(
            num_candidates=3,
            input_dim=9,
            hidden_dim=16,
            dropout=0.0,
        )
        features = torch.randn(2, 5, 9)
        available = torch.zeros(2, 5, 3, dtype=torch.bool)

        output = router(features, available)

        self.assertFalse(output["active_mask"].any().item())
        self.assertTrue(torch.allclose(output["weights"], torch.zeros(2, 5, 3)))
        self.assertTrue(torch.allclose(output["weights"].sum(dim=-1), torch.zeros(2, 5)))

    def test_router_rejects_wrong_feature_width(self):
        router = TwoStageRAGRouter(
            num_candidates=4,
            input_dim=9,
            hidden_dim=16,
            dropout=0.0,
        )
        features = torch.randn(2, 5, 8)
        available = torch.ones(2, 5, 4, dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "feature"):
            router(features, available)


class RAGMoEFeatureTest(unittest.TestCase):
    def test_build_router_features_uses_history_and_candidate_errors(self):
        history = torch.randn(2, 6, 4, 1)
        baseline = torch.zeros(2, 3, 4, 1)
        experts = {
            "itsc": torch.ones(2, 3, 4, 1),
            "raft": torch.full((2, 3, 4, 1), 2.0),
        }

        features = build_router_features(history, baseline, experts)

        self.assertEqual(tuple(features.shape), (2, 4, 7))
        self.assertTrue(torch.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
