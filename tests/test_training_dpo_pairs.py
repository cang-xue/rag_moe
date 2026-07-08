import unittest

import torch


class TrainingDPOPairTest(unittest.TestCase):
    def test_compute_candidate_node_errors_averages_horizon_and_channel(self):
        from experiments.training.dpo_pairs import compute_candidate_node_errors

        candidates = torch.tensor(
            [
                [
                    [[[1.0], [2.0]], [[0.0], [10.0]]],
                    [[[3.0], [4.0]], [[2.0], [8.0]]],
                ]
            ]
        )
        labels = torch.tensor([[[[2.0], [3.0]], [[2.0], [7.0]]]])

        errors = compute_candidate_node_errors(candidates, labels)

        self.assertEqual(tuple(errors.shape), (1, 2, 2))
        self.assertTrue(torch.allclose(errors[0, 0], torch.tensor([1.0, 1.0])))
        self.assertTrue(torch.allclose(errors[0, 1], torch.tensor([2.0, 4.0])))

    def test_build_dpo_pairs_prefers_expert_only_when_margin_beats_none(self):
        from experiments.training.dpo_pairs import build_dpo_pairs

        errors = torch.tensor([[[10.0, 8.0, 9.9], [5.0, 4.99, 5.5]]])
        available = torch.ones(1, 2, 3, dtype=torch.bool)

        pairs = build_dpo_pairs(
            errors,
            available,
            rel_margin=0.05,
            abs_margin=0.1,
        )

        self.assertEqual(pairs.chosen.tolist(), [1, 0])
        self.assertEqual(pairs.rejected.tolist(), [0, 1])
        self.assertEqual(pairs.batch_index.tolist(), [0, 0])
        self.assertEqual(pairs.node_index.tolist(), [0, 1])

    def test_build_dpo_pairs_ignores_unavailable_experts(self):
        from experiments.training.dpo_pairs import build_dpo_pairs

        errors = torch.tensor([[[10.0, 1.0, 8.0]]])
        available = torch.tensor([[[True, False, True]]])

        pairs = build_dpo_pairs(errors, available, rel_margin=0.05, abs_margin=0.1)

        self.assertEqual(pairs.chosen.tolist(), [2])
        self.assertEqual(pairs.rejected.tolist(), [0])


if __name__ == "__main__":
    unittest.main()
