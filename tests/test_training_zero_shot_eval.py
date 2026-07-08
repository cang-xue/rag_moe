import tempfile
import unittest
from pathlib import Path

import torch


class TrainingZeroShotEvalTest(unittest.TestCase):
    def test_evaluate_cached_candidates_reports_soft_and_guarded_hard_metrics(self):
        from experiments.training.evaluate_zero_shot import evaluate_cached_candidates

        with tempfile.TemporaryDirectory() as tmpdir:
            candidates = torch.zeros(2, 3, 2, 4, 1)
            candidates[:, :, 0, :, :] = 2.0
            candidates[:, :, 1, :, :] = 1.0
            labels = torch.ones(2, 3, 4, 1)
            weights = torch.zeros(2, 4, 2)
            weights[..., 1] = 1.0
            select_prob = weights.clone()
            available = torch.ones(2, 4, 2, dtype=torch.bool)

            summary = evaluate_cached_candidates(
                candidates=candidates,
                labels=labels,
                weights=weights,
                select_prob=select_prob,
                available=available,
                candidate_names=["none", "toy"],
                null_value=-1.0,
                run_dir=tmpdir,
            )

            self.assertAlmostEqual(summary["soft_mae"], 0.0)
            self.assertAlmostEqual(summary["guarded_hard_mae"], 0.0)
            self.assertEqual(summary["per_expert_selected_rate"]["toy"], 1.0)
            self.assertTrue((Path(tmpdir) / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
