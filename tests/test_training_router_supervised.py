import tempfile
import unittest
from pathlib import Path

import torch


class TrainingRouterSupervisedTest(unittest.TestCase):
    def test_build_good_expert_targets_keeps_baseline_and_masks_unavailable(self):
        from experiments.training.train_router_supervised import build_good_expert_targets

        candidate_errors = torch.tensor([[[5.0, 4.8, 4.0], [3.0, 2.0, 2.5]]])
        available = torch.tensor([[[True, True, False], [True, False, True]]])

        good = build_good_expert_targets(candidate_errors, available, oracle_margin=0.98)

        self.assertTrue(torch.equal(
            good,
            torch.tensor([[[True, True, False], [True, False, True]]]),
        ))

    def test_train_router_supervised_from_cache_writes_router_artifacts(self):
        from experiments.training.train_router_supervised import train_router_supervised_from_cache

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cache_path = tmp / "cache.pt"
            torch.save(
                {
                    "features": torch.randn(6, 3, 5),
                    "available": torch.ones(6, 3, 2, dtype=torch.bool),
                    "candidate_errors": torch.tensor(
                        [
                            [[5.0, 3.0], [5.0, 5.2], [5.0, 2.5]],
                            [[4.0, 2.0], [4.0, 4.4], [4.0, 3.0]],
                            [[6.0, 1.0], [6.0, 6.5], [6.0, 4.0]],
                            [[5.0, 3.0], [5.0, 5.1], [5.0, 2.0]],
                            [[4.0, 2.5], [4.0, 4.2], [4.0, 3.3]],
                            [[6.0, 3.0], [6.0, 6.1], [6.0, 2.5]],
                        ]
                    ),
                    "candidate_names": ["none", "toy"],
                    "metadata": {"split": ["train", "train", "train", "val", "val", "val"]},
                },
                cache_path,
            )

            summary = train_router_supervised_from_cache(
                cache_path=str(cache_path),
                run_dir=str(tmp / "router"),
                hidden_dim=6,
                max_epochs=3,
                lr=0.05,
                oracle_margin=0.98,
                lambda_sparse=0.01,
                patience=3,
                dropout=0.0,
                device="cpu",
            )

            self.assertEqual(summary["candidate_names"], ["none", "toy"])
            self.assertTrue((tmp / "router" / "best_router.pt").exists())
            self.assertTrue((tmp / "router" / "history.csv").exists())
            self.assertGreaterEqual(summary["best_epoch"], 0)


if __name__ == "__main__":
    unittest.main()
