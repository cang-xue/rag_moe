import tempfile
import unittest
from pathlib import Path

import torch

from src.rag_moe.router import TwoStageRAGRouter


class TrainRouterDPOTest(unittest.TestCase):
    def test_train_router_dpo_updates_heads_and_writes_checkpoint(self):
        from experiments.training.train_router_dpo import train_router_dpo_from_cache

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            router = TwoStageRAGRouter(num_candidates=2, input_dim=5, hidden_dim=6, dropout=0.0)
            router_path = tmp / "router.pt"
            torch.save(router.state_dict(), router_path)
            cache_path = tmp / "cache.pt"
            torch.save(
                {
                    "features": torch.randn(4, 3, 5),
                    "available": torch.ones(4, 3, 2, dtype=torch.bool),
                    "candidate_errors": torch.tensor(
                        [
                            [[5.0, 3.0], [5.0, 5.1], [5.0, 2.5]],
                            [[4.0, 2.0], [4.0, 4.5], [4.0, 3.0]],
                            [[6.0, 1.0], [6.0, 6.4], [6.0, 4.0]],
                            [[7.0, 5.0], [7.0, 7.2], [7.0, 3.0]],
                        ]
                    ),
                    "candidate_names": ["none", "itsc"],
                },
                cache_path,
            )

            summary = train_router_dpo_from_cache(
                cache_path=str(cache_path),
                router_ckpt=str(router_path),
                run_dir=str(tmp / "run"),
                hidden_dim=6,
                beta=0.2,
                rel_margin=0.05,
                abs_margin=0.1,
                max_epochs=3,
                lr=0.05,
                train_scope="heads",
            )

            self.assertGreater(summary["num_pairs"], 0)
            self.assertTrue((tmp / "run" / "best_router.pt").exists())
            self.assertTrue((tmp / "run" / "history.csv").exists())


if __name__ == "__main__":
    unittest.main()
