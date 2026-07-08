import tempfile
import unittest
from pathlib import Path

import torch


class TrainingCandidateCacheTest(unittest.TestCase):
    def test_collect_candidate_cache_pads_variable_node_cities_and_preserves_metadata(self):
        from experiments.training.cache_candidates import collect_candidate_cache
        from experiments.training.multisource_data import CityDataContext

        class Loader(list):
            pass

        class FakeModel(torch.nn.Module):
            correction_names = ["none", "toy"]

            def forward(self, history_data, supports, llm_encoding, batch_meta=None):
                batch, horizon, nodes, channels = history_data.shape
                baseline = torch.zeros(batch, horizon, nodes, channels)
                delta = torch.ones_like(baseline)
                available = torch.ones(batch, nodes, dtype=torch.bool)
                return {
                    "baseline_pred": baseline,
                    "residual_deltas": {"none": torch.zeros_like(baseline), "toy": delta},
                    "expert_outputs": {"toy": type("Output", (), {"available": available})()},
                    "candidate_names": ["none", "toy"],
                }

        loader_a = Loader([
            (
                torch.zeros(1, 2, 3, 1),
                torch.ones(1, 2, 3, 1),
                torch.tensor([7]),
            )
        ])
        loader_b = Loader([
            (
                torch.zeros(1, 2, 2, 1),
                torch.ones(1, 2, 2, 1) * 2,
                torch.tensor([9]),
            )
        ])
        loader_a.batch_meta_keys = ["sample_ids"]
        loader_b.batch_meta_keys = ["sample_ids"]
        contexts = {
            "A": CityDataContext(
                city="A",
                city_id=0,
                loaders={"train_loader": loader_a, "scalers": []},
                llm_encoding=torch.zeros(3, 4),
                num_nodes=3,
                null_value=-1.0,
                scalers=[],
                supports=[torch.eye(3)],
                unknown_set={0},
                known_set={1, 2},
                num_masked_nodes=1,
            ),
            "B": CityDataContext(
                city="B",
                city_id=1,
                loaders={"train_loader": loader_b, "scalers": []},
                llm_encoding=torch.zeros(2, 4),
                num_nodes=2,
                null_value=-1.0,
                scalers=[],
                supports=[torch.eye(2)],
                unknown_set={0},
                known_set={1},
                num_masked_nodes=1,
            ),
        }

        payload = collect_candidate_cache(FakeModel(), contexts, split="train", device=torch.device("cpu"))

        self.assertEqual(payload["features"].shape[:2], torch.Size([2, 3]))
        self.assertEqual(payload["available"].shape, torch.Size([2, 3, 2]))
        self.assertEqual(payload["candidates"].shape, torch.Size([2, 2, 2, 3, 1]))
        self.assertFalse(bool(payload["available"][1, 2, 0]))
        self.assertEqual(payload["candidate_names"], ["none", "toy"])
        self.assertEqual(payload["metadata"]["city"], ["A", "B"])
        self.assertEqual(payload["metadata"]["city_id"], [0, 1])
        self.assertEqual(payload["metadata"]["sample_ids"], [7, 9])

    def test_save_candidate_cache_cli_payload_roundtrip(self):
        from experiments.training.cache_candidates import build_candidate_cache_payload, save_candidate_cache

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.pt"
            payload = build_candidate_cache_payload(
                features=torch.zeros(1, 2, 5),
                available=torch.ones(1, 2, 2, dtype=torch.bool),
                candidates=torch.zeros(1, 3, 2, 2, 1),
                labels=torch.zeros(1, 3, 2, 1),
                candidate_names=["none", "toy"],
                metadata={"city": ["A"]},
            )
            save_candidate_cache(path, payload)
            loaded = torch.load(path, map_location="cpu")
            self.assertEqual(loaded["metadata"]["city"], ["A"])


if __name__ == "__main__":
    unittest.main()
