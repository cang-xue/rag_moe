import tempfile
import unittest
from pathlib import Path

import torch

from src.rag_moe.full_model_utils import (
    extract_state_dict,
    load_torch_artifact,
    require_artifact,
    require_config_keys,
)


class FullModelUtilsTest(unittest.TestCase):
    def test_require_artifact_rejects_missing_file(self):
        with self.assertRaisesRegex(FileNotFoundError, "ITSCExpert checkpoint_path"):
            require_artifact("ITSCExpert", "checkpoint_path", "missing-file.pt")

    def test_require_config_keys_rejects_missing_key(self):
        with self.assertRaisesRegex(ValueError, "RAFTExpert full_model requires"):
            require_config_keys("RAFTExpert", {}, ["checkpoint_path", "retrieval_cache_path"])

    def test_load_torch_artifact_loads_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "artifact.pt"
            torch.save({"value": torch.tensor([3.0])}, path)

            loaded = load_torch_artifact("TPBExpert", "checkpoint_path", str(path), map_location="cpu")

        self.assertTrue(torch.equal(loaded["value"], torch.tensor([3.0])))

    def test_extract_state_dict_accepts_plain_and_wrapped_state(self):
        plain = {"weight": torch.tensor([1.0])}
        wrapped = {"state_dict": {"weight": torch.tensor([2.0])}}

        self.assertTrue(torch.equal(extract_state_dict(plain)["weight"], torch.tensor([1.0])))
        self.assertTrue(torch.equal(extract_state_dict(wrapped)["weight"], torch.tensor([2.0])))


if __name__ == "__main__":
    unittest.main()
