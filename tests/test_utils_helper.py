import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.utils.helper import get_dataloader, reassign_train_val_to_source_test, split_batch


class HelperDataloaderMetadataTest(unittest.TestCase):
    def test_get_dataloader_return_time_meta_keeps_legacy_tuple_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for split, offset in (("train", 0), ("val", 10), ("test", 20)):
                x = np.ones((2, 3, 2, 1), dtype=np.float32) * offset
                y = np.ones((2, 2, 2, 1), dtype=np.float32) * (offset + 1)
                x_hour = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64) + offset
                x_minute = np.array([[0, 10, 20], [30, 40, 50]], dtype=np.int64)
                x_weekday = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
                np.savez(
                    root / f"{split}.npz",
                    x=x,
                    y=y,
                    x_hour=x_hour,
                    x_minute=x_minute,
                    x_weekday=x_weekday,
                )

            data = get_dataloader(
                str(root),
                batch_size=2,
                input_dim=1,
                output_dim=1,
                return_time_meta=True,
            )
            batch = next(iter(data["train_loader"]))

        self.assertEqual(data["batch_meta_keys"], ["x_hour", "x_minute", "x_weekday", "sample_ids"])
        self.assertEqual(len(batch), 6)
        self.assertEqual(tuple(batch[2].shape), (2, 3))
        self.assertEqual(tuple(batch[3].shape), (2, 3))
        self.assertEqual(tuple(batch[4].shape), (2, 3))
        self.assertEqual(sorted(batch[5].tolist()), [0, 1])

    def test_get_dataloader_can_include_sample_and_time_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for category, offset, length in (("train", 10, 4), ("val", 20, 2), ("test", 30, 2)):
                x = np.arange(length * 3 * 2, dtype=np.float32).reshape(length, 3, 2, 1)
                y = np.arange(length * 2 * 2, dtype=np.float32).reshape(length, 2, 2, 1)
                x_hour = np.arange(offset, offset + length * 3, dtype=np.int64).reshape(length, 3)
                x_weekday = np.full((length, 3), 2, dtype=np.int64)
                sample_idx = np.arange(offset, offset + length, dtype=np.int64)
                np.savez_compressed(
                    root / f"{category}.npz",
                    x=x,
                    y=y,
                    x_hour=x_hour,
                    x_weekday=x_weekday,
                    sample_idx=sample_idx,
                )

            data = get_dataloader(str(root), batch_size=2, input_dim=1, output_dim=1, include_metadata=True)

            self.assertEqual(data["batch_meta_keys"], ["sample_ids", "x_hour", "x_weekday"])
            batch = next(iter(data["train_loader"]))
            x, y, batch_meta = split_batch(batch, data["batch_meta_keys"])

            self.assertEqual(tuple(x.shape), (2, 3, 2, 1))
            self.assertEqual(tuple(y.shape), (2, 2, 2, 1))
            self.assertEqual(set(batch_meta), {"sample_ids", "x_hour", "x_weekday"})
            self.assertTrue(torch.equal(batch_meta["x_weekday"], torch.full((2, 3), 2)))

    def test_get_dataloader_synthesizes_sample_ids_without_weekday(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for category, length in (("train", 3), ("val", 2), ("test", 2)):
                np.savez_compressed(
                    root / f"{category}.npz",
                    x=np.arange(length * 3 * 2, dtype=np.float32).reshape(length, 3, 2, 1),
                    y=np.arange(length * 2 * 2, dtype=np.float32).reshape(length, 2, 2, 1),
                    x_hour=np.zeros((length, 3), dtype=np.int64),
                )

            data = get_dataloader(str(root), batch_size=3, input_dim=1, output_dim=1, include_metadata=True)
            _, _, batch_meta = split_batch(next(iter(data["train_loader"])), data["batch_meta_keys"])

            self.assertEqual(data["batch_meta_keys"], ["sample_ids", "x_hour"])
            self.assertEqual(sorted(batch_meta["sample_ids"].tolist()), [0, 1, 2])

    def test_reassign_train_val_to_source_test_merges_train_and_val(self):
        train = torch.utils.data.TensorDataset(torch.arange(4).view(4, 1), torch.arange(4).view(4, 1))
        val = torch.utils.data.TensorDataset(torch.arange(10, 12).view(2, 1), torch.arange(10, 12).view(2, 1))
        test = torch.utils.data.TensorDataset(torch.arange(20, 23).view(3, 1), torch.arange(20, 23).view(3, 1))
        data = {
            "train_loader": torch.utils.data.DataLoader(train, batch_size=2, shuffle=False),
            "val_loader": torch.utils.data.DataLoader(val, batch_size=2, shuffle=False),
            "test_loader": torch.utils.data.DataLoader(test, batch_size=2, shuffle=False),
            "batch_meta_keys": [],
        }

        updated = reassign_train_val_to_source_test(data, batch_size=2)

        self.assertEqual(len(updated["train_loader"].dataset), 6)
        self.assertEqual(len(updated["val_loader"].dataset), 3)
        self.assertIs(updated["test_loader"].dataset, test)


if __name__ == "__main__":
    unittest.main()
