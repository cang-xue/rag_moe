import unittest

import torch

from src.rag_moe.experts.custom_features import horizon_fraction, normalized_entropy


class CustomFeatureTest(unittest.TestCase):
    def test_normalized_entropy_normalizes_positive_raw_weights(self):
        weights = torch.tensor([[10.0, 0.0, 0.0]])

        entropy = normalized_entropy(weights)

        self.assertEqual(entropy.shape, (1,))
        self.assertTrue(torch.allclose(entropy, torch.zeros(1)))

    def test_normalized_entropy_uniform_weights_are_max_entropy(self):
        weights = torch.tensor([[1.0, 1.0]])

        entropy = normalized_entropy(weights)

        self.assertTrue(torch.allclose(entropy, torch.ones(1)))

    def test_normalized_entropy_float16_sparse_row_is_finite_zero(self):
        weights = torch.tensor([[1.0, 0.0]], dtype=torch.float16)

        entropy = normalized_entropy(weights)

        self.assertEqual(entropy.dtype, weights.dtype)
        self.assertTrue(torch.isfinite(entropy).all())
        self.assertTrue(torch.equal(entropy, torch.zeros(1, dtype=weights.dtype)))

    def test_normalized_entropy_all_zero_row_returns_zero(self):
        weights = torch.zeros(2, 3)

        entropy = normalized_entropy(weights)

        self.assertTrue(torch.equal(entropy, torch.zeros(2)))

    def test_normalized_entropy_rejects_negative_weights(self):
        weights = torch.tensor([[1.0, -0.1]])

        with self.assertRaisesRegex(ValueError, "non-negative"):
            normalized_entropy(weights)

    def test_normalized_entropy_rejects_single_negative_weight(self):
        weights = torch.tensor([[-1.0]])

        with self.assertRaisesRegex(ValueError, "non-negative"):
            normalized_entropy(weights)

    def test_horizon_fraction_zero_horizon_preserves_empty_horizon(self):
        reference = torch.zeros(2, 0, 3, 1)

        fraction = horizon_fraction(reference)

        self.assertEqual(tuple(fraction.shape), (1, 0, 1, 1))
        self.assertEqual(fraction.dtype, reference.dtype)
        self.assertEqual(fraction.device, reference.device)


if __name__ == "__main__":
    unittest.main()
