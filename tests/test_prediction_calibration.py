import unittest

import torch

from src.utils.prediction_calibration import (
    GlobalSoftmaxMixer,
    HorizonAffineCalibrator,
    SharedAffineCalibrator,
    add_negative_residual_candidate,
    apply_source_robust_baseline_gate,
    apply_horizon_candidate_selector,
    apply_residual_magnitude_gate,
    fit_horizon_source_consistent_selector,
    fit_residual_magnitude_gate,
    make_temporal_source_splits,
)


class PredictionCalibrationTest(unittest.TestCase):
    def test_horizon_affine_calibrator_applies_scale_and_bias(self):
        calibrator = HorizonAffineCalibrator(horizon=2, output_dim=1)
        with torch.no_grad():
            calibrator.scale.copy_(torch.tensor([[[[2.0]], [[3.0]]]]))
            calibrator.bias.copy_(torch.tensor([[[[-1.0]], [[4.0]]]]))

        preds = torch.tensor(
            [
                [[[[2.0]], [[5.0]]]],
            ]
        ).squeeze(1)

        corrected = calibrator(preds)

        expected = torch.tensor(
            [
                [[[3.0]], [[19.0]]],
            ]
        )
        self.assertTrue(torch.equal(corrected, expected))

    def test_shared_affine_calibrator_applies_one_scale_and_bias(self):
        calibrator = SharedAffineCalibrator()
        with torch.no_grad():
            calibrator.scale.fill_(0.5)
            calibrator.bias.fill_(2.0)

        preds = torch.tensor(
            [
                [[[2.0]], [[6.0]]],
            ]
        )

        corrected = calibrator(preds)

        expected = torch.tensor(
            [
                [[[3.0]], [[5.0]]],
            ]
        )
        self.assertTrue(torch.equal(corrected, expected))

    def test_global_softmax_mixer_fuses_candidate_axis(self):
        mixer = GlobalSoftmaxMixer(num_candidates=2)
        with torch.no_grad():
            mixer.logits.copy_(torch.tensor([0.0, 0.0]))

        candidates = torch.tensor(
            [
                [
                    [[[2.0], [6.0]], [[4.0], [8.0]]],
                ]
            ]
        ).unsqueeze(-1)

        mixed = mixer(candidates)

        expected = torch.tensor(
            [
                [[[3.0], [7.0]]],
            ]
        ).unsqueeze(-1)
        self.assertTrue(torch.equal(mixed, expected))

    def test_add_negative_residual_candidate_reflects_around_baseline(self):
        names = ["none", "raft"]
        candidates = torch.tensor(
            [
                [
                    [[[10.0]], [[13.0]]],
                ]
            ]
        )

        updated_candidates, updated_names = add_negative_residual_candidate(candidates, names, "raft", "raft_neg")

        self.assertEqual(updated_names, ["none", "raft", "raft_neg"])
        self.assertEqual(tuple(updated_candidates.shape), (1, 1, 3, 1, 1))
        self.assertEqual(updated_candidates[0, 0, 2, 0, 0].item(), 7.0)

    def test_source_robust_gate_falls_back_when_any_split_is_not_better(self):
        mixer = GlobalSoftmaxMixer(num_candidates=2)
        with torch.no_grad():
            mixer.logits.copy_(torch.tensor([-2.0, 2.0]))
        good_candidates = torch.tensor([[[[[1.0]], [[0.0]]]]])
        good_labels = torch.tensor([[[[0.0]]]])
        bad_candidates = torch.tensor([[[[[0.0]], [[1.0]]]]])
        bad_labels = torch.tensor([[[[0.0]]]])

        kept = apply_source_robust_baseline_gate(
            mixer,
            [(good_candidates, good_labels), (bad_candidates, bad_labels)],
            null_value=-1.0,
            margin=0.0,
        )

        self.assertFalse(kept)
        self.assertGreater(torch.softmax(mixer.logits, dim=0)[0].item(), 0.999)

    def test_horizon_selector_requires_all_source_splits_to_improve(self):
        names = ["none", "raft"]
        labels = torch.zeros(1, 2, 1, 1)
        good_split = torch.tensor(
            [
                [
                    [[[2.0]], [[0.0]]],
                    [[[2.0]], [[0.0]]],
                ]
            ]
        )
        bad_second_horizon = torch.tensor(
            [
                [
                    [[[2.0]], [[0.0]]],
                    [[[0.0]], [[2.0]]],
                ]
            ]
        )

        selection, summary = fit_horizon_source_consistent_selector(
            [(good_split, labels), (bad_second_horizon, labels)],
            names,
            null_value=-1.0,
            margin=0.1,
        )

        self.assertEqual(selection, [1, 0])
        self.assertEqual(summary[0]["selected_name"], "raft")
        self.assertEqual(summary[1]["selected_name"], "none")

        selected = apply_horizon_candidate_selector(good_split, selection)

        expected = torch.tensor(
            [
                [[[0.0]], [[2.0]]],
            ]
        )
        self.assertTrue(torch.equal(selected, expected))

    def test_temporal_source_splits_make_partial_time_gain_fail_consistency(self):
        names = ["none", "raft"]
        labels = torch.zeros(4, 1, 1, 1)
        candidates = torch.tensor(
            [
                [[[[2.0]], [[0.0]]]],
                [[[[2.0]], [[0.0]]]],
                [[[[0.0]], [[2.0]]]],
                [[[[0.0]], [[2.0]]]],
            ]
        )

        splits = make_temporal_source_splits([(candidates, labels)], num_blocks=2)
        selection, summary = fit_horizon_source_consistent_selector(
            splits,
            names,
            null_value=-1.0,
            margin=0.0,
        )

        self.assertEqual(len(splits), 2)
        self.assertEqual(selection, [0])
        self.assertEqual(summary[0]["selected_name"], "none")

    def test_residual_magnitude_gate_uses_candidate_only_below_threshold(self):
        baseline = torch.tensor([[[[10.0], [10.0]]]])
        candidate = torch.tensor([[[[11.0], [14.0]]]])

        gated = apply_residual_magnitude_gate(baseline, candidate, max_abs_delta=2.0)

        expected = torch.tensor([[[[11.0], [10.0]]]])
        self.assertTrue(torch.equal(gated, expected))

    def test_fit_residual_magnitude_gate_requires_all_splits_to_improve(self):
        labels = torch.tensor([[[[1.0], [10.0]]]])
        good_candidates = torch.tensor([[[[[0.0], [10.0]], [[1.0], [14.0]]]]])
        bad_candidates = torch.tensor([[[[[0.0], [10.0]], [[1.0], [6.0]]]]])

        threshold, summary = fit_residual_magnitude_gate(
            [(good_candidates, labels), (bad_candidates, labels)],
            candidate_index=1,
            null_value=-1.0,
            thresholds=[4.0],
            margin=0.0,
        )

        self.assertEqual(threshold, 0.0)
        self.assertFalse(summary["kept"])


if __name__ == "__main__":
    unittest.main()
