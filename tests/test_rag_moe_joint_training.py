import unittest

import torch
from torch import nn

from src.trainers.rag_moe_joint_trainer import (
    TrainableRAFTResidual,
    apply_joint_input_mask,
    collect_joint_trainable_parameters,
    compute_baseline_inclusive_pseudo_dist,
    compute_expert_improvement_targets,
)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))


class TinyITSCExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.encoder = nn.Linear(1, 1)
        self.model.prior_alpha = nn.Parameter(torch.tensor(0.0))
        self.model.prior_out_proj = nn.Linear(1, 1)
        self.model.prior_out_gate = nn.Linear(2, 1)


class RAGMoEJointTrainingTest(unittest.TestCase):
    def test_collect_joint_parameters_freezes_backbone_and_keeps_residual_trainable(self):
        backbone = TinyBackbone()
        router = nn.Linear(2, 2)
        itsc = TinyITSCExtractor()
        raft = TrainableRAFTResidual(initial_alpha=0.25)

        groups = collect_joint_trainable_parameters(
            backbone=backbone,
            router=router,
            itsc_extractor=itsc,
            raft_residual=raft,
            router_lr=1e-3,
            itsc_lr=1e-4,
            raft_lr=1e-3,
        )

        self.assertFalse(backbone.weight.requires_grad)
        self.assertFalse(itsc.model.encoder.weight.requires_grad)
        self.assertTrue(itsc.model.prior_alpha.requires_grad)
        self.assertTrue(itsc.model.prior_out_proj.weight.requires_grad)
        self.assertTrue(raft.prior_alpha.requires_grad)
        self.assertEqual(len(groups), 3)

    def test_collect_joint_parameters_can_unfreeze_backbone(self):
        backbone = TinyBackbone()
        router = nn.Linear(2, 2)
        itsc = TinyITSCExtractor()
        raft = TrainableRAFTResidual(initial_alpha=0.25)

        groups = collect_joint_trainable_parameters(
            backbone=backbone,
            router=router,
            itsc_extractor=itsc,
            raft_residual=raft,
            router_lr=1e-3,
            itsc_lr=1e-4,
            raft_lr=1e-3,
            backbone_lr=1e-5,
            train_backbone=True,
        )

        self.assertTrue(backbone.weight.requires_grad)
        self.assertEqual(len(groups), 4)
        self.assertEqual(groups[0]["lr"], 1e-5)

    def test_trainable_raft_residual_backprops_to_alpha(self):
        raft = TrainableRAFTResidual(initial_alpha=0.5)
        baseline = torch.zeros(2, 3, 1, 1)
        prior = torch.ones(2, 3, 1, 1)

        delta = raft(prior, baseline)
        loss = delta.mean()
        loss.backward()

        self.assertIsNotNone(raft.prior_alpha.grad)
        self.assertGreater(abs(float(raft.prior_alpha.grad)), 0.0)

    def test_baseline_inclusive_pseudo_dist_prefers_baseline_with_margin(self):
        losses = torch.tensor([[[1.0, 0.98, 1.1]]])

        pseudo = compute_baseline_inclusive_pseudo_dist(
            losses,
            temperature=0.1,
            baseline_margin=0.05,
        )

        self.assertEqual(tuple(pseudo.shape), (1, 1, 3))
        self.assertGreater(float(pseudo[0, 0, 0]), float(pseudo[0, 0, 1]))

    def test_expert_improvement_targets_require_margin_over_baseline(self):
        losses = torch.tensor([[[1.0, 0.96, 0.80], [1.0, 0.99, 1.10]]])

        targets = compute_expert_improvement_targets(
            losses,
            expert_improvement_margin=0.03,
        )

        expected = torch.tensor([[[True, True, True], [True, False, False]]])
        self.assertTrue(torch.equal(targets, expected))

    def test_apply_joint_input_mask_matches_unknown_and_random_known_nodes(self):
        torch.manual_seed(0)
        x = torch.ones(2, 3, 5, 1)

        masked = apply_joint_input_mask(
            x,
            unknown_nodes={1},
            known_nodes={0, 2, 3, 4},
            num_masked_nodes=2,
            mask_unknown_inputs=True,
            training=True,
        )

        self.assertTrue(torch.equal(masked[:, :, 1, :], torch.zeros_like(masked[:, :, 1, :])))
        zero_known_counts = (masked[:, 0, [0, 2, 3, 4], 0] == 0).sum(dim=1)
        self.assertTrue(torch.equal(zero_known_counts, torch.tensor([2, 2])))


if __name__ == "__main__":
    unittest.main()
