import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.rag_moe_impel import RAGMoEIMPEL
from src.rag_moe.experts.base import RAGCorrectionOutput, RAGExpertAdapter
from src.trainers.rag_moe_staged_trainer import (
    apply_staged_input_mask,
    compute_counterfactual_pseudo_dist,
    train_staged_router_epoch,
)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, history_data, supports, llm_encoding):
        return history_data[:, -1:, :, :1].repeat(1, 3, 1, 1) + self.bias


class DeltaExpert(RAGExpertAdapter):
    def __init__(self, name, delta):
        super().__init__()
        self.name = name
        self.delta_value = float(delta)

    @torch.no_grad()
    def forward_correction(self, history_data, supports, llm_encoding, batch_meta, baseline_pred):
        delta = torch.full_like(baseline_pred, self.delta_value)
        available = torch.ones(
            baseline_pred.shape[0],
            baseline_pred.shape[2],
            dtype=torch.bool,
            device=baseline_pred.device,
        )
        return RAGCorrectionOutput(
            name=self.name,
            delta=delta,
            available=available,
            raw_prior=baseline_pred + delta,
        )


class RAGMoEStagedTrainingTest(unittest.TestCase):
    def test_counterfactual_pseudo_dist_prefers_lower_loss_expert(self):
        itsc_loss = torch.tensor([[0.1, 2.0]])
        raft_loss = torch.tensor([[1.0, 0.2]])

        pseudo = compute_counterfactual_pseudo_dist(itsc_loss, raft_loss, temperature=1.0)

        self.assertEqual(tuple(pseudo.shape), (1, 2, 2))
        self.assertGreater(pseudo[0, 0, 0].item(), pseudo[0, 0, 1].item())
        self.assertGreater(pseudo[0, 1, 1].item(), pseudo[0, 1, 0].item())

    def test_apply_staged_input_mask_matches_unknown_and_random_known_nodes(self):
        x = torch.ones(3, 2, 5, 1)

        masked = apply_staged_input_mask(
            x,
            unknown_nodes={1},
            known_nodes={0, 2, 3, 4},
            num_masked_nodes=2,
            mask_unknown_inputs=True,
            training=True,
        )

        self.assertTrue(torch.equal(masked[:, :, 1, :], torch.zeros_like(masked[:, :, 1, :])))
        zero_known_counts = (masked[:, 0, [0, 2, 3, 4], 0] == 0).sum(dim=1)
        self.assertTrue(torch.equal(zero_known_counts, torch.full_like(zero_known_counts, 2)))

    def test_stage2_updates_router_only(self):
        model = RAGMoEIMPEL(
            TinyBackbone(),
            [DeltaExpert("itsc", 1.0), DeltaExpert("raft", 2.0)],
            output_len=3,
            output_dim=1,
            router_hidden_dim=8,
            router_dropout=0.0,
        )
        before_backbone = model.backbone.bias.detach().clone()
        before_router = [p.detach().clone() for p in model.router.parameters()]

        x = torch.zeros(4, 6, 2, 1)
        y = torch.ones(4, 3, 2, 1)
        loader = DataLoader(TensorDataset(x, y), batch_size=2)
        optimizer = torch.optim.Adam(model.router.parameters(), lr=0.01)

        metrics = train_staged_router_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            supports=[torch.eye(2)],
            llm_encoding=torch.zeros(2, 8),
            null_value=-1.0,
            stage="stage2",
            lambda_itsc=0.0,
            lambda_raft=0.0,
            lambda_entropy=0.0,
            lambda_balance=0.0,
            pseudo_temperature=1.0,
            device=torch.device("cpu"),
        )

        self.assertGreater(metrics["loss"], 0.0)
        self.assertGreater(metrics["ce_loss"], 0.0)
        self.assertTrue(torch.equal(model.backbone.bias.detach(), before_backbone))
        self.assertTrue(
            any(not torch.equal(param.detach(), before_router[i]) for i, param in enumerate(model.router.parameters()))
        )


if __name__ == "__main__":
    unittest.main()
