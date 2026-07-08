import unittest

import torch

from src.rag_moe.router import TwoStageRAGRouter


class TrainingRouterDPOTest(unittest.TestCase):
    def test_configure_head_only_tuning_freezes_encoder(self):
        from experiments.training.router_dpo import configure_router_dpo_tuning

        router = TwoStageRAGRouter(num_candidates=3, input_dim=5, hidden_dim=8, dropout=0.0)

        trainable = configure_router_dpo_tuning(router, train_scope="heads")

        self.assertTrue(all(not p.requires_grad for p in router.encoder.parameters()))
        self.assertTrue(all(p.requires_grad for p in router.selector.parameters()))
        self.assertTrue(all(p.requires_grad for p in router.weighter.parameters()))
        self.assertEqual(
            {name for name, _ in trainable},
            {
                "selector.weight",
                "selector.bias",
                "weighter.weight",
                "weighter.bias",
            },
        )

    def test_dpo_loss_decreases_when_chosen_logit_is_higher_than_rejected(self):
        from experiments.training.router_dpo import dpo_preference_loss

        chosen = torch.tensor([1])
        rejected = torch.tensor([0])
        ref_logits = torch.zeros(1, 3)
        bad_logits = torch.tensor([[2.0, 0.0, 0.0]])
        good_logits = torch.tensor([[0.0, 2.0, 0.0]])

        bad_loss = dpo_preference_loss(bad_logits, ref_logits, chosen, rejected)
        good_loss = dpo_preference_loss(good_logits, ref_logits, chosen, rejected)

        self.assertLess(float(good_loss), float(bad_loss))

    def test_guarded_hard_selection_falls_back_to_none_when_not_confident(self):
        from experiments.training.router_dpo import guarded_hard_selection

        probabilities = torch.tensor([[[0.45, 0.47, 0.08], [0.2, 0.75, 0.05]]])

        selected = guarded_hard_selection(
            probabilities,
            none_index=0,
            min_best_prob=0.5,
            min_margin_over_none=0.1,
        )

        self.assertEqual(selected.tolist(), [[0, 1]])

    def test_apply_guarded_hard_candidates_gathers_per_node_candidate(self):
        from experiments.training.evaluate_zero_shot import apply_guarded_hard_candidates

        candidates = torch.zeros(1, 2, 3, 2, 1)
        candidates[:, :, 0, :, :] = 1.0
        candidates[:, :, 1, :, :] = 5.0
        candidates[:, :, 2, :, :] = 9.0
        probabilities = torch.tensor([[[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]]])

        prediction = apply_guarded_hard_candidates(
            candidates,
            probabilities,
            none_index=0,
            min_best_prob=0.5,
            min_margin_over_none=0.1,
        )

        self.assertTrue(torch.equal(prediction[0, :, 0, 0], torch.tensor([1.0, 1.0])))
        self.assertTrue(torch.equal(prediction[0, :, 1, 0], torch.tensor([9.0, 9.0])))


if __name__ == "__main__":
    unittest.main()
