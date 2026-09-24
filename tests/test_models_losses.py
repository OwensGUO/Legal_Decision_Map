from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # The project keeps non-model tooling usable before GPU deps are installed.
    torch = None  # type: ignore[assignment]


@unittest.skipIf(torch is None, "PyTorch is not installed in this interpreter")
class ModelLossTests(unittest.TestCase):
    def test_soft_charge_marginalization_weights_sum_to_one(self) -> None:
        from legal_landscape.models.heads import marginalize_sentence

        logits = torch.tensor([[0.0, 1.0]])
        conditional = torch.tensor([[10.0, 20.0]])
        months, weights = marginalize_sentence(logits, conditional)
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(1)))
        self.assertTrue(torch.allclose(months, torch.tensor([15.9385]), atol=1e-4))

    def test_invariant_pair_only_activates_stability_loss(self) -> None:
        from legal_landscape.models.losses import compute_typed_losses

        losses = compute_typed_losses(
            pair_types=("invariant",),
            parent_charge_logits=torch.tensor([[2.0, -1.0]]),
            counterfactual_charge_logits=torch.tensor([[1.0, 0.0]]),
        )
        self.assertGreater(losses.invariant.item(), 0.0)
        self.assertEqual(losses.boundary.item(), 0.0)
        self.assertEqual(losses.response.item(), 0.0)
        self.assertEqual(losses.active_counts["invariant"], 1)

    def test_charge_flip_never_activates_consistency(self) -> None:
        from legal_landscape.models.losses import compute_typed_losses

        losses = compute_typed_losses(
            pair_types=("charge_flip",),
            parent_charge_logits=torch.tensor([[3.0, -2.0]]),
            counterfactual_charge_logits=torch.tensor([[1.0, -1.0]]),
            target_charge_indices=torch.tensor([1]),
        )
        self.assertEqual(losses.invariant.item(), 0.0)
        self.assertGreater(losses.boundary.item(), 0.0)
        self.assertEqual(losses.active_counts["invariant"], 0)

    def test_rank_loss_rewards_the_declared_direction(self) -> None:
        from legal_landscape.models.losses import compute_typed_losses

        correct = compute_typed_losses(
            pair_types=("sentence_rank",),
            parent_sentence=torch.tensor([12.0]),
            counterfactual_sentence=torch.tensor([8.0]),
            rank_direction=torch.tensor([-1.0]),
            rank_margin=1.0,
        )
        wrong = compute_typed_losses(
            pair_types=("sentence_rank",),
            parent_sentence=torch.tensor([8.0]),
            counterfactual_sentence=torch.tensor([12.0]),
            rank_direction=torch.tensor([-1.0]),
            rank_margin=1.0,
        )
        self.assertEqual(correct.response.item(), 0.0)
        self.assertGreater(wrong.response.item(), correct.response.item())

    def test_empty_supervision_masks_produce_exact_zero(self) -> None:
        from legal_landscape.models.losses import compute_typed_losses

        losses = compute_typed_losses(
            charge_logits=torch.zeros((2, 3)),
            charge_targets=torch.zeros((2, 3)),
            charge_mask=torch.tensor([False, False]),
            sentence_predictions=torch.zeros(2),
            sentence_targets=torch.zeros(2),
            sentence_mask=torch.tensor([False, False]),
        )
        self.assertEqual(losses.charge.item(), 0.0)
        self.assertEqual(losses.sentence.item(), 0.0)
        self.assertEqual(losses.total.item(), 0.0)

    def test_predictor_runs_with_dummy_backbone(self) -> None:
        from legal_landscape.models.predictor import DummyBackbone, LegalLandscapePredictor

        model = LegalLandscapePredictor(
            DummyBackbone(vocab_size=20, hidden_size=8),
            hidden_size=8,
            num_charges=3,
            num_penalty_types=6,
            num_factors=5,
        )
        with torch.no_grad():
            model.heads.charge.weight.zero_()
            model.heads.charge.bias.zero_()
        output = model(torch.tensor([[1, 2, 3], [4, 5, 0]]))
        self.assertEqual(tuple(output["charge_logits"].shape), (2, 3))
        self.assertEqual(tuple(output["sentence_months"].shape), (2,))
        self.assertTrue(
            torch.allclose(output["charge_probabilities"].sum(-1), torch.full((2,), 1.5))
        )
        self.assertTrue(torch.allclose(output["charge_weights"].sum(-1), torch.ones(2)))


if __name__ == "__main__":
    unittest.main()
