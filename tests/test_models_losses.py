from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # The project keeps non-model tooling usable before GPU deps are installed.
    torch = None  # type: ignore[assignment]


@unittest.skipIf(torch is None, "PyTorch is not installed in this interpreter")
class ModelLossTests(unittest.TestCase):
    def test_unknown_factor_fields_do_not_supervise_and_known_booleans_do(self) -> None:
        from legal_landscape.models.losses import compute_typed_losses

        logits = torch.zeros(2, 5, requires_grad=True)
        targets = torch.tensor([
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
        ])
        # amount existence, surrender, restitution, confession, accessory role
        mask = torch.tensor([
            [True, False, True, True, False],
            [False, False, False, False, False],
        ])
        losses = compute_typed_losses(
            factor_logits=logits, factor_targets=targets, factor_mask=mask,
        )
        losses.factor.backward()
        grad = logits.grad
        assert grad is not None
        self.assertEqual(grad[0, 1].item(), 0.0)
        self.assertEqual(grad[0, 4].item(), 0.0)
        self.assertNotEqual(grad[0, 2].item(), 0.0)
        self.assertNotEqual(grad[0, 3].item(), 0.0)
        self.assertTrue(torch.all(grad[1] == 0))
        self.assertGreater(losses.factor.item(), 0.0)

        unknown = torch.zeros(1, 5, requires_grad=True)
        empty = compute_typed_losses(
            factor_logits=unknown,
            factor_targets=torch.zeros(1, 5),
            factor_mask=torch.zeros(1, 5, dtype=torch.bool),
        )
        empty.factor.backward()
        self.assertEqual(empty.factor.item(), 0.0)
        assert unknown.grad is not None
        self.assertTrue(torch.all(unknown.grad == 0))

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

    def test_article_supervision_uses_multilabel_bce_and_empty_mask(self) -> None:
        from legal_landscape.models.losses import compute_typed_losses

        active = compute_typed_losses(
            article_logits=torch.tensor([[2.0, -2.0], [-1.0, 1.0]]),
            article_targets=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        )
        empty = compute_typed_losses(
            article_logits=torch.zeros((2, 2)),
            article_targets=torch.zeros((2, 2)),
            article_mask=torch.tensor([False, False]),
        )
        self.assertGreater(active.article.item(), 0.0)
        self.assertEqual(active.active_counts["article"], 2)
        self.assertEqual(empty.article.item(), 0.0)
        self.assertEqual(empty.active_counts["article"], 0)

    def test_multilabel_losses_sum_over_labels_by_default(self) -> None:
        import math

        from legal_landscape.models.losses import compute_typed_losses

        targets = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        summed = compute_typed_losses(
            charge_logits=torch.zeros((2, 4)), charge_targets=targets,
            article_logits=torch.zeros((2, 4)), article_targets=targets,
            pair_types=("charge_flip",),
            counterfactual_charge_logits=torch.zeros((1, 3)),
            target_charge_indices=torch.tensor([1]),
        )
        self.assertAlmostEqual(summed.charge.item(), 4 * math.log(2), places=5)
        self.assertAlmostEqual(summed.article.item(), 4 * math.log(2), places=5)
        self.assertAlmostEqual(summed.boundary.item(), 3 * math.log(2), places=5)
        averaged = compute_typed_losses(
            charge_logits=torch.zeros((2, 4)), charge_targets=targets,
            multilabel_reduction="mean",
        )
        self.assertAlmostEqual(averaged.charge.item(), math.log(2), places=5)

    def test_invariant_loss_sums_squared_probability_changes_over_labels(self) -> None:
        from legal_landscape.models.losses import compute_typed_losses

        parent = torch.tensor([[2.0, -1.0]])
        counterfactual = torch.tensor([[1.0, 0.0]])
        losses = compute_typed_losses(
            pair_types=("invariant",),
            parent_charge_logits=parent,
            counterfactual_charge_logits=counterfactual,
        )
        expected = ((torch.sigmoid(parent) - torch.sigmoid(counterfactual)) ** 2).sum()
        self.assertAlmostEqual(losses.invariant.item(), expected.item(), places=6)

    def test_unknown_multilabel_reduction_is_rejected(self) -> None:
        from legal_landscape.models.losses import compute_typed_losses

        with self.assertRaises(ValueError):
            compute_typed_losses(
                charge_logits=torch.zeros((1, 2)), charge_targets=torch.zeros((1, 2)),
                multilabel_reduction="median",
            )

    def test_predictor_runs_with_dummy_backbone(self) -> None:
        from legal_landscape.models.predictor import DummyBackbone, LegalLandscapePredictor

        model = LegalLandscapePredictor(
            DummyBackbone(vocab_size=20, hidden_size=8),
            hidden_size=8,
            num_charges=3,
            num_articles=4,
            num_penalty_types=6,
            num_factors=5,
        )
        with torch.no_grad():
            model.heads.charge.weight.zero_()
            model.heads.charge.bias.zero_()
        output = model(torch.tensor([[1, 2, 3], [4, 5, 0]]))
        self.assertEqual(tuple(output["charge_logits"].shape), (2, 3))
        self.assertEqual(tuple(output["article_logits"].shape), (2, 4))
        self.assertEqual(model.heads.article.in_features, 8 + 3)
        self.assertEqual(model.heads.sentence_by_charge.in_features, 8 + 4)
        self.assertEqual(tuple(output["sentence_months"].shape), (2,))
        self.assertTrue(
            torch.allclose(output["charge_probabilities"].sum(-1), torch.full((2,), 1.5))
        )
        self.assertTrue(torch.allclose(output["charge_weights"].sum(-1), torch.ones(2)))


def _ddp_factor_mask_worker(rank, world_size, init_path, root):
    import os
    from pathlib import Path

    import torch.distributed as distributed

    os.environ.update({
        "RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(world_size),
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29521",
    })
    distributed.init_process_group(
        "gloo", init_method=f"file://{init_path}", rank=rank, world_size=world_size,
    )
    try:
        from legal_landscape.models.losses import compute_typed_losses

        torch.manual_seed(0)
        layer = torch.nn.Linear(2, 5)
        model = torch.nn.parallel.DistributedDataParallel(layer)
        logits = model(torch.ones(2, 2))
        mask = torch.tensor([
            [True, False, True, True, False],
            [False, False, False, False, False],
        ])
        targets = torch.zeros(2, 5)
        targets[0, 3] = 1.0
        loss = compute_typed_losses(
            factor_logits=logits, factor_targets=targets, factor_mask=mask,
        )
        loss.factor.backward()
        weight_grad = model.module.weight.grad.detach().cpu()
        torch.save(weight_grad, Path(root) / f"rank-{rank}.pt")
    finally:
        distributed.destroy_process_group()


def test_ddp_unknown_factor_fields_have_no_gradient(tmp_path):
    import pytest

    pytest.importorskip("torch")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("Torch Gloo distributed runtime is not installed")
    torch.multiprocessing.spawn(
        _ddp_factor_mask_worker,
        args=(2, str(tmp_path / "init"), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    grads = [
        torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True) for rank in range(2)
    ]
    # Output rows are amount, surrender, restitution, confession, role.
    assert torch.allclose(grads[0], grads[1])
    assert torch.count_nonzero(grads[0][1]).item() == 0
    assert torch.count_nonzero(grads[0][4]).item() == 0
    assert torch.count_nonzero(grads[0][2]).item() > 0
    assert torch.count_nonzero(grads[0][3]).item() > 0


if __name__ == "__main__":
    unittest.main()
