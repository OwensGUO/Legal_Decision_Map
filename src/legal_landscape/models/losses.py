"""Explicit routing for supervised and typed-counterfactual losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class LossBreakdown:
    charge: Tensor
    article: Tensor
    sentence: Tensor
    invariant: Tensor
    boundary: Tensor
    response: Tensor
    factor: Tensor
    total: Tensor
    active_counts: dict[str, int]


def _anchor(values: tuple[Any, ...]) -> Tensor:
    for value in values:
        if isinstance(value, Tensor):
            return value.sum() * 0.0
    return torch.tensor(0.0)


def _masked_mean(values: Tensor, mask: Tensor, zero: Tensor) -> Tensor:
    return values[mask].mean() if bool(mask.any()) else zero


def compute_typed_losses(
    *,
    charge_logits: Tensor | None = None,
    charge_targets: Tensor | None = None,
    charge_mask: Tensor | None = None,
    article_logits: Tensor | None = None,
    article_targets: Tensor | None = None,
    article_mask: Tensor | None = None,
    penalty_logits: Tensor | None = None,
    penalty_targets: Tensor | None = None,
    penalty_mask: Tensor | None = None,
    sentence_predictions: Tensor | None = None,
    sentence_targets: Tensor | None = None,
    sentence_mask: Tensor | None = None,
    factor_logits: Tensor | None = None,
    factor_targets: Tensor | None = None,
    factor_mask: Tensor | None = None,
    pair_types: tuple[str, ...] = (),
    parent_charge_logits: Tensor | None = None,
    counterfactual_charge_logits: Tensor | None = None,
    target_charge_indices: Tensor | None = None,
    parent_sentence: Tensor | None = None,
    counterfactual_sentence: Tensor | None = None,
    rank_direction: Tensor | None = None,
    rank_margin: float = 0.0,
    weights: dict[str, float] | None = None,
) -> LossBreakdown:
    """Compute the supervised and counterfactual objective with empty-mask safeguards."""
    zero = _anchor(
        (
            charge_logits,
            article_logits,
            penalty_logits,
            sentence_predictions,
            factor_logits,
            parent_charge_logits,
            counterfactual_charge_logits,
            parent_sentence,
        )
    )
    counts = {
        name: 0
        for name in (
            "charge",
            "article",
            "sentence",
            "invariant",
            "boundary",
            "response",
            "factor",
        )
    }
    charge = zero
    if charge_logits is not None and charge_targets is not None:
        mask = (
            torch.ones(charge_logits.shape[0], dtype=torch.bool, device=charge_logits.device)
            if charge_mask is None
            else charge_mask.bool()
        )
        per_item = F.binary_cross_entropy_with_logits(
            charge_logits, charge_targets, reduction="none"
        ).mean(-1)
        charge = _masked_mean(per_item, mask, zero)
        counts["charge"] = int(mask.sum().item())

    article = zero
    if article_logits is not None and article_targets is not None:
        mask = (
            torch.ones(article_logits.shape[0], dtype=torch.bool, device=article_logits.device)
            if article_mask is None
            else article_mask.bool()
        )
        per_item = F.binary_cross_entropy_with_logits(
            article_logits, article_targets, reduction="none"
        ).mean(-1)
        article = _masked_mean(per_item, mask, zero)
        counts["article"] = int(mask.sum().item())

    sentence = zero
    sentence_count = 0
    if penalty_logits is not None and penalty_targets is not None:
        mask = (
            torch.ones(penalty_logits.shape[0], dtype=torch.bool, device=penalty_logits.device)
            if penalty_mask is None
            else penalty_mask.bool()
        )
        sentence = sentence + _masked_mean(
            F.cross_entropy(penalty_logits, penalty_targets, reduction="none"), mask, zero
        )
        sentence_count += int(mask.sum().item())
    if sentence_predictions is not None and sentence_targets is not None:
        mask = (
            torch.ones_like(sentence_predictions, dtype=torch.bool)
            if sentence_mask is None
            else sentence_mask.bool()
        )
        # Log months keep the regression on the same scale as the classification losses.
        sentence = sentence + _masked_mean(
            F.smooth_l1_loss(
                torch.log1p(sentence_predictions.clamp_min(0.0)),
                torch.log1p(sentence_targets.clamp_min(0.0)),
                reduction="none",
            ),
            mask,
            zero,
        )
        sentence_count += int(mask.sum().item())
    counts["sentence"] = sentence_count

    factor = zero
    if factor_logits is not None and factor_targets is not None:
        mask = (
            torch.ones(factor_logits.shape[0], dtype=torch.bool, device=factor_logits.device)
            if factor_mask is None
            else factor_mask.bool()
        )
        per_item = F.binary_cross_entropy_with_logits(
            factor_logits, factor_targets, reduction="none"
        ).mean(-1)
        factor = _masked_mean(per_item, mask, zero)
        counts["factor"] = int(mask.sum().item())

    invariant = boundary = response = zero
    if pair_types:
        device = next(
            value.device
            for value in (parent_charge_logits, counterfactual_charge_logits, parent_sentence)
            if isinstance(value, Tensor)
        )
        invariant_mask = torch.tensor([item == "invariant" for item in pair_types], device=device)
        flip_mask = torch.tensor([item == "charge_flip" for item in pair_types], device=device)
        rank_mask = torch.tensor([item == "sentence_rank" for item in pair_types], device=device)
        if bool(invariant_mask.any()):
            if parent_charge_logits is None or counterfactual_charge_logits is None:
                raise ValueError("invariant pairs require both charge-logit tensors")
            parent_prob = torch.sigmoid(parent_charge_logits[invariant_mask])
            counterfactual_prob = torch.sigmoid(counterfactual_charge_logits[invariant_mask])
            invariant = F.mse_loss(parent_prob, counterfactual_prob)
        if bool(flip_mask.any()):
            if counterfactual_charge_logits is None or target_charge_indices is None:
                raise ValueError("charge_flip pairs require logits and target indices")
            flipped_logits = counterfactual_charge_logits[flip_mask]
            # Multi-label target: the flipped case carries the target charge and nothing else.
            flipped_targets = F.one_hot(
                target_charge_indices[flip_mask], num_classes=flipped_logits.shape[-1]
            ).to(flipped_logits.dtype)
            boundary = F.binary_cross_entropy_with_logits(flipped_logits, flipped_targets)
        if bool(rank_mask.any()):
            if parent_sentence is None or counterfactual_sentence is None or rank_direction is None:
                raise ValueError("sentence_rank pairs require both predictions and directions")
            signed_change = rank_direction[rank_mask] * (
                counterfactual_sentence[rank_mask] - parent_sentence[rank_mask]
            )
            response = torch.relu(
                torch.as_tensor(rank_margin, device=device) - signed_change
            ).mean()
        counts["invariant"] = int(invariant_mask.sum().item())
        counts["boundary"] = int(flip_mask.sum().item())
        counts["response"] = int(rank_mask.sum().item())

    scale = {
        "article": 1.0,
        "sentence": 1.0,
        "invariant": 1.0,
        "boundary": 1.0,
        "response": 1.0,
        "factor": 1.0,
        **(weights or {}),
    }
    total = (
        charge
        + scale["article"] * article
        + scale["sentence"] * sentence
        + scale["invariant"] * invariant
        + scale["boundary"] * boundary
        + scale["response"] * response
        + scale["factor"] * factor
    )
    return LossBreakdown(
        charge,
        article,
        sentence,
        invariant,
        boundary,
        response,
        factor,
        total,
        counts,
    )
