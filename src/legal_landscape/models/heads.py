"""Prediction heads and probabilistic charge marginalization."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def marginalize_sentence(
    charge_logits: Tensor,
    conditional_months: Tensor,
    *,
    hard: bool = False,
) -> tuple[Tensor, Tensor]:
    """Marginalize sentence months over charges.

    Charges are multi-label, so each charge probability is an independent sigmoid; the
    mixture weights are those probabilities renormalized to sum to one.
    """
    probabilities = torch.sigmoid(charge_logits)
    weights = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    if hard:
        indices = weights.argmax(dim=-1, keepdim=True)
        months = conditional_months.gather(-1, indices).squeeze(-1)
    else:
        months = (weights * conditional_months).sum(dim=-1)
    return months, weights


class MultiTaskHeads(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_charges: int,
        num_articles: int,
        num_penalty_types: int,
        num_factors: int,
    ) -> None:
        super().__init__()
        self.charge = nn.Linear(hidden_size, num_charges)
        self.article = nn.Linear(hidden_size + num_charges, num_articles)
        self.penalty_type = nn.Linear(hidden_size, num_penalty_types)
        self.sentence_by_charge = nn.Linear(hidden_size + num_articles, num_charges)
        self.factors = nn.Linear(hidden_size, num_factors)

    def forward(self, hidden: Tensor, *, hard_charge: bool = False) -> dict[str, Tensor]:
        charge_logits = self.charge(hidden)
        charge_probabilities = torch.sigmoid(charge_logits)
        article_logits = self.article(torch.cat((hidden, charge_probabilities), dim=-1))
        article_probabilities = torch.sigmoid(article_logits)
        conditional_months = torch.nn.functional.softplus(
            self.sentence_by_charge(torch.cat((hidden, article_probabilities), dim=-1))
        )
        sentence_months, weights = marginalize_sentence(
            charge_logits, conditional_months, hard=hard_charge
        )
        return {
            "charge_logits": charge_logits,
            "charge_probabilities": charge_probabilities,
            "charge_weights": weights,
            "article_logits": article_logits,
            "article_probabilities": article_probabilities,
            "penalty_type_logits": self.penalty_type(hidden),
            "sentence_by_charge": conditional_months,
            "sentence_months": sentence_months,
            "factor_logits": self.factors(hidden),
        }
