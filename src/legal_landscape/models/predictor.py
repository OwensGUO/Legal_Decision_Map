"""Multitask predictor with a replaceable text backbone."""

from __future__ import annotations

from torch import Tensor, nn

from legal_landscape.models.heads import MultiTaskHeads


class DummyBackbone(nn.Module):
    """Small CPU backbone used only for tests and bounded dry-runs."""

    def __init__(self, vocab_size: int = 256, hidden_size: int = 32) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        tokens = self.embedding(input_ids)
        mask = (input_ids != 0) if attention_mask is None else attention_mask.bool()
        weights = mask.unsqueeze(-1).to(tokens.dtype)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class LegalLandscapePredictor(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        hidden_size: int,
        num_charges: int,
        num_articles: int,
        num_penalty_types: int,
        num_factors: int,
        hard_charge: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.heads = MultiTaskHeads(
            hidden_size,
            num_charges,
            num_articles,
            num_penalty_types,
            num_factors,
        )
        self.hard_charge = hard_charge

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> dict[str, Tensor]:
        encoded = self.backbone(input_ids, attention_mask=attention_mask)
        if isinstance(encoded, Tensor):
            pooled = encoded
        else:
            hidden = encoded.last_hidden_state
            mask = (input_ids != 0) if attention_mask is None else attention_mask.bool()
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        return self.heads(pooled, hard_charge=self.hard_charge)
