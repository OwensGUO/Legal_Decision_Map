from __future__ import annotations

import itertools
import json
import math

import pytest

from legal_landscape.training.train import (
    TokenBudgetBatchSampler,
    focus_fact,
    format_model_input,
    inference_batches,
    learning_rate_factor,
    prune_checkpoints,
    select_best_checkpoint,
    shard_batches,
    tune_threshold,
)


def test_factor_supervision_masks_unknown_booleans_and_keeps_amount_existence() -> None:
    from legal_landscape.training.train import factor_supervision

    partial_target, partial_mask = factor_supervision({
        "amount": None,
        "surrender": None,
        "restitution": False,
        "confession": True,
        "role": "unknown",
    })
    assert partial_target == [0.0, 0.0, 0.0, 1.0, 0.0]
    assert partial_mask == [True, False, True, True, False]
    known_target, known_mask = factor_supervision({
        "amount": 12.0,
        "surrender": True,
        "restitution": False,
        "confession": False,
        "role": "principal",
    })
    assert known_target == [1.0, 1.0, 0.0, 0.0, 0.0]
    assert known_mask == [True, True, True, True, True]
    absent_target, absent_mask = factor_supervision({})
    assert absent_target[0] == 0.0
    assert absent_mask == [True, False, False, False, False]


def test_learning_rate_warms_up_then_decays_by_cosine_to_the_floor() -> None:
    factors = [learning_rate_factor(step, 100, 10, 0.1) for step in range(100)]
    assert factors[0] == pytest.approx(0.1)
    assert factors[9] == pytest.approx(1.0)
    assert factors[10] == pytest.approx(1.0)
    assert factors[55] == pytest.approx(0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * 45 / 90)))
    assert factors[-1] == pytest.approx(0.1, abs=1e-3)
    assert all(a >= b for a, b in itertools.pairwise(factors[10:]))
    assert learning_rate_factor(5, 0, 0, 0.1) == 1.0


def test_token_budget_batches_cover_every_index_once_within_budget() -> None:
    lengths = [10, 500, 20, 480, 30, 15, 900, 12] * 5
    sampler = TokenBudgetBatchSampler(lengths, max_tokens=1000, max_batch=4, seed=3)
    batches = list(sampler)
    assert sorted(index for batch in batches for index in batch) == list(range(len(lengths)))
    for batch in batches:
        assert len(batch) <= 4
        assert len(batch) * max(lengths[index] for index in batch) <= 1000 or len(batch) == 1
    assert list(TokenBudgetBatchSampler(lengths, 1000, 4, seed=3)) == batches
    sampler.next_epoch()
    assert list(sampler) != batches


def test_token_budget_sampler_resumes_from_a_batch_position() -> None:
    lengths = list(range(1, 61))
    sampler = TokenBudgetBatchSampler(lengths, max_tokens=120, max_batch=8, seed=1)
    full = list(sampler)
    sampler.advance(5)
    state = sampler.state_dict()
    restored = TokenBudgetBatchSampler(lengths, max_tokens=120, max_batch=8, seed=1)
    restored.load_state_dict(state)
    assert list(restored) == full[5:]
    assert len(restored) == len(full) - 5
    assert restored.batches_per_epoch == len(full)
    with pytest.raises(ValueError, match="seed"):
        TokenBudgetBatchSampler(lengths, 120, 8, seed=2).load_state_dict(state)


def test_fixed_budget_of_one_sample_per_batch_matches_plain_shuffling() -> None:
    sampler = TokenBudgetBatchSampler([5] * 10, max_tokens=5, max_batch=1, seed=0)
    assert all(len(batch) == 1 for batch in sampler)


def test_inference_batches_sort_by_length_and_respect_budget() -> None:
    lengths = [300, 10, 20, 290, 15]
    batches = inference_batches(lengths, max_tokens=600, max_batch=8)
    assert sorted(index for batch in batches for index in batch) == list(range(5))
    assert batches[0] == [1, 4, 2]
    assert all(len(batch) * max(lengths[i] for i in batch) <= 600 for batch in batches)


def test_shard_batches_gives_every_process_the_same_number_of_batches() -> None:
    batches = [[0], [1], [2], [3], [4]]
    shards = [shard_batches(batches, 2, rank) for rank in range(2)]
    assert [len(shard) for shard in shards] == [3, 3]
    assert sorted(index for shard in shards for batch in shard for index in batch) == [
        0, 1, 2, 3, 4, 4,
    ]


def test_focus_fact_keeps_target_and_shared_sentences_in_order_within_budget() -> None:
    text = "甲与乙共谋。乙单独盗走手机。丙另案处理。甲退赃。经鉴定价值1000元。"
    focused = focus_fact(text, "甲", ("乙", "丙"), max_chars=22)
    assert focused == "甲与乙共谋。甲退赃。经鉴定价值1000元。"
    assert focus_fact(text, "甲", ("乙",), max_chars=1000) == text
    assert len(focus_fact("甲" * 50 + "。", "甲", (), max_chars=10)) == 10


def test_model_input_focuses_long_multi_defendant_facts() -> None:
    record = {
        "dataset": "cmdl", "target_defendant": "甲", "co_defendants": ["乙"],
        "fact_strict": "乙单独盗走手机。" * 10 + "甲退赃。",
    }
    text = format_model_input(record, use_factors=False, max_fact_chars=20)
    assert "[FACT]\n甲退赃。" in text
    assert "乙单独" not in text


def test_threshold_tuning_maximizes_micro_f1() -> None:
    probabilities = [[0.9, 0.35], [0.4, 0.8], [0.2, 0.3]]
    truth = [[1, 1], [0, 1], [0, 0]]
    assert tune_threshold(probabilities, truth, grid=(0.32, 0.5, 0.7)) == 0.32
    assert tune_threshold([[0.6]], [[1]], grid=(0.5, 0.7)) == 0.5


def test_best_checkpoint_selection_and_pruning(tmp_path) -> None:
    for name, score in (("checkpoint-step-00000100", 0.5), ("checkpoint-step-00000200", 0.7),
                        ("checkpoint-step-00000300", 0.6), ("checkpoint-final", None)):
        (tmp_path / name).mkdir()
        progress = {"steps": 1, "validation_score": score}
        (tmp_path / name / "training_progress.json").write_text(json.dumps(progress))
    best = select_best_checkpoint(tmp_path)
    assert best == tmp_path / "checkpoint-step-00000200"
    removed = prune_checkpoints(tmp_path, keep={best, tmp_path / "checkpoint-step-00000300"})
    assert removed == [tmp_path / "checkpoint-step-00000100"]
    assert (tmp_path / "checkpoint-final").exists()
    assert select_best_checkpoint(tmp_path / "missing") is None


def test_best_checkpoint_ties_prefer_more_training_steps(tmp_path) -> None:
    for name, steps in (("checkpoint-final", 20), ("checkpoint-step-00000010", 10)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "training_progress.json").write_text(
            json.dumps({"steps": steps, "validation_score": 0.5})
        )
    assert select_best_checkpoint(tmp_path) == tmp_path / "checkpoint-final"
