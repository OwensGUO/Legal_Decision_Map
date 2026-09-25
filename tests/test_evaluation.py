from __future__ import annotations

import numpy as np
import pytest

from legal_landscape.evaluation.bootstrap import (
    bootstrap_metric_set,
    cluster_bootstrap,
    holm_adjust,
    paired_cluster_test,
)
from legal_landscape.evaluation.cmdl_case_metrics import cmdl_metrics
from legal_landscape.evaluation.counterfactual_metrics import counterfactual_metrics
from legal_landscape.evaluation.static_metrics import (
    charge_metrics,
    multiclass_metrics,
    multilabel_metrics,
    sentence_class,
    sentence_metrics,
)


def test_charge_metrics_have_hand_checked_values() -> None:
    truth = np.array([[1, 0], [1, 1], [0, 1]])
    prediction = np.array([[1, 0], [1, 0], [0, 1]])
    result = charge_metrics(truth, prediction)
    assert result["accuracy"] == 2 / 3
    assert result["macro_precision"] == 1.0
    assert result["macro_recall"] == 0.75
    assert result["micro_f1"] == 6 / 7
    assert result["macro_f1"] == (1.0 + 2 / 3) / 2
    assert result["exact_match"] == 2 / 3


def test_multilabel_metrics_report_required_and_supplementary_values() -> None:
    truth = np.array([[1, 0], [1, 1], [0, 1]])
    prediction = np.array([[1, 0], [1, 0], [0, 1]])
    result = multilabel_metrics(truth, prediction)
    assert result["accuracy"] == 2 / 3
    assert result["macro_precision"] == 1.0
    assert result["macro_recall"] == 0.75
    assert result["macro_f1"] == (1.0 + 2 / 3) / 2
    assert result["hamming_accuracy"] == 5 / 6


def test_multiclass_metrics_and_sentence_bins_have_hand_checked_values() -> None:
    result = multiclass_metrics(
        ("a", "b", "c", "c"),
        ("a", "b", "b", "c"),
        labels=("a", "b", "c"),
    )
    assert result["accuracy"] == 0.75
    assert result["macro_precision"] == (1.0 + 0.5 + 1.0) / 3
    assert result["macro_recall"] == (1.0 + 1.0 + 0.5) / 3
    assert result["macro_f1"] == (1.0 + 2 / 3 + 2 / 3) / 3
    assert sentence_class("fixed_term", 6) == "fixed_term_0_6"
    assert sentence_class("fixed_term", 7) == "fixed_term_7_12"
    assert sentence_class("fixed_term", 121) == "fixed_term_121_plus"
    assert sentence_class("life", None) == "life"


def test_sentence_metrics_exclude_life_and_death() -> None:
    result = sentence_metrics(
        predicted_months=np.array([10.0, 99.0, 99.0, 24.0]),
        true_months=np.array([12.0, 0.0, 0.0, 18.0]),
        penalty_types=("fixed_term", "life", "death", "fixed_term"),
        tolerance_months=3.0,
    )
    assert result["eligible_count"] == 2
    assert result["mae"] == 4.0
    assert result["tolerance_accuracy"] == 0.5


def test_counterfactual_metrics_are_type_specific() -> None:
    rows = [
        {
            "intervention_type": "charge_flip",
            "target_charge": 1,
            "parent_charge_probabilities": [0.9, 0.1],
            "counterfactual_charge_probabilities": [0.2, 0.8],
        },
        {
            "intervention_type": "invariant",
            "parent_charge_probabilities": [0.8, 0.2],
            "counterfactual_charge_probabilities": [0.7, 0.3],
        },
        {
            "intervention_type": "sentence_rank",
            "parent_sentence": 12.0,
            "counterfactual_sentence": 8.0,
            "rank_direction": -1,
        },
    ]
    result = counterfactual_metrics(rows, invariant_drift_threshold=0.25)
    assert result["flip_accuracy"] == 1.0
    assert result["invariant_violation_rate"] == 0.0
    assert result["sentence_pairwise_accuracy"] == 1.0
    assert result["monotonicity_violation_rate"] == 0.0
    assert abs(result["non_target_prediction_drift"] - 0.1) < 1e-12


def test_cmdl_reports_defendant_case_and_count_strata() -> None:
    rows = [
        {"group_id": "g1", "correct": True},
        {"group_id": "g1", "correct": False},
        {"group_id": "g2", "correct": True},
    ]
    result = cmdl_metrics(rows)
    assert result["defendant_accuracy"] == 2 / 3
    assert result["case_exact_accuracy"] == 0.5
    assert result["by_defendant_count"]["2"]["case_count"] == 1


def test_cluster_bootstrap_resamples_whole_groups() -> None:
    rows = [
        {"group_id": "g1", "value": 0.0},
        {"group_id": "g1", "value": 0.0},
        {"group_id": "g2", "value": 1.0},
    ]
    seen_lengths: list[int] = []

    def metric(sample):
        seen_lengths.append(len(sample))
        return float(np.mean([item["value"] for item in sample]))

    result = cluster_bootstrap(rows, metric, iterations=40, seed=42)
    assert result["estimate"] == 1 / 3
    assert set(seen_lengths).issubset({2, 3, 4})
    assert result["lower"] <= result["estimate"] <= result["upper"]


def test_holm_adjust_is_monotone_in_sorted_p_values() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.2, "e": 0.5})
    ordered = [adjusted[key] for key in ("a", "c", "b", "d", "e")]
    assert ordered == sorted(ordered)
    assert adjusted["a"] == 0.05


def test_bootstrap_metric_set_emits_intervals_for_named_metrics() -> None:
    rows = [
        {"group_id": "g1", "value": 0.0},
        {"group_id": "g1", "value": 0.0},
        {"group_id": "g2", "value": 1.0},
    ]

    def metrics(sample):
        values = np.asarray([row["value"] for row in sample], dtype=float)
        return {"mean": float(values.mean()), "positive_rate": float((values > 0).mean())}

    result = bootstrap_metric_set(
        rows,
        metrics,
        ("mean", "positive_rate"),
        iterations=40,
        seed=42,
    )
    assert result["mean"]["estimate"] == 1 / 3
    assert result["mean"]["lower"] <= 1 / 3 <= result["mean"]["upper"]
    assert result["positive_rate"]["groups"] == 2


def test_paired_cluster_test_requires_matching_groups_and_reports_difference() -> None:
    candidate = [
        {"group_id": "g1", "value": 2.0},
        {"group_id": "g2", "value": 4.0},
    ]
    reference = [
        {"group_id": "g1", "value": 1.0},
        {"group_id": "g2", "value": 3.0},
    ]
    def metric(sample):
        return float(np.mean([row["value"] for row in sample]))

    result = paired_cluster_test(candidate, reference, metric, iterations=40, seed=7)
    assert result["difference"] == 1.0
    assert result["groups"] == 2
    assert 0.0 < result["p_value"] <= 1.0

    with pytest.raises(ValueError, match="identical group IDs"):
        paired_cluster_test(candidate, reference[:1], metric, iterations=10)


def test_sentence_metrics_can_match_single_charge_training_eligibility() -> None:
    result = sentence_metrics(
        predicted_months=np.array([10.0, 30.0]),
        true_months=np.array([12.0, 60.0]),
        penalty_types=("fixed_term", "fixed_term"),
        charge_counts=np.array([1, 2]),
    )
    assert result["eligible_count"] == 1
    assert result["mae"] == 2.0
