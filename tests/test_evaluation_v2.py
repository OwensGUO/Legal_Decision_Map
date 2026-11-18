from __future__ import annotations

import gzip
import json
import math
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from legal_landscape.evaluation.bootstrap import (
    bootstrap_metric_set,
    fast_bootstrap_metric_set,
    fast_paired_test,
    paired_cluster_test,
)
from legal_landscape.evaluation.prepared import (
    METRIC_VERSION,
    prepare_cmdl,
    prepare_counterfactual,
    prepare_static,
)

ROOT = Path(__file__).resolve().parents[1]


def _static_rows(seed: int, count: int = 60) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(count):
        true = [int(value) for value in rng.random(3) < 0.4]
        predicted = [int(value) for value in rng.random(3) < 0.4]
        rows.append({
            "group_id": f"g{index // 3}",
            "true_charges": true,
            "predicted_charges": predicted,
            "true_articles": predicted,
            "predicted_articles": true,
            "penalty_type": "fixed_term",
            "predicted_penalty_type": "fixed_term",
            "true_months": float(rng.integers(1, 40)),
            "predicted_months": float(rng.integers(1, 40)),
        })
    return rows


def test_fast_bootstrap_reproduces_row_resampling_for_label_set_free_metrics() -> None:
    rows = _static_rows(1)
    names = ("charge_accuracy", "charge_micro_f1", "sentence_mae", "sentence_accuracy")
    fast = fast_bootstrap_metric_set(prepare_static(rows), names, iterations=60, seed=7)
    slow = bootstrap_metric_set(
        rows, lambda sample: prepare_static(sample).metrics(None), names, iterations=60, seed=7
    )
    for name in names:
        for key in ("estimate", "lower", "upper"):
            assert math.isclose(fast[name][key], slow[name][key], abs_tol=1e-12)


def test_fast_bootstrap_macro_averages_over_classes_present_in_each_resample() -> None:
    """A rare class missing from a resample is not averaged in as zero.

    Recomputing a macro metric on a resampled dataset averages over the classes that
    dataset's truth contains; the vectorized replicate must do the same, or the interval
    sits below the point estimate.
    """
    rows = _static_rows(4)
    for row in rows:
        row["true_charges"] = [*row["true_charges"], 0]
        row["predicted_charges"] = [*row["predicted_charges"], 0]
    rows[0]["true_charges"][3] = 1
    rows[0]["predicted_charges"][3] = 1
    rows[0]["true_months"], rows[0]["predicted_months"] = 100.0, 100.0
    names = (
        "charge_macro_precision", "charge_macro_recall", "charge_macro_f1",
        "article_macro_f1", "sentence_macro_precision", "sentence_macro_recall",
        "sentence_macro_f1",
    )
    fast = fast_bootstrap_metric_set(prepare_static(rows), names, iterations=60, seed=7)
    slow = bootstrap_metric_set(
        rows, lambda sample: prepare_static(sample).metrics(None), names, iterations=60, seed=7
    )
    for name in names:
        for key in ("estimate", "lower", "upper"):
            assert math.isclose(fast[name][key], slow[name][key], abs_tol=1e-12), (name, key)


def test_paired_macro_difference_reproduces_row_resampling() -> None:
    reference = _static_rows(5)
    rng = np.random.default_rng(6)
    candidate = []
    for index, row in enumerate(reference):
        row["true_charges"] = [*row["true_charges"], int(index == 0)]
        row["predicted_charges"] = [*row["predicted_charges"], int(index == 0)]
        row["case_id"] = f"c{index}"
        row["charge_vocabulary"] = ["a", "b", "c", "d"]
        row["article_vocabulary"] = ["a", "b", "c"]
        row["articles_available"] = True
        predicted = [int(value) for value in rng.random(4) < 0.4]
        candidate.append({**row, "predicted_charges": predicted})
    names = ("charge_macro_f1", "sentence_macro_f1")
    fast = fast_paired_test(
        prepare_static(candidate), prepare_static(reference), names, iterations=60, seed=11
    )
    for name in names:
        slow = paired_cluster_test(
            candidate, reference,
            lambda sample, metric=name: prepare_static(sample).metrics(None)[metric],
            iterations=60, seed=11,
        )
        for key in ("difference", "lower", "upper", "p_value"):
            assert math.isclose(fast[name][key], slow[key], abs_tol=1e-12), (name, key)


def test_fast_paired_test_reproduces_row_resampling() -> None:
    reference = _static_rows(2)
    rng = np.random.default_rng(3)
    candidate = []
    for index, row in enumerate(reference):
        row["case_id"] = f"c{index}"
        row["charge_vocabulary"] = ["a", "b", "c"]
        row["article_vocabulary"] = ["a", "b", "c"]
        row["unseen_true_charge_labels"] = []
        row["unseen_true_article_labels"] = []
        row["articles_available"] = True
        predicted = [int(value) for value in rng.random(3) < 0.4]
        candidate.append({
            **row,
            "predicted_charges": predicted,
            "predicted_months": float(rng.integers(1, 40)),
        })
    fast = fast_paired_test(
        prepare_static(candidate), prepare_static(reference), ("charge_accuracy", "sentence_mae"),
        iterations=60, seed=11,
    )
    for name in ("charge_accuracy", "sentence_mae"):
        slow = paired_cluster_test(
            candidate, reference,
            lambda sample, metric=name: prepare_static(sample).metrics(None)[metric],
            iterations=60, seed=11,
        )
        for key in ("difference", "lower", "upper", "p_value"):
            assert math.isclose(fast[name][key], slow[key], abs_tol=1e-12)


def _row(true, predicted, group="g1", **extra):
    return {
        "group_id": group, "true_charges": true, "predicted_charges": predicted,
        "penalty_type": "fixed_term", "predicted_penalty_type": "fixed_term",
        "true_months": 6.0, "predicted_months": 6.0,
        **extra,
    }


def test_complete_cmdl_limit_and_paired_identity(tmp_path) -> None:
    from legal_landscape.evaluation.bootstrap import assert_paired_rows

    module = runpy.run_path(str(ROOT / "scripts" / "evaluate_model.py"))
    rows = [
        {
            "group_id": "case-a",
            "case_id": "a-1",
            "true_charges": [1],
            "charge_vocabulary": ["盗窃"],
        },
        {
            "group_id": "case-a",
            "case_id": "a-2",
            "true_charges": [0],
            "charge_vocabulary": ["盗窃"],
        },
        {
            "group_id": "case-b",
            "case_id": "b-1",
            "true_charges": [1],
            "charge_vocabulary": ["盗窃"],
        },
    ]
    destination = tmp_path / "cmdl.jsonl"
    destination.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    selected = module["read_rows"](destination, 1, complete_groups=True)
    assert [row["case_id"] for row in selected] == ["a-1", "a-2"]
    swapped = [rows[1], rows[0]]
    assert_paired_rows(rows[:2], swapped)
    with pytest.raises(ValueError, match="sample sets differ"):
        assert_paired_rows(rows[:2], rows[:1])
    with pytest.raises(ValueError, match="repeated"):
        assert_paired_rows(rows[:2], [*rows[:2], rows[0]])
    changed = [{**rows[0], "true_charges": [0]}, rows[1]]
    with pytest.raises(ValueError, match="different truth"):
        assert_paired_rows(rows[:2], changed)


def test_missing_article_labels_stop_scoring(tmp_path) -> None:
    module = runpy.run_path(str(ROOT / "scripts" / "evaluate_model.py"))
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({
        "group_id": "g", "case_id": "c",
        "true_charges": [1], "predicted_charges": [1],
        "true_articles": [0, 0], "predicted_articles": [0, 0],
        "articles_available": False,
        "penalty_type": "fixed_term", "predicted_penalty_type": "fixed_term",
        "true_months": 6, "predicted_months": 6,
    }) + "\n", encoding="utf-8")
    args = type("Args", (), {
        "limit": 10, "kind": "static", "parent_records": None, "metadata": None,
    })()
    with pytest.raises(SystemExit, match="missing article"):
        module["_load"](path, args)


def test_distinct_unseen_labels_are_separate_macro_classes() -> None:
    prepared = prepare_static([
        _row([1], [1], unseen_true_charge_labels=["X", "Y"]),
        _row([0], [0], "g2", unseen_true_charge_labels=["X"]),
    ])
    # Sample 1: A,X,Y true and only A predicted. Sample 2 shares unseen label X.
    # Micro: tp=1, fn=3 (X,Y,X) → F1 = 2/5? User asked one sample A/X/Y predict A → 0.5.
    one = prepare_static([_row([1], [1], unseen_true_charge_labels=["X", "Y"])])
    one_metrics = one.metrics(None)
    assert one_metrics["charge_micro_f1"] == 0.5
    assert one_metrics["charge_macro_f1"] == pytest.approx(1 / 3)
    # Shared unseen label X is one column. Resampling cannot add a class.
    assert prepared.charge.tp.shape[1] == 3
    assert prepared.metrics(None)["charge_macro_f1"] == pytest.approx(1 / 3)
    assert prepared.metrics(np.array([2.0, 0.0]))["charge_macro_f1"] != 0.0


def test_macro_average_uses_classes_present_in_the_weighted_truth() -> None:
    prepared = prepare_static([_row([1, 0, 0], [1, 0, 0]), _row([0, 1, 0], [0, 1, 0], "g2")])
    metrics = prepared.metrics(None)
    assert metrics["charge_macro_f1"] == 1.0
    assert metrics["charge_macro_f1_all_labels"] == 2 / 3
    assert metrics["metric_version"] == METRIC_VERSION
    # A resample without the second group has no class-1 truth, so class 1 leaves the average.
    without_second_group = prepared.metrics(np.array([1.0, 0.0]))
    assert without_second_group["charge_macro_f1"] == 1.0
    assert without_second_group["charge_macro_f1_all_labels"] == 1 / 3


def test_sentence_macro_average_uses_classes_present_in_the_weighted_truth() -> None:
    first = {**_row([1], [1]), "true_months": 3.0, "predicted_months": 30.0}
    second = {**_row([1], [1], "g2"), "true_months": 30.0, "predicted_months": 30.0}
    prepared = prepare_static([first, second])
    # Classes 0-6 (recall 0) and 25-36 (precision 1/2) are both in the full truth.
    assert prepared.metrics(None)["sentence_macro_f1"] == pytest.approx((0 + 2 / 3) / 2)
    # Without the first group only 25-36 is true; the 0-6 class leaves the average.
    assert prepared.metrics(np.array([0.0, 1.0]))["sentence_macro_f1"] == 1.0


def _pair(kind, parent, counterfactual, **extra):
    return {
        "group_id": extra.pop("group_id", "g1"), "intervention_type": kind,
        "parent_charge_probabilities": parent,
        "counterfactual_charge_probabilities": counterfactual,
        "parent_sentence": extra.pop("parent_sentence", 12.0),
        "counterfactual_sentence": extra.pop("counterfactual_sentence", 12.0),
        **extra,
    }


def test_counterfactual_v2_definitions() -> None:
    rows = [
        _pair("charge_flip", [0.9, 0.1, 0.0], [0.6, 0.7, 0.0],
              target_charge=1, parent_charge_indices=[0]),
        _pair("charge_flip", [0.9, 0.1, 0.0], [0.2, 0.55, 0.6],
              target_charge=1, parent_charge_indices=[0]),
        _pair("invariant", [0.8, 0.2, 0.0], [0.7, 0.3, 0.0],
              parent_sentence=10.0, counterfactual_sentence=12.0),
        _pair("sentence_rank", [0.5, 0.5, 0.0], [0.5, 0.5, 0.0], rank_direction=-1,
              parent_sentence=12.0, counterfactual_sentence=12.0),
        _pair("sentence_rank", [0.5, 0.5, 0.0], [0.5, 0.5, 0.0], rank_direction=-1,
              parent_sentence=12.0, counterfactual_sentence=11.5),
        _pair("sentence_rank", [0.5, 0.5, 0.0], [0.5, 0.5, 0.0], rank_direction=1,
              parent_sentence=12.0, counterfactual_sentence=14.0),
    ]
    metrics = prepare_counterfactual(rows).metrics(None)
    assert metrics["flip_accuracy"] == 0.5
    assert metrics["flip_top1_accuracy"] == 0.5
    assert math.isclose(metrics["flip_margin"], ((0.7 - 0.6) + (0.55 - 0.2)) / 2)
    assert math.isclose(metrics["invariant_charge_drift"], 0.1)
    assert math.isclose(metrics["invariant_sentence_drift"], 0.2)
    assert math.isclose(metrics["sentence_pairwise_accuracy"], (0.5 + 1.0 + 1.0) / 3)
    assert math.isclose(metrics["sentence_rank_tie_rate"], 1 / 3)
    assert math.isclose(metrics["sentence_pairwise_accuracy_margin"], 1 / 3)
    assert math.isclose(metrics["monotonicity_violation_rate"], 1 - (2.5 / 3))
    assert "non_target_prediction_drift" not in metrics


def test_flip_accuracy_requires_parent_charges() -> None:
    row = _pair("charge_flip", [0.9, 0.1], [0.2, 0.8], target_charge=1)
    with pytest.raises(ValueError, match="parent_charge_indices"):
        prepare_counterfactual([row])


def test_cmdl_reports_charge_article_and_full_judgment_levels() -> None:
    def row(group, correct, article, months):
        return {
            "group_id": group, "correct": correct, "article_correct": article,
            "penalty_type": "fixed_term", "true_months": 12, "predicted_penalty_type": "fixed_term",
            "predicted_months": months,
        }

    metrics = prepare_cmdl([
        row("g1", True, True, 12.0), row("g1", True, False, 12.0),
        row("g2", True, True, 40.0),
    ]).metrics(None)
    assert metrics["defendant_accuracy"] == 1.0
    assert metrics["case_exact_accuracy"] == 1.0
    assert metrics["defendant_charge_article_accuracy"] == 2 / 3
    assert metrics["case_charge_article_accuracy"] == 0.5
    assert metrics["defendant_full_accuracy"] == 1 / 3
    assert metrics["case_full_accuracy"] == 0.0
    assert metrics["by_defendant_count"]["2"]["case_charge_article_accuracy"] == 0.0


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def test_cli_backfills_parent_charges_for_legacy_counterfactual_rows(tmp_path) -> None:
    rows = [
        {**_pair("charge_flip", [0.9, 0.1], [0.2, 0.8], target_charge=1), "case_id": "c1"},
        {**_pair("invariant", [0.8, 0.2], [0.8, 0.2], group_id="g2"), "case_id": "c2"},
    ]
    predictions = tmp_path / "counterfactual.jsonl"
    _write_jsonl(predictions, rows)
    parents = tmp_path / "train-labels.jsonl.gz"
    with gzip.open(parents, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"case_id": "c1", "charges": ["盗窃"]}, ensure_ascii=False) + "\n")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"charge_vocabulary": ["盗窃", "诈骗"]}, ensure_ascii=False))
    command = [
        sys.executable, str(ROOT / "scripts/evaluate_model.py"), "--kind", "counterfactual",
        "--input", str(predictions), "--output", str(tmp_path / "result.json"),
        "--bootstrap-iterations", "5",
    ]
    env = {"PYTHONPATH": str(ROOT / "src"), "PROGRESS": "never"}
    missing = subprocess.run(command, capture_output=True, text=True, env=env)
    assert missing.returncode != 0
    assert "--parent-records" in missing.stderr
    result = subprocess.run(
        [*command, "--parent-records", str(parents), "--metadata", str(metadata)],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads((tmp_path / "result.json").read_text())
    assert payload["metric_version"] == METRIC_VERSION
    assert payload["flip_accuracy"] == 1.0
    assert set(payload["confidence_intervals"]) == {
        "flip_accuracy", "invariant_violation_rate", "sentence_pairwise_accuracy",
        "invariant_charge_drift",
    }


def test_counterfactual_primary_endpoints_exclude_redundant_monotonicity() -> None:
    module = runpy.run_path(str(ROOT / "scripts/evaluate_model.py"))
    endpoints = module["DEFAULT_PRIMARY_ENDPOINTS"]["counterfactual"]
    assert endpoints == (
        "flip_accuracy", "invariant_violation_rate", "sentence_pairwise_accuracy",
        "invariant_charge_drift",
    )


def _keyed_static_rows(count: int = 6) -> list[dict]:
    rows = _static_rows(3, count)
    for index, row in enumerate(rows):
        row["case_id"] = f"c{index}"
        row["group_id"] = f"c{index}"
        row["charge_vocabulary"] = ["a", "b", "c"]
        row["article_vocabulary"] = ["a", "b", "c"]
        row["unseen_true_charge_labels"] = []
        row["unseen_true_article_labels"] = []
        row["articles_available"] = True
    return rows


def test_paired_test_keeps_repeated_samples_including_conflicting_truth() -> None:
    """CAIL derives case_id from the fact text, so a repeated judgment shares one id.

    Both systems predict every repeated row. A verbatim repeat and a repeat whose source
    label disagrees (same fact, 3 versus 9 months) are both kept: the paired test pairs the
    two sides by sample key plus truth, and its point difference is the difference of the
    full-row metrics that the headline results report.
    """
    from legal_landscape.evaluation.bootstrap import paired_sample_summary

    reference = _keyed_static_rows()
    reference.append(dict(reference[0]))
    reference.append({**reference[1], "true_months": float(reference[1]["true_months"]) + 6.0})
    candidate = [
        {**row, "predicted_months": float(row["predicted_months"]) + 0.5} for row in reference
    ]
    # Row order within a file is not part of the pairing.
    candidate = [*candidate[-1:], *candidate[:-1]]

    summary = paired_sample_summary(candidate, reference)
    assert summary == {
        "rows": 8,
        "unique_samples": 6,
        "repeated_samples": 2,
        "repeated_rows": 4,
        "conflicting_truth_samples": 1,
    }

    prepared_candidate, prepared_reference = prepare_static(candidate), prepare_static(reference)
    result = fast_paired_test(
        prepared_candidate, prepared_reference, ("sentence_mae",), iterations=20, seed=5
    )
    expected = (
        prepared_candidate.metrics(None)["sentence_mae"]
        - prepared_reference.metrics(None)["sentence_mae"]
    )
    assert math.isclose(result["sentence_mae"]["difference"], expected, abs_tol=1e-12)
    assert result["sentence_mae"]["groups"] == 6


def test_paired_test_refuses_repeats_that_differ_between_sides() -> None:
    from legal_landscape.evaluation.bootstrap import paired_sample_summary

    base = _keyed_static_rows()
    conflicting = [*base, {**base[1], "true_months": float(base[1]["true_months"]) + 6.0}]
    verbatim = [*base, dict(base[1])]
    with pytest.raises(ValueError, match="differs between the two sides"):
        paired_sample_summary(conflicting, verbatim)
    with pytest.raises(ValueError, match="differs between the two sides"):
        paired_sample_summary(verbatim, base)
    with pytest.raises(ValueError, match="sample sets differ"):
        paired_sample_summary(base, base[:-1])
