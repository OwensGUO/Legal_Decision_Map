"""Vectorized evaluation metrics over precomputed row arrays.

Each ``prepare_*`` function converts prediction rows into arrays once. ``metrics`` then takes
optional per-group counts (a cluster-bootstrap replicate) and returns the full metric set as
weighted reductions, which is equivalent to recomputing on the resampled rows.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from legal_landscape.evaluation.static_metrics import SENTENCE_CLASS_VOCABULARY, sentence_class

METRIC_VERSION = "v2"
_SENTENCE_INDEX = {label: index for index, label in enumerate(SENTENCE_CLASS_VOCABULARY)}


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0
    )


def _mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    return float((values * weights).sum() / total) if total > 0 else float("nan")


def _macro(values: np.ndarray, columns: np.ndarray) -> float:
    return float(values[columns].mean()) if columns.any() else 0.0


class _Grouped:
    def __init__(self, rows: list[dict[str, Any]], group_key: str = "group_id") -> None:
        if not rows:
            raise ValueError("evaluation input contains no rows")
        positions: dict[str, int] = {}
        self.row_group = np.asarray(
            [positions.setdefault(str(row[group_key]), len(positions)) for row in rows]
        )
        self.group_keys = tuple(positions)

    def row_weights(self, group_counts: np.ndarray | None) -> np.ndarray:
        if group_counts is None:
            return np.ones(len(self.row_group))
        return np.asarray(group_counts, dtype=float)[self.row_group]


class _MultiLabel:
    """Multi-label metrics; macro averages use the classes present in the full truth."""

    def __init__(self, rows: list[dict[str, Any]], prefix: str) -> None:
        truth = np.asarray([row[f"true_{prefix}"] for row in rows], dtype=bool)
        prediction = np.asarray([row[f"predicted_{prefix}"] for row in rows], dtype=bool)
        unseen_key = f"unseen_true_{prefix[:-1]}_labels"
        unseen_index: dict[str, int] = {}
        for row in rows:
            for label in row.get(unseen_key) or ():
                unseen_index.setdefault(str(label), len(unseen_index))
        if unseen_index:
            # Each distinct unseen label is its own class. Predictions stay zero.
            extra = np.zeros((len(rows), len(unseen_index)), dtype=bool)
            for row_index, row in enumerate(rows):
                for label in row.get(unseen_key) or ():
                    extra[row_index, unseen_index[str(label)]] = True
            truth = np.hstack([truth, extra])
            prediction = np.hstack([prediction, np.zeros_like(extra)])
        self.tp = (truth & prediction).astype(float)
        self.fp = (~truth & prediction).astype(float)
        self.fn = (truth & ~prediction).astype(float)
        self.exact = (truth == prediction).all(axis=1).astype(float)
        self.hamming = (truth == prediction).mean(axis=1)
        self.supported = truth.any(axis=0)
        self.all_columns = np.ones(truth.shape[1], dtype=bool)

    def metrics(self, weights: np.ndarray) -> dict[str, float]:
        tp, fp, fn = weights @ self.tp, weights @ self.fp, weights @ self.fn
        precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
        f1 = _ratio(2 * tp, 2 * tp + fp + fn)
        total_tp, total_fp, total_fn = tp.sum(), fp.sum(), fn.sum()
        result = {
            "accuracy": _mean(self.exact, weights),
            "hamming_accuracy": _mean(self.hamming, weights),
            "micro_precision": float(_ratio(np.array(total_tp), np.array(total_tp + total_fp))),
            "micro_recall": float(_ratio(np.array(total_tp), np.array(total_tp + total_fn))),
            "micro_f1": float(
                _ratio(np.array(2 * total_tp), np.array(2 * total_tp + total_fp + total_fn))
            ),
        }
        for suffix, columns in (("", self.supported), ("_all_labels", self.all_columns)):
            result[f"macro_precision{suffix}"] = _macro(precision, columns)
            result[f"macro_recall{suffix}"] = _macro(recall, columns)
            result[f"macro_f1{suffix}"] = _macro(f1, columns)
        return result


class StaticPrepared(_Grouped):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        super().__init__(rows)
        self.charge = _MultiLabel(rows, "charges")
        self.article = (
            _MultiLabel(rows, "articles")
            if all("true_articles" in row and "predicted_articles" in row for row in rows)
            else None
        )
        true_class = np.asarray([
            _SENTENCE_INDEX[sentence_class(row["penalty_type"], row.get("true_months"))]
            for row in rows
        ])
        predicted_class = np.asarray([
            _SENTENCE_INDEX[sentence_class(
                row.get("predicted_penalty_type", row["penalty_type"]), row.get("predicted_months")
            )]
            for row in rows
        ])
        width = len(SENTENCE_CLASS_VOCABULARY)
        self.class_truth = np.eye(width)[true_class]
        self.class_prediction = np.eye(width)[predicted_class]
        self.class_equal = (true_class == predicted_class).astype(float)
        self.class_supported = self.class_truth.any(axis=0)
        predicted = np.asarray([row["predicted_months"] for row in rows], dtype=float)
        truth = np.asarray(
            [math.nan if row.get("true_months") is None else row["true_months"] for row in rows],
            dtype=float,
        )
        charge_counts = np.asarray([
            sum(row["true_charges"]) + len(row.get("unseen_true_charge_labels") or ())
            for row in rows
        ])
        penalties = np.asarray([row["penalty_type"] for row in rows])
        # Training regresses months only for single-charge fixed-term cases.
        self.eligible = (
            (penalties == "fixed_term") & np.isfinite(predicted) & np.isfinite(truth)
            & (charge_counts == 1)
        ).astype(float)
        safe_predicted = np.where(np.isfinite(predicted), predicted, 0.0)
        safe_truth = np.where(np.isfinite(truth), truth, 0.0)
        self.errors = np.abs(safe_predicted - safe_truth)
        self.log_errors = np.abs(
            np.log1p(np.maximum(safe_predicted, 0)) - np.log1p(np.maximum(safe_truth, 0))
        )

    def metrics(self, group_counts: np.ndarray | None) -> dict[str, Any]:
        weights = self.row_weights(group_counts)
        charge = self.charge.metrics(weights)
        tp = weights @ (self.class_truth * self.class_equal[:, None])
        predicted, actual = weights @ self.class_prediction, weights @ self.class_truth
        accuracy = _mean(self.class_equal, weights)
        sentence_class_metrics = {
            "accuracy": accuracy,
            "macro_precision": _macro(_ratio(tp, predicted), self.class_supported),
            "macro_recall": _macro(_ratio(tp, actual), self.class_supported),
            "macro_f1": _macro(_ratio(2 * tp, predicted + actual), self.class_supported),
            "micro_precision": accuracy,
            "micro_recall": accuracy,
            "micro_f1": accuracy,
        }
        eligible = weights * self.eligible
        count = float(eligible.sum())
        regression = {
            "eligible_count": round(count),
            "mae": _mean(self.errors, eligible),
            "log_mae": _mean(self.log_errors, eligible),
            "tolerance_accuracy": _mean((self.errors <= 3.0).astype(float), eligible),
        }
        result: dict[str, Any] = {
            "metric_version": METRIC_VERSION,
            **{f"charge_{name}": value for name, value in charge.items()},
            "charge_exact_match": charge["accuracy"],
            **{f"sentence_{name}": value for name, value in sentence_class_metrics.items()},
            **{f"sentence_{name}": value for name, value in regression.items()},
            # Legacy aliases retained for existing result consumers.
            "macro_f1": charge["macro_f1"],
            "micro_f1": charge["micro_f1"],
            "exact_match": charge["accuracy"],
            **regression,
        }
        if self.article is not None:
            article = self.article.metrics(weights)
            result.update({f"article_{name}": value for name, value in article.items()})
        return result


class CounterfactualPrepared(_Grouped):
    def __init__(self, rows: list[dict[str, Any]], *, invariant_drift_threshold: float) -> None:
        super().__init__(rows)
        count = len(rows)
        kinds = np.asarray([row.get("intervention_type") for row in rows])
        self.flip, self.invariant, self.rank = (
            (kinds == "charge_flip").astype(float),
            (kinds == "invariant").astype(float),
            (kinds == "sentence_rank").astype(float),
        )
        values = {name: np.zeros(count) for name in (
            "flip_correct", "flip_top1", "flip_margin", "invariant_violation",
            "invariant_charge_drift", "invariant_sentence_drift", "rank_score", "rank_tie",
            "rank_margin",
        )}
        for index, row in enumerate(rows):
            kind = row.get("intervention_type")
            parent_sentence = float(row["parent_sentence"])
            counterfactual_sentence = float(row["counterfactual_sentence"])
            if kind == "charge_flip":
                probabilities = np.asarray(row["counterfactual_charge_probabilities"], dtype=float)
                target = int(row["target_charge"])
                sources = row.get("parent_charge_indices")
                if not sources:
                    raise ValueError(
                        "charge_flip rows need parent_charge_indices; regenerate predictions or "
                        "pass --parent-records and --metadata to evaluate_model.py"
                    )
                threshold = float(row.get("charge_threshold", 0.5))
                source = float(probabilities[np.asarray(sources, dtype=int)].max())
                values["flip_correct"][index] = (
                    probabilities[target] >= threshold and source < threshold
                )
                values["flip_top1"][index] = int(probabilities.argmax()) == target
                values["flip_margin"][index] = probabilities[target] - source
            elif kind == "invariant":
                parent = np.asarray(row["parent_charge_probabilities"], dtype=float)
                counterfactual = np.asarray(row["counterfactual_charge_probabilities"], dtype=float)
                drift = float(np.max(np.abs(parent - counterfactual)))
                values["invariant_charge_drift"][index] = drift
                values["invariant_violation"][index] = (
                    parent.argmax() != counterfactual.argmax() or drift > invariant_drift_threshold
                )
                values["invariant_sentence_drift"][index] = abs(
                    counterfactual_sentence - parent_sentence
                ) / max(parent_sentence, 1.0)
            elif kind == "sentence_rank":
                change = int(row["rank_direction"]) * (counterfactual_sentence - parent_sentence)
                # Ties earn half credit, as in concordance statistics.
                values["rank_score"][index] = 1.0 if change > 0 else 0.5 if change == 0 else 0.0
                values["rank_tie"][index] = change == 0
                values["rank_margin"][index] = change > 1.0
        self.values = values

    def metrics(self, group_counts: np.ndarray | None) -> dict[str, Any]:
        weights = self.row_weights(group_counts)
        flip, invariant, rank = weights * self.flip, weights * self.invariant, weights * self.rank
        pairwise = _mean(self.values["rank_score"], rank)
        return {
            "metric_version": METRIC_VERSION,
            "flip_count": round(flip.sum()),
            "flip_accuracy": _mean(self.values["flip_correct"], flip),
            "flip_top1_accuracy": _mean(self.values["flip_top1"], flip),
            "flip_margin": _mean(self.values["flip_margin"], flip),
            "invariant_count": round(invariant.sum()),
            "invariant_violation_rate": _mean(self.values["invariant_violation"], invariant),
            "invariant_charge_drift": _mean(self.values["invariant_charge_drift"], invariant),
            "invariant_sentence_drift": _mean(self.values["invariant_sentence_drift"], invariant),
            "sentence_rank_count": round(rank.sum()),
            "sentence_pairwise_accuracy": pairwise,
            "sentence_rank_tie_rate": _mean(self.values["rank_tie"], rank),
            "sentence_pairwise_accuracy_margin": _mean(self.values["rank_margin"], rank),
            "monotonicity_violation_rate": 1.0 - pairwise if rank.sum() > 0 else float("nan"),
        }


class CmdlPrepared(_Grouped):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        super().__init__(rows)
        charge = np.asarray([bool(row["correct"]) for row in rows])
        levels = {"": charge}
        if all("article_correct" in row for row in rows):
            charge_article = charge & np.asarray([bool(row["article_correct"]) for row in rows])
            levels["_charge_article"] = charge_article
            if all("predicted_months" in row and "penalty_type" in row for row in rows):
                sentence = np.asarray([
                    sentence_class(row["penalty_type"], row.get("true_months"))
                    == sentence_class(
                        row.get("predicted_penalty_type", row["penalty_type"]),
                        row.get("predicted_months"),
                    )
                    for row in rows
                ])
                levels["_full"] = charge_article & sentence
        groups = len(self.group_keys)
        self.defendants = np.bincount(self.row_group, minlength=groups)
        self.levels = {
            suffix: (
                correct.astype(float),
                (
                    np.bincount(self.row_group, weights=(~correct).astype(float), minlength=groups)
                    == 0
                ).astype(float),
            )
            for suffix, correct in levels.items()
        }

    def metrics(self, group_counts: np.ndarray | None) -> dict[str, Any]:
        weights = self.row_weights(group_counts)
        case_weights = (
            np.ones(len(self.group_keys)) if group_counts is None
            else np.asarray(group_counts, dtype=float)
        )
        names = {"": ("defendant_accuracy", "case_exact_accuracy")}
        names["_charge_article"] = (
            "defendant_charge_article_accuracy", "case_charge_article_accuracy"
        )
        names["_full"] = ("defendant_full_accuracy", "case_full_accuracy")
        result: dict[str, Any] = {
            "metric_version": METRIC_VERSION,
            "defendant_count": round(weights.sum()),
            "case_count": round(case_weights.sum()),
        }
        for suffix, (defendant, case) in self.levels.items():
            defendant_name, case_name = names[suffix]
            result[defendant_name] = _mean(defendant, weights)
            result[case_name] = _mean(case, case_weights)
        if group_counts is None:
            strata: dict[str, dict[str, float | int]] = {}
            for size in sorted(set(self.defendants.tolist())):
                members = (self.defendants == size).astype(float)
                stratum: dict[str, float | int] = {"case_count": int(members.sum())}
                for suffix, (_defendant, case) in self.levels.items():
                    stratum[names[suffix][1]] = _mean(case, members)
                strata[str(size)] = stratum
            result["by_defendant_count"] = strata
        return result


def prepare_static(rows: list[dict[str, Any]]) -> StaticPrepared:
    return StaticPrepared(rows)


def prepare_counterfactual(
    rows: list[dict[str, Any]], *, invariant_drift_threshold: float = 0.1
) -> CounterfactualPrepared:
    return CounterfactualPrepared(rows, invariant_drift_threshold=invariant_drift_threshold)


def prepare_cmdl(rows: list[dict[str, Any]]) -> CmdlPrepared:
    return CmdlPrepared(rows)
