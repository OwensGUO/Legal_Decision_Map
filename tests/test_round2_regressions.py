"""Second-round regressions. Each test enters the production call chain."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from legal_landscape.counterfactual.validators import validate_generation
from legal_landscape.data.sanitize import SANITIZER_VERSION
from legal_landscape.factors.extract import extract_factors
from legal_landscape.factors.schema import InterventionSpec, LegalFactors
from tests.support import cache_identity

ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def _amount_spec(
    source: float,
    target: float,
    *,
    defendant: str = "某甲",
    co_defendants: tuple[str, ...] = (),
) -> InterventionSpec:
    return InterventionSpec(
        "c1",
        defendant,
        "invariant",
        LegalFactors(amount=source, conduct="秘密窃取"),
        LegalFactors(amount=target, conduct="秘密窃取"),
        None,
        None,
        ("amount",),
        "amount",
        co_defendants,
    )


def _payload(text: str, spec: InterventionSpec) -> str:
    return json.dumps({"counterfactual_text": text, "target_defendant": spec.target_defendant})


@pytest.mark.parametrize(
    ("text", "surrender", "confession"),
    [
        ("张三不但主动投案，还如实供述。", True, True),
        ("张三无前科且主动投案。", True, None),
        ("张三未成年时主动投案。", True, None),
    ],
)
def test_non_target_negation_does_not_mark_surrender_false(text, surrender, confession) -> None:
    factors = extract_factors(text, target_defendant="张三")
    assert factors.surrender is surrender
    assert factors.confession is confession


def test_direct_negation_still_marks_the_factor_false() -> None:
    factors = extract_factors(
        "张三未主动投案，未如实供述，未退赔。", target_defendant="张三"
    )
    assert factors.surrender is False
    assert factors.confession is False
    assert factors.restitution is False


def test_unreliable_negation_stays_unknown() -> None:
    factors = extract_factors("张三不很主动投案。", target_defendant="张三")
    assert factors.surrender is None


def test_validator_accepts_affirmative_surrender_after_non_target_negation() -> None:
    source = LegalFactors(surrender=False)
    target = LegalFactors(surrender=True)
    spec = InterventionSpec(
        "c1", "张三", "invariant", source, target, None, None, ("surrender",), "surrender"
    )
    parent = "张三未主动投案。"
    accepted = validate_generation(
        parent, spec, _payload("张三不但主动投案。", spec)
    )
    assert accepted.valid, accepted.errors
    refused = validate_generation(
        parent, spec, _payload("张三未主动投案，后被抓获。", spec)
    )
    assert "target_change_missing" in refused.errors


def test_amount_tolerance_rejects_an_unedited_small_gap_and_accepts_the_edit() -> None:
    spec = _amount_spec(50, 50.5)
    parent = "某甲秘密窃取他人财物，价值50元。"
    unchanged = validate_generation(
        parent, spec, _payload("某甲秘密窃取他人财物，价值50元，后被抓获。", spec)
    )
    assert unchanged.valid is False
    assert "target_change_missing" in unchanged.errors
    edited = validate_generation(
        parent, spec, _payload("某甲秘密窃取他人财物，价值50.5元。", spec)
    )
    assert edited.valid, edited.errors


def test_amount_check_covers_smaller_values_wan_multiple_amounts_and_defendants() -> None:
    tiny = _amount_spec(1, 1.01)
    tiny_parent = "某甲秘密窃取他人财物，价值1元。"
    assert "target_change_missing" in validate_generation(
        tiny_parent, tiny, _payload("某甲秘密窃取他人财物，价值1元，随即离开。", tiny)
    ).errors
    assert validate_generation(
        tiny_parent, tiny, _payload("某甲秘密窃取他人财物，价值1.01元。", tiny)
    ).valid

    wan = _amount_spec(10000, 10100)
    wan_parent = "某甲秘密窃取他人财物，价值1万元。"
    assert validate_generation(
        wan_parent, wan, _payload("某甲秘密窃取他人财物，价值1.01万元。", wan)
    ).valid
    leftover = validate_generation(
        wan_parent,
        wan,
        _payload("某甲秘密窃取他人财物，价值1.01万元，原记载1万元。", wan),
    )
    assert "target_change_missing" in leftover.errors

    spec = _amount_spec(50, 50.5)
    parent = "某甲秘密窃取他人财物，价值50元。"
    both = validate_generation(
        parent, spec, _payload("某甲秘密窃取他人财物，价值50.5元，另有50元。", spec)
    )
    assert "target_change_missing" in both.errors

    shared = _amount_spec(50, 50.5, co_defendants=("某乙",))
    shared_parent = "某甲秘密窃取他人财物，价值50元。某乙涉案50元。"
    wrong_person = validate_generation(
        shared_parent,
        shared,
        _payload("某甲秘密窃取他人财物，价值50元。某乙涉案50.5元。", shared),
    )
    assert wrong_person.valid is False
    right_person = validate_generation(
        shared_parent,
        shared,
        _payload("某甲秘密窃取他人财物，价值50.5元。某乙涉案50元。", shared),
    )
    assert right_person.valid, right_person.errors


def _parent(text: str = "某甲涉案50元。") -> dict:
    return {
        "case_id": "c1",
        "dataset": "cail",
        "target_defendant": "某甲",
        "fact_strict": text,
        "factors": {"amount": 50.0, "restitution": False},
        "charges": ["盗窃"],
        "penalty_type": "fixed_term",
        "imprisonment_months": 12,
        "co_defendants": [],
    }


def _row(
    text: str,
    target: dict,
    changed: list[str],
    *,
    parent_text: str,
    valid: bool = True,
) -> dict:
    return {
        "parent_case_id": "c1",
        "intervention_type": "invariant" if changed == ["amount"] else "sentence_rank",
        "changed_fields": changed,
        "target_factors": target,
        "validation": {"valid": valid, "errors": [] if valid else ["target_change_missing"]},
        "counterfactual_strict": text,
        "rank_direction": None if changed == ["amount"] else -1,
        **cache_identity(parent_text),
    }


def test_loader_uses_the_same_amount_rule_and_does_not_keep_an_unedited_amount(tmp_path) -> None:
    from legal_landscape.training.train import load_counterfactual_pairs

    parent_text = "某甲涉案50元。"
    path = tmp_path / "cf.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in (
                _row(
                    "某甲涉案50元，后被抓获。",
                    {"amount": 50.5},
                    ["amount"],
                    parent_text=parent_text,
                ),
                _row("某甲涉案50.5元。", {"amount": 50.5}, ["amount"], parent_text=parent_text),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    pairs = load_counterfactual_pairs(path, {"c1": _parent(parent_text)}, filter_valid=True)
    assert len(pairs) == 1
    assert pairs[0]["counterfactual_factors"]["amount"] == 50.5
    assert pairs[0]["counterfactual_factors"] is not pairs[0]["expected_factors"]


def test_boolean_factor_swap_does_not_match_on_either_filter_path(tmp_path) -> None:
    from legal_landscape.training.train import load_counterfactual_pairs

    parent_text = "某甲未退赃。"
    path = tmp_path / "cf.jsonl"
    rows = [
        _row("某甲未退赃。", {"restitution": True}, ["restitution"], parent_text=parent_text),
        _row(
            "某甲未退赃。",
            {"restitution": True},
            ["restitution"],
            parent_text=parent_text,
            valid=False,
        ),
        _row("某甲不很想退赃。", {"restitution": False}, ["restitution"], parent_text=parent_text),
        _row("某甲已退赃。", {"restitution": True}, ["restitution"], parent_text=parent_text),
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    parents = {"c1": _parent(parent_text)}
    filtered = load_counterfactual_pairs(path, parents, filter_valid=True)
    unfiltered = load_counterfactual_pairs(path, parents, filter_valid=False)
    assert [pair["counterfactual_text"] for pair in filtered] == ["某甲已退赃。"]
    assert [pair["counterfactual_text"] for pair in unfiltered] == ["某甲已退赃。"]
    assert filtered[0]["counterfactual_factors"]["restitution"] is True


def _write_cache(path: Path, row: dict) -> None:
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def test_loader_rejects_old_incomplete_and_stale_parent_caches(tmp_path) -> None:
    from legal_landscape.counterfactual.prompts import PROMPT_VERSION
    from legal_landscape.counterfactual.validators import VALIDATOR_VERSION
    from legal_landscape.training.train import load_counterfactual_pairs

    parent_text = "某甲涉案50.5元。"
    parents = {"c1": _parent(parent_text)}
    current = _row("某甲涉案50.5元。", {"amount": 50.5}, ["amount"], parent_text=parent_text)
    path = tmp_path / "cf.jsonl"
    before = None
    for damage, reason in (
        (
            {"prompt_version": "v3", "sanitizer_version": "v2", "validator_version": "v1"},
            "prompt_version",
        ),
        ({"validator_version": "missing"}, "validator_version"),
        ({"parent_text_sha256": "0" * 64}, "parent_text_sha256"),
    ):
        row = {**current, **damage}
        if damage.get("validator_version") == "missing":
            row.pop("validator_version")
        _write_cache(path, row)
        before = path.read_bytes()
        with pytest.raises(ValueError, match="Regenerate"):
            load_counterfactual_pairs(path, parents, filter_valid=True)
        assert path.read_bytes() == before
        assert reason
    assert current["prompt_version"] == PROMPT_VERSION
    assert current["sanitizer_version"] == SANITIZER_VERSION
    assert current["validator_version"] == VALIDATOR_VERSION
    _write_cache(path, current)
    loaded = load_counterfactual_pairs(path, parents, filter_valid=True)
    assert len(loaded) == 1


def test_training_cli_rejects_an_old_counterfactual_cache(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    module = runpy.run_path(str(ROOT / "scripts" / "train_model.py"))
    train = tmp_path / "train.jsonl"
    record = {
        "case_id": "c1",
        "group_id": "g1",
        "dataset": "cail",
        "target_defendant": "某甲",
        "fact_strict": "某甲涉案50元。",
        "charges": ["盗窃"],
        "conviction_articles": ["criminal_law:264"],
        "penalty_type": "fixed_term",
        "imprisonment_months": 12,
        "factors": {},
    }
    train.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "charge_vocabulary": ["盗窃"],
                "article_vocabulary": ["criminal_law:264"],
                "penalty_vocabulary": ["fixed_term"],
            }
        ),
        encoding="utf-8",
    )
    cache = tmp_path / "old.jsonl"
    row = _row("某甲涉案50.5元。", {"amount": 50.5}, ["amount"], parent_text=record["fact_strict"])
    row["prompt_version"] = "v3"
    row["sanitizer_version"] = "v2"
    cache.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    before = cache.read_bytes()
    calls = []
    monkeypatch.setattr(
        "legal_landscape.training.train.load_backbone",
        lambda config: calls.append(config) or (_ for _ in ()).throw(AssertionError("backbone")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_model.py",
            "--execute",
            "--experiment",
            "B3",
            "--train-data",
            str(train),
            "--counterfactual-data",
            str(cache),
            "--output-dir",
            str(tmp_path / "out"),
            "--limit",
            "1",
        ],
    )
    with pytest.raises(ValueError, match="Regenerate"):
        module["main"]()
    assert cache.read_bytes() == before
    assert calls == []


def _resume_config(**training) -> dict:
    return {
        "model": {"path": "unused", "max_length": 32},
        "training": {"seed": 7, "epochs": 1, "max_steps": 4, **training},
        "loss": {"article": 1.0},
    }


def test_resume_rejects_changes_that_move_batches_or_loss(tmp_path, monkeypatch) -> None:
    import builtins

    from legal_landscape.training.train import (
        _file_identity,
        resume_compatibility,
        run_real_training,
        validate_resume_metadata,
    )

    base = _resume_config()
    saved = resume_compatibility(
        config=base,
        experiment_name="B3",
        limit=4,
        num_processes=1,
        charge_vocabulary=["盗窃"],
        article_vocabulary=["criminal_law:264"],
        penalty_vocabulary=["fixed_term"],
        validation_identity=None,
    )
    progress = {"seed": 7, "input_identity": {"train": {"sha256": "a"}}, "compatibility": saved}

    def expect_reject(**training_changes):
        current = resume_compatibility(
            config=_resume_config(**training_changes) if training_changes else base,
            experiment_name="B3",
            limit=4,
            num_processes=1,
            charge_vocabulary=["盗窃"],
            article_vocabulary=["criminal_law:264"],
            penalty_vocabulary=["fixed_term"],
            validation_identity=None,
        )
        with pytest.raises(ValueError, match="incompatible"):
            validate_resume_metadata(
                progress,
                expected_seed=7,
                expected_input_identity=progress["input_identity"],
                expected_compatibility=current,
            )

    expect_reject(pair_every=3)
    expect_reject(rank_margin=2.0)
    longer = resume_compatibility(
        config={**base, "model": {**base["model"], "max_length": 64}},
        experiment_name="B3",
        limit=4,
        num_processes=1,
        charge_vocabulary=["盗窃"],
        article_vocabulary=["criminal_law:264"],
        penalty_vocabulary=["fixed_term"],
        validation_identity=None,
    )
    with pytest.raises(ValueError, match="incompatible"):
        validate_resume_metadata(
            progress, expected_seed=7, expected_input_identity=progress["input_identity"],
            expected_compatibility=longer,
        )
    limited = resume_compatibility(
        config=base, experiment_name="B3", limit=2, num_processes=1,
        charge_vocabulary=["盗窃"], article_vocabulary=["criminal_law:264"],
        penalty_vocabulary=["fixed_term"], validation_identity=None,
    )
    with pytest.raises(ValueError, match="incompatible"):
        validate_resume_metadata(
            progress, expected_seed=7, expected_input_identity=progress["input_identity"],
            expected_compatibility=limited,
        )
    loss_changed = resume_compatibility(
        config={**base, "loss": {"article": 2.0}},
        experiment_name="B3", limit=4, num_processes=1,
        charge_vocabulary=["盗窃"], article_vocabulary=["criminal_law:264"],
        penalty_vocabulary=["fixed_term"], validation_identity=None,
    )
    with pytest.raises(ValueError, match="incompatible"):
        validate_resume_metadata(
            progress, expected_seed=7, expected_input_identity=progress["input_identity"],
            expected_compatibility=loss_changed,
        )
    other_experiment = resume_compatibility(
        config=base, experiment_name="B4", limit=4, num_processes=1,
        charge_vocabulary=["盗窃"], article_vocabulary=["criminal_law:264"],
        penalty_vocabulary=["fixed_term"], validation_identity=None,
    )
    with pytest.raises(ValueError, match="incompatible"):
        validate_resume_metadata(
            progress, expected_seed=7, expected_input_identity=progress["input_identity"],
            expected_compatibility=other_experiment,
        )
    extended = resume_compatibility(
        config=_resume_config(max_steps=8),
        experiment_name="B3", limit=4, num_processes=1,
        charge_vocabulary=["盗窃"], article_vocabulary=["criminal_law:264"],
        penalty_vocabulary=["fixed_term"], validation_identity=None,
    )
    validate_resume_metadata(
        progress, expected_seed=7, expected_input_identity=progress["input_identity"],
        expected_compatibility=extended,
    )
    validate_resume_metadata(
        progress, expected_seed=7, expected_input_identity=progress["input_identity"],
        expected_compatibility=saved,
    )

    train_path = tmp_path / "train.jsonl"
    train_path.write_text(json.dumps({
        "case_id": "c1", "group_id": "g1", "dataset": "cail", "target_defendant": "某甲",
        "fact_strict": "案件事实", "charges": ["盗窃"],
        "conviction_articles": ["criminal_law:264"], "penalty_type": "fixed_term",
        "imprisonment_months": 12, "factors": {},
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    (tmp_path / "metadata.json").write_text(json.dumps({
        "charge_vocabulary": ["盗窃"],
        "article_vocabulary": ["criminal_law:264"],
        "penalty_vocabulary": ["fixed_term"],
    }), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "training_progress.json").write_text(json.dumps({
        "seed": 42,
        "steps": 1,
        "input_identity": {"train": _file_identity(train_path)},
        "compatibility": resume_compatibility(
            config={"model": {"max_length": 32}, "training": {"seed": 42, "max_steps": 4}},
            experiment_name="B3", limit=None, num_processes=1,
            charge_vocabulary=["盗窃"], article_vocabulary=["criminal_law:264"],
            penalty_vocabulary=["fixed_term"], validation_identity=None,
        ),
    }), encoding="utf-8")
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {"torch", "accelerate", "transformers"}:
            raise AssertionError("loaded training dependencies before resume validation")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(ValueError, match="incompatible"):
        run_real_training(
            {"model": {"max_length": 64}, "training": {"seed": 42, "max_steps": 4}},
            train_data=train_path,
            output_dir=tmp_path / "output",
            experiment_name="B3",
            resume_from_checkpoint=checkpoint,
        )


def _prediction_record(**extra) -> dict:
    return {
        "case_id": "c1",
        "group_id": "g1",
        "dataset": "cail",
        "target_defendant": "甲",
        "charges": ["盗窃"],
        "conviction_articles": ["criminal_law:264"],
        "penalty_type": "fixed_term",
        "imprisonment_months": 12,
        **extra,
    }


def _predict(records):
    from legal_landscape.training.train import make_static_prediction_rows

    return make_static_prediction_rows(
        records,
        charge_probabilities=[[0.9]],
        article_probabilities=[[0.2]],
        penalty_indices=[0],
        sentence_months=[12.0],
        charge_vocabulary=["盗窃"],
        article_vocabulary=["criminal_law:264"],
        penalty_vocabulary=["fixed_term"],
    )


def test_missing_article_labels_survive_export_and_stop_evaluation(tmp_path) -> None:
    from legal_landscape.data.cail import iter_cail

    source = tmp_path / "data_train.json"
    source.write_text(json.dumps({
        "fact": "被告人甲盗走手机。",
        "meta": {
            "criminals": ["甲"],
            "accusation": ["盗窃"],
            "term_of_imprisonment": {"imprisonment": 12},
        },
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    case = next(iter_cail(source, split="train"))
    assert case.articles_available is False
    rows = _predict([case.to_dict()])
    assert rows[0]["articles_available"] is False
    assert rows[0]["true_articles"] is None
    destination = tmp_path / "predictions.jsonl"
    destination.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n", encoding="utf-8")
    module = runpy.run_path(str(ROOT / "scripts" / "evaluate_model.py"))
    args = type("Args", (), {
        "limit": 10, "kind": "static", "parent_records": None, "metadata": None,
    })()
    with pytest.raises(SystemExit, match="missing article"):
        module["_load"](destination, args)
    args.kind = "cmdl"
    with pytest.raises(SystemExit, match="missing article"):
        module["_load"](destination, args)


def test_unseen_articles_remain_scored_when_labels_are_available(tmp_path) -> None:
    record = _prediction_record(
        articles_available=True,
        conviction_articles=["criminal_law:225"],
    )
    [row] = _predict([record])
    assert row["articles_available"] is True
    assert row["unseen_true_article_labels"] == ["criminal_law:225"]
    destination = tmp_path / "predictions.jsonl"
    destination.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    module = runpy.run_path(str(ROOT / "scripts" / "evaluate_model.py"))
    prepared = module["_load"](destination, type("Args", (), {
        "limit": 10, "kind": "static", "parent_records": None, "metadata": None,
    })())
    assert prepared.source_rows[0]["unseen_true_article_labels"] == ["criminal_law:225"]
    assert prepared.metrics(None)["article_macro_f1"] < 1


def test_historical_prediction_without_article_marker_is_not_assumed_labeled(tmp_path) -> None:
    module = runpy.run_path(str(ROOT / "scripts" / "evaluate_model.py"))
    path = tmp_path / "old.jsonl"
    path.write_text(json.dumps({
        "case_id": "c1", "group_id": "g1", "charge_vocabulary": ["盗窃"],
        "true_charges": [0], "predicted_charges": [0],
        "true_articles": [0], "predicted_articles": [0],
        "penalty_type": "fixed_term", "predicted_penalty_type": "fixed_term",
        "true_months": 12, "predicted_months": 12,
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="lack articles_available"):
        module["_load"](path, type("Args", (), {
            "limit": 10, "kind": "static", "parent_records": None, "metadata": None,
        })())


def _static_pair(case_id: str, **extra) -> dict:
    row = {
        "case_id": case_id,
        "group_id": "g1",
        "charge_vocabulary": ["盗窃", "诈骗"],
        "article_vocabulary": ["criminal_law:264"],
        "articles_available": True,
        "true_charges": [1, 0],
        "predicted_charges": [1, 0],
        "true_articles": [1],
        "predicted_articles": [1],
        "unseen_true_charge_labels": [],
        "unseen_true_article_labels": [],
        "penalty_type": "fixed_term",
        "predicted_penalty_type": "fixed_term",
        "true_months": 12.0,
        "predicted_months": 12.0,
    }
    row.update(extra)
    return row


def test_fast_and_legacy_pairing_reject_identity_truth_and_definition_mismatches() -> None:
    from legal_landscape.evaluation.bootstrap import (
        assert_paired_rows,
        fast_paired_test,
        paired_cluster_test,
    )
    from legal_landscape.evaluation.prepared import prepare_counterfactual, prepare_static

    left = [_static_pair("c1"), _static_pair("c2", group_id="g2", true_charges=[0, 1])]
    right = [left[1], left[0]]
    assert_paired_rows(left, right)
    fast_paired_test(
        prepare_static(left), prepare_static(right), ("charge_accuracy",), iterations=5, seed=1
    )
    with pytest.raises(ValueError, match="lacks case_id"):
        assert_paired_rows([{"group_id": "g", "value": 1}], [{"group_id": "g", "value": 1}])
    with pytest.raises(ValueError, match="sample sets differ"):
        assert_paired_rows(left, left[:1])
    with pytest.raises(ValueError, match="repeated"):
        assert_paired_rows(left, [*left, left[0]])
    with pytest.raises(ValueError, match="different truth"):
        changed = [{**left[0], "unseen_true_charge_labels": ["X"]}, left[1]]
        other = [{**left[0], "unseen_true_charge_labels": ["Y"]}, left[1]]
        assert_paired_rows(changed, other)
    with pytest.raises(ValueError, match="different truth"):
        assert_paired_rows(
            left,
            [{**left[0], "charge_vocabulary": ["诈骗", "盗窃"]}, left[1]],
        )
    with pytest.raises(ValueError, match="source row identity"):
        fast_paired_test(
            type("Bare", (), {"group_keys": ("g1",), "metrics": lambda self, counts: {}})(),
            type("Bare", (), {"group_keys": ("g1",), "metrics": lambda self, counts: {}})(),
            ("charge_accuracy",),
            iterations=1,
        )

    def cf(generation_id, **extra):
        row = {
            "case_id": "c1",
            "group_id": "g1",
            "generation_id": generation_id,
            "charge_vocabulary": ["盗窃", "诈骗"],
            "intervention_type": "sentence_rank",
            "rank_direction": 1,
            "target_charge": None,
            "rule_id": "rank",
            "parent_charge_probabilities": [0.5, 0.5],
            "counterfactual_charge_probabilities": [0.5, 0.5],
            "parent_sentence": 12.0,
            "counterfactual_sentence": 14.0,
        }
        row.update(extra)
        return row

    same = [cf("gen-1"), cf("gen-2", group_id="g2")]
    paired_cluster_test(
        [same[1], same[0]], same, lambda sample: 1.0, iterations=5, seed=1
    )
    with pytest.raises(ValueError, match="generation_id"):
        assert_paired_rows(
            [{**same[0], "generation_id": None}],
            [{**same[0], "generation_id": None}],
        )
    with pytest.raises(ValueError, match="sample sets differ"):
        assert_paired_rows(same, [cf("gen-9"), same[1]])
    with pytest.raises(ValueError, match="different truth"):
        assert_paired_rows(same, [{**same[0], "rank_direction": -1}, same[1]])
    prepare_counterfactual(same)


def test_evaluate_cli_pairs_reordered_rows_and_rejects_different_cases(tmp_path) -> None:
    module = str(ROOT / "scripts" / "evaluate_model.py")
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    rows = [_static_pair("c1"), _static_pair("c2", group_id="g2")]
    left.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    right.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reversed(rows)))
    command = [
        sys.executable, module, "--kind", "static", "--input", str(left),
        "--reference-input", str(right), "--output", str(tmp_path / "out.json"),
        "--limit", "10", "--bootstrap-iterations", "5",
    ]
    ok = subprocess.run(command, env=_env(), capture_output=True, text=True, check=False)
    assert ok.returncode == 0, ok.stderr
    assert "comparison" in json.loads(ok.stdout)
    broken = [_static_pair("c9", group_id="g2"), rows[0]]
    right.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in broken))
    failed = subprocess.run(command, env=_env(), capture_output=True, text=True, check=False)
    assert failed.returncode != 0
    assert "sample sets differ" in failed.stderr


def _judgment(accusation: str, penalty: dict, article: list[str] | None = None) -> dict:
    return {
        "accusation": accusation,
        "standard_accusation": accusation,
        "article": article or ["264"],
        "penalty": penalty,
    }


def _complete_penalty(**overrides) -> dict:
    penalty = {
        "surveillance": 0,
        "detention": 0,
        "imprisonment": 0,
        "death_penalty": False,
        "life_imprisonment": False,
    }
    penalty.update(overrides)
    return penalty


def test_placeholder_final_penalties_do_not_become_exemption(tmp_path) -> None:
    from legal_landscape.data.cmdl import iter_cmdl

    path = tmp_path / "train_small.jsonl"
    outcomes = []
    for final in (
        {"imprisonment": None},
        {"imprisonment": ""},
        {"death_penalty": False},
    ):
        outcomes.append({
            "fact": "被告人甲盗走财物。",
            "defendants": ["甲"],
            "outcomes": [{
                "name": "甲",
                "judgment": [_judgment("盗窃罪", _complete_penalty(imprisonment=12))],
                "final_penalty": final,
            }],
        })
    outcomes.append({
        "fact": "被告人乙被免予刑事处罚。",
        "defendants": ["乙"],
        "outcomes": [{
            "name": "乙",
            "judgment": [_judgment("盗窃罪", _complete_penalty())],
            "final_penalty": _complete_penalty(),
        }],
    })
    outcomes.append({
        "fact": "被告人丙犯两罪。",
        "defendants": ["丙"],
        "outcomes": [{
            "name": "丙",
            "judgment": [
                _judgment("盗窃罪", _complete_penalty(imprisonment=12)),
                _judgment("诈骗罪", _complete_penalty(imprisonment=6), ["266"]),
            ],
            "final_penalty": {"imprisonment": None},
        }],
    })
    outcomes.append({
        "fact": "被告人丁被判处有期徒刑。",
        "defendants": ["丁"],
        "outcomes": [{
            "name": "丁",
            "judgment": [_judgment("盗窃罪", _complete_penalty(imprisonment=6))],
            "final_penalty": _complete_penalty(imprisonment=18),
        }],
    })
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outcomes))
    cases = list(iter_cmdl(path, split="train"))
    observed = [
        (case.penalty_type, case.imprisonment_months, case.penalty_fallback) for case in cases[:3]
    ]
    assert observed == [
        ("fixed_term", 12, "unusable_final_penalty_fallback"),
        ("fixed_term", 12, "unusable_final_penalty_fallback"),
        ("fixed_term", 12, "unusable_final_penalty_fallback"),
    ]
    assert (cases[3].penalty_type, cases[3].imprisonment_months, cases[3].mask_penalty) == (
        "exempt", None, False,
    )
    assert (cases[4].penalty_type, cases[4].mask_penalty, cases[4].penalty_exclusion) == (
        "unknown", True, "missing_final_penalty",
    )
    assert (cases[5].penalty_type, cases[5].imprisonment_months, cases[5].penalty_fallback) == (
        "fixed_term", 18, None,
    )


def _processed_parent(
    case_id: str,
    defendant: str,
    charges: list[str],
    fact: str,
    *,
    co_defendants: list[str] | None = None,
    amount: float | None = 1500,
    restitution: bool | None = False,
    penalty: str = "fixed_term",
    months: int | None = 12,
    conduct: str | None = "秘密窃取",
) -> dict:
    factors = {"restitution": restitution, "conduct": conduct}
    if amount is not None:
        factors["amount"] = amount
    return {
        "case_id": case_id,
        "target_defendant": defendant,
        "co_defendants": co_defendants or [],
        "fact_strict": fact,
        "charges": charges,
        "factors": factors,
        "penalty_type": penalty,
        "imprisonment_months": months,
    }


def test_generation_report_covers_filters_and_matches_after_resume(tmp_path) -> None:
    source = tmp_path / "processed" / "train.jsonl"
    source.parent.mkdir()
    rows = [
        _processed_parent("c1", "甲", ["盗窃"], "甲秘密窃取财物，价值1500元，未退赃。"),
        _processed_parent(
            "c2", "甲", ["诈骗"], "甲谎称借款。",
            co_defendants=["乙"], amount=4000, restitution=True, months=6,
            conduct="虚构事实使被害人自愿交付",
        ),
        _processed_parent("c3", "甲", ["盗窃"], "长" * 5000, amount=1500),
        _processed_parent(
            "c4", "甲", ["故意伤害"], "甲因琐事致人轻伤。",
            amount=None, penalty="life", months=None, restitution=False,
            conduct="因个人纠纷定向伤害",
        ),
    ]
    source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    (source.parent / "metadata.json").write_text(json.dumps({
        "sanitizer": {"version": SANITIZER_VERSION, "charge_lexicon": ["盗窃", "诈骗", "故意伤害"]},
    }), encoding="utf-8")

    def run(output: Path, view: Path) -> dict:
        result = subprocess.run(
            [
                sys.executable, str(ROOT / "scripts" / "generate_counterfactuals.py"),
                "--mock", "--execute", "--resume",
                "--input", str(source), "--output", str(output),
                "--view-output", str(view), "--limit", "20",
            ],
            env=_env(), capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    fresh = run(tmp_path / "fresh" / "cache.jsonl", tmp_path / "fresh" / "view.jsonl")
    coverage = fresh["coverage"]
    assert coverage["units"]["parent_records"] == "defendant_record"
    assert coverage["units"]["generation_requests"] == "generation_request"
    assert coverage["parents"]["before_filters"]["count"] == 4
    assert coverage["parents"]["after_filters"]["count"] == 3
    assert coverage["parents"]["before_filters"]["charges"] == {
        "盗窃": 2, "诈骗": 1, "故意伤害": 1,
    }
    assert coverage["parents"]["after_filters"]["charges"] == {
        "盗窃": 1, "诈骗": 1, "故意伤害": 1,
    }
    assert coverage["parents"]["before_filters"]["defendants_per_case"]["unit"] == "case"
    before_counts = coverage["parents"]["before_filters"]["defendants_per_case"]["counts"]
    assert before_counts == {"1": 3, "2": 1}
    assert coverage["parents"]["after_filters"]["defendants_per_case"]["counts"] == {"1": 2, "2": 1}
    assert coverage["parents"]["before_filters"]["length_buckets"]["ge_4096"] == 1
    assert coverage["parents"]["excluded"]["length"]["unit"] == "defendant_record"
    assert coverage["parents"]["excluded"]["length"]["count"] == 1
    assert coverage["parents"]["excluded"]["length"]["affected_cases"] == 1
    assert coverage["parents"]["excluded"]["length"]["partial_cases"] == 0
    assert coverage["parents"]["excluded"]["sentence_rank"]["reasons"] == {"penalty_life": 1}
    assert coverage["generation"]["unit"] == "generation_request"
    assert coverage["generation"]["new"] == 9
    assert coverage["generation"]["resumed"] == 0
    assert coverage["generation"]["current_total"] == 9
    assert coverage["view"]["unit"] == "generation_request"
    assert coverage["view"]["records"] == 9
    assert coverage["view"]["passed"] + coverage["view"]["failed_validation"] == 9

    resumed = run(tmp_path / "fresh" / "cache.jsonl", tmp_path / "fresh" / "view-again.jsonl")
    assert resumed["coverage"]["generation"]["new"] == 0
    assert resumed["coverage"]["generation"]["resumed"] == 9
    assert resumed["coverage"]["generation"]["current_total"] == 9
    assert resumed["coverage"]["view"]["records"] == fresh["coverage"]["view"]["records"]
    assert resumed["coverage"]["view"]["passed"] == fresh["coverage"]["view"]["passed"]
    assert resumed["coverage"]["view"]["length_buckets"] == fresh["coverage"]["view"][
        "length_buckets"
    ]
    assert resumed["coverage"]["parents"] == fresh["coverage"]["parents"]


def test_longformer_uses_eager_and_other_models_keep_sdpa(tmp_path, monkeypatch) -> None:
    from legal_landscape.training.train import load_backbone

    seen = {}

    class Config:
        use_cache = True

    class Model:
        def __init__(self):
            self.config = Config()

        def named_parameters(self):
            return []

    def from_pretrained(path, **kwargs):
        seen["attn"] = kwargs["attn_implementation"]
        return Model()

    loader = type("Loader", (), {"from_pretrained": staticmethod(from_pretrained)})
    monkeypatch.setitem(sys.modules, "transformers", type("T", (), {
        "AutoModel": loader,
        "AutoModelForCausalLM": loader,
        "AutoModelForImageTextToText": loader,
        "BitsAndBytesConfig": staticmethod(lambda **kwargs: kwargs),
    }))
    longformer = tmp_path / "lawformer"
    longformer.mkdir()
    (longformer / "config.json").write_text(json.dumps({
        "model_type": "longformer",
        "architectures": ["LongformerModel"],
        "hidden_size": 8,
    }))
    qwen = tmp_path / "qwen"
    qwen.mkdir()
    (qwen / "config.json").write_text(json.dumps({
        "model_type": "bert",
        "architectures": ["BertModel"],
        "hidden_size": 8,
    }))
    load_backbone({"path": str(longformer), "attn_implementation": "sdpa"})
    assert seen["attn"] == "eager"
    load_backbone({"path": str(qwen), "attn_implementation": "sdpa"})
    assert seen["attn"] == "sdpa"


def test_b2_run_selects_eager_attention_without_changing_other_experiments() -> None:
    result = subprocess.run(
        ["bash", str(ROOT / "run.sh"), "--dry-run"],
        cwd=ROOT,
        env={**_env(), "MODE": "matrix", "EXPERIMENTS": "B2 B3", "SEEDS": "42"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    commands = [
        line for line in result.stdout.splitlines() if "scripts/train_model.py" in line
    ]
    b2 = [line for line in commands if "--experiment B2" in line or "--experiment 'B2'" in line]
    b3 = [line for line in commands if "--experiment B3" in line]
    assert b2 and b3
    assert all("model.attn_implementation=eager" in line for line in b2)
    assert all("model.attn_implementation=eager" not in line for line in b3)
