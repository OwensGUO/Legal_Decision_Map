from __future__ import annotations

import hashlib
import json

import pytest

from legal_landscape.data.cail import iter_cail
from legal_landscape.data.cmdl import iter_cmdl
from legal_landscape.data.manifest import build_manifest
from legal_landscape.data.sanitize import sanitize_text
from legal_landscape.data.schema import assert_group_split_integrity


def _write_jsonl(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def test_cail_maps_one_row_to_one_case_and_preserves_multilabel(tmp_path) -> None:
    path = tmp_path / "data_train.json"
    _write_jsonl(
        path,
        [
            {
                "fact": "被告人段某造成轻伤。",
                "meta": {
                    "criminals": ["段某"],
                    "accusation": ["故意伤害", "寻衅滋事"],
                    "relevant_articles": [234, "293-1"],
                    "term_of_imprisonment": {
                        "death_penalty": False,
                        "life_imprisonment": False,
                        "imprisonment": 12,
                    },
                },
            }
        ],
    )
    [case] = list(iter_cail(path, split="train"))
    assert case.case_id == case.group_id
    assert case.target_defendant == "段某"
    assert case.charges == ("故意伤害", "寻衅滋事")
    assert case.conviction_articles == ("criminal_law:234", "criminal_law:293:1")
    assert case.sentencing_articles == ()
    assert case.imprisonment_months == 12


def test_cmdl_expands_defendants_and_keeps_group(tmp_path) -> None:
    path = tmp_path / "train_small.jsonl"
    penalty = {
        "surveillance": 0,
        "detention": 0,
        "imprisonment": 18,
        "death_penalty": False,
        "life_imprisonment": False,
    }
    _write_jsonl(
        path,
        [
            {
                "fact": "甲乙共同实施行为。",
                "defendants": ["甲", "乙"],
                "relevant_articles": ["25-1", "27"],
                "outcomes": [
                    {
                        "name": "甲",
                        "judgment": [
                            {"accusation": "盗窃罪", "article": ["264"], "penalty": penalty}
                        ],
                    },
                    {
                        "name": "乙",
                        "judgment": [
                            {"accusation": "诈骗罪", "article": ["266-1"], "penalty": penalty}
                        ],
                    },
                ],
            }
        ],
    )
    cases = list(iter_cmdl(path, split="train"))
    assert [item.target_defendant for item in cases] == ["甲", "乙"]
    assert len({item.group_id for item in cases}) == 1
    assert len({item.case_id for item in cases}) == 2
    assert cases[0].charges == ("盗窃",)
    assert cases[0].conviction_articles == ("criminal_law:264",)
    assert cases[1].conviction_articles == ("criminal_law:266:1",)
    assert all(item.sentencing_articles == () for item in cases)


def test_life_sentence_never_gets_months(tmp_path) -> None:
    path = tmp_path / "data.json"
    _write_jsonl(
        path,
        [
            {
                "fact": "事实",
                "meta": {
                    "criminals": ["某甲"],
                    "accusation": ["抢劫"],
                    "term_of_imprisonment": {
                        "death_penalty": False,
                        "life_imprisonment": True,
                        "imprisonment": 999,
                    },
                },
            }
        ],
    )
    [case] = list(iter_cail(path, split="test"))
    assert case.penalty_type == "life"
    assert case.imprisonment_months is None


def test_group_split_integrity_rejects_leakage(tmp_path) -> None:
    path = tmp_path / "data.json"
    base = {
        "fact": "事实",
        "meta": {
            "criminals": ["某甲"],
            "accusation": ["盗窃"],
            "term_of_imprisonment": {
                "death_penalty": False,
                "life_imprisonment": False,
                "imprisonment": 6,
            },
        },
    }
    _write_jsonl(path, [base])
    [train_case] = list(iter_cail(path, split="train"))
    leaked = train_case.to_dict()
    leaked["split"] = "test"
    from legal_landscape.data.schema import CaseUnit

    with pytest.raises(ValueError, match="multiple splits"):
        assert_group_split_integrity([train_case, CaseUnit.from_dict(leaked)])


def test_identical_source_case_has_stable_group_across_split_files(tmp_path) -> None:
    row = {
        "fact": "同一案件事实",
        "meta": {
            "criminals": ["某甲"],
            "accusation": ["盗窃"],
            "term_of_imprisonment": {
                "death_penalty": False,
                "life_imprisonment": False,
                "imprisonment": 6,
            },
        },
    }
    train_path = tmp_path / "train.json"
    test_path = tmp_path / "test.json"
    _write_jsonl(train_path, [row])
    differently_labelled = json.loads(json.dumps(row, ensure_ascii=False))
    differently_labelled["meta"]["accusation"] = ["诈骗"]
    _write_jsonl(test_path, [differently_labelled])
    [train_case] = list(iter_cail(train_path, split="train"))
    [test_case] = list(iter_cail(test_path, split="test"))
    assert train_case.group_id == test_case.group_id
    with pytest.raises(ValueError, match="multiple splits"):
        assert_group_split_integrity([train_case, test_case])


def test_sanitizer_records_categories_without_mutating_raw() -> None:
    text = "公诉机关指控被告人犯盗窃罪，依据刑法第二百六十四条，建议判处有期徒刑三年。其主动退赃。"
    result = sanitize_text(text, charges=("盗窃",))
    assert result.raw == text
    assert "建议判处" not in result.conservative
    assert "盗窃" not in result.strict
    assert {item.category for item in result.redactions} >= {
        "result_statement",
        "charge",
        "article",
    }
    assert "主动退赃" in result.conservative


def test_manifest_contains_size_hash_and_record_count(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'{"a": 1}\n{"a": 2}\n')
    [entry] = build_manifest([path])
    assert entry.bytes == 18
    assert entry.records == 2
    assert entry.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_cmdl_fixed_term_absorbs_concurrent_detention(tmp_path) -> None:
    path = tmp_path / "train_small.jsonl"

    def penalty(imprisonment: int, detention: int) -> dict:
        return {
            "surveillance": 0,
            "detention": detention,
            "imprisonment": imprisonment,
            "death_penalty": False,
            "life_imprisonment": False,
        }

    _write_jsonl(
        path,
        [
            {
                "fact": "甲乙共同实施行为。",
                "defendants": ["甲", "乙"],
                "outcomes": [
                    {
                        "name": "甲",
                        "judgment": [
                            {"accusation": "盗窃罪", "penalty": penalty(18, 0)},
                            {"accusation": "危险驾驶罪", "penalty": penalty(0, 4)},
                        ],
                    },
                    {
                        "name": "乙",
                        "judgment": [
                            {"accusation": "盗窃罪", "penalty": penalty(0, 4)}
                        ],
                    },
                ],
            }
        ],
    )
    first, second = iter_cmdl(path, split="train")
    assert (first.penalty_type, first.imprisonment_months) == ("fixed_term", 18)
    assert (second.penalty_type, second.imprisonment_months) == ("detention", None)
