from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from legal_landscape.data.cail import iter_cail
from legal_landscape.data.cmdl import iter_cmdl
from legal_landscape.data.manifest import build_manifest
from legal_landscape.data.sanitize import charge_lexicon, sanitize_text
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
    text = "公诉机关指控被告人犯盗窃罪，违反刑法第二百六十四条，建议判处有期徒刑三年。其主动退赃。"
    result = sanitize_text(text, charge_lexicon=("盗窃",))
    assert result.raw == text
    assert "建议判处" not in result.conservative
    assert "盗窃" not in result.strict
    assert {item.category for item in result.redactions} >= {
        "result_statement",
        "charge",
        "article",
        "conclusion",
    }
    assert "主动退赃" in result.conservative


def test_charge_lexicon_expands_bracketed_cail_labels() -> None:
    lexicon = charge_lexicon(["[走私、贩卖、运输、制造]毒品", "非法[持有、私藏][枪支、弹药]"])
    assert {"贩卖毒品", "走私、贩卖、运输、制造毒品", "非法持有枪支", "非法私藏弹药"} <= set(
        lexicon
    )
    assert all("[" not in item and "]" not in item for item in lexicon)
    assert list(lexicon) == sorted(lexicon, key=lambda item: (-len(item), item))


def test_sanitizer_masks_bracketed_charge_names_without_case_labels() -> None:
    lexicon = charge_lexicon(["[走私、贩卖、运输、制造]毒品"])
    result = sanitize_text("被告人甲因贩卖毒品罪被刑事拘留。", charge_lexicon=lexicon)
    assert "贩卖毒品" not in result.strict
    assert "被告人甲因" in result.strict


def test_sanitizer_masks_unknown_charge_mentions_after_legal_cues() -> None:
    result = sanitize_text("被告人甲涉嫌合同诈骗罪被刑事拘留，到案后当庭认罪。")
    assert "合同诈骗" not in result.strict
    assert "当庭认罪" in result.strict


def test_sanitizer_truncates_prosecution_conclusion_but_keeps_preceding_facts() -> None:
    text = (
        "被告人甲到案后如实供述，公诉机关认为其行为已构成××罪，系累犯。"
        "公诉机关建议对被告人甲在××至一年八个月幅度内量刑。案发后甲已退赃。"
    )
    result = sanitize_text(text)
    assert "被告人甲到案后如实供述" in result.conservative
    assert "案发后甲已退赃" in result.conservative
    assert "公诉机关认为" not in result.conservative
    assert "一年八个月" not in result.conservative


def test_sanitizer_removes_court_addressed_sentencing_recommendations() -> None:
    result = sanitize_text(
        "被告人甲认罪。公诉机关建议法庭对被告人甲在××至三年幅度内量刑。甲已退赃。"
    )
    assert "三年" not in result.conservative
    assert "被告人甲认罪" in result.conservative
    assert "甲已退赃" in result.conservative


def test_sanitizer_masks_charge_headings_but_keeps_factual_verbs() -> None:
    lexicon = charge_lexicon(["强迫交易", "盗窃"])
    result = sanitize_text(
        "一、强迫交易；2014年甲强行收费。二、盗窃：甲盗窃白酒六箱，盗窃犯罪事实清楚。",
        charge_lexicon=lexicon,
    )
    assert "强迫交易" not in result.strict
    assert "盗窃：" not in result.strict
    assert "盗窃犯罪" not in result.strict
    assert "甲盗窃白酒六箱" in result.strict


def test_sanitizer_keeps_prior_record_facts_but_masks_its_charge_name() -> None:
    result = sanitize_text("被告人甲曾因犯盗窃罪于2010年被判处有期徒刑一年。")
    assert "盗窃" not in result.strict
    assert "2010年被判处有期徒刑一年" in result.strict


def test_sanitizer_masks_article_paragraph_suffixes() -> None:
    result = sanitize_text("甲违反刑法第一百三十三条之一第二款的规定驾驶机动车。")
    assert "第一百三十三条" not in result.strict
    assert "第二款" not in result.strict
    assert "驾驶机动车" in result.strict


def test_sanitizer_keeps_historical_facts_negations_and_false_conclusion_cues() -> None:
    text = (
        "双方根据此前约定交付货款。被告人当庭答应以现金赔偿被害人损失。"
        "甲曾被依法判处有期徒刑一年。甲否认指控，称自己没有盗窃，也没有自首。"
        "辩护人提出不应当从重处罚。"
    )
    result = sanitize_text(text, charge_lexicon=("盗窃",))
    assert result.conservative == text
    assert "没有自首" in result.strict
    assert "有期徒刑一年" in result.strict
    assert "根据此前约定" in result.strict
    assert "答应以现金赔偿" in result.strict


def test_sanitizer_removes_current_sentencing_evaluation_and_keeps_same_sentence_facts() -> None:
    text = "被告人甲到案后如实供述并退赃，可以从轻处罚。其曾于2010年被判处拘役，应当从重处罚。"
    result = sanitize_text(text)
    assert "如实供述并退赃" in result.conservative
    assert "2010年被判处拘役" in result.conservative
    assert "可以从轻处罚" not in result.conservative
    assert "应当从重处罚" not in result.conservative
    evaluations = [item for item in result.redactions if item.category == "conclusion"]
    assert evaluations
    assert all(item.text == text[item.start : item.end] for item in evaluations)
    assert any("可以从轻处罚" in item.text for item in evaluations)
    assert any("应当从重处罚" in item.text for item in evaluations)


def test_sanitizer_does_not_treat_negated_or_promised_phrasing_as_a_charge_cue() -> None:
    result = sanitize_text("甲辩称不以犯罪论处。被告人当庭答应以合同诈骗罪的名义写下说明。")
    assert "不以犯罪论处" in result.strict
    assert "合同诈骗罪" in result.strict
    assert "答应以合同诈骗罪" in result.conservative


def test_sanitizer_is_idempotent_on_strict_text() -> None:
    lexicon = charge_lexicon(["盗窃", "[走私、贩卖、运输、制造]毒品"])
    text = (
        "公诉机关指控，甲盗窃手机一部，曾因犯贩卖毒品罪被判刑。"
        "公诉机关认为其行为已构成盗窃罪，依照《中华人民共和国刑法》第二百六十四条之规定。"
    )
    strict = sanitize_text(text, charge_lexicon=lexicon).strict
    again = sanitize_text(strict, charge_lexicon=lexicon)
    assert again.strict == strict
    assert again.redactions == ()


def _cail_row(fact: str, charge: str, defendant: str = "某甲") -> dict:
    return {
        "fact": fact,
        "meta": {
            "criminals": [defendant],
            "accusation": [charge],
            "relevant_articles": [264],
            "term_of_imprisonment": {
                "death_penalty": False,
                "life_imprisonment": False,
                "imprisonment": 6,
            },
        },
    }


def _build(tmp_path, dataset: str, splits: dict[str, list[dict]]) -> tuple[dict, dict]:
    import yaml

    from legal_landscape.data.build import build_dataset

    names = {
        "cail": ("exercise_contest", "data_{}.json"),
        "cmdl": ("small", "{}_small.jsonl"),
    }
    variant, pattern = names[dataset]
    source = tmp_path / "source" / variant
    source.mkdir(parents=True)
    for split in ("train", "valid", "test"):
        _write_jsonl(source / pattern.format(split), splits.get(split, []))
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({"data": {"dataset": dataset, "root": str(tmp_path / "source")}}),
        encoding="utf-8",
    )
    output = tmp_path / "processed"
    metadata = build_dataset(config, output, limit=100)
    records = {
        split: [
            json.loads(line)
            for line in (output / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        for split in ("train", "valid", "test")
    }
    return metadata, records


_LONG_FACT = (
    "2015年9月24日6时20分许，被告人某甲在本市渭滨区高新一路一网吧内趁被害人刘某某熟睡之际，"
    "将其正在充电的苹果手机盗走，后变卖，赃款已挥霍。案发后，被告人某甲被公安机关抓获，"
    "其到案后如实供述了上述事实，涉案手机经鉴定价值人民币4450元，现已追回并发还被害人。"
)


def test_build_uses_train_only_label_vocabulary_and_reports_unseen_test_labels(tmp_path) -> None:
    metadata, records = _build(
        tmp_path,
        "cail",
        {
            "train": [_cail_row("某甲盗走手机一部。", "盗窃")],
            "test": [_cail_row("某乙谎称借款骗取现金。", "诈骗", "某乙")],
        },
    )
    assert metadata["charge_vocabulary"] == ["盗窃"]
    assert metadata["label_coverage"]["test"]["units_with_unseen_charge"] == 1
    assert metadata["label_coverage"]["train"]["units_with_unseen_charge"] == 0
    assert records["test"][0]["charges"] == ["诈骗"]


def test_build_records_label_agnostic_sanitizer_metadata(tmp_path) -> None:
    from legal_landscape.data.sanitize import SANITIZER_VERSION

    metadata, _ = _build(
        tmp_path, "cail", {"train": [_cail_row("某甲盗走手机一部。", "[走私、贩卖]毒品")]}
    )
    assert metadata["sanitizer"]["version"] == SANITIZER_VERSION
    assert "贩卖毒品" in metadata["sanitizer"]["charge_lexicon"]


def test_build_moves_near_duplicate_facts_to_the_held_out_split(tmp_path) -> None:
    metadata, records = _build(
        tmp_path,
        "cail",
        {
            "train": [_cail_row(_LONG_FACT, "盗窃")],
            "test": [_cail_row(_LONG_FACT.replace("4450元", "4460元"), "盗窃")],
        },
    )
    assert records["train"] == []
    assert len(records["test"]) == 1
    assert metadata["split_integrity"]["near_duplicate_pairs"] == 1
    assert metadata["split_integrity"]["dropped_units"]["train"] == 1


def test_build_same_fact_with_different_defendants_shares_one_group(tmp_path) -> None:
    _, records = _build(
        tmp_path,
        "cail",
        {
            "train": [
                _cail_row("某甲与某乙共同盗走手机一部。", "盗窃", "某甲"),
                _cail_row("某甲与某乙共同盗走手机一部。", "盗窃", "某乙"),
            ]
        },
    )
    first, second = records["train"]
    assert first["case_id"] != second["case_id"]
    assert first["group_id"] == second["group_id"]


def test_build_gives_co_defendants_identical_inputs_and_local_factors(tmp_path) -> None:
    penalty = {"imprisonment": 12}
    row = {
        "fact": "被告人甲犯盗窃罪，被告人乙犯诈骗罪。被告人甲主动投案自首。被告人乙拒不认罪。",
        "defendants": ["甲", "乙"],
        "outcomes": [
            {"name": "甲", "judgment": [{"accusation": "盗窃罪", "penalty": penalty}]},
            {"name": "乙", "judgment": [{"accusation": "诈骗罪", "penalty": penalty}]},
        ],
    }
    _, records = _build(tmp_path, "cmdl", {"train": [row]})
    first, second = records["train"]
    assert first["fact_strict"] == second["fact_strict"]
    assert "盗窃罪" not in first["fact_strict"]
    assert "诈骗罪" not in first["fact_strict"]
    assert first["co_defendants"] == ["乙"]
    assert first["factors"]["surrender"] is True
    assert second["factors"]["surrender"] is None
    assert second["factors"]["confession"] is False


def test_manifest_contains_size_hash_and_record_count(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'{"a": 1}\n{"a": 2}\n')
    [entry] = build_manifest([path])
    assert entry.bytes == 18
    assert entry.records == 2
    assert entry.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def _judgment(accusation: str, penalty: dict, article: list[str] | None = None) -> dict:
    return {
        "accusation": accusation,
        "standard_accusation": accusation,
        "article": article or ["264"],
        "penalty": penalty,
    }


def _cmdl_penalty(**overrides) -> dict:
    penalty = {
        "surveillance": 0,
        "detention": 0,
        "imprisonment": 0,
        "death_penalty": False,
        "life_imprisonment": False,
    }
    penalty.update(overrides)
    return penalty


def _one_cmdl(tmp_path, outcome: dict):
    path = tmp_path / "train_small.jsonl"
    _write_jsonl(
        path,
        [{"fact": "被告人甲实施了行为。", "defendants": ["甲"], "outcomes": [outcome]}],
    )
    return next(iter_cmdl(path, split="train"))


@pytest.mark.parametrize(
    ("final_penalty", "expected_type", "expected_months"),
    [
        (_cmdl_penalty(imprisonment=10), "fixed_term", 10),
        (_cmdl_penalty(life_imprisonment=True, imprisonment=120), "life", None),
        (_cmdl_penalty(death_penalty=True, imprisonment=120), "death", None),
        (_cmdl_penalty(detention=3), "detention", None),
        (_cmdl_penalty(surveillance=6), "control", None),
        (_cmdl_penalty(), "exempt", None),
    ],
)
def test_cmdl_final_penalty_overrides_summed_judgments(
    tmp_path, final_penalty, expected_type, expected_months
) -> None:
    case = _one_cmdl(
        tmp_path,
        {
            "name": "甲",
            "judgment": [
                _judgment("盗窃罪", _cmdl_penalty(imprisonment=18), ["264"]),
                _judgment("诈骗罪", _cmdl_penalty(imprisonment=12), ["266"]),
            ],
            "final_penalty": final_penalty,
        },
    )
    assert case.charges == ("盗窃", "诈骗")
    assert case.penalty_type == expected_type
    assert case.imprisonment_months == expected_months
    assert case.mask_penalty is False
    assert case.penalty_exclusion is None


def test_cmdl_single_charge_without_final_penalty_uses_that_charge(tmp_path) -> None:
    case = _one_cmdl(
        tmp_path,
        {"name": "甲", "judgment": [_judgment("盗窃罪", _cmdl_penalty(imprisonment=14))]},
    )
    assert (case.penalty_type, case.imprisonment_months, case.mask_penalty) == (
        "fixed_term",
        14,
        False,
    )


def test_cmdl_multi_charge_without_final_penalty_is_not_summed(tmp_path) -> None:
    path = tmp_path / "train_small.jsonl"
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
                            _judgment("盗窃罪", _cmdl_penalty(imprisonment=18)),
                            _judgment("危险驾驶罪", _cmdl_penalty(detention=4), ["133"]),
                        ],
                    },
                    {
                        "name": "乙",
                        "judgment": [_judgment("盗窃罪", _cmdl_penalty(detention=4))],
                    },
                ],
            }
        ],
    )
    first, second = iter_cmdl(path, split="train")
    assert first.charges == ("盗窃", "危险驾驶")
    assert (first.penalty_type, first.imprisonment_months) == ("unknown", None)
    assert first.mask_penalty is True
    assert first.penalty_exclusion == "missing_final_penalty"
    assert (second.penalty_type, second.imprisonment_months, second.mask_penalty) == (
        "detention",
        None,
        False,
    )


def test_cmdl_missing_penalty_fields_are_not_treated_as_exemption(tmp_path) -> None:
    case = _one_cmdl(
        tmp_path, {"name": "甲", "judgment": [{"accusation": "盗窃罪", "article": ["264"]}]}
    )
    assert (case.penalty_type, case.imprisonment_months, case.mask_penalty) == (
        "unknown",
        None,
        True,
    )
    assert case.penalty_exclusion == "missing_penalty_fields"


def test_build_counts_penalty_labels_refused_for_missing_final_penalty(tmp_path) -> None:
    metadata, records = _build(
        tmp_path,
        "cmdl",
        {
            "train": [
                {
                    "fact": "甲犯两罪，判决未给出最终刑罚。",
                    "defendants": ["甲"],
                    "outcomes": [
                        {
                            "name": "甲",
                            "judgment": [
                                _judgment("盗窃罪", _cmdl_penalty(imprisonment=18)),
                                _judgment("诈骗罪", _cmdl_penalty(imprisonment=12), ["266"]),
                            ],
                        }
                    ],
                }
            ]
        },
    )
    assert records["train"][0]["mask_penalty"] is True
    assert records["train"][0]["penalty_type"] == "unknown"
    assert metadata["penalty_label_exclusions"] == {
        "missing_final_penalty": 1,
        "missing_penalty_fields": 0,
    }


def test_real_cmdl_records_with_final_penalty_match_that_penalty() -> None:
    root = Path("/Users/gilbert/NEFU/datasets/CMDL/small")
    if not (root / "train_small.jsonl").is_file():
        pytest.skip("local CMDL sample is not present")
    checked = 0
    for name in ("train_small.jsonl", "valid_small.jsonl", "test_small.jsonl"):
        path = root / name
        if not path.is_file():
            continue
        split = name.split("_", 1)[0]
        cases = iter_cmdl(path, split=split)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                for outcome in row.get("outcomes") or []:
                    case = next(cases)
                    final = outcome.get("final_penalty")
                    if not isinstance(final, dict):
                        continue
                    expected = _spec_penalty(final)
                    if expected is None:
                        continue
                    assert (case.penalty_type, case.imprisonment_months) == expected, outcome.get(
                        "name"
                    )
                    assert case.mask_penalty is False
                    assert case.penalty_exclusion is None
                    checked += 1
    assert checked > 0


def _spec_penalty(penalty: dict) -> tuple[str, int | None] | None:
    recognized = {"death_penalty", "life_imprisonment", "imprisonment", "detention", "surveillance"}
    if not recognized & set(penalty):
        return None
    if bool(penalty.get("death_penalty")):
        return "death", None
    if bool(penalty.get("life_imprisonment")):
        return "life", None
    months = int(penalty.get("imprisonment") or 0)
    if months > 0:
        return "fixed_term", months
    if int(penalty.get("detention") or 0) > 0:
        return "detention", None
    if int(penalty.get("surveillance") or 0) > 0:
        return "control", None
    return "exempt", None


def test_cmdl_multi_charge_without_final_penalty_does_not_absorb_detention(tmp_path) -> None:
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
    assert (first.penalty_type, first.imprisonment_months, first.mask_penalty) == (
        "unknown",
        None,
        True,
    )
    assert first.penalty_exclusion == "missing_final_penalty"
    assert (second.penalty_type, second.imprisonment_months, second.mask_penalty) == (
        "detention",
        None,
        False,
    )
