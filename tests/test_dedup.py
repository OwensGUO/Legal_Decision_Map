from __future__ import annotations

import numpy as np

from legal_landscape.data.dedup import cross_split_near_duplicates, minhash_signature

_TEXT = (
    "2015年9月24日6时20分许，被告人郑某某在本市渭滨区高新一路一网吧内趁被害人刘某某熟睡之际，"
    "将其正在充电的苹果手机盗走，后变卖，赃款已挥霍。案发后，被告人郑某某被公安机关抓获，"
    "其到案后如实供述了上述事实，涉案手机经鉴定价值人民币4450元，现已追回并发还被害人。"
)


def test_identical_texts_share_a_signature_and_punctuation_is_ignored() -> None:
    first = minhash_signature(_TEXT)
    second = minhash_signature(_TEXT.replace("，", " "))
    assert first.dtype == np.uint64
    assert np.array_equal(first, second)


def test_cross_split_pairs_find_near_duplicates_only() -> None:
    near = _TEXT.replace("4450元", "4460元")
    unrelated = (
        "被告人王某于2018年驾驶机动车在道路上醉酒行驶，经检测血液酒精含量为每百毫升150毫克。"
    )
    signatures = np.stack([minhash_signature(item) for item in (_TEXT, near, unrelated, _TEXT)])
    pairs = cross_split_near_duplicates(signatures, ["train", "test", "test", "train"])
    assert pairs == {(0, 1), (1, 3)}
