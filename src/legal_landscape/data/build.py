"""Build traceable processed JSONL without modifying source datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from legal_landscape.config import load_config
from legal_landscape.data.cail import iter_cail
from legal_landscape.data.cmdl import iter_cmdl
from legal_landscape.data.dedup import (
    NEAR_DUPLICATE_THRESHOLD,
    cross_split_near_duplicates,
    minhash_signature,
    normalize_fact,
)
from legal_landscape.data.manifest import build_manifest
from legal_landscape.data.sanitize import SANITIZER_VERSION, charge_lexicon, sanitize_text
from legal_landscape.factors.extract import extract_factors
from legal_landscape.progress import NullProgressReporter, ProgressReporter

_SPLIT_PRIORITY = {"train": 0, "valid": 1, "test": 2}


def _source_paths(data: dict[str, Any]) -> dict[str, Path]:
    root = Path(data["root"])
    if data["dataset"] == "cail":
        directory = root / data.get("variant", "exercise_contest")
        return {
            "train": directory / "data_train.json",
            "valid": directory / "data_valid.json",
            "test": directory / "data_test.json",
        }
    if data["dataset"] == "cmdl":
        directory = root / data.get("variant", "small")
        return {
            "train": directory / "train_small.jsonl",
            "valid": directory / "valid_small.jsonl",
            "test": directory / "test_small.jsonl",
        }
    raise ValueError(f"unsupported dataset: {data['dataset']}")


def _iter_cases(
    data: dict[str, Any], source: Path, *, split: str, limit: int | None
):
    if data["dataset"] == "cail":
        return iter_cail(source, split=split, limit=limit)
    return iter_cmdl(source, split=split, limit=limit)


def _content_key(fact: str) -> str:
    return hashlib.sha256(normalize_fact(fact).encode("utf-8")).hexdigest()


@dataclass
class _SplitPlan:
    """Case clusters (same normalized fact or cross-split near duplicates) and their split."""

    cluster_of: dict[str, int] = field(default_factory=dict)
    cluster_split: dict[int, str] = field(default_factory=dict)
    cluster_group: dict[int, str] = field(default_factory=dict)
    cross_split_clusters: int = 0
    near_duplicate_pairs: int = 0
    train_charges: set[str] = field(default_factory=set)
    train_articles: set[str] = field(default_factory=set)

    def assigned_split(self, fact: str) -> str:
        return self.cluster_split[self.cluster_of[_content_key(fact)]]

    def group_id(self, fact: str) -> str:
        return self.cluster_group[self.cluster_of[_content_key(fact)]]


def _plan_splits(
    data: dict[str, Any], paths: dict[str, Path], *, limit: int | None,
    progress: ProgressReporter,
) -> _SplitPlan:
    """Assign each case cluster to the most held-out official split it appears in."""
    keys: list[str] = []
    index_of: dict[str, int] = {}
    splits_of: list[set[str]] = []
    signatures: list[Any] = []
    charges_of: list[set[str]] = []
    articles_of: list[set[str]] = []
    processed = 0
    with progress.task(
        f"Assign groups · {data['dataset']}",
        total=len(paths) * limit if data["dataset"] == "cail" and limit is not None else None,
        dataset=data["dataset"],
        pass_name="assignment",
        split="all",
        accepted=0,
        dropped=0,
    ) as task:
        for split, source in paths.items():
            for case in _iter_cases(data, source, split=split, limit=limit):
                key = _content_key(case.fact_raw)
                index = index_of.get(key)
                if index is None:
                    index = index_of[key] = len(keys)
                    keys.append(key)
                    splits_of.append(set())
                    signatures.append(minhash_signature(case.fact_raw))
                    charges_of.append(set())
                    articles_of.append(set())
                splits_of[index].add(split)
                if split == "train":
                    charges_of[index].update(case.charges)
                    articles_of[index].update(case.conviction_articles)
                processed += 1
                task.advance(split=split, accepted=processed)
        task.succeed(processed=processed)

    def held_out(splits: set[str]) -> str:
        return max(splits, key=_SPLIT_PRIORITY.__getitem__)

    parent = list(range(len(keys)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    pairs: set[tuple[int, int]] = set()
    if keys:
        pairs = cross_split_near_duplicates(
            np.stack(signatures), [held_out(item) for item in splits_of]
        )
    for first, second in pairs:
        parent[find(first)] = find(second)
    members: dict[int, list[int]] = {}
    for index in range(len(keys)):
        members.setdefault(find(index), []).append(index)

    plan = _SplitPlan(near_duplicate_pairs=len(pairs))
    for cluster, indices in enumerate(members.values()):
        cluster_splits = set().union(*(splits_of[index] for index in indices))
        split = held_out(cluster_splits)
        plan.cluster_split[cluster] = split
        plan.cluster_group[cluster] = f"{data['dataset']}-g{min(keys[i] for i in indices)[:20]}"
        plan.cross_split_clusters += len(cluster_splits) > 1
        for index in indices:
            plan.cluster_of[keys[index]] = cluster
            if split == "train":
                plan.train_charges.update(charges_of[index])
                plan.train_articles.update(articles_of[index])
    return plan


def build_dataset(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    limit: int | None = None,
    overrides: dict[str, Any] | None = None,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Stream all configured splits into auditable processed records."""
    config = load_config(config_path, overrides=overrides)
    reporter = progress if progress is not None else NullProgressReporter()
    data = config["data"]
    paths = _source_paths(data)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    split_units: dict[str, int] = {}
    plan = _plan_splits(data, paths, limit=limit, progress=reporter)
    charges = sorted(plan.train_charges)
    articles = sorted(plan.train_articles)
    lexicon = charge_lexicon(charges)
    coverage = {
        split: {"units_with_unseen_charge": 0, "units_with_unseen_article": 0} for split in paths
    }
    dropped_units = dict.fromkeys(paths, 0)
    for split, source in paths.items():
        iterator = _iter_cases(data, source, split=split, limit=limit)
        count = 0
        output_path = destination / f"{split}.jsonl"
        with reporter.task(
            f"Build {split} · {data['dataset']}",
            total=limit if data["dataset"] == "cail" else None,
            dataset=data["dataset"],
            pass_name="build",
            split=split,
            accepted=0,
            dropped=0,
        ) as task:
            with output_path.open("w", encoding="utf-8") as handle:
                for case in iterator:
                    if plan.assigned_split(case.fact_raw) != split:
                        dropped_units[split] += 1
                    else:
                        sanitized = sanitize_text(case.fact_raw, charge_lexicon=lexicon)
                        factors = extract_factors(
                            sanitized.strict,
                            target_defendant=case.target_defendant,
                            co_defendants=case.co_defendants,
                        )
                        record = {
                            **case.to_dict(),
                            "group_id": plan.group_id(case.fact_raw),
                            "source_group_id": case.group_id,
                            "fact_conservative": sanitized.conservative,
                            "fact_strict": sanitized.strict,
                            "redactions": [asdict(item) for item in sanitized.redactions],
                            "factors": factors.to_dict(),
                        }
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        coverage[split]["units_with_unseen_charge"] += any(
                            item not in plan.train_charges for item in case.charges
                        )
                        coverage[split]["units_with_unseen_article"] += any(
                            item not in plan.train_articles for item in case.conviction_articles
                        )
                        count += 1
                    task.advance(accepted=count, dropped=dropped_units[split])
            task.succeed(processed=count + dropped_units[split])
        split_units[split] = count
    manifest = [entry.to_dict() for entry in build_manifest(list(paths.values()))]
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "dataset": data["dataset"],
        "split_units": split_units,
        "split_integrity": {
            "assignment_policy": "prefer_test_then_valid_then_train",
            "grouping": "normalized_fact_and_minhash",
            "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
            "near_duplicate_pairs": plan.near_duplicate_pairs,
            "cross_split_groups": plan.cross_split_clusters,
            "dropped_units": dropped_units,
        },
        "label_vocabulary_source": "train",
        "label_coverage": coverage,
        "sanitizer": {"version": SANITIZER_VERSION, "charge_lexicon": list(lexicon)},
        "charge_vocabulary": charges,
        "article_vocabulary": articles,
        "penalty_vocabulary": [
            "fixed_term",
            "life",
            "death",
            "detention",
            "control",
            "exempt",
            "unknown",
        ],
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata
