"""MinHash near-duplicate detection used to keep near-identical cases out of training."""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

NUM_PERMUTATIONS = 64
SHINGLE_SIZE = 5
NEAR_DUPLICATE_THRESHOLD = 0.9
_PRIME = np.uint64((1 << 31) - 1)
_BASE = np.uint64(1000003)
_RNG = np.random.default_rng(20261024)
_A = _RNG.integers(1, int(_PRIME), NUM_PERMUTATIONS, dtype=np.uint64)
_B = _RNG.integers(0, int(_PRIME), NUM_PERMUTATIONS, dtype=np.uint64)
_NON_WORD = re.compile(r"[\W_]+")


def normalize_fact(text: str) -> str:
    """Drop whitespace and punctuation so formatting differences do not split a case."""
    return _NON_WORD.sub("", text)


def minhash_signature(text: str) -> np.ndarray:
    """MinHash over character shingles of the normalized text."""
    codes = np.frombuffer(normalize_fact(text).encode("utf-32-le"), dtype=np.uint32).astype(
        np.uint64
    )
    width = min(SHINGLE_SIZE, len(codes)) or 1
    if len(codes) == 0:
        codes = np.zeros(1, dtype=np.uint64)
    count = len(codes) - width + 1
    hashes = np.zeros(count, dtype=np.uint64)
    for offset in range(width):
        hashes = (hashes * _BASE + codes[offset : offset + count]) % _PRIME
    shingles = np.unique(hashes)
    return ((_A[:, None] * shingles[None, :] + _B[:, None]) % _PRIME).min(axis=1)


def cross_split_near_duplicates(
    signatures: np.ndarray,
    splits: list[str],
    *,
    bands: int = 8,
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> set[tuple[int, int]]:
    """Index pairs from different splits whose estimated Jaccard reaches ``threshold``."""
    count, permutations = signatures.shape
    rows = permutations // bands
    pairs: set[tuple[int, int]] = set()
    for band in range(bands):
        block = np.ascontiguousarray(signatures[:, band * rows : (band + 1) * rows])
        buckets: dict[bytes, list[int]] = defaultdict(list)
        for index in range(count):
            buckets[block[index].tobytes()].append(index)
        for members in buckets.values():
            by_split: dict[str, list[int]] = defaultdict(list)
            for index in members:
                by_split[splits[index]].append(index)
            names = sorted(by_split)
            for position, first_split in enumerate(names):
                for second_split in names[position + 1 :]:
                    small, large = sorted(
                        (by_split[first_split], by_split[second_split]), key=len
                    )
                    candidates = np.asarray(large)
                    for index in small:
                        similarity = (signatures[candidates] == signatures[index]).mean(axis=1)
                        for match in candidates[similarity >= threshold]:
                            pairs.add(tuple(sorted((index, int(match)))))
    return pairs
