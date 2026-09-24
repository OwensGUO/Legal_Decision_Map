"""Source-file manifests without copying source data."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    bytes: int
    sha256: str
    records: int

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def build_manifest(paths: list[str | Path]) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append(
            ManifestEntry(
                path=str(path),
                bytes=path.stat().st_size,
                sha256=_sha256(path),
                records=_line_count(path),
            )
        )
    return entries
