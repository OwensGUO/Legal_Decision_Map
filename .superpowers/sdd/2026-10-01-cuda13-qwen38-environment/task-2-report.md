# Task 2 report: GPU peer-access diagnostics

## Status

Complete. Pairwise peer-access diagnostics are included in successful CUDA probe payloads, and the CLI can write the emitted JSON to a caller-selected path.

## Files

- `scripts/probe_gpu_stack.py`
  - Added `probe_peer_access(torch, device_count)`, recording every ordered pair of distinct visible devices and calculating `all_pairs_accessible`.
  - Added `--output PATH` using `pathlib.Path`; stdout and file receive the same formatted JSON. The probe does not create directories.
  - Diagnostics are added only after CUDA availability and requested device-count validation succeed.
- `tests/test_cli.py`
  - Added the specified pure unit test for the pairwise access matrix.

## RED/GREEN evidence

- RED: `pytest -q tests/test_cli.py::test_gpu_probe_reports_pairwise_peer_access` failed with `KeyError: 'probe_peer_access'`, confirming the missing feature.
- GREEN: after implementation, the peer-access test passed.
- The CLI output path was exercised with `--dry-run --output PATH`; parsed file JSON matched parsed stdout JSON, with the directory created in advance by the check harness.

## Checks

- `pytest -q tests/test_cli.py::test_gpu_probe_reports_pairwise_peer_access tests/test_cli.py::test_environment_and_requirement_dry_runs_are_read_only` — 2 passed.
- `pytest -q tests/test_cli.py::test_all_cli_help_paths tests/test_cli.py::test_gpu_probe_reports_pairwise_peer_access tests/test_cli.py::test_environment_and_requirement_dry_runs_are_read_only` — 3 passed.
- `git diff --check` — passed.
- Manual CLI output consistency check — passed.

## Commit

- `788ea26c8d9777703c6daf79cb2da7bd8c347fd2` — `feat: report GPU peer access`

## Self-review

- Peer pairs are directional and omit self-pairs, matching the specified interface.
- A zero or one-device input produces an empty pair mapping and `all_pairs_accessible: false`, as defined by `bool(pairs) and all(pairs.values())`.
- Output writing uses the caller-provided path and does not create its parent directory.
- Only the two scoped implementation files were staged for the implementation commit.

## Concerns

- CUDA hardware was not available for an end-to-end hardware probe. Pair calculations were verified with a fake CUDA object, and CLI file output was verified through dry-run mode.
- Two unrelated untracked plan/spec files were preserved and left unstaged.
