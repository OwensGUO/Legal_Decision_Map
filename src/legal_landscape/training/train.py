"""Experiment configuration, model inspection, and safe training helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import shutil
from collections.abc import Callable, Iterable, Sized
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from legal_landscape.progress import ProgressReporter, ProgressTask, create_progress_reporter


@dataclass(frozen=True)
class Experiment:
    name: str
    backbone: str
    use_factors: bool
    typed_counterfactuals: bool
    ordinary_augmentation: bool
    use_invariant: bool
    use_boundary: bool
    use_rank: bool
    hard_charge_sentence: bool
    filter_counterfactuals: bool


@dataclass(frozen=True)
class InspectedModelConfig:
    model_type: str
    architectures: tuple[str, ...]
    loader_class: str
    architecture_kind: str
    text_backbone_path: tuple[str, ...]
    has_visual_components: bool
    hidden_size: int | None


class DeterministicEpochSampler:
    """Deterministic shuffled indices with explicit resumable epoch position."""

    def __init__(self, size: int, *, seed: int) -> None:
        if size < 0:
            raise ValueError("sampler size must be non-negative")
        self.size = size
        self.seed = seed
        self.epoch = 0
        self.start_index = 0

    def __iter__(self):
        indices = list(range(self.size))
        random.Random(self.seed + self.epoch).shuffle(indices)
        return iter(indices[self.start_index :])

    def __len__(self) -> int:
        return max(0, self.size - self.start_index)

    def advance(self, count: int) -> None:
        if count < 0:
            raise ValueError("sampler advance must be non-negative")
        self.start_index = min(self.size, self.start_index + count)

    def next_epoch(self) -> None:
        self.epoch += 1
        self.start_index = 0

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "start_index": self.start_index, "seed": self.seed}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        state_seed = int(state["seed"])
        if state_seed != self.seed:
            raise ValueError(f"sampler seed mismatch: checkpoint={state_seed}, current={self.seed}")
        epoch = int(state["epoch"])
        start_index = int(state["start_index"])
        if epoch < 0 or not 0 <= start_index <= self.size:
            raise ValueError("invalid sampler checkpoint state")
        self.epoch = epoch
        self.start_index = start_index


class TokenBudgetBatchSampler:
    """Deterministic, resumable length-bucketed batches under a padded-token budget.

    Each epoch shuffles indices, sorts windows of ``max_batch * 50`` indices by length, packs
    them greedily so that ``batch size * longest length <= max_tokens``, and shuffles the batch
    order. Position is tracked in batches so a resumed run yields the same remaining batches.
    """

    def __init__(self, lengths: list[int], max_tokens: int, max_batch: int, *, seed: int) -> None:
        if max_tokens < 1 or max_batch < 1:
            raise ValueError("max_tokens and max_batch must be positive")
        self.lengths = [max(1, int(length)) for length in lengths]
        self.max_tokens = max_tokens
        self.max_batch = max_batch
        self.seed = seed
        self.epoch = 0
        self.start_batch = 0
        self._cache: tuple[int, list[list[int]]] | None = None

    def _batches(self) -> list[list[int]]:
        if self._cache is not None and self._cache[0] == self.epoch:
            return self._cache[1]
        generator = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        generator.shuffle(indices)
        window = self.max_batch * 50
        batches: list[list[int]] = []
        for start in range(0, len(indices), window):
            ordered = sorted(indices[start : start + window], key=self.lengths.__getitem__)
            batches.extend(_pack(ordered, self.lengths, self.max_tokens, self.max_batch))
        generator.shuffle(batches)
        self._cache = (self.epoch, batches)
        return batches

    @property
    def batches_per_epoch(self) -> int:
        return len(self._batches())

    def __iter__(self):
        return iter(self._batches()[self.start_batch :])

    def __len__(self) -> int:
        return max(0, self.batches_per_epoch - self.start_batch)

    def advance(self, count: int) -> None:
        if count < 0:
            raise ValueError("sampler advance must be non-negative")
        self.start_batch = min(self.batches_per_epoch, self.start_batch + count)

    def next_epoch(self) -> None:
        self.epoch += 1
        self.start_batch = 0

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "start_batch": self.start_batch, "seed": self.seed}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["seed"]) != self.seed:
            raise ValueError(
                f"sampler seed mismatch: checkpoint={state['seed']}, current={self.seed}"
            )
        self.epoch = int(state["epoch"])
        self._cache = None
        start = int(state["start_batch"])
        if self.epoch < 0 or not 0 <= start <= self.batches_per_epoch:
            raise ValueError("invalid sampler checkpoint state")
        self.start_batch = start


def _pack(
    ordered: list[int], lengths: list[int], max_tokens: int, max_batch: int
) -> list[list[int]]:
    batches: list[list[int]] = []
    current: list[int] = []
    longest = 0
    for index in ordered:
        candidate = max(longest, lengths[index])
        if current and (len(current) >= max_batch or (len(current) + 1) * candidate > max_tokens):
            batches.append(current)
            current, candidate = [], lengths[index]
        current.append(index)
        longest = candidate
    if current:
        batches.append(current)
    return batches


def inference_batches(lengths: list[int], max_tokens: int, max_batch: int) -> list[list[int]]:
    """Length-sorted batches for prediction; order is restored through prediction indices."""
    ordered = sorted(range(len(lengths)), key=lambda index: lengths[index])
    return _pack(ordered, [max(1, int(length)) for length in lengths], max_tokens, max_batch)


def shard_batches(
    batches: list[list[int]], num_processes: int, process_index: int
) -> list[list[int]]:
    """Round-robin batches for one process, padded so every process runs the same number of
    forward passes (DDP forwards are collective); duplicated rows are deduplicated by index."""
    if not batches:
        return []
    per_process = math.ceil(len(batches) / num_processes)
    padded = batches + [batches[-1]] * (per_process * num_processes - len(batches))
    return padded[process_index::num_processes]


def main_accumulation_should_sync(
    microbatches_completed_in_epoch: int,
    accumulation_steps: int,
    *,
    at_epoch_end: bool,
) -> bool:
    """Whether this main microbatch must update parameters and the learning rate.

    ``microbatches_completed_in_epoch`` counts only main batches yielded in the current
    epoch pass, starting at 1. Full windows sync on multiples of ``accumulation_steps``.
    A short final window syncs at the main epoch boundary so its gradients are applied.
    Counterfactual batches are not part of this decision. Resuming is valid only from a
    completed update, so a restored epoch pass starts a new window.
    """
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if microbatches_completed_in_epoch < 1:
        raise ValueError("main microbatch count must be positive")
    if microbatches_completed_in_epoch % accumulation_steps == 0:
        return True
    return bool(at_epoch_end)


def learning_rate_factor(
    step: int, total_steps: int, warmup_steps: int, min_ratio: float
) -> float:
    """Linear warmup to 1, then cosine decay to ``min_ratio`` at ``total_steps``."""
    if total_steps <= 0:
        return 1.0
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


_SENTENCE = re.compile(r"[^。；！？\n]*[。；！？\n]?")


def focus_fact(text: str, target: str, co_defendants: Iterable[str], max_chars: int) -> str:
    """Fit a long fact into ``max_chars`` by keeping sentences about the target defendant,
    then sentences naming no defendant, in original order; sentences that mention only other
    defendants are dropped first."""
    if len(text) <= max_chars:
        return text
    others = [name for name in co_defendants if name and name != target]
    sentences = [sentence for sentence in _SENTENCE.findall(text) if sentence]
    tiers: list[tuple[int, int]] = []
    for position, sentence in enumerate(sentences):
        stripped = sentence
        for name in others:
            if target and target in name:
                stripped = stripped.replace(name, "")
        if target and target in stripped:
            tiers.append((0, position))
        elif not any(name in sentence for name in others):
            tiers.append((1, position))
    kept: set[int] = set()
    used = 0
    for _tier, position in sorted(tiers):
        size = len(sentences[position])
        if used + size <= max_chars:
            kept.add(position)
            used += size
    focused = "".join(sentences[position] for position in sorted(kept))
    return (focused or text)[:max_chars]


def tune_threshold(
    probabilities: list[list[float]],
    truth: list[list[int]],
    grid: Iterable[float] = tuple(round(0.05 * step, 2) for step in range(1, 20)),
) -> float:
    """Global decision threshold maximizing micro-F1; ties keep the lowest threshold."""
    import numpy as np

    scores = np.asarray(probabilities, dtype=float)
    labels = np.asarray(truth, dtype=bool)
    best_threshold, best_score = 0.5, -1.0
    for threshold in grid:
        predicted = scores >= threshold
        empty = ~predicted.any(axis=1)
        if empty.any():
            predicted[empty, scores[empty].argmax(axis=1)] = True
        tp = float((predicted & labels).sum())
        denominator = float(predicted.sum() + labels.sum())
        score = 2 * tp / denominator if denominator else 0.0
        if score > best_score:
            best_threshold, best_score = float(threshold), score
    return best_threshold


def select_best_checkpoint(destination: str | Path) -> Path | None:
    """The checkpoint with the highest recorded validation score; ties prefer more steps."""
    root = Path(destination)
    best: tuple[float, int, Path] | None = None
    for progress_path in sorted(root.glob("checkpoint-*/training_progress.json")):
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        score = progress.get("validation_score")
        if score is None:
            continue
        candidate = (float(score), int(progress.get("steps", 0)), progress_path.parent)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return best[2] if best else None


def prune_checkpoints(destination: str | Path, *, keep: set[Path]) -> list[Path]:
    """Delete periodic checkpoints that are neither kept (best/latest) nor final."""
    removed = []
    for checkpoint in sorted(Path(destination).glob("checkpoint-step-*")):
        if checkpoint not in keep and checkpoint.is_dir():
            shutil.rmtree(checkpoint)
            removed.append(checkpoint)
    return removed


def _file_identity(path: str | Path) -> dict[str, int | str]:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(source.resolve()),
        "bytes": source.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("torch", "transformers", "accelerate", "peft", "bitsandbytes"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


PROCESSING_VERSION = "2026-10-27-label-protocol"
_LOSS_WEIGHT_NAMES = ("article", "sentence", "invariant", "boundary", "response", "factor")
# Logging and checkpoint selection. They do not change the sample stream or the loss.
RESUME_LOG_ONLY_KEYS = (
    "checkpoint_every",
    "validation_limit",
    "inference_batch_size",
    "inference_max_tokens_per_batch",
    "select_best_checkpoint",
    "keep_best_and_latest_checkpoints",
)


def _is_longformer(inspected: InspectedModelConfig) -> bool:
    if inspected.model_type.lower() == "longformer":
        return True
    return any("longformer" in name.lower() for name in inspected.architectures)


def configured_attention(model_config: dict[str, Any]) -> str:
    """Attention implementation actually used for this checkpoint.

    Longformer on the current stack has no SDPA kernel, so it always uses eager.
    Other architectures keep the configured implementation, defaulting to SDPA.
    """
    requested = str(model_config.get("attn_implementation", "sdpa"))
    path = model_config.get("path")
    if path and (Path(path) / "config.json").is_file():
        if _is_longformer(inspect_model_config(path)):
            return "eager"
    return requested


def _resolved_max_length(config: dict[str, Any]) -> int:
    model = config.get("model") or {}
    return int(model.get("max_length", config.get("max_length", 4096)))


_TOKENIZER_AND_CONFIG_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "tokenizer.model",
    "config.json",
)


def _canonical_file_bytes(path: Path) -> bytes:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    return path.read_bytes()


def tokenizer_and_config_identity(model_path: str | Path | None) -> dict[str, Any]:
    """Fingerprint tokenizer resources and config.json, not model weights.

    The same files at another path match. A changed vocab, special token, added token,
    or config does not. A missing directory is an explicit unavailable identity, not a
    silent skip.
    """
    if not model_path:
        return {"status": "unavailable", "files": {}}
    root = Path(model_path)
    if not root.is_dir():
        return {"status": "unavailable", "files": {}}
    files: dict[str, str] = {}
    for name in _TOKENIZER_AND_CONFIG_FILES:
        file = root / name
        if not file.is_file():
            continue
        files[name] = hashlib.sha256(_canonical_file_bytes(file)).hexdigest()
    return {"status": "present", "files": files}


def _resolved_model_identity(config: dict[str, Any]) -> dict[str, Any]:
    model = dict(config.get("model") or {})
    training = config.get("training") or {}
    quantization = model.get("quantization", "none")
    identity: dict[str, Any] = {
        "max_length": _resolved_max_length(config),
        "quantization": quantization,
        "attn_implementation": configured_attention(model),
        "gradient_checkpointing": bool(
            training.get("gradient_checkpointing", model.get("gradient_checkpointing", True))
        ),
        "tokenizer_identity": tokenizer_and_config_identity(model.get("path")),
    }
    if quantization == "nf4":
        lora = dict(model.get("lora") or {})
        identity["lora"] = {
            "rank": int(lora.get("rank", 32)),
            "alpha": int(lora.get("alpha", 64)),
            "dropout": float(lora.get("dropout", 0.05)),
        }
    else:
        identity["lora"] = None
    return identity


def _resolved_training_identity(training: dict[str, Any], *, max_length: int) -> dict[str, Any]:
    batch = int(training.get("per_device_batch_size", 1))
    return {
        "learning_rate": float(training.get("learning_rate", 2e-4)),
        "epochs": int(training.get("epochs", 1)),
        "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps", 8)),
        "per_device_batch_size": batch,
        "max_tokens_per_batch": int(training.get("max_tokens_per_batch", batch * max_length)),
        "pair_every": max(1, int(training.get("pair_every", 1))),
        "rank_margin": float(training.get("rank_margin", 1.0)),
        "mixed_precision": str(training.get("mixed_precision", "bf16")),
        "seed": int(training.get("seed", 42)),
    }


def _resolved_schedule(training: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_steps": int(training.get("max_steps", 1000)),
        "warmup_ratio": float(training.get("warmup_ratio", 0.0)),
        "min_learning_rate_ratio": float(training.get("min_learning_rate_ratio", 1.0)),
    }


def _resolved_loss(loss: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(loss or {})
    reduction = str(payload.pop("multilabel_reduction", "sum"))
    weights = {name: float(payload.pop(name, 1.0)) for name in _LOSS_WEIGHT_NAMES}
    for name, value in payload.items():
        weights[str(name)] = float(value)
    return {"multilabel_reduction": reduction, "weights": weights}


def resume_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def resume_compatibility(
    *,
    config: dict[str, Any],
    experiment_name: str,
    limit: int | None,
    num_processes: int,
    charge_vocabulary: list[str],
    article_vocabulary: list[str],
    penalty_vocabulary: list[str],
    validation_identity: dict[str, Any] | None,
) -> dict[str, Any]:
    """Normalized identity of everything that changes samples, loss, or parameters.

    ``max_steps`` is part of the schedule, not a log setting: it sets the warmup and
    cosine horizon. Resume may extend a positive cap; the restored run keeps its
    optimizer step and then follows the new horizon. Shortening the cap, or switching
    between a numeric cap and ``max_steps=0``, is rejected. Checkpoint frequency and
    validation-display limits stay outside this identity.
    """
    from legal_landscape.counterfactual.prompts import PROMPT_VERSION
    from legal_landscape.data.sanitize import SANITIZER_VERSION

    training = dict(config.get("training") or {})
    return {
        "processing_version": PROCESSING_VERSION,
        "sanitizer_version": SANITIZER_VERSION,
        "prompt_version": PROMPT_VERSION,
        "charge_vocabulary": list(charge_vocabulary),
        "article_vocabulary": list(article_vocabulary),
        "penalty_vocabulary": list(penalty_vocabulary),
        "experiment": asdict(build_experiment(experiment_name)),
        "model": _resolved_model_identity(config),
        "data_limit": None if limit is None else int(limit),
        "num_processes": int(num_processes),
        "training_config": _resolved_training_identity(
            training, max_length=_resolved_max_length(config)
        ),
        "schedule": _resolved_schedule(training),
        "loss": _resolved_loss(config.get("loss")),
        "validation_identity": validation_identity,
        "log_only_keys": list(RESUME_LOG_ONLY_KEYS),
    }


def _schedule_extension_error(
    saved: dict[str, Any] | None, current: dict[str, Any] | None
) -> str | None:
    saved_schedule = dict(saved or {})
    current_schedule = dict(current or {})
    saved_steps = saved_schedule.pop("max_steps", None)
    current_steps = current_schedule.pop("max_steps", None)
    if saved_schedule != current_schedule:
        return "warmup or minimum learning-rate ratio changed"
    if saved_steps == current_steps:
        return None
    if (
        isinstance(saved_steps, int)
        and isinstance(current_steps, int)
        and not isinstance(saved_steps, bool)
        and not isinstance(current_steps, bool)
        and saved_steps > 0
        and current_steps > saved_steps
    ):
        return None
    return (
        "max_steps changes the warmup/cosine horizon. Resume may extend a positive cap "
        "and then follow the new horizon from the saved step; it cannot shorten the cap "
        "or switch between a numeric cap and max_steps=0"
    )


def _tokenizer_identity_error(
    saved: dict[str, Any], expected: dict[str, Any]
) -> str | None:
    saved_model = saved.get("model") if isinstance(saved.get("model"), dict) else {}
    expected_model = expected.get("model") if isinstance(expected.get("model"), dict) else {}
    saved_has = "tokenizer_identity" in saved_model
    expected_has = "tokenizer_identity" in expected_model
    if expected_has and not saved_has:
        return (
            "checkpoint has no tokenizer and model-config fingerprint; start a new run, "
            "this checkpoint cannot be migrated in place"
        )
    if saved_has or expected_has:
        if saved_model.get("tokenizer_identity") != expected_model.get("tokenizer_identity"):
            return "tokenizer or model config content changed"
    return None


def _compared_compatibility(payload: dict[str, Any]) -> dict[str, Any]:
    body = {
        key: value
        for key, value in payload.items()
        if key not in {"log_only_keys", "schedule"}
    }
    schedule = dict(payload.get("schedule") or {})
    schedule.pop("max_steps", None)
    body["schedule_without_horizon"] = schedule
    return body


def validate_resume_metadata(
    progress: dict[str, Any],
    *,
    expected_seed: int,
    expected_input_identity: dict[str, Any],
    expected_compatibility: dict[str, Any] | None = None,
    check_compatibility: bool = True,
) -> None:
    """Reject checkpoints produced from a different deterministic run input."""
    if int(progress.get("seed", -1)) != expected_seed:
        raise ValueError(
            f"checkpoint seed mismatch: checkpoint={progress.get('seed')}, current={expected_seed}"
        )
    saved_identity = progress.get("input_identity")
    if saved_identity != expected_input_identity:
        differences = []
        if isinstance(saved_identity, dict):
            for name in sorted(saved_identity.keys() | expected_input_identity.keys()):
                saved = saved_identity.get(name)
                current = expected_input_identity.get(name)
                if saved != current:
                    differences.append(f"{name}: checkpoint={saved!r}, current={current!r}")
        else:
            differences.append(
                f"input_identity: checkpoint={saved_identity!r}, "
                f"current={expected_input_identity!r}"
            )
        raise ValueError(
            "checkpoint input identity does not match the current training inputs. "
            + "; ".join(differences)
            + ". Restore the original input files to resume, or restart with "
            "RESUME_TRAINING=0 bash run.sh (archives selected runs and preserves "
            "the counterfactual generation cache). For scripts/train_model.py, use "
            "a new --output-dir without --resume-from-checkpoint."
        )
    if not check_compatibility:
        return
    saved_compatibility = progress.get("compatibility")
    if not isinstance(saved_compatibility, dict) or not isinstance(expected_compatibility, dict):
        raise ValueError(
            "checkpoint is incompatible with the current processing and training setup: "
            f"checkpoint={saved_compatibility!r}, current={expected_compatibility!r}. "
            "Use a new OUTPUT_ROOT; do not delete this checkpoint to reuse it."
        )
    extension_error = _schedule_extension_error(
        saved_compatibility.get("schedule"), expected_compatibility.get("schedule")
    )
    tokenizer_error = _tokenizer_identity_error(saved_compatibility, expected_compatibility)
    if (
        tokenizer_error is not None
        or extension_error is not None
        or _compared_compatibility(saved_compatibility)
        != _compared_compatibility(expected_compatibility)
    ):
        detail = tokenizer_error or extension_error or (
            "configuration that affects inputs, bucketing, loss, sampling, or model "
            "structure changed"
        )
        raise ValueError(
            "checkpoint is incompatible with the current processing and training setup: "
            f"{detail}. checkpoint={saved_compatibility!r}, current={expected_compatibility!r}. "
            "Use a new OUTPUT_ROOT; do not delete this checkpoint to reuse it."
        )


def build_experiment(name: str) -> Experiment:
    """Resolve the required baseline/full/ablation matrix."""
    base = Experiment(name, "qwen35", False, False, False, False, False, False, False, True)
    if name == "B0":
        return replace(base, backbone="majority_mean")
    if name == "B1":
        return replace(base, backbone="hfl/chinese-roberta-wwm-ext")
    if name == "B2":
        return replace(base, backbone="thunlp/Lawformer")
    if name == "B3":
        return base
    if name == "B4":
        return replace(base, use_factors=True)
    if name == "B5":
        return replace(base, use_factors=True, ordinary_augmentation=True)
    full = replace(
        base,
        name="M",
        use_factors=True,
        typed_counterfactuals=True,
        use_invariant=True,
        use_boundary=True,
        use_rank=True,
    )
    if name == "M":
        return full
    ablations = {
        "A1": replace(full, name="A1", use_invariant=False),
        "A2": replace(full, name="A2", use_boundary=False),
        "A3": replace(full, name="A3", use_rank=False),
        "A4": replace(full, name="A4", use_factors=False),
        "A5": replace(full, name="A5", hard_charge_sentence=True),
        "A6": replace(full, name="A6", filter_counterfactuals=False),
    }
    try:
        return ablations[name]
    except KeyError as exc:
        raise ValueError(f"unknown experiment {name!r}") from exc


def inspect_model_config(model_path: str | Path) -> InspectedModelConfig:
    """Read local config.json before selecting a Transformers loader class."""
    path = Path(model_path) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"model config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    architectures = tuple(str(item) for item in config.get("architectures") or ())
    conditional = any("ForConditionalGeneration" in item for item in architectures)
    causal = any("ForCausalLM" in item for item in architectures)
    serialized = json.dumps(config, ensure_ascii=False).lower()
    visual = any(key in serialized for key in ('"vision_config"', '"visual"', '"image_token'))
    hidden_size = config.get("hidden_size") or config.get("text_config", {}).get("hidden_size")
    if conditional and visual:
        loader_class = "AutoModelForImageTextToText"
        architecture_kind = "conditional_generation"
        text_backbone_path = ("model", "language_model")
    elif causal:
        loader_class = "AutoModelForCausalLM"
        architecture_kind = "causal_lm"
        text_backbone_path = ("model",)
    else:
        loader_class = "AutoModel"
        architecture_kind = "base_model"
        text_backbone_path = ()
    return InspectedModelConfig(
        model_type=str(config.get("model_type", "unknown")),
        architectures=architectures,
        loader_class=loader_class,
        architecture_kind=architecture_kind,
        text_backbone_path=text_backbone_path,
        has_visual_components=visual,
        hidden_size=int(hidden_size) if hidden_size is not None else None,
    )


def extract_text_backbone(model: Any, inspected: InspectedModelConfig) -> Any:
    """Extract the text transformer selected from the checkpoint architecture."""
    current = model
    for attribute in inspected.text_backbone_path:
        if not hasattr(current, attribute):
            path = ".".join(inspected.text_backbone_path)
            raise RuntimeError(
                f"{inspected.architecture_kind} model has no expected text backbone {path!r}"
            )
        current = getattr(current, attribute)
    # PEFT uses wrapper-level quantization flags for checkpointing and LoRA dispatch.
    if current is not model:
        for attribute in (
            "is_loaded_in_4bit", "is_loaded_in_8bit", "is_quantized", "quantization_method",
        ):
            if hasattr(model, attribute):
                setattr(current, attribute, getattr(model, attribute))
    return current


def training_plan(config: dict[str, Any], *, experiment: str) -> dict[str, Any]:
    resolved = build_experiment(experiment)
    return {
        "experiment": asdict(resolved),
        "mode": "full" if experiment == "M" else "baseline_or_ablation",
        "model_path": config["model"]["path"],
        "quantization": config["model"].get("quantization", "none"),
        "mixed_precision": config.get("training", {}).get("mixed_precision", "bf16"),
        "seed": int(config.get("training", {}).get("seed", 42)),
        "requires_execute": True,
    }


def factor_supervision(factors: dict[str, Any] | None) -> tuple[list[float], list[bool]]:
    """Targets and a per-field mask for the five factor outputs.

    Amount is an existence observation: every case is supervised as present or
    absent. Surrender, restitution, confession, and accessory status are boolean
    tasks and are masked while unknown. ``False`` remains a real negative label.
    """
    values = factors or {}
    amount = values.get("amount")
    targets = [1.0 if amount is not None else 0.0]
    mask = [True]
    for key in ("surrender", "restitution", "confession"):
        value = values.get(key)
        known = isinstance(value, bool)
        targets.append(1.0 if value is True else 0.0)
        mask.append(known)
    role = values.get("role")
    targets.append(1.0 if role == "accessory" else 0.0)
    mask.append(role in {"principal", "accessory"})
    return targets, mask


def format_model_input(
    record: dict[str, Any], *, use_factors: bool, max_fact_chars: int | None = None
) -> str:
    """Render explicit input sections shared by all real-model experiments."""
    fact = (
        record.get("fact_strict")
        or record.get("fact_conservative")
        or record.get("fact_raw")
        or ""
    )
    if max_fact_chars is not None:
        fact = focus_fact(
            fact,
            str(record.get("target_defendant") or ""),
            record.get("co_defendants") or (),
            max_fact_chars,
        )
    sections = [
        f"[DATASET]\n{record['dataset']}",
        f"[TARGET_DEFENDANT]\n{record['target_defendant']}",
        f"[FACT]\n{fact}",
    ]
    if use_factors:
        sections.append(
            "[FACTORS]\n"
            + json.dumps(record.get("factors") or {}, ensure_ascii=False, sort_keys=True)
        )
    return "\n\n".join(sections)


def masked_mean_pool(hidden: Any, attention_mask: Any = None) -> Any:
    """Reduce token-level hidden states to one vector per input."""
    if hidden.ndim == 2:
        return hidden
    if hidden.ndim != 3:
        raise ValueError(f"expected 2D or 3D hidden states, got shape {tuple(hidden.shape)}")
    if attention_mask is None:
        return hidden.mean(dim=1)
    weights = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def load_processed_records(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid processed JSON at {path}:{line_number}") from exc
            if limit is not None and len(records) >= limit:
                break
    return records


def _factor_values_match(observed: Any, expected: Any, *, numeric: bool = False) -> bool:
    """Compare one factor without treating bool as a number or None as False."""
    if isinstance(observed, bool) or isinstance(expected, bool):
        return isinstance(observed, bool) and isinstance(expected, bool) and observed == expected
    if observed is None or expected is None:
        return observed is None and expected is None
    if numeric:
        if isinstance(observed, int | float) and isinstance(expected, int | float):
            from legal_landscape.counterfactual.validators import amounts_match

            return amounts_match(float(observed), float(expected))
        return False
    return observed == expected


def _observed_counterfactual_factors(
    row: dict[str, Any], parent: dict[str, Any], text: str
) -> dict[str, Any] | None:
    """Factors read from the final text. Rule targets are not observations.

    When the cache names the fields it claims to have changed, a text that does
    not realize those values is rejected. Extraction failure does not fall back
    to the rule target.
    """
    from legal_landscape.factors.extract import extract_factors

    target = str(row.get("target_defendant") or parent.get("target_defendant") or "")
    others = tuple(row.get("co_defendants") or parent.get("co_defendants") or ())
    observed = extract_factors(text, target_defendant=target, co_defendants=others).to_dict()
    changed = tuple(row.get("changed_fields") or ())
    expected = row.get("target_factors")
    if not changed:
        return observed
    if not isinstance(expected, dict):
        return None
    for field in changed:
        if not _factor_values_match(
            observed.get(field), expected.get(field), numeric=field == "amount"
        ):
            return None
    return observed


def _canonical_parent_text(parent: dict[str, Any]) -> str:
    return str(
        parent.get("fact_strict")
        or parent.get("fact_conservative")
        or parent.get("fact_raw")
        or ""
    )


def _reject_counterfactual_cache(path: str | Path, line_number: int, reason: str) -> None:
    raise ValueError(
        f"counterfactual cache at {path}:{line_number} cannot be used ({reason}). "
        "Regenerate the counterfactual cache with the current prompt, sanitizer, and "
        "validator; the existing file was not modified."
    )


def _require_current_counterfactual(
    path: str | Path, line_number: int, row: dict[str, Any], parent: dict[str, Any]
) -> None:
    """Reject old, incomplete, or parent-mismatched cache rows before they can train."""
    from legal_landscape.counterfactual.generate import text_sha256
    from legal_landscape.counterfactual.prompts import PROMPT_VERSION
    from legal_landscape.counterfactual.validators import VALIDATOR_VERSION
    from legal_landscape.data.sanitize import SANITIZER_VERSION

    versions = {
        "prompt_version": PROMPT_VERSION,
        "sanitizer_version": SANITIZER_VERSION,
        "validator_version": VALIDATOR_VERSION,
    }
    for field, expected in versions.items():
        if row.get(field) != expected:
            _reject_counterfactual_cache(
                path,
                line_number,
                f"{field} is {row.get(field)!r}, expected {expected!r}",
            )
    parent_text = _canonical_parent_text(parent)
    if row.get("parent_text_sha256") != text_sha256(parent_text):
        _reject_counterfactual_cache(
            path, line_number, "parent_text_sha256 does not match the current parent text"
        )
    defendant = row.get("target_defendant")
    if not defendant or defendant != parent.get("target_defendant"):
        _reject_counterfactual_cache(
            path, line_number, "target defendant does not match the current parent case"
        )
    if not isinstance(row.get("intervention_type"), str) or not row.get("intervention_type"):
        _reject_counterfactual_cache(path, line_number, "intervention_type is missing")
    changed = row.get("changed_fields")
    fields_ok = (
        isinstance(changed, list)
        and bool(changed)
        and all(isinstance(item, str) for item in changed)
    )
    if not fields_ok:
        _reject_counterfactual_cache(path, line_number, "changed_fields are missing")
    if not isinstance(row.get("target_factors"), dict):
        _reject_counterfactual_cache(path, line_number, "target_factors are missing")


def load_counterfactual_pairs(
    path: str | Path,
    parents: dict[str, dict[str, Any]],
    *,
    filter_valid: bool,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Join generated counterfactual records back to original cases by ID.

    Only the sanitized ``counterfactual_strict`` text is used. Records with label or outcome
    leakage are skipped even when ``filter_valid`` is false. Version, parent-text, and
    intervention identity mismatches are errors: the cache file is left unchanged.
    """
    from legal_landscape.counterfactual.validators import has_leakage

    pairs: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
                parent_id = str(row["parent_case_id"])
                if parent_id not in parents:
                    raise ValueError(
                        f"parent case {parent_id} is not among the supplied parent records; "
                        "counterfactuals must come from the same split as their parents"
                    )
                parent = parents[parent_id]
                _require_current_counterfactual(path, line_number, row, parent)
                validation = row.get("validation") or {}
                if filter_valid and not validation.get("valid", False):
                    continue
                if row.get("leakage") or has_leakage(validation.get("errors") or ()):
                    continue
                counterfactual_text = row.get("counterfactual_strict")
                if not isinstance(counterfactual_text, str) or not counterfactual_text:
                    continue
                observed = _observed_counterfactual_factors(row, parent, counterfactual_text)
                if observed is None:
                    continue
                if row.get("intervention_type") == "sentence_rank":
                    from legal_landscape.counterfactual.interventions import (
                        sentence_rank_exclusion,
                    )

                    parent_factors = parent.get("factors") or {}
                    reason = sentence_rank_exclusion(
                        charges=tuple(parent.get("charges") or ()),
                        penalty_type=parent.get("penalty_type"),
                        imprisonment_months=parent.get("imprisonment_months"),
                        restitution=(
                            parent_factors.get("restitution")
                            if isinstance(parent_factors, dict)
                            else None
                        ),
                    )
                    if reason is not None:
                        continue
                pairs.append(
                    {
                        "parent_record": parent,
                        "parent_text": parent.get("fact_strict")
                        or parent.get("fact_conservative")
                        or parent.get("fact_raw")
                        or "",
                        "counterfactual_text": counterfactual_text,
                        "counterfactual_factors": observed,
                        "expected_factors": row.get("target_factors")
                        if isinstance(row.get("target_factors"), dict)
                        else {},
                        "intervention_type": row["intervention_type"],
                        "target_charge": row.get("target_charge"),
                        "rank_direction": row.get("rank_direction"),
                        "generation_id": row.get("generation_id"),
                    }
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid counterfactual record at {path}:{line_number}: {exc}"
                ) from exc
            if limit is not None and len(pairs) >= limit:
                break
    return pairs


def counterfactual_record(pair: dict[str, Any]) -> dict[str, Any]:
    """The parent record with the sanitized counterfactual text and intervened factors."""
    return {
        **pair["parent_record"],
        "fact_conservative": pair["counterfactual_text"],
        "fact_strict": pair["counterfactual_text"],
        "factors": pair.get("counterfactual_factors", pair["parent_record"].get("factors")),
    }


def augmentation_records(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ordinary-augmentation records; their penalty and sentence are unknown, so masked."""
    records = []
    for pair in pairs:
        record = counterfactual_record(pair)
        if pair["intervention_type"] == "charge_flip" and pair.get("target_charge"):
            record["charges"] = [pair["target_charge"]]
            record["conviction_articles"] = []
        record["penalty_type"] = "unknown"
        record["imprisonment_months"] = None
        record["mask_penalty"] = True
        records.append(record)
    return records


def make_static_prediction_rows(
    records: list[dict[str, Any]],
    *,
    charge_probabilities: list[list[float]],
    article_probabilities: list[list[float]],
    penalty_indices: list[int],
    sentence_months: list[float],
    charge_vocabulary: list[str],
    article_vocabulary: list[str],
    penalty_vocabulary: list[str],
    threshold: float = 0.5,
    article_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Build evaluation JSON rows while retaining original case/group IDs."""
    article_cutoff = threshold if article_threshold is None else article_threshold
    if not (
        len(records)
        == len(charge_probabilities)
        == len(article_probabilities)
        == len(penalty_indices)
        == len(sentence_months)
    ):
        raise ValueError("prediction arrays must match the number of records")
    charge_to_index = {label: index for index, label in enumerate(charge_vocabulary)}
    article_to_index = {label: index for index, label in enumerate(article_vocabulary)}
    rows: list[dict[str, Any]] = []
    for record, probabilities, article_probs, penalty_index, predicted_months in zip(
        records,
        charge_probabilities,
        article_probabilities,
        penalty_indices,
        sentence_months,
        strict=True,
    ):
        if len(probabilities) != len(charge_vocabulary):
            raise ValueError("charge probability width does not match vocabulary")
        if len(article_probs) != len(article_vocabulary):
            raise ValueError("article probability width does not match vocabulary")
        # The vocabulary comes from the training split; held-out labels outside it can never
        # be predicted, so they are reported and count against exact-match accuracy.
        true_charges = [0] * len(charge_vocabulary)
        unseen_charges = [item for item in record["charges"] if item not in charge_to_index]
        for charge in record["charges"]:
            if charge in charge_to_index:
                true_charges[charge_to_index[charge]] = 1
        predicted_charges = [int(value >= threshold) for value in probabilities]
        if not any(predicted_charges):
            predicted_charges[max(range(len(probabilities)), key=probabilities.__getitem__)] = 1
        articles_available = (
            record["articles_available"] if "articles_available" in record else None
        )
        predicted_articles = [int(value >= article_cutoff) for value in article_probs]
        if article_probs and not any(predicted_articles):
            predicted_articles[max(range(len(article_probs)), key=article_probs.__getitem__)] = 1
        if articles_available is False:
            # Missing annotations must not become an all-negative article target.
            true_articles = None
            record_articles = ()
            unseen_articles: list[str] = []
            article_correct = None
        else:
            true_articles = [0] * len(article_vocabulary)
            record_articles = record.get("conviction_articles") or ()
            unseen_articles = [item for item in record_articles if item not in article_to_index]
            for article in record_articles:
                if article in article_to_index:
                    true_articles[article_to_index[article]] = 1
            article_correct = predicted_articles == true_articles and not unseen_articles
        rows.append(
            {
                "case_id": record["case_id"],
                "group_id": record["group_id"],
                "charge_vocabulary": list(charge_vocabulary),
                "article_vocabulary": list(article_vocabulary),
                "articles_available": articles_available,
                "true_charges": true_charges,
                "predicted_charges": predicted_charges,
                "charge_probabilities": probabilities,
                "true_charge_labels": list(record["charges"]),
                "unseen_true_charge_labels": unseen_charges,
                "predicted_charge_labels": [
                    label
                    for label, selected in zip(charge_vocabulary, predicted_charges, strict=True)
                    if selected
                ],
                "true_articles": true_articles,
                "predicted_articles": predicted_articles,
                "article_probabilities": article_probs,
                "true_article_labels": list(record_articles),
                "unseen_true_article_labels": unseen_articles,
                "predicted_article_labels": [
                    label
                    for label, selected in zip(
                        article_vocabulary, predicted_articles, strict=True
                    )
                    if selected
                ],
                "penalty_type": record["penalty_type"],
                "predicted_penalty_type": penalty_vocabulary[penalty_index],
                "true_months": record.get("imprisonment_months"),
                "predicted_months": float(predicted_months),
                "correct": predicted_charges == true_charges and not unseen_charges,
                "article_correct": article_correct,
            }
        )
    return rows


def make_counterfactual_prediction_rows(
    pairs: list[dict[str, Any]],
    *,
    parent_probabilities: list[list[float]],
    counterfactual_probabilities: list[list[float]],
    parent_sentences: list[float],
    counterfactual_sentences: list[float],
    charge_vocabulary: list[str],
    charge_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Build type-specific counterfactual evaluation JSON rows."""
    if not (
        len(pairs)
        == len(parent_probabilities)
        == len(counterfactual_probabilities)
        == len(parent_sentences)
        == len(counterfactual_sentences)
    ):
        raise ValueError("counterfactual prediction arrays must match pair count")
    charge_to_index = {label: index for index, label in enumerate(charge_vocabulary)}
    rows: list[dict[str, Any]] = []
    for pair, parent_prob, counterfactual_prob, parent_sentence, counterfactual_sentence in zip(
        pairs,
        parent_probabilities,
        counterfactual_probabilities,
        parent_sentences,
        counterfactual_sentences,
        strict=True,
    ):
        target = pair.get("target_charge")
        rows.append(
            {
                "case_id": pair["parent_record"]["case_id"],
                "group_id": pair["parent_record"]["group_id"],
                "generation_id": pair.get("generation_id"),
                "charge_vocabulary": list(charge_vocabulary),
                "intervention_type": pair["intervention_type"],
                "rule_id": pair.get("rule_id"),
                "target_charge": charge_to_index[target] if target is not None else None,
                "parent_charge_indices": [
                    charge_to_index[label]
                    for label in pair["parent_record"].get("charges") or ()
                    if label in charge_to_index
                ],
                "charge_threshold": charge_threshold,
                "rank_direction": pair.get("rank_direction"),
                "parent_charge_probabilities": parent_prob,
                "counterfactual_charge_probabilities": counterfactual_prob,
                "parent_sentence": float(parent_sentence),
                "counterfactual_sentence": float(counterfactual_sentence),
                "charges": list(pair["parent_record"].get("charges") or ()),
                "penalty_type": pair["parent_record"].get("penalty_type"),
                "imprisonment_months": pair["parent_record"].get("imprisonment_months"),
                "restitution": (pair["parent_record"].get("factors") or {}).get("restitution"),
            }
        )
    return rows


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_backbone(model_config: dict[str, Any]) -> tuple[Any, InspectedModelConfig]:
    """Load the inspected local architecture and attach QLoRA when requested."""
    model_path = Path(model_config["path"])
    inspected = inspect_model_config(model_path)
    try:
        from transformers import (
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            BitsAndBytesConfig,
        )
    except ImportError as exc:
        raise RuntimeError("transformers is required for real-model training") from exc
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": bool(model_config.get("trust_remote_code", False)),
        "attn_implementation": (
            "eager"
            if _is_longformer(inspected)
            else model_config.get("attn_implementation", "sdpa")
        ),
    }
    if model_config.get("quantization") == "nf4":
        import torch

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        if torch.cuda.is_available():
            kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}
    loaders = {
        "AutoModel": AutoModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoModelForImageTextToText": AutoModelForImageTextToText,
    }
    loader = loaders[inspected.loader_class]
    model = loader.from_pretrained(model_path, **kwargs)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if inspected.has_visual_components:
        for module_name, parameter in model.named_parameters():
            if any(token in module_name.lower() for token in ("vision", "visual", "image")):
                parameter.requires_grad_(False)
    model = extract_text_backbone(model, inspected)
    if model_config.get("quantization") == "nf4":
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError as exc:
            raise RuntimeError("peft is required for QLoRA training") from exc
        lora = model_config.get("lora", {})
        gradient_checkpointing = bool(model_config.get("gradient_checkpointing", True))
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=int(lora.get("rank", 32)),
                lora_alpha=int(lora.get("alpha", 64)),
                lora_dropout=float(lora.get("dropout", 0.05)),
                bias="none",
                task_type="FEATURE_EXTRACTION",
                target_modules="all-linear",
            ),
        )
        if gradient_checkpointing and not model.is_gradient_checkpointing:
            raise RuntimeError(
                "gradient checkpointing was requested but is not enabled on the QLoRA "
                "text backbone; check quantization metadata and PEFT model preparation"
            )
    return model, inspected


def run_dummy_train_step(*, seed: int = 42) -> float:
    """Run a real optimizer step against the dummy backbone on CPU."""
    import torch

    from legal_landscape.models.losses import compute_typed_losses
    from legal_landscape.models.predictor import DummyBackbone, LegalLandscapePredictor

    set_deterministic_seed(seed)
    model = LegalLandscapePredictor(
        DummyBackbone(vocab_size=32, hidden_size=16),
        hidden_size=16,
        num_charges=3,
        num_articles=4,
        num_penalty_types=6,
        num_factors=5,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    output = model(torch.tensor([[1, 2, 3], [3, 4, 0]]))
    losses = compute_typed_losses(
        charge_logits=output["charge_logits"],
        charge_targets=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        article_logits=output["article_logits"],
        article_targets=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        ),
        penalty_logits=output["penalty_type_logits"],
        penalty_targets=torch.tensor([0, 0]),
        sentence_predictions=output["sentence_months"],
        sentence_targets=torch.tensor([12.0, 8.0]),
        sentence_mask=torch.tensor([True, True]),
    )
    optimizer.zero_grad()
    losses.total.backward()
    optimizer.step()
    return float(losses.total.detach().item())


_LOSS_VALUE_NAMES = (
    "charge",
    "article",
    "sentence",
    "invariant",
    "boundary",
    "response",
    "factor",
    "total",
)

_PREDICTOR_BACKWARD_OUTPUT_NAMES = (
    "charge_logits",
    "article_logits",
    "penalty_type_logits",
    "sentence_by_charge",
    "sentence_months",
    "factor_logits",
)


@dataclass(frozen=True)
class _RngSnapshot:
    cpu_state: Any
    cuda_state: Any | None
    cuda_device: Any | None


def _capture_rng_state(torch_module: Any, device: Any) -> _RngSnapshot:
    cuda_device = device if getattr(device, "type", None) == "cuda" else None
    cuda_state = None
    if cuda_device is not None and torch_module.cuda.is_available():
        cuda_state = torch_module.cuda.get_rng_state(cuda_device).clone()
    return _RngSnapshot(
        cpu_state=torch_module.get_rng_state().clone(),
        cuda_state=cuda_state,
        cuda_device=cuda_device,
    )


@contextmanager
def _replay_rng_state(torch_module: Any, snapshot: _RngSnapshot):
    current = _capture_rng_state(torch_module, snapshot.cuda_device)
    try:
        torch_module.set_rng_state(snapshot.cpu_state)
        if snapshot.cuda_state is not None:
            torch_module.cuda.set_rng_state(snapshot.cuda_state, device=snapshot.cuda_device)
        yield
    finally:
        try:
            torch_module.set_rng_state(current.cpu_state)
        finally:
            if current.cuda_state is not None:
                torch_module.cuda.set_rng_state(current.cuda_state, device=current.cuda_device)


def _distributed_parent_requirements(
    torch_module: Any,
    accelerator: Any,
    pair_types: tuple[str, ...],
    device: Any,
) -> tuple[bool, bool]:
    local_requirements = torch_module.tensor(
        (
            int("invariant" in pair_types),
            int("sentence_rank" in pair_types),
        ),
        device=device,
        dtype=torch_module.int64,
    )
    global_requirements = accelerator.reduce(local_requirements, reduction="sum")
    return (
        bool(global_requirements[0].item()),
        bool(global_requirements[1].item()),
    )


def _zero_output_anchor(outputs: dict[str, Any]) -> Any:
    anchor = outputs[_PREDICTOR_BACKWARD_OUTPUT_NAMES[0]].sum() * 0.0
    for name in _PREDICTOR_BACKWARD_OUTPUT_NAMES[1:]:
        anchor = anchor + outputs[name].sum() * 0.0
    return anchor


def _backward_recomputed_parent(
    torch_module: Any,
    outputs: dict[str, Any],
    *,
    charge_gradient: Any | None,
    sentence_gradient: Any | None,
) -> None:
    gradients = tuple(
        (
            charge_gradient
            if name == "charge_logits" and charge_gradient is not None
            else sentence_gradient
            if name == "sentence_months" and sentence_gradient is not None
            else torch_module.zeros_like(outputs[name])
        )
        for name in _PREDICTOR_BACKWARD_OUTPUT_NAMES
    )
    torch_module.autograd.backward(
        tuple(outputs[name] for name in _PREDICTOR_BACKWARD_OUTPUT_NAMES),
        grad_tensors=gradients,
    )


def _detached_loss_values(losses: Any) -> dict[str, float]:
    return {name: float(getattr(losses, name).detach()) for name in _LOSS_VALUE_NAMES}


def _sum_loss_values(*parts: dict[str, float]) -> dict[str, float]:
    if not parts:
        raise ValueError("at least one loss-value mapping is required")
    return {name: sum(part[name] for part in parts) for name in _LOSS_VALUE_NAMES}


def validate_loss_breakdown(losses: Any, diagnostic_path: str | Path, *, step: int) -> None:
    """Fail fast on NaN/Inf or a nonzero loss with no eligible samples."""
    errors: list[str] = []
    values: dict[str, float] = {}
    for name in _LOSS_VALUE_NAMES:
        value = float(getattr(losses, name).detach().item())
        values[name] = value
        if not math.isfinite(value):
            errors.append(name)
        if name != "total" and losses.active_counts.get(name, 0) == 0 and value != 0.0:
            errors.append(f"{name}:nonzero_without_samples")
    if errors:
        destination = Path(diagnostic_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": step,
            "errors": errors,
            "losses": values,
            "active_counts": losses.active_counts,
        }
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        raise FloatingPointError(f"invalid loss state at step {step}: {', '.join(errors)}")


def _fit_majority_mean(records: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    charge_counts: dict[str, int] = {}
    article_counts: dict[str, int] = {}
    finite_terms: list[float] = []
    penalty_counts: dict[str, int] = {}
    for record in records:
        for charge in record["charges"]:
            charge_counts[charge] = charge_counts.get(charge, 0) + 1
        for article in record.get("conviction_articles") or ():
            article_counts[article] = article_counts.get(article, 0) + 1
        penalty = str(record["penalty_type"])
        penalty_counts[penalty] = penalty_counts.get(penalty, 0) + 1
        if penalty == "fixed_term" and record.get("imprisonment_months") is not None:
            finite_terms.append(float(record["imprisonment_months"]))
    payload = {
        "majority_charge": max(charge_counts, key=charge_counts.get),
        "majority_article": max(article_counts, key=article_counts.get),
        "majority_penalty_type": max(penalty_counts, key=penalty_counts.get),
        "mean_imprisonment_months": (
            sum(finite_terms) / len(finite_terms) if finite_terms else 0.0
        ),
        "training_units": len(records),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _training_progress_start(step: int, max_steps: int) -> tuple[int, int]:
    total = max(0, max_steps)
    return min(max(0, step), total), total


def _progress_enabled_for_process(*, is_main_process: bool) -> bool:
    return is_main_process


@contextmanager
def _main_accumulation_context(accelerator: Any, model: Any, should_sync: bool):
    """Skip DDP allreduce except on a main-batch update boundary.

    ``accelerator.backward`` still divides every loss by the configured accumulation
    steps, including the short final window. That matches Accelerate's own tail scaling.
    """
    if should_sync or not getattr(accelerator, "use_distributed", False):
        yield
        return
    with accelerator.no_sync(model):
        yield


def _loader_generator(seed: int, epoch: int, stream: str):
    """RNG used only to build a DataLoader iterator.

    Iterator creation draws one integer. Keeping that draw off the global CPU generator
    leaves dropout on the same sequence after a checkpoint restores that global state.
    The seed is a function of the training seed, epoch, and stream, so recreating an
    iterator for the same epoch does not depend on how many iterators were built earlier.
    """
    import torch

    generator = torch.Generator(device="cpu")
    stream_code = {"main": 1, "counterfactual": 2, "prediction": 3}[stream]
    generator.manual_seed((int(seed) + 1) * 1_000_003 + int(epoch) * 97 + stream_code)
    return generator


def _use_loader_generator(loader: Any, seed: int, epoch: int, stream: str) -> None:
    target = getattr(loader, "base_dataloader", loader)
    target.generator = _loader_generator(seed, epoch, stream)


def _untracked_counterfactual_loader(
    dataset: Any,
    batch_sampler: Any,
    collate_fn: Callable[[list[dict[str, Any]]], dict[str, Any]],
    accelerator: Any,
) -> Any:
    """Yield counterfactual samples without joining Accelerate's gradient-state stack.

    Preparing this loader would make its ``end_of_dataloader`` flag the active one and
    could sync or skip the main tail. Ranks still see the same shard a prepared loader
    would have produced.
    """
    num_processes = int(getattr(accelerator, "num_processes", 1) or 1)
    process_index = int(getattr(accelerator, "process_index", 0) or 0)
    sampler = batch_sampler
    if num_processes > 1:
        from accelerate.data_loader import BatchSamplerShard

        sampler = BatchSamplerShard(
            batch_sampler,
            num_processes=num_processes,
            process_index=process_index,
            split_batches=False,
            even_batches=True,
        )
    from torch.utils.data import DataLoader

    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)


def _report_optimizer_step(
    task: ProgressTask, step: int, max_steps: int, total_loss: float
) -> None:
    completed, total = _training_progress_start(step, max_steps)
    task.update(completed=completed, total=total, step=step, loss=total_loss)


def _prediction_progress_total(loader: Sized) -> int:
    return len(loader)


def _report_prediction_batch(task: ProgressTask, kind: str, batches: int) -> None:
    task.advance(kind=kind, batches=batches)


def run_real_training(
    config: dict[str, Any],
    *,
    train_data: str | Path,
    output_dir: str | Path,
    experiment_name: str,
    counterfactual_data: str | Path | None = None,
    evaluation_data: str | Path | None = None,
    evaluation_counterfactual_data: str | Path | None = None,
    validation_data: str | Path | None = None,
    prediction_dir: str | Path | None = None,
    limit: int | None = None,
    resume_from_checkpoint: str | Path | None = None,
    progress_factory: Callable[..., ProgressReporter] = create_progress_reporter,
) -> dict[str, Any]:
    """Run a local-only Accelerate loop for baseline, full, or ablation training."""
    seed = int(config.get("training", {}).get("seed", 42))
    input_identity: dict[str, Any] = {"train": _file_identity(train_data)}
    if counterfactual_data is not None:
        input_identity["counterfactual"] = _file_identity(counterfactual_data)
    step = 0
    completed_microbatches = 0
    progress: dict[str, Any] = {}
    progress_path: Path | None = None
    if resume_from_checkpoint is not None:
        progress_path = Path(resume_from_checkpoint) / "training_progress.json"
        if not progress_path.is_file():
            raise FileNotFoundError(f"checkpoint progress not found: {progress_path}")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        try:
            validate_resume_metadata(
                progress,
                expected_seed=seed,
                expected_input_identity=input_identity,
                check_compatibility=False,
            )
        except ValueError as exc:
            raise ValueError(f"{progress_path}: {exc}") from exc

    records = load_processed_records(train_data, limit=limit)
    if not records:
        raise ValueError("training data contains no records")
    metadata_path = Path(train_data).parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    charges = list(metadata["charge_vocabulary"])
    articles = list(metadata.get("article_vocabulary") or ())
    if not articles:
        articles = sorted(
            {
                str(article)
                for record in records
                for article in (record.get("conviction_articles") or ())
            }
        )
    if not articles:
        raise ValueError(
            "training data has no conviction-article labels; rebuild the processed dataset"
        )
    penalties = list(metadata["penalty_vocabulary"])
    compatibility = resume_compatibility(
        config=config,
        experiment_name=experiment_name,
        limit=limit,
        num_processes=resume_world_size(),
        charge_vocabulary=charges,
        article_vocabulary=articles,
        penalty_vocabulary=penalties,
        validation_identity=(
            _file_identity(validation_data) if validation_data is not None else None
        ),
    )
    if resume_from_checkpoint is not None:
        try:
            validate_resume_metadata(
                progress,
                expected_seed=seed,
                expected_input_identity=input_identity,
                expected_compatibility=compatibility,
            )
        except ValueError as exc:
            raise ValueError(f"{progress_path}: {exc}") from exc
        step = int(progress.get("steps", 0))
        completed_microbatches = int(progress.get("completed_microbatches", 0))

    try:
        import torch
        from accelerate import Accelerator
        from torch import nn
        from torch.utils.data import DataLoader
        from torch.utils.tensorboard import SummaryWriter
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "real training requires torch, accelerate, transformers, and tensorboard"
        ) from exc

    from legal_landscape.models.losses import compute_typed_losses
    from legal_landscape.models.predictor import LegalLandscapePredictor

    set_deterministic_seed(seed)
    experiment = build_experiment(experiment_name)
    destination = Path(output_dir)
    charge_to_index = {label: index for index, label in enumerate(charges)}
    article_to_index = {label: index for index, label in enumerate(articles)}
    penalty_to_index = {label: index for index, label in enumerate(penalties)}
    parents = {str(record["case_id"]): record for record in records}
    evaluation_records = (
        load_processed_records(evaluation_data, limit=limit) if evaluation_data is not None else []
    )
    validation_records: list[dict[str, Any]] = []
    if validation_data is not None:
        validation_records = load_processed_records(validation_data)
        validation_limit = int(config.get("training", {}).get("validation_limit", 2000))
        if 0 < validation_limit < len(validation_records):
            validation_records = random.Random(seed).sample(validation_records, validation_limit)
    training_groups = {str(record["group_id"]) for record in records}
    for name, held_out in (("evaluation", evaluation_records), ("validation", validation_records)):
        shared_groups = training_groups & {str(record["group_id"]) for record in held_out}
        if shared_groups:
            raise ValueError(
                f"training and {name} records share group_id values, e.g. "
                f"{sorted(shared_groups)[0]}; rebuild the processed dataset"
            )
    pairs: list[dict[str, Any]] = []
    if counterfactual_data is not None:
        pairs = load_counterfactual_pairs(
            counterfactual_data,
            parents,
            filter_valid=experiment.filter_counterfactuals,
            limit=limit,
        )
    # Counterfactual metrics come only from held-out parents, never from training pairs.
    evaluation_pairs: list[dict[str, Any]] = []
    if evaluation_counterfactual_data is not None:
        evaluation_pairs = load_counterfactual_pairs(
            evaluation_counterfactual_data,
            {str(record["case_id"]): record for record in evaluation_records},
            filter_valid=True,
            limit=limit,
        )

    def known_charge_pairs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            pair
            for pair in items
            if pair["intervention_type"] != "charge_flip"
            or pair.get("target_charge") in charge_to_index
        ]

    evaluation_pairs = known_charge_pairs(evaluation_pairs)
    pairs = known_charge_pairs(pairs)
    if experiment.backbone == "majority_mean":
        result = _fit_majority_mean(records, destination)
        majority_charge = str(result["majority_charge"])
        majority_article = str(result["majority_article"])
        majority_penalty = str(result["majority_penalty_type"])
        majority_probability = [float(charge == majority_charge) for charge in charges]
        majority_article_probability = [
            float(article == majority_article) for article in articles
        ]
        prediction_root = Path(prediction_dir or destination / "predictions")
        if evaluation_data is not None:
            static_path = prediction_root / "static.jsonl"
            _write_jsonl(
                static_path,
                make_static_prediction_rows(
                    evaluation_records,
                    charge_probabilities=[majority_probability] * len(evaluation_records),
                    article_probabilities=[majority_article_probability]
                    * len(evaluation_records),
                    penalty_indices=[penalties.index(majority_penalty)] * len(evaluation_records),
                    sentence_months=[float(result["mean_imprisonment_months"])]
                    * len(evaluation_records),
                    charge_vocabulary=charges,
                    article_vocabulary=articles,
                    penalty_vocabulary=penalties,
                ),
            )
            result["static_predictions"] = str(static_path)
        if evaluation_pairs:
            counterfactual_path = prediction_root / "counterfactual.jsonl"
            _write_jsonl(
                counterfactual_path,
                make_counterfactual_prediction_rows(
                    evaluation_pairs,
                    parent_probabilities=[majority_probability] * len(evaluation_pairs),
                    counterfactual_probabilities=[majority_probability] * len(evaluation_pairs),
                    parent_sentences=[float(result["mean_imprisonment_months"])]
                    * len(evaluation_pairs),
                    counterfactual_sentences=[float(result["mean_imprisonment_months"])]
                    * len(evaluation_pairs),
                    charge_vocabulary=charges,
                ),
            )
            result["counterfactual_predictions"] = str(counterfactual_path)
        return result
    if experiment.ordinary_augmentation:
        records.extend(augmentation_records(pairs))
        pairs = []
    allowed_pair_types: set[str] = set()
    if experiment.typed_counterfactuals:
        if experiment.use_invariant:
            allowed_pair_types.add("invariant")
        if experiment.use_boundary:
            allowed_pair_types.add("charge_flip")
        if experiment.use_rank:
            allowed_pair_types.add("sentence_rank")
    pairs = [pair for pair in pairs if pair["intervention_type"] in allowed_pair_types]
    if experiment.typed_counterfactuals and not pairs:
        raise ValueError(
            f"{experiment.name} requires at least one eligible counterfactual pair; "
            "check the current request view, validation results, target charges, "
            f"and allowed intervention types {sorted(allowed_pair_types)}"
        )

    model_config = dict(config["model"])
    model_config["gradient_checkpointing"] = config.get("training", {}).get(
        "gradient_checkpointing", True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["path"],
        local_files_only=True,
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    text_model, inspected = load_backbone(model_config)

    class TextBackbone(nn.Module):
        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, input_ids: Any, attention_mask: Any = None) -> Any:
            forward_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "return_dict": True,
                "use_cache": False,
            }
            output = self.model(**forward_kwargs)
            if not hasattr(output, "last_hidden_state"):
                raise RuntimeError(
                    "selected text backbone did not return last_hidden_state; "
                    "check the local model architecture"
                )
            hidden = output.last_hidden_state
            return masked_mean_pool(hidden, attention_mask)

    hidden_size = inspected.hidden_size or int(text_model.config.hidden_size)
    predictor = LegalLandscapePredictor(
        TextBackbone(text_model),
        hidden_size=hidden_size,
        num_charges=len(charges),
        num_articles=len(articles),
        num_penalty_types=len(penalties),
        num_factors=5,
        hard_charge=experiment.hard_charge_sentence,
    )
    max_length = int(config.get("model", {}).get("max_length", config.get("max_length", 4096)))

    def model_text(record: dict[str, Any]) -> str:
        # Characters bound tokens from above for Chinese text, so focusing to max_length
        # characters keeps the defendant's facts inside the tokenizer's truncation window.
        return format_model_input(
            record, use_factors=experiment.use_factors, max_fact_chars=max_length
        )

    def text_length(record: dict[str, Any]) -> int:
        return min(len(model_text(record)), max_length)

    def tokenize_texts(texts: list[str]) -> dict[str, Any]:
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def collate_cases(batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenize_texts(
            [model_text(item) for item in batch]
        )
        charge_targets = torch.zeros((len(batch), len(charges)), dtype=torch.float32)
        article_targets = torch.zeros((len(batch), len(articles)), dtype=torch.float32)
        article_mask = torch.zeros(len(batch), dtype=torch.bool)
        penalty_targets = torch.zeros(len(batch), dtype=torch.long)
        penalty_mask = torch.ones(len(batch), dtype=torch.bool)
        sentence_targets = torch.zeros(len(batch), dtype=torch.float32)
        sentence_mask = torch.zeros(len(batch), dtype=torch.bool)
        factor_targets = torch.zeros((len(batch), 5), dtype=torch.float32)
        factor_mask = torch.zeros((len(batch), 5), dtype=torch.bool)
        for index, item in enumerate(batch):
            # Evaluation labels outside the training vocabulary have no output unit.
            for charge in item["charges"]:
                if charge in charge_to_index:
                    charge_targets[index, charge_to_index[charge]] = 1.0
            item_articles = [
                article
                for article in item.get("conviction_articles") or ()
                if article in article_to_index
            ]
            for article in item_articles:
                article_targets[index, article_to_index[article]] = 1.0
            article_mask[index] = (
                bool(item_articles) and item.get("articles_available") is not False
            )
            penalty_targets[index] = penalty_to_index[item["penalty_type"]]
            penalty_mask[index] = not item.get("mask_penalty", False)
            eligible = (
                item["penalty_type"] == "fixed_term"
                and item.get("imprisonment_months") is not None
                and len(item["charges"]) == 1
            )
            if eligible:
                sentence_targets[index] = float(item["imprisonment_months"])
                sentence_mask[index] = True
            factors = item.get("factors") or {}
            supervised_targets, supervised_mask = factor_supervision(factors)
            factor_targets[index] = torch.tensor(supervised_targets)
            factor_mask[index] = torch.tensor(supervised_mask)
        result = {
            **encoded,
            "charge_targets": charge_targets,
            "article_targets": article_targets,
            "article_mask": article_mask,
            "penalty_targets": penalty_targets,
            "penalty_mask": penalty_mask,
            "sentence_targets": sentence_targets,
            "sentence_mask": sentence_mask,
            "factor_targets": factor_targets,
            "factor_mask": factor_mask,
        }
        if all("_prediction_index" in item for item in batch):
            result["record_indices"] = torch.tensor(
                [int(item["_prediction_index"]) for item in batch], dtype=torch.long
            )
        return result

    def collate_pairs(batch: list[dict[str, Any]]) -> dict[str, Any]:
        parent_records = [item["parent_record"] for item in batch]
        cf_records = [counterfactual_record(item) for item in batch]
        result = {
            "parent": tokenize_texts(
                [
                    model_text(item)
                    for item in parent_records
                ]
            ),
            "counterfactual": tokenize_texts(
                [
                    model_text(item)
                    for item in cf_records
                ]
            ),
            "pair_types": tuple(str(item["intervention_type"]) for item in batch),
            "target_charge_indices": torch.tensor(
                [charge_to_index.get(str(item.get("target_charge")), 0) for item in batch],
                dtype=torch.long,
            ),
            "rank_direction": torch.tensor(
                [float(item.get("rank_direction") or 0) for item in batch], dtype=torch.float32
            ),
        }
        if all("_prediction_index" in item for item in batch):
            result["pair_indices"] = torch.tensor(
                [int(item["_prediction_index"]) for item in batch], dtype=torch.long
            )
        return result

    training = config.get("training", {})
    batch_size = int(training.get("per_device_batch_size", 1))
    max_tokens = int(training.get("max_tokens_per_batch", batch_size * max_length))
    inference_batch_size = int(training.get("inference_batch_size", batch_size))
    inference_max_tokens = int(training.get("inference_max_tokens_per_batch", max_tokens))
    pair_every = max(1, int(training.get("pair_every", 1)))
    case_sampler = TokenBudgetBatchSampler(
        [text_length(record) for record in records], max_tokens, batch_size, seed=seed
    )
    pair_sampler = (
        TokenBudgetBatchSampler(
            [
                max(text_length(pair["parent_record"]), text_length(counterfactual_record(pair)))
                for pair in pairs
            ],
            max_tokens,
            batch_size,
            seed=seed + 1,
        )
        if pairs
        else None
    )
    if progress:
        case_sampler.load_state_dict(progress["case_sampler"])
        if pair_sampler is not None and progress.get("pair_sampler") is not None:
            pair_sampler.load_state_dict(progress["pair_sampler"])

    loader = DataLoader(
        records,
        batch_sampler=case_sampler,
        collate_fn=collate_cases,
        generator=_loader_generator(seed, case_sampler.epoch, "main"),
    )
    pair_loader = (
        DataLoader(
            pairs,
            batch_sampler=pair_sampler,
            collate_fn=collate_pairs,
            generator=_loader_generator(seed, pair_sampler.epoch, "counterfactual"),
        )
        if pair_sampler is not None
        else None
    )
    mixed_precision = str(training.get("mixed_precision", "bf16"))
    gradient_accumulation_steps = int(training.get("gradient_accumulation_steps", 8))
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    base_learning_rate = float(training.get("learning_rate", 2e-4))
    optimizer = torch.optim.AdamW(
        (parameter for parameter in predictor.parameters() if parameter.requires_grad),
        lr=base_learning_rate,
    )
    predictor, optimizer, loader = accelerator.prepare(predictor, optimizer, loader)
    if pair_sampler is not None:
        # Do not prepare this loader. Its iterator must not become the active
        # gradient-state dataloader or it will move optimizer boundaries.
        pair_loader = _untracked_counterfactual_loader(
            pairs, pair_sampler, collate_pairs, accelerator
        )
    if resume_from_checkpoint is not None:
        if int(accelerator.num_processes) != int(compatibility["num_processes"]):
            raise ValueError(
                "refusing to resume under a different process count: "
                f"checkpoint={compatibility['num_processes']}, "
                f"current={accelerator.num_processes}. Sampler cursors are not valid "
                "across a world-size change; start a new run."
            )
        accelerator.load_state(str(resume_from_checkpoint))
    destination.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(destination / "tensorboard") if accelerator.is_main_process else None
    log_path = destination / "train.jsonl"
    epochs = int(training.get("epochs", 1))
    steps_per_epoch = math.ceil(
        case_sampler.batches_per_epoch / accelerator.num_processes / gradient_accumulation_steps
    )
    max_steps = epochs * steps_per_epoch
    configured_max_steps = int(training.get("max_steps", 1000))
    if configured_max_steps > 0:
        max_steps = min(max_steps, configured_max_steps)
    warmup_steps = round(float(training.get("warmup_ratio", 0.0)) * max_steps)
    min_learning_rate_ratio = float(training.get("min_learning_rate_ratio", 1.0))
    checkpoint_every = int(training.get("checkpoint_every", 100))
    if pair_loader is not None and pair_sampler is not None:
        _use_loader_generator(pair_loader, seed, pair_sampler.epoch, "counterfactual")
    pair_iterator = iter(pair_loader) if pair_loader is not None else None
    loss_config = dict(config.get("loss", {}))
    multilabel_reduction = str(loss_config.pop("multilabel_reduction", "sum"))
    weights = {name: float(value) for name, value in loss_config.items()}

    def save_checkpoint(path: Path) -> None:
        accelerator.wait_for_everyone()
        accelerator.save_state(str(path))
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            (path / "training_progress.json").write_text(
                json.dumps(
                    {
                        "steps": step,
                        "completed_microbatches": completed_microbatches,
                        "seed": seed,
                        "case_sampler": case_sampler.state_dict(),
                        "pair_sampler": (
                            pair_sampler.state_dict() if pair_sampler is not None else None
                        ),
                        "input_identity": input_identity,
                        "compatibility": compatibility,
                        "dependency_versions": _dependency_versions(),
                        "model_config": asdict(inspected),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            os.utime(path)
        accelerator.wait_for_everyone()

    def device_batch(value: Any) -> Any:
        device = getattr(accelerator, "device", None)
        if isinstance(value, dict):
            return {name: device_batch(item) for name, item in value.items()}
        if device is not None and hasattr(value, "to"):
            return value.to(device)
        return value

    def forward_static(batch: dict[str, Any]) -> list[tuple[int, tuple[Any, ...]]]:
        output = predictor(batch["input_ids"], batch.get("attention_mask"))
        return list(zip(
            batch["record_indices"].tolist(),
            zip(
                torch.sigmoid(output["charge_logits"]).float().cpu().tolist(),
                torch.sigmoid(output["article_logits"]).float().cpu().tolist(),
                output["penalty_type_logits"].argmax(dim=-1).cpu().tolist(),
                output["sentence_months"].float().cpu().tolist(),
                strict=True,
            ),
            strict=True,
        ))

    def forward_pairs(batch: dict[str, Any]) -> list[tuple[int, tuple[Any, ...]]]:
        parent = predictor(batch["parent"]["input_ids"], batch["parent"].get("attention_mask"))
        counterfactual = predictor(
            batch["counterfactual"]["input_ids"], batch["counterfactual"].get("attention_mask")
        )
        return list(zip(
            batch["pair_indices"].tolist(),
            zip(
                torch.sigmoid(parent["charge_logits"]).float().cpu().tolist(),
                torch.sigmoid(counterfactual["charge_logits"]).float().cpu().tolist(),
                parent["sentence_months"].float().cpu().tolist(),
                counterfactual["sentence_months"].float().cpu().tolist(),
                strict=True,
            ),
            strict=True,
        ))

    def predict(
        items: list[dict[str, Any]],
        *,
        kind: str,
        collate: Callable[[list[dict[str, Any]]], dict[str, Any]],
        lengths: list[int],
        forward: Callable[[dict[str, Any]], list[tuple[int, tuple[Any, ...]]]],
        progress_reporter: ProgressReporter,
    ) -> dict[int, tuple[Any, ...]] | None:
        """Length-bucketed prediction; every process runs the same number of batches and the
        main process receives all outputs keyed by item position."""
        from accelerate.utils import gather_object

        indexed = [{**item, "_prediction_index": index} for index, item in enumerate(items)]
        batches = shard_batches(
            inference_batches(lengths, inference_max_tokens, inference_batch_size),
            accelerator.num_processes,
            accelerator.process_index,
        )
        local_loader = DataLoader(
            indexed,
            batch_sampler=batches,
            collate_fn=collate,
            generator=_loader_generator(seed, 0, "prediction"),
        )
        results: list[tuple[int, tuple[Any, ...]]] = []
        with (
            progress_reporter.task(
                f"Predict {kind}",
                total=_prediction_progress_total(local_loader),
                kind=kind,
                batches=0,
            ) as prediction_progress,
            torch.no_grad(),
        ):
            for batches_done, batch in enumerate(local_loader, start=1):
                results.extend(forward(device_batch(batch)))
                _report_prediction_batch(prediction_progress, kind, batches_done)
        gathered = gather_object(results)
        if not accelerator.is_main_process:
            return None
        return {int(index): payload for index, payload in gathered}

    def label_matrix(
        items: list[dict[str, Any]], key: str, vocabulary: dict[str, int]
    ) -> list[list[int]]:
        matrix = []
        for item in items:
            row = [0] * len(vocabulary)
            for label in item.get(key) or ():
                if label in vocabulary:
                    row[vocabulary[label]] = 1
            matrix.append(row)
        return matrix

    def micro_f1(probabilities: list[list[float]], truth: list[list[int]]) -> float:
        import numpy as np

        predicted = np.asarray(probabilities) >= 0.5
        labels = np.asarray(truth, dtype=bool)
        denominator = float(predicted.sum() + labels.sum())
        return 2 * float((predicted & labels).sum()) / denominator if denominator else 0.0

    def validation_outputs(progress_reporter: ProgressReporter) -> list[tuple[Any, ...]] | None:
        predictor.eval()
        try:
            outputs = predict(
                validation_records,
                kind="validation",
                collate=collate_cases,
                lengths=[text_length(record) for record in validation_records],
                forward=forward_static,
                progress_reporter=progress_reporter,
            )
        finally:
            predictor.train()
        if outputs is None:
            return None
        return [outputs[index] for index in range(len(validation_records))]

    def score_checkpoint(path: Path, progress_reporter: ProgressReporter) -> None:
        """Record the mean charge/article micro-F1 on the validation subset and keep only the
        best and latest periodic checkpoints."""
        if not validation_records:
            return
        outputs = validation_outputs(progress_reporter)
        if accelerator.is_main_process and outputs is not None:
            score = (
                micro_f1(
                    [item[0] for item in outputs],
                    label_matrix(validation_records, "charges", charge_to_index),
                )
                + micro_f1(
                    [item[1] for item in outputs],
                    label_matrix(validation_records, "conviction_articles", article_to_index),
                )
            ) / 2
            progress_path = path / "training_progress.json"
            saved = json.loads(progress_path.read_text(encoding="utf-8"))
            saved["validation_score"] = score
            progress_path.write_text(json.dumps(saved, ensure_ascii=False, indent=2), "utf-8")
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step, "validation_score": score}) + "\n")
            if bool(training.get("keep_best_and_latest_checkpoints", True)):
                best = select_best_checkpoint(destination)
                prune_checkpoints(destination, keep={path, *([best] if best else [])})
        accelerator.wait_for_everyone()

    reporter = progress_factory(
        enabled=_progress_enabled_for_process(is_main_process=accelerator.is_main_process)
    )
    initial_step, display_total = _training_progress_start(step, max_steps)
    with reporter as active_progress:
        with active_progress.task(
            f"Train {experiment.name}",
            total=display_total,
            completed=initial_step,
            experiment=experiment.name,
            step=step,
            loss="—",
        ) as train_progress:
            predictor.train()
            for _epoch in range(case_sampler.epoch, epochs) if step < max_steps else ():
                microbatches_in_epoch = 0
                _use_loader_generator(loader, seed, case_sampler.epoch, "main")
                for batch in loader:
                    microbatches_in_epoch += 1
                    # The prepared main loader sets this before yielding its last batch.
                    # Counterfactual iteration must not be consulted here.
                    should_sync = main_accumulation_should_sync(
                        microbatches_in_epoch,
                        gradient_accumulation_steps,
                        at_epoch_end=bool(getattr(loader, "end_of_dataloader", False)),
                    )
                    accelerator.sync_gradients = should_sync
                    with _main_accumulation_context(accelerator, predictor, should_sync):
                        output = predictor(batch["input_ids"], batch.get("attention_mask"))
                        supervised_losses = compute_typed_losses(
                            charge_logits=output["charge_logits"],
                            charge_targets=batch["charge_targets"],
                            article_logits=output["article_logits"],
                            article_targets=batch["article_targets"],
                            article_mask=batch["article_mask"],
                            penalty_logits=output["penalty_type_logits"],
                            penalty_targets=batch["penalty_targets"],
                            penalty_mask=batch.get("penalty_mask"),
                            sentence_predictions=output["sentence_months"],
                            sentence_targets=batch["sentence_targets"],
                            sentence_mask=batch["sentence_mask"],
                            factor_logits=output["factor_logits"]
                            if experiment.use_factors
                            else None,
                            factor_targets=batch["factor_targets"]
                            if experiment.use_factors
                            else None,
                            factor_mask=batch.get("factor_mask")
                            if experiment.use_factors
                            else None,
                            weights=weights,
                            multilabel_reduction=multilabel_reduction,
                        )
                        validate_loss_breakdown(
                            supervised_losses,
                            destination / "loss_diagnostic.json",
                            step=step,
                        )
                        # Inactive or fully masked heads still need DDP gradient hooks.
                        supervised_backward = supervised_losses.total + _zero_output_anchor(output)
                        accelerator.backward(supervised_backward)
                        loss_values = _detached_loss_values(supervised_losses)
                        del output, supervised_backward, supervised_losses
                        # Every process uses the same microbatch counter, so pair steps (and
                        # their DDP backward passes) stay aligned across ranks.
                        if pair_iterator is not None and completed_microbatches % pair_every == 0:
                            try:
                                pair_batch = device_batch(next(pair_iterator))
                            except StopIteration:
                                if pair_sampler is None:
                                    raise RuntimeError(
                                        "counterfactual sampler unexpectedly missing"
                                    ) from None
                                pair_sampler.next_epoch()
                                _use_loader_generator(
                                    pair_loader, seed, pair_sampler.epoch, "counterfactual"
                                )
                                pair_iterator = iter(pair_loader)  # type: ignore[arg-type]
                                pair_batch = device_batch(next(pair_iterator))
                            pair_types = pair_batch["pair_types"]
                            needs_parent_charge, needs_parent_sentence = (
                                _distributed_parent_requirements(
                                    torch,
                                    accelerator,
                                    pair_types,
                                    pair_batch["parent"]["input_ids"].device,
                                )
                            )
                            parent_charge = None
                            parent_sentence = None
                            parent_rng = None
                            if needs_parent_charge or needs_parent_sentence:
                                parent_rng = _capture_rng_state(
                                    torch,
                                    pair_batch["parent"]["input_ids"].device,
                                )
                                with torch.no_grad():
                                    parent_output = predictor(
                                        pair_batch["parent"]["input_ids"],
                                        pair_batch["parent"].get("attention_mask"),
                                    )
                                if needs_parent_charge:
                                    parent_charge = (
                                        parent_output["charge_logits"]
                                        .detach()
                                        .requires_grad_(True)
                                    )
                                if needs_parent_sentence:
                                    parent_sentence = (
                                        parent_output["sentence_months"]
                                        .detach()
                                        .requires_grad_(True)
                                    )
                                del parent_output
                            cf_output = predictor(
                                pair_batch["counterfactual"]["input_ids"],
                                pair_batch["counterfactual"].get("attention_mask"),
                            )
                            paired_losses = compute_typed_losses(
                                pair_types=pair_types,
                                parent_charge_logits=parent_charge,
                                counterfactual_charge_logits=cf_output["charge_logits"],
                                target_charge_indices=pair_batch["target_charge_indices"],
                                parent_sentence=parent_sentence,
                                counterfactual_sentence=cf_output["sentence_months"],
                                rank_direction=pair_batch["rank_direction"],
                                rank_margin=float(training.get("rank_margin", 1.0)),
                                weights=weights,
                                multilabel_reduction=multilabel_reduction,
                            )
                            validate_loss_breakdown(
                                paired_losses,
                                destination / "loss_diagnostic.json",
                                step=step,
                            )
                            paired_backward = paired_losses.total + _zero_output_anchor(cf_output)
                            accelerator.backward(paired_backward)
                            loss_values = _sum_loss_values(
                                loss_values, _detached_loss_values(paired_losses)
                            )
                            parent_charge_gradient = None
                            if parent_charge is not None:
                                parent_charge_gradient = (
                                    parent_charge.grad.detach()
                                    if parent_charge.grad is not None
                                    else torch.zeros_like(parent_charge)
                                )
                            parent_sentence_gradient = None
                            if parent_sentence is not None:
                                parent_sentence_gradient = (
                                    parent_sentence.grad.detach()
                                    if parent_sentence.grad is not None
                                    else torch.zeros_like(parent_sentence)
                                )
                            del cf_output, paired_backward, paired_losses
                            del parent_charge, parent_sentence
                            if (
                                parent_charge_gradient is not None
                                or parent_sentence_gradient is not None
                            ):
                                if parent_rng is None:
                                    raise RuntimeError(
                                        "parent gradients exist without an RNG snapshot"
                                    )
                                with _replay_rng_state(torch, parent_rng):
                                    parent_output = predictor(
                                        pair_batch["parent"]["input_ids"],
                                        pair_batch["parent"].get("attention_mask"),
                                    )
                                    _backward_recomputed_parent(
                                        torch,
                                        parent_output,
                                        charge_gradient=parent_charge_gradient,
                                        sentence_gradient=parent_sentence_gradient,
                                    )
                                del parent_output
                            del parent_charge_gradient, parent_sentence_gradient, parent_rng
                            del pair_batch
                            if pair_sampler is not None:
                                pair_sampler.advance(accelerator.num_processes)
                        completed_optimizer_step = should_sync
                        learning_rate = base_learning_rate * learning_rate_factor(
                            step, max_steps, warmup_steps, min_learning_rate_ratio
                        )
                        for group in optimizer.param_groups:
                            group["lr"] = learning_rate
                        optimizer.step()
                        optimizer.zero_grad()
                    completed_microbatches += 1
                    # Each process consumes one batch per microbatch.
                    case_sampler.advance(accelerator.num_processes)
                    if completed_optimizer_step:
                        step += 1
                        _report_optimizer_step(
                            train_progress, step, max_steps, loss_values["total"]
                        )
                    if completed_optimizer_step and accelerator.is_main_process:
                        log = {"step": step, **loss_values, "learning_rate": learning_rate}
                        with log_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(log) + "\n")
                        if writer is not None:
                            for name, value in log.items():
                                if name != "step":
                                    writer.add_scalar(f"loss/{name}", value, step)
                    if (
                        completed_optimizer_step
                        and checkpoint_every > 0
                        and step % checkpoint_every == 0
                        and step < max_steps
                    ):
                        periodic = destination / f"checkpoint-step-{step:08d}"
                        save_checkpoint(periodic)
                        score_checkpoint(periodic, active_progress)
                    if step >= max_steps:
                        break
                if step >= max_steps:
                    break
                case_sampler.next_epoch()
            final_checkpoint = destination / "checkpoint-final"
            save_checkpoint(final_checkpoint)
            score_checkpoint(final_checkpoint, active_progress)
            if writer is not None:
                writer.close()
        selected_checkpoint = final_checkpoint
        if validation_records and bool(training.get("select_best_checkpoint", True)):
            from accelerate.utils import broadcast_object_list

            choice = [
                str(select_best_checkpoint(destination) or final_checkpoint)
                if accelerator.is_main_process
                else None
            ]
            broadcast_object_list(choice)
            if Path(str(choice[0])) != final_checkpoint:
                accelerator.load_state(str(choice[0]))
                selected_checkpoint = Path(str(choice[0]))
        result: dict[str, Any] = {
            "steps": step,
            "output_dir": str(destination),
            "counterfactual_pairs": len(pairs),
            "evaluation_counterfactual_pairs": len(evaluation_pairs),
            "selected_checkpoint": str(selected_checkpoint),
        }
        prediction_root = Path(prediction_dir or destination / "predictions")
        charge_threshold = article_threshold = 0.5
        if validation_records:
            outputs = validation_outputs(active_progress)
            if accelerator.is_main_process and outputs is not None:
                charge_threshold = tune_threshold(
                    [item[0] for item in outputs],
                    label_matrix(validation_records, "charges", charge_to_index),
                )
                article_threshold = tune_threshold(
                    [item[1] for item in outputs],
                    label_matrix(validation_records, "conviction_articles", article_to_index),
                )
                prediction_root.mkdir(parents=True, exist_ok=True)
                (prediction_root / "thresholds.json").write_text(
                    json.dumps(
                        {
                            "charge": charge_threshold,
                            "article": article_threshold,
                            "validation_records": len(validation_records),
                            "selected_checkpoint": str(selected_checkpoint),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        result["thresholds"] = {"charge": charge_threshold, "article": article_threshold}
        if accelerator.is_main_process:
            (destination / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "experiment": asdict(experiment),
                        "steps": step,
                        "max_steps": max_steps,
                        "completed_microbatches": completed_microbatches,
                        "charge_vocabulary": charges,
                        "article_vocabulary": articles,
                        "penalty_vocabulary": penalties,
                        "counterfactual_pairs": len(pairs),
                        "evaluation_counterfactual_pairs": len(evaluation_pairs),
                        "selected_checkpoint": str(selected_checkpoint),
                        "thresholds": result["thresholds"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        predictor.eval()
        if evaluation_data is not None:
            static_outputs = predict(
                evaluation_records,
                kind="static",
                collate=collate_cases,
                lengths=[text_length(record) for record in evaluation_records],
                forward=forward_static,
                progress_reporter=active_progress,
            )
            if accelerator.is_main_process and static_outputs is not None:
                ordered_static = [static_outputs[index] for index in range(len(evaluation_records))]
                static_path = prediction_root / "static.jsonl"
                _write_jsonl(
                    static_path,
                    make_static_prediction_rows(
                        evaluation_records,
                        charge_probabilities=[item[0] for item in ordered_static],
                        article_probabilities=[item[1] for item in ordered_static],
                        penalty_indices=[item[2] for item in ordered_static],
                        sentence_months=[float(item[3]) for item in ordered_static],
                        charge_vocabulary=charges,
                        article_vocabulary=articles,
                        penalty_vocabulary=penalties,
                        threshold=charge_threshold,
                        article_threshold=article_threshold,
                    ),
                )
                result["static_predictions"] = str(static_path)

        if evaluation_pairs:
            pair_outputs = predict(
                evaluation_pairs,
                kind="counterfactual",
                collate=collate_pairs,
                lengths=[
                    max(
                        text_length(pair["parent_record"]),
                        text_length(counterfactual_record(pair)),
                    )
                    for pair in evaluation_pairs
                ],
                forward=forward_pairs,
                progress_reporter=active_progress,
            )
            if accelerator.is_main_process and pair_outputs is not None:
                ordered_pairs = [pair_outputs[index] for index in range(len(evaluation_pairs))]
                counterfactual_path = prediction_root / "counterfactual.jsonl"
                _write_jsonl(
                    counterfactual_path,
                    make_counterfactual_prediction_rows(
                        evaluation_pairs,
                        parent_probabilities=[item[0] for item in ordered_pairs],
                        counterfactual_probabilities=[item[1] for item in ordered_pairs],
                        parent_sentences=[float(item[2]) for item in ordered_pairs],
                        counterfactual_sentences=[float(item[3]) for item in ordered_pairs],
                        charge_vocabulary=charges,
                        charge_threshold=charge_threshold,
                    ),
                )
                result["counterfactual_predictions"] = str(counterfactual_path)
        accelerator.wait_for_everyone()
    return result
