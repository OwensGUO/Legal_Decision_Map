"""Experiment configuration, model inspection, and safe training helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


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


def validate_resume_metadata(
    progress: dict[str, Any],
    *,
    expected_seed: int,
    expected_input_identity: dict[str, Any],
) -> None:
    """Reject checkpoints produced from a different deterministic run input."""
    if int(progress.get("seed", -1)) != expected_seed:
        raise ValueError(
            f"checkpoint seed mismatch: checkpoint={progress.get('seed')}, current={expected_seed}"
        )
    if progress.get("input_identity") != expected_input_identity:
        raise ValueError("checkpoint input identity does not match the current training inputs")


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


def format_model_input(record: dict[str, Any], *, use_factors: bool) -> str:
    """Render explicit input sections shared by all real-model experiments."""
    sections = [
        f"[DATASET]\n{record['dataset']}",
        f"[TARGET_DEFENDANT]\n{record['target_defendant']}",
        f"[FACT]\n{record.get('fact_conservative') or record.get('fact_raw') or ''}",
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


def load_counterfactual_pairs(
    path: str | Path,
    parents: dict[str, dict[str, Any]],
    *,
    filter_valid: bool,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Join generated counterfactual records back to original cases by ID."""
    pairs: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
                validation = row.get("validation") or {}
                if filter_valid and not validation.get("valid", False):
                    continue
                parent = parents[str(row["parent_case_id"])]
                parsed = validation.get("parsed") or {}
                counterfactual_text = parsed.get("counterfactual_text")
                if not counterfactual_text and not filter_valid:
                    try:
                        counterfactual_text = json.loads(row.get("raw_response") or "{}").get(
                            "counterfactual_text"
                        )
                    except json.JSONDecodeError:
                        counterfactual_text = None
                if not isinstance(counterfactual_text, str) or not counterfactual_text:
                    continue
                pairs.append(
                    {
                        "parent_record": parent,
                        "parent_text": parent.get("fact_conservative")
                        or parent.get("fact_raw")
                        or "",
                        "counterfactual_text": counterfactual_text,
                        "intervention_type": row["intervention_type"],
                        "target_charge": row.get("target_charge"),
                        "rank_direction": row.get("rank_direction"),
                    }
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid counterfactual record at {path}:{line_number}: {exc}"
                ) from exc
            if limit is not None and len(pairs) >= limit:
                break
    return pairs


def make_static_prediction_rows(
    records: list[dict[str, Any]],
    *,
    charge_probabilities: list[list[float]],
    penalty_indices: list[int],
    sentence_months: list[float],
    charge_vocabulary: list[str],
    penalty_vocabulary: list[str],
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Build evaluation JSON rows while retaining original case/group IDs."""
    if not (
        len(records) == len(charge_probabilities) == len(penalty_indices) == len(sentence_months)
    ):
        raise ValueError("prediction arrays must match the number of records")
    charge_to_index = {label: index for index, label in enumerate(charge_vocabulary)}
    rows: list[dict[str, Any]] = []
    for record, probabilities, penalty_index, predicted_months in zip(
        records,
        charge_probabilities,
        penalty_indices,
        sentence_months,
        strict=True,
    ):
        if len(probabilities) != len(charge_vocabulary):
            raise ValueError("charge probability width does not match vocabulary")
        true_charges = [0] * len(charge_vocabulary)
        for charge in record["charges"]:
            true_charges[charge_to_index[charge]] = 1
        predicted_charges = [int(value >= threshold) for value in probabilities]
        if not any(predicted_charges):
            predicted_charges[max(range(len(probabilities)), key=probabilities.__getitem__)] = 1
        rows.append(
            {
                "case_id": record["case_id"],
                "group_id": record["group_id"],
                "true_charges": true_charges,
                "predicted_charges": predicted_charges,
                "charge_probabilities": probabilities,
                "true_charge_labels": list(record["charges"]),
                "predicted_charge_labels": [
                    label
                    for label, selected in zip(charge_vocabulary, predicted_charges, strict=True)
                    if selected
                ],
                "penalty_type": record["penalty_type"],
                "predicted_penalty_type": penalty_vocabulary[penalty_index],
                "true_months": record.get("imprisonment_months"),
                "predicted_months": float(predicted_months),
                "correct": predicted_charges == true_charges,
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
                "intervention_type": pair["intervention_type"],
                "target_charge": charge_to_index[target] if target is not None else None,
                "rank_direction": pair.get("rank_direction"),
                "parent_charge_probabilities": parent_prob,
                "counterfactual_charge_probabilities": counterfactual_prob,
                "parent_sentence": float(parent_sentence),
                "counterfactual_sentence": float(counterfactual_sentence),
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
        "attn_implementation": model_config.get("attn_implementation", "sdpa"),
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
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(model_config.get("gradient_checkpointing", True)),
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
        num_penalty_types=6,
        num_factors=5,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    output = model(torch.tensor([[1, 2, 3], [3, 4, 0]]))
    losses = compute_typed_losses(
        charge_logits=output["charge_logits"],
        charge_targets=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
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


def validate_loss_breakdown(losses: Any, diagnostic_path: str | Path, *, step: int) -> None:
    """Fail fast on NaN/Inf or a nonzero loss with no eligible samples."""
    errors: list[str] = []
    values: dict[str, float] = {}
    for name in ("charge", "sentence", "invariant", "boundary", "response", "factor", "total"):
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
    finite_terms: list[float] = []
    penalty_counts: dict[str, int] = {}
    for record in records:
        for charge in record["charges"]:
            charge_counts[charge] = charge_counts.get(charge, 0) + 1
        penalty = str(record["penalty_type"])
        penalty_counts[penalty] = penalty_counts.get(penalty, 0) + 1
        if penalty == "fixed_term" and record.get("imprisonment_months") is not None:
            finite_terms.append(float(record["imprisonment_months"]))
    payload = {
        "majority_charge": max(charge_counts, key=charge_counts.get),
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


def run_real_training(
    config: dict[str, Any],
    *,
    train_data: str | Path,
    output_dir: str | Path,
    experiment_name: str,
    counterfactual_data: str | Path | None = None,
    evaluation_data: str | Path | None = None,
    prediction_dir: str | Path | None = None,
    limit: int | None = None,
    resume_from_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Run a local-only Accelerate loop for baseline, full, or ablation training."""
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

    seed = int(config.get("training", {}).get("seed", 42))
    set_deterministic_seed(seed)
    experiment = build_experiment(experiment_name)
    records = load_processed_records(train_data, limit=limit)
    if not records:
        raise ValueError("training data contains no records")
    destination = Path(output_dir)
    metadata_path = Path(train_data).parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    charges = list(metadata["charge_vocabulary"])
    penalties = list(metadata["penalty_vocabulary"])
    charge_to_index = {label: index for index, label in enumerate(charges)}
    penalty_to_index = {label: index for index, label in enumerate(penalties)}
    parents = {str(record["case_id"]): record for record in records}
    evaluation_pairs: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    if counterfactual_data is not None:
        evaluation_pairs = load_counterfactual_pairs(
            counterfactual_data,
            parents,
            filter_valid=True,
            limit=limit,
        )
        pairs = (
            list(evaluation_pairs)
            if experiment.filter_counterfactuals
            else load_counterfactual_pairs(
                counterfactual_data,
                parents,
                filter_valid=False,
                limit=limit,
            )
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
        majority_penalty = str(result["majority_penalty_type"])
        majority_probability = [float(charge == majority_charge) for charge in charges]
        prediction_root = Path(prediction_dir or destination / "predictions")
        if evaluation_data is not None:
            evaluation_records = load_processed_records(evaluation_data, limit=limit)
            static_path = prediction_root / "static.jsonl"
            _write_jsonl(
                static_path,
                make_static_prediction_rows(
                    evaluation_records,
                    charge_probabilities=[majority_probability] * len(evaluation_records),
                    penalty_indices=[penalties.index(majority_penalty)] * len(evaluation_records),
                    sentence_months=[float(result["mean_imprisonment_months"])]
                    * len(evaluation_records),
                    charge_vocabulary=charges,
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
        for pair in pairs:
            parent = dict(pair["parent_record"])
            parent["fact_conservative"] = pair["counterfactual_text"]
            if pair["intervention_type"] == "charge_flip" and pair["target_charge"]:
                parent["charges"] = [pair["target_charge"]]
            parent["penalty_type"] = "unknown"
            parent["imprisonment_months"] = None
            records.append(parent)
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
        num_penalty_types=len(penalties),
        num_factors=5,
        hard_charge=experiment.hard_charge_sentence,
    )
    max_length = int(config.get("model", {}).get("max_length", config.get("max_length", 4096)))

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
            [format_model_input(item, use_factors=experiment.use_factors) for item in batch]
        )
        charge_targets = torch.zeros((len(batch), len(charges)), dtype=torch.float32)
        penalty_targets = torch.zeros(len(batch), dtype=torch.long)
        sentence_targets = torch.zeros(len(batch), dtype=torch.float32)
        sentence_mask = torch.zeros(len(batch), dtype=torch.bool)
        factor_targets = torch.zeros((len(batch), 5), dtype=torch.float32)
        for index, item in enumerate(batch):
            for charge in item["charges"]:
                charge_targets[index, charge_to_index[charge]] = 1.0
            penalty_targets[index] = penalty_to_index[item["penalty_type"]]
            eligible = (
                item["penalty_type"] == "fixed_term"
                and item.get("imprisonment_months") is not None
                and len(item["charges"]) == 1
            )
            if eligible:
                sentence_targets[index] = float(item["imprisonment_months"])
                sentence_mask[index] = True
            factors = item.get("factors") or {}
            factor_targets[index] = torch.tensor(
                [
                    float(factors.get("amount") is not None),
                    float(factors.get("surrender") is True),
                    float(factors.get("restitution") is True),
                    float(factors.get("confession") is True),
                    float(factors.get("role") == "accessory"),
                ]
            )
        result = {
            **encoded,
            "charge_targets": charge_targets,
            "penalty_targets": penalty_targets,
            "sentence_targets": sentence_targets,
            "sentence_mask": sentence_mask,
            "factor_targets": factor_targets,
        }
        if all("_prediction_index" in item for item in batch):
            result["record_indices"] = torch.tensor(
                [int(item["_prediction_index"]) for item in batch], dtype=torch.long
            )
        return result

    def collate_pairs(batch: list[dict[str, Any]]) -> dict[str, Any]:
        parent_records = [item["parent_record"] for item in batch]
        cf_records = [
            {
                **item["parent_record"],
                "fact_conservative": item["counterfactual_text"],
            }
            for item in batch
        ]
        result = {
            "parent": tokenize_texts(
                [
                    format_model_input(item, use_factors=experiment.use_factors)
                    for item in parent_records
                ]
            ),
            "counterfactual": tokenize_texts(
                [
                    format_model_input(item, use_factors=experiment.use_factors)
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
    input_identity: dict[str, Any] = {"train": _file_identity(train_data)}
    if counterfactual_data is not None:
        input_identity["counterfactual"] = _file_identity(counterfactual_data)
    step = 0
    completed_microbatches = 0
    progress: dict[str, Any] = {}
    if resume_from_checkpoint is not None:
        progress_path = Path(resume_from_checkpoint) / "training_progress.json"
        if not progress_path.is_file():
            raise FileNotFoundError(f"checkpoint progress not found: {progress_path}")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        validate_resume_metadata(
            progress,
            expected_seed=seed,
            expected_input_identity=input_identity,
        )
        step = int(progress.get("steps", 0))
        completed_microbatches = int(progress.get("completed_microbatches", 0))

    case_sampler = DeterministicEpochSampler(len(records), seed=seed)
    pair_sampler = DeterministicEpochSampler(len(pairs), seed=seed + 1) if pairs else None
    if progress:
        case_sampler.load_state_dict(progress["case_sampler"])
        if pair_sampler is not None and progress.get("pair_sampler") is not None:
            pair_sampler.load_state_dict(progress["pair_sampler"])

    loader = DataLoader(
        records,
        batch_size=batch_size,
        sampler=case_sampler,
        collate_fn=collate_cases,
    )
    pair_loader = (
        DataLoader(
            pairs,
            batch_size=batch_size,
            sampler=pair_sampler,
            collate_fn=collate_pairs,
        )
        if pair_sampler is not None
        else None
    )
    mixed_precision = str(training.get("mixed_precision", "bf16"))
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 8)),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in predictor.parameters() if parameter.requires_grad),
        lr=float(training.get("learning_rate", 2e-4)),
    )
    predictor, optimizer, loader = accelerator.prepare(predictor, optimizer, loader)
    if pair_loader is not None:
        pair_loader = accelerator.prepare(pair_loader)
    if resume_from_checkpoint is not None:
        accelerator.load_state(str(resume_from_checkpoint))
    destination.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(destination / "tensorboard") if accelerator.is_main_process else None
    log_path = destination / "train.jsonl"
    max_steps = int(training.get("max_steps", 1000))
    checkpoint_every = int(training.get("checkpoint_every", 100))
    epochs = int(training.get("epochs", 1))
    pair_iterator = iter(pair_loader) if pair_loader is not None else None
    weights = config.get("loss", {})

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

    predictor.train()
    for _epoch in range(case_sampler.epoch, epochs) if step < max_steps else ():
        for batch in loader:
            with accelerator.accumulate(predictor):
                output = predictor(batch["input_ids"], batch.get("attention_mask"))
                pair_kwargs: dict[str, Any] = {}
                if pair_iterator is not None:
                    try:
                        pair_batch = next(pair_iterator)
                    except StopIteration:
                        if pair_sampler is None:
                            raise RuntimeError(
                                "counterfactual sampler unexpectedly missing"
                            ) from None
                        pair_sampler.next_epoch()
                        pair_iterator = iter(pair_loader)  # type: ignore[arg-type]
                        pair_batch = next(pair_iterator)
                    parent_output = predictor(
                        pair_batch["parent"]["input_ids"],
                        pair_batch["parent"].get("attention_mask"),
                    )
                    cf_output = predictor(
                        pair_batch["counterfactual"]["input_ids"],
                        pair_batch["counterfactual"].get("attention_mask"),
                    )
                    pair_kwargs = {
                        "pair_types": pair_batch["pair_types"],
                        "parent_charge_logits": parent_output["charge_logits"],
                        "counterfactual_charge_logits": cf_output["charge_logits"],
                        "target_charge_indices": pair_batch["target_charge_indices"],
                        "parent_sentence": parent_output["sentence_months"],
                        "counterfactual_sentence": cf_output["sentence_months"],
                        "rank_direction": pair_batch["rank_direction"],
                        "rank_margin": float(training.get("rank_margin", 1.0)),
                    }
                    if pair_sampler is not None:
                        pair_sampler.advance(
                            len(pair_batch["pair_types"]) * accelerator.num_processes
                        )
                losses = compute_typed_losses(
                    charge_logits=output["charge_logits"],
                    charge_targets=batch["charge_targets"],
                    penalty_logits=output["penalty_type_logits"],
                    penalty_targets=batch["penalty_targets"],
                    sentence_predictions=output["sentence_months"],
                    sentence_targets=batch["sentence_targets"],
                    sentence_mask=batch["sentence_mask"],
                    factor_logits=output["factor_logits"] if experiment.use_factors else None,
                    factor_targets=batch["factor_targets"] if experiment.use_factors else None,
                    weights=weights,
                    **pair_kwargs,
                )
                validate_loss_breakdown(losses, destination / "loss_diagnostic.json", step=step)
                accelerator.backward(losses.total)
                completed_optimizer_step = accelerator.sync_gradients
                optimizer.step()
                optimizer.zero_grad()
            completed_microbatches += 1
            case_sampler.advance(batch["input_ids"].shape[0] * accelerator.num_processes)
            if completed_optimizer_step:
                step += 1
            if completed_optimizer_step and accelerator.is_main_process:
                log = {
                    "step": step,
                    **{
                        name: float(getattr(losses, name).detach())
                        for name in (
                            "charge",
                            "sentence",
                            "invariant",
                            "boundary",
                            "response",
                            "factor",
                            "total",
                        )
                    },
                }
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
                save_checkpoint(destination / f"checkpoint-step-{step:08d}")
            if step >= max_steps:
                break
        if step >= max_steps:
            break
        case_sampler.next_epoch()
    save_checkpoint(destination / "checkpoint-final")
    if writer is not None:
        writer.close()
    if accelerator.is_main_process:
        (destination / "run_metadata.json").write_text(
            json.dumps(
                {
                    "experiment": asdict(experiment),
                    "steps": step,
                    "completed_microbatches": completed_microbatches,
                    "charge_vocabulary": charges,
                    "penalty_vocabulary": penalties,
                    "counterfactual_pairs": len(pairs),
                    "evaluation_counterfactual_pairs": len(evaluation_pairs),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    result: dict[str, Any] = {
        "steps": step,
        "output_dir": str(destination),
        "counterfactual_pairs": len(pairs),
        "evaluation_counterfactual_pairs": len(evaluation_pairs),
    }
    prediction_root = Path(prediction_dir or destination / "predictions")
    predictor.eval()
    if evaluation_data is not None:
        evaluation_records = load_processed_records(evaluation_data, limit=limit)
        indexed_records = [
            {**record, "_prediction_index": index}
            for index, record in enumerate(evaluation_records)
        ]
        evaluation_loader = accelerator.prepare(
            DataLoader(
                indexed_records,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_cases,
            )
        )
        gathered_static: dict[int, tuple[list[float], int, float]] = {}
        with torch.no_grad():
            for batch in evaluation_loader:
                output = predictor(batch["input_ids"], batch.get("attention_mask"))
                gathered = accelerator.gather_for_metrics(
                    (
                        batch["record_indices"],
                        torch.sigmoid(output["charge_logits"]),
                        output["penalty_type_logits"].argmax(dim=-1),
                        output["sentence_months"],
                    )
                )
                indices, probabilities, penalty_indices, sentence_months = gathered
                if accelerator.is_main_process:
                    for index, probability, penalty, months in zip(
                        indices.cpu().tolist(),
                        probabilities.float().cpu().tolist(),
                        penalty_indices.cpu().tolist(),
                        sentence_months.float().cpu().tolist(),
                        strict=True,
                    ):
                        gathered_static[int(index)] = (probability, int(penalty), float(months))
        if accelerator.is_main_process:
            ordered_static = [gathered_static[index] for index in range(len(evaluation_records))]
            static_path = prediction_root / "static.jsonl"
            _write_jsonl(
                static_path,
                make_static_prediction_rows(
                    evaluation_records,
                    charge_probabilities=[item[0] for item in ordered_static],
                    penalty_indices=[item[1] for item in ordered_static],
                    sentence_months=[item[2] for item in ordered_static],
                    charge_vocabulary=charges,
                    penalty_vocabulary=penalties,
                ),
            )
            result["static_predictions"] = str(static_path)

    if evaluation_pairs:
        indexed_pairs = [
            {**pair, "_prediction_index": index} for index, pair in enumerate(evaluation_pairs)
        ]
        prediction_pair_loader = accelerator.prepare(
            DataLoader(
                indexed_pairs,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_pairs,
            )
        )
        gathered_pairs: dict[int, tuple[list[float], list[float], float, float]] = {}
        with torch.no_grad():
            for batch in prediction_pair_loader:
                parent_output = predictor(
                    batch["parent"]["input_ids"], batch["parent"].get("attention_mask")
                )
                counterfactual_output = predictor(
                    batch["counterfactual"]["input_ids"],
                    batch["counterfactual"].get("attention_mask"),
                )
                gathered = accelerator.gather_for_metrics(
                    (
                        batch["pair_indices"],
                        torch.sigmoid(parent_output["charge_logits"]),
                        torch.sigmoid(counterfactual_output["charge_logits"]),
                        parent_output["sentence_months"],
                        counterfactual_output["sentence_months"],
                    )
                )
                indices, parent_prob, counterfactual_prob, parent_months, cf_months = gathered
                if accelerator.is_main_process:
                    for index, first_prob, second_prob, first_months, second_months in zip(
                        indices.cpu().tolist(),
                        parent_prob.float().cpu().tolist(),
                        counterfactual_prob.float().cpu().tolist(),
                        parent_months.float().cpu().tolist(),
                        cf_months.float().cpu().tolist(),
                        strict=True,
                    ):
                        gathered_pairs[int(index)] = (
                            first_prob,
                            second_prob,
                            float(first_months),
                            float(second_months),
                        )
        if accelerator.is_main_process:
            ordered_pairs = [gathered_pairs[index] for index in range(len(evaluation_pairs))]
            counterfactual_path = prediction_root / "counterfactual.jsonl"
            _write_jsonl(
                counterfactual_path,
                make_counterfactual_prediction_rows(
                    evaluation_pairs,
                    parent_probabilities=[item[0] for item in ordered_pairs],
                    counterfactual_probabilities=[item[1] for item in ordered_pairs],
                    parent_sentences=[item[2] for item in ordered_pairs],
                    counterfactual_sentences=[item[3] for item in ordered_pairs],
                    charge_vocabulary=charges,
                ),
            )
            result["counterfactual_predictions"] = str(counterfactual_path)
    accelerator.wait_for_everyone()
    return result
