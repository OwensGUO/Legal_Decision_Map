"""Exercise the real training loop with two CPU workers and tiny local models."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from legal_landscape.progress import NullProgressReporter
from legal_landscape.training.train import InspectedModelConfig, run_real_training

torch = pytest.importorskip("torch")
pytest.importorskip("accelerate")
pytest.importorskip("transformers")
pytest.importorskip("tensorboard")

class TinyTokenizer:
    pad_token_id = 0

    def __call__(self, texts, **kwargs):
        return {
            "input_ids": torch.tensor([[1, 2, 3]] * len(texts)),
            "attention_mask": torch.ones(len(texts), 3, dtype=torch.long),
        }


class TinyCheckpointedBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 4)
        self.embedding.requires_grad_(False)
        self.projection = torch.nn.Linear(4, 4)

    def forward(self, input_ids, **kwargs):
        from torch.utils.checkpoint import checkpoint

        hidden = self.embedding(input_ids)
        hidden = checkpoint(self.projection, hidden, use_reentrant=False)
        return SimpleNamespace(last_hidden_state=hidden)


def _training_worker(rank, world_size, init_path, root, experiment, extra=None):
    import torch.distributed as distributed
    from accelerate.state import AcceleratorState, PartialState
    from accelerate.utils import DistributedType

    os.environ.update({
        "RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(world_size),
        "LOCAL_WORLD_SIZE": str(world_size), "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500", "ACCELERATE_USE_CPU": "true",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    })
    torch.set_num_threads(1)
    distributed.init_process_group(
        "gloo", init_method=f"file://{init_path}", rank=rank, world_size=world_size,
    )
    AcceleratorState._reset_state(reset_partial_state=True)
    state = PartialState(cpu=True)
    assert state.distributed_type == DistributedType.MULTI_CPU
    root = Path(root)
    inspected = InspectedModelConfig("tiny", (), "tiny", "base_model", (), False, 4)
    try:
        # Only pretrained loading is substituted; losses, autograd, DDP, accumulation,
        # optimizer updates, checkpoint saves, and prediction gathers are real.
        with (
            # Force CPU-only hardware discovery for Gloo barriers on MPS/CUDA hosts.
            patch("torch._C._get_accelerator", return_value=torch.device("cpu")),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=TinyTokenizer()),
            patch(
                "legal_landscape.training.train.load_backbone",
                side_effect=lambda config: (TinyCheckpointedBackbone(), inspected),
            ),
        ):
            result = run_real_training(
                {"model": {"path": "tiny"}, "training": {
                    "seed": 42, "mixed_precision": "no", "max_steps": 2,
                    "epochs": 1, "per_device_batch_size": 1,
                    "gradient_accumulation_steps": 2, "checkpoint_every": 1,
                    **((extra or {}).get("training") or {}),
                }},
                train_data=root / "train.jsonl", output_dir=root / "output",
                experiment_name=experiment,
                counterfactual_data=root / "pairs.jsonl" if experiment != "B3" else None,
                evaluation_data=root / "evaluation.jsonl",
                evaluation_counterfactual_data=root / "heldout_pairs.jsonl",
                validation_data=(extra or {}).get("validation"),
                progress_factory=lambda **kwargs: NullProgressReporter(),
            )
        (root / f"rank-{rank}.json").write_text(json.dumps(result))
    finally:
        distributed.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="Torch Gloo distributed runtime is not installed",
)
@pytest.mark.parametrize(("experiment", "labels_present"), [
    ("B3", True), ("B3", False), ("M", True), ("A4", False),
])
def test_real_training_completes_ddp_accumulation_with_inactive_heads(
    tmp_path, experiment, labels_present,
):
    records = [{
        "case_id": f"c{index}", "group_id": f"g{index}", "dataset": "cail",
        "target_defendant": "某甲", "fact_strict": "案件事实", "factors": {"restitution": False},
        "charges": ["盗窃"],
        "conviction_articles": ["criminal_law:264"] if labels_present else [],
        "penalty_type": "fixed_term" if labels_present else "life",
        "imprisonment_months": 12 if labels_present else None,
    } for index in range(8)]
    (tmp_path / "train.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    evaluation = [
        {**record, "case_id": f"e{index}", "group_id": f"eg{index}"}
        for index, record in enumerate(records)
    ]
    (tmp_path / "evaluation.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in evaluation)
    )
    (tmp_path / "metadata.json").write_text(json.dumps({
        "charge_vocabulary": ["盗窃", "诈骗"],
        "article_vocabulary": ["criminal_law:264"],
        "penalty_vocabulary": ["fixed_term", "life"],
    }))
    def pairs(parents):
        return "".join(json.dumps({
            "parent_case_id": record["case_id"],
            "intervention_type": ("invariant", "charge_flip", "sentence_rank")[index % 3],
            "target_charge": "诈骗" if index % 3 == 1 else None,
            "rank_direction": 1 if index % 3 == 2 else None,
            "target_factors": {},
            "validation": {"valid": True, "errors": []},
            "counterfactual_strict": "改写事实",
        }) + "\n" for index, record in enumerate(parents))

    (tmp_path / "pairs.jsonl").write_text(pairs(records))
    (tmp_path / "heldout_pairs.jsonl").write_text(pairs(evaluation))

    torch.multiprocessing.spawn(
        _training_worker,
        args=(2, str(tmp_path / "gloo-init"), str(tmp_path), experiment),
        nprocs=2, join=True,
    )

    for rank in range(2):
        assert json.loads((tmp_path / f"rank-{rank}.json").read_text())["steps"] == 2
    progress = json.loads((tmp_path / "output/checkpoint-final/training_progress.json").read_text())
    assert progress["steps"] == 2
    assert progress["completed_microbatches"] == 4
    assert (tmp_path / "output/checkpoint-step-00000001/training_progress.json").is_file()
    rows = [json.loads(line) for line in (
        tmp_path / "output/predictions/static.jsonl"
    ).read_text().splitlines()]
    assert [row["case_id"] for row in rows] == [f"e{index}" for index in range(8)]
    pair_rows = [json.loads(line) for line in (
        tmp_path / "output/predictions/counterfactual.jsonl"
    ).read_text().splitlines()]
    # Life or missing-month cases are outside the sentence-rank task.
    expected_pairs = [
        f"e{index}" for index in range(8) if labels_present or index % 3 != 2
    ]
    assert [row["case_id"] for row in pair_rows] == expected_pairs


def _write_fixture(tmp_path):
    records = [{
        "case_id": f"c{index}", "group_id": f"g{index}", "dataset": "cail",
        "target_defendant": "某甲", "fact_strict": "案件事实" * (index + 1),
        "factors": {"restitution": False},
        "charges": ["盗窃"], "conviction_articles": ["criminal_law:264"],
        "penalty_type": "fixed_term", "imprisonment_months": 12,
    } for index in range(8)]
    held_out = {
        name: [
            {**record, "case_id": f"{prefix}{index}", "group_id": f"{prefix}g{index}"}
            for index, record in enumerate(records)
        ]
        for name, prefix in (("evaluation", "e"), ("validation", "v"))
    }
    for name, rows in (("train", records), *held_out.items()):
        (tmp_path / f"{name}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in rows)
        )
    (tmp_path / "metadata.json").write_text(json.dumps({
        "charge_vocabulary": ["盗窃", "诈骗"],
        "article_vocabulary": ["criminal_law:264"],
        "penalty_vocabulary": ["fixed_term", "life"],
    }))

    def pairs(parents):
        return "".join(json.dumps({
            "parent_case_id": record["case_id"],
            "intervention_type": ("invariant", "charge_flip", "sentence_rank")[index % 3],
            "target_charge": "诈骗" if index % 3 == 1 else None,
            "rank_direction": 1 if index % 3 == 2 else None,
            "target_factors": {},
            "validation": {"valid": True, "errors": []},
            "counterfactual_strict": "改写事实",
        }) + "\n" for index, record in enumerate(parents))

    (tmp_path / "pairs.jsonl").write_text(pairs(records))
    (tmp_path / "heldout_pairs.jsonl").write_text(pairs(held_out["evaluation"]))


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="Torch Gloo distributed runtime is not installed",
)
def test_real_training_validates_selects_checkpoint_and_tunes_thresholds(tmp_path):
    _write_fixture(tmp_path)
    extra = {
        "validation": str(tmp_path / "validation.jsonl"),
        "training": {
            "pair_every": 2, "warmup_ratio": 0.5, "min_learning_rate_ratio": 0.1,
            "max_steps": 0, "inference_batch_size": 3, "inference_max_tokens_per_batch": 4096,
        },
    }
    torch.multiprocessing.spawn(
        _training_worker,
        args=(2, str(tmp_path / "gloo-init"), str(tmp_path), "M", extra),
        nprocs=2, join=True,
    )
    results = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert results[0]["steps"] == results[1]["steps"] == 2
    assert results[0]["selected_checkpoint"] == results[1]["selected_checkpoint"]
    output = tmp_path / "output"
    thresholds = json.loads((output / "predictions/thresholds.json").read_text())
    assert 0.0 < thresholds["charge"] < 1.0
    scored = [
        json.loads(path.read_text())
        for path in output.glob("checkpoint-*/training_progress.json")
    ]
    assert scored and all("validation_score" in item for item in scored)
    logs = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert [round(item["learning_rate"], 8) for item in logs if "learning_rate" in item] == [
        0.0002, 0.0002,
    ]
    static = [
        json.loads(line)
        for line in (output / "predictions/static.jsonl").read_text().splitlines()
    ]
    assert [row["case_id"] for row in static] == [f"e{index}" for index in range(8)]
    pairs = [
        json.loads(line)
        for line in (output / "predictions/counterfactual.jsonl").read_text().splitlines()
    ]
    assert [row["case_id"] for row in pairs] == [f"e{index}" for index in range(8)]
    assert all(row["parent_charge_indices"] == [0] for row in pairs)
    assert all(row["charge_threshold"] == thresholds["charge"] for row in pairs)


def _adam_update_count(checkpoint: Path) -> int:
    """Adam's own step counter, incremented only inside a real optimizer.step()."""
    state = torch.load(checkpoint / "optimizer.bin", map_location="cpu", weights_only=False)
    counts = []
    for slot in state["state"].values():
        step = slot["step"]
        counts.append(int(step.item() if hasattr(step, "item") else step))
    if not counts or len(set(counts)) != 1:
        raise AssertionError(f"optimizer steps diverged in {checkpoint}: {counts}")
    return counts[0]


def _write_accumulation_case(root: Path, *, records: int, pairs: int) -> None:
    rows = [{
        "case_id": f"c{index}", "group_id": f"g{index}", "dataset": "cail",
        "target_defendant": "某甲", "fact_strict": "案件事实", "factors": {"restitution": False},
        "charges": ["盗窃"], "conviction_articles": ["criminal_law:264"],
        "penalty_type": "fixed_term", "imprisonment_months": 12,
    } for index in range(records)]
    (root / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (root / "metadata.json").write_text(json.dumps({
        "charge_vocabulary": ["盗窃", "诈骗"],
        "article_vocabulary": ["criminal_law:264"],
        "penalty_vocabulary": ["fixed_term", "life"],
    }))
    parents = rows[:pairs] if pairs else []
    (root / "pairs.jsonl").write_text("".join(json.dumps({
        "parent_case_id": row["case_id"],
        "intervention_type": "invariant",
        "target_charge": None,
        "rank_direction": None,
        "target_factors": {},
        "validation": {"valid": True, "errors": []},
        "counterfactual_strict": "改写事实",
    }) + "\n" for row in parents))


def _run_accumulation_case(
    root: Path,
    *,
    experiment: str,
    epochs: int,
    max_steps: int,
    resume: Path | None = None,
    checkpoint_every: int = 0,
) -> dict:
    """Real Accelerator loop. The returned weight is the live parameter after optimizer updates."""
    from accelerate.state import AcceleratorState, GradientState

    inspected = InspectedModelConfig("tiny", (), "tiny", "base_model", (), False, 4)
    holder: dict = {}

    def load_backbone(config):
        model = TinyCheckpointedBackbone()
        holder["initial"] = model.projection.weight.detach().cpu().clone()
        holder["model"] = model
        return model, inspected

    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()
    try:
        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=TinyTokenizer()),
            patch("legal_landscape.training.train.load_backbone", side_effect=load_backbone),
        ):
            result = run_real_training(
                {"model": {"path": "tiny"}, "training": {
                    "seed": 42, "mixed_precision": "no", "max_steps": max_steps,
                    "epochs": epochs, "per_device_batch_size": 1,
                    "gradient_accumulation_steps": 2, "checkpoint_every": checkpoint_every,
                }},
                train_data=root / "train.jsonl",
                output_dir=root / "output",
                experiment_name=experiment,
                counterfactual_data=root / "pairs.jsonl" if experiment != "B3" else None,
                resume_from_checkpoint=resume,
                progress_factory=lambda **kwargs: NullProgressReporter(),
            )
    finally:
        AcceleratorState._reset_state(reset_partial_state=True)
        GradientState._reset_state()
    final = holder["model"].projection.weight.detach().cpu()
    result = dict(result)
    result["weight_changed"] = not torch.allclose(final, holder["initial"])
    result["weight"] = final.view(-1)[:4].tolist()
    return result


def _assert_update_schedule(root: Path, result: dict, *, steps: int, microbatches: int) -> None:
    progress = json.loads((root / "output/checkpoint-final/training_progress.json").read_text())
    logs = [
        json.loads(line)
        for line in (root / "output/train.jsonl").read_text().splitlines()
        if line.strip()
    ]
    learning_rates = [row for row in logs if "learning_rate" in row]
    assert result["steps"] == steps
    assert progress["steps"] == steps
    assert progress["completed_microbatches"] == microbatches
    assert _adam_update_count(root / "output/checkpoint-final") == steps
    assert [row["step"] for row in learning_rates] == list(range(1, steps + 1))
    assert result["weight_changed"] is True


@pytest.mark.parametrize(("_name", "experiment", "records", "pairs", "epochs", "steps", "micro"), [
    ("no_counterfactual", "B3", 8, 0, 1, 4, 8),
    ("with_counterfactual", "M", 8, 8, 1, 4, 8),
    ("counterfactual_exhausted", "M", 8, 1, 1, 4, 8),
    ("tail_without_counterfactual", "B3", 3, 0, 1, 2, 3),
    ("tail_with_counterfactual", "M", 3, 1, 1, 2, 3),
    ("multi_epoch_tail", "M", 3, 1, 2, 4, 6),
])
def test_main_microbatches_decide_real_optimizer_updates(
    tmp_path, _name, experiment, records, pairs, epochs, steps, micro,
):
    """8 main batches accumulated by 2 update 4 times.

    3 main batches use the tail protocol: sync after batch 2 and again on the
    incomplete final batch, so the epoch applies 2 updates. A second epoch
    starts a new window. Counterfactual exhaustion must not add or drop updates.
    """
    _write_accumulation_case(tmp_path, records=records, pairs=pairs)
    result = _run_accumulation_case(
        tmp_path, experiment=experiment, epochs=epochs, max_steps=0,
    )
    _assert_update_schedule(tmp_path, result, steps=steps, microbatches=micro)


def test_resume_continues_main_accumulation_window(tmp_path):
    """Stopping on an optimizer step and restoring it finishes the same 8-batch schedule.

    The counterfactual stream has a single pair, so it is exhausted and restarted while
    the main window is still open. Restoration must keep 4 updates and 8 main microbatches.
    """
    _write_accumulation_case(tmp_path, records=8, pairs=1)
    _run_accumulation_case(tmp_path, experiment="M", epochs=1, max_steps=2)
    midpoint = json.loads((tmp_path / "output/checkpoint-final/training_progress.json").read_text())
    assert midpoint["steps"] == 2
    assert midpoint["completed_microbatches"] == 4
    assert _adam_update_count(tmp_path / "output/checkpoint-final") == 2
    resumed = _run_accumulation_case(
        tmp_path,
        experiment="M",
        epochs=1,
        max_steps=4,
        resume=tmp_path / "output/checkpoint-final",
    )
    _assert_update_schedule(tmp_path, resumed, steps=4, microbatches=8)


def _ddp_accumulation_worker(rank, world_size, init_path, root, spec):
    import torch.distributed as distributed
    import torch.nn.parallel
    from accelerate.state import AcceleratorState, GradientState, PartialState
    from accelerate.utils import DistributedType

    os.environ.update({
        "RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(world_size),
        "LOCAL_WORLD_SIZE": str(world_size), "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29511", "ACCELERATE_USE_CPU": "true",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    })
    torch.set_num_threads(1)
    distributed.init_process_group(
        "gloo", init_method=f"file://{init_path}", rank=rank, world_size=world_size,
    )
    nosync_entries = {"count": 0}
    original_no_sync = torch.nn.parallel.DistributedDataParallel.no_sync

    def _counting_no_sync(self):
        nosync_entries["count"] += 1
        return original_no_sync(self)

    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()
    state = PartialState(cpu=True)
    assert state.distributed_type == DistributedType.MULTI_CPU
    root = Path(root)
    inspected = InspectedModelConfig("tiny", (), "tiny", "base_model", (), False, 4)
    holder: dict = {}

    def load_backbone(config):
        model = TinyCheckpointedBackbone()
        holder["initial"] = model.projection.weight.detach().cpu().clone()
        holder["model"] = model
        return model, inspected

    try:
        with (
            patch("torch._C._get_accelerator", return_value=torch.device("cpu")),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=TinyTokenizer()),
            patch("legal_landscape.training.train.load_backbone", side_effect=load_backbone),
            patch.object(
                torch.nn.parallel.DistributedDataParallel, "no_sync", _counting_no_sync,
            ),
        ):
            result = run_real_training(
                {"model": {"path": "tiny"}, "training": {
                    "seed": 42, "mixed_precision": "no", "max_steps": spec["max_steps"],
                    "epochs": spec["epochs"], "per_device_batch_size": 1,
                    "gradient_accumulation_steps": 2, "checkpoint_every": 0,
                }},
                train_data=root / "train.jsonl",
                output_dir=root / "output",
                experiment_name=spec["experiment"],
                counterfactual_data=(
                    root / "pairs.jsonl" if spec["experiment"] != "B3" else None
                ),
                progress_factory=lambda **kwargs: NullProgressReporter(),
            )
        weight = holder["model"].projection.weight.detach().cpu()
        (root / f"rank-{rank}.json").write_text(json.dumps({
            "steps": result["steps"],
            "nosync_entries": nosync_entries["count"],
            "weight_changed": not torch.allclose(weight, holder["initial"]),
            "weight": [round(float(value), 8) for value in weight.view(-1)[:8]],
        }))
    finally:
        distributed.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="Torch Gloo distributed runtime is not installed",
)
@pytest.mark.parametrize(("experiment", "records", "pairs", "epochs", "steps", "micro", "nosync"), [
    # 16 global batches / 2 ranks = 8 main microbatches, accumulation 2 → 4 updates.
    ("B3", 16, 0, 1, 4, 8, 4),
    ("M", 16, 16, 1, 4, 8, 4),
    # One counterfactual pair is exhausted on every fetch and must not force extra updates.
    ("M", 16, 1, 1, 4, 8, 4),
    # 6 global batches / 2 ranks = 3 main microbatches. Tail protocol: updates on 2 and 3.
    ("M", 6, 1, 1, 2, 3, 1),
    # Each epoch flushes its own remainder, so two short epochs update 4 times.
    ("M", 6, 1, 2, 4, 6, 2),
])
def test_ddp_optimizer_updates_follow_main_batches_not_counterfactual_iterator(
    tmp_path, experiment, records, pairs, epochs, steps, micro, nosync,
):
    _write_accumulation_case(tmp_path, records=records, pairs=pairs)
    torch.multiprocessing.spawn(
        _ddp_accumulation_worker,
        args=(
            2,
            str(tmp_path / "gloo-init"),
            str(tmp_path),
            {
                "experiment": experiment,
                "epochs": epochs,
                "max_steps": 0,
            },
        ),
        nprocs=2,
        join=True,
    )
    ranks = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    progress = json.loads((tmp_path / "output/checkpoint-final/training_progress.json").read_text())
    logs = [
        json.loads(line)
        for line in (tmp_path / "output/train.jsonl").read_text().splitlines()
        if line.strip() and "learning_rate" in line
    ]
    assert [rank["steps"] for rank in ranks] == [steps, steps]
    assert progress["steps"] == steps
    assert progress["completed_microbatches"] == micro
    assert _adam_update_count(tmp_path / "output/checkpoint-final") == steps
    assert [row["step"] for row in logs] == list(range(1, steps + 1))
    assert [rank["nosync_entries"] for rank in ranks] == [nosync, nosync]
    assert all(rank["weight_changed"] for rank in ranks)
    assert ranks[0]["weight"] == ranks[1]["weight"]
