from __future__ import annotations

import json
import unittest
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from legal_landscape.training.train import (
    DeterministicEpochSampler,
    build_experiment,
    format_model_input,
    inspect_model_config,
    load_counterfactual_pairs,
    make_counterfactual_prediction_rows,
    make_static_prediction_rows,
    training_plan,
    validate_resume_metadata,
)


def _distributed_parent_requirement_worker(rank, world_size, init_path, output_dir):
    import torch as worker_torch
    import torch.distributed as distributed

    from legal_landscape.training.train import _distributed_parent_requirements

    distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )

    class Accelerator:
        @staticmethod
        def reduce(value, reduction):
            assert reduction == "sum"
            distributed.all_reduce(value, op=distributed.ReduceOp.SUM)
            return value

    try:
        pair_types = ("charge_flip",) if rank == 0 else ("invariant",)
        requirements = _distributed_parent_requirements(
            worker_torch,
            Accelerator(),
            pair_types,
            worker_torch.device("cpu"),
        )
        from pathlib import Path

        (Path(output_dir) / f"rank-{rank}.txt").write_text(
            f"{int(requirements[0])},{int(requirements[1])}",
            encoding="utf-8",
        )
    finally:
        distributed.destroy_process_group()


class RecordingProgress:
    def __init__(self, events):
        self.events = events
        self.tasks = []
        self.started_tasks = []
        self.closed = False

    def __enter__(self):
        self.events.append("reporter:enter")
        return self

    def __exit__(self, *exc):
        self.closed = True
        self.events.append("reporter:exit")
        return False

    @contextmanager
    def task(self, description, *, total, completed=0, **fields):
        state = {"description": description, "total": total, "completed": completed, **fields}
        self.tasks.append(state)
        self.started_tasks.append(dict(state))
        self.events.append(f"task:enter:{description}")

        def advance(amount=1, **updates):
            state["completed"] += amount
            state.update(updates)
            self.events.append(("advance", description, dict(state)))

        def update(**updates):
            state.update(updates)
            self.events.append(("update", description, dict(state)))

        try:
            yield SimpleNamespace(advance=advance, update=update)
        finally:
            self.events.append(f"task:exit:{description}")


def test_detached_loss_values_detaches_every_reporting_scalar():
    from legal_landscape.training.train import _detached_loss_values

    class Scalar:
        def __init__(self, value):
            self.value = value
            self.detached = False

        def detach(self):
            self.detached = True
            return self

        def __float__(self):
            return float(self.value)

    names = (
        "charge",
        "article",
        "sentence",
        "invariant",
        "boundary",
        "response",
        "factor",
        "total",
    )
    scalars = {name: Scalar(index / 10) for index, name in enumerate(names, start=1)}

    values = _detached_loss_values(SimpleNamespace(**scalars))

    assert values == {name: index / 10 for index, name in enumerate(names, start=1)}
    assert all(scalar.detached for scalar in scalars.values())


def test_sum_loss_values_adds_each_component_without_mutating_inputs():
    from legal_landscape.training.train import _sum_loss_values

    first = {
        "charge": 1.0,
        "article": 2.0,
        "sentence": 3.0,
        "invariant": 0.0,
        "boundary": 0.0,
        "response": 0.0,
        "factor": 4.0,
        "total": 10.0,
    }
    second = {
        "charge": 0.0,
        "article": 0.0,
        "sentence": 0.0,
        "invariant": 5.0,
        "boundary": 6.0,
        "response": 7.0,
        "factor": 0.0,
        "total": 18.0,
    }

    combined = _sum_loss_values(first, second)

    assert combined == {
        "charge": 1.0,
        "article": 2.0,
        "sentence": 3.0,
        "invariant": 5.0,
        "boundary": 6.0,
        "response": 7.0,
        "factor": 4.0,
        "total": 28.0,
    }
    assert first["total"] == 10.0
    assert second["total"] == 18.0


def test_sum_loss_values_rejects_empty_input():
    from legal_landscape.training.train import _sum_loss_values

    with pytest.raises(ValueError, match="at least one"):
        _sum_loss_values()


def test_rng_replay_restores_the_state_present_at_context_entry():
    from legal_landscape.training.train import _capture_rng_state, _replay_rng_state

    class State:
        def __init__(self, value):
            self.value = value

        def clone(self):
            return State(self.value)

    class Cuda:
        def __init__(self):
            self.state = State(20)

        def is_available(self):
            return True

        def get_rng_state(self, device):
            assert device.type == "cuda"
            return self.state

        def set_rng_state(self, state, device):
            assert device.type == "cuda"
            self.state = state.clone()

    class FakeTorch:
        def __init__(self):
            self.state = State(10)
            self.cuda = Cuda()

        def get_rng_state(self):
            return self.state

        def set_rng_state(self, state):
            self.state = state.clone()

    fake_torch = FakeTorch()
    device = SimpleNamespace(type="cuda")
    snapshot = _capture_rng_state(fake_torch, device)
    fake_torch.state = State(30)
    fake_torch.cuda.state = State(40)

    with _replay_rng_state(fake_torch, snapshot):
        assert fake_torch.state.value == 10
        assert fake_torch.cuda.state.value == 20
        fake_torch.state = State(50)
        fake_torch.cuda.state = State(60)

    assert fake_torch.state.value == 30
    assert fake_torch.cuda.state.value == 40


def test_rng_replay_restores_state_when_recomputation_raises():
    from legal_landscape.training.train import _capture_rng_state, _replay_rng_state

    class State:
        def __init__(self, value):
            self.value = value

        def clone(self):
            return State(self.value)

    class FakeTorch:
        cuda = SimpleNamespace(is_available=lambda: False)

        def __init__(self):
            self.state = State(1)

        def get_rng_state(self):
            return self.state

        def set_rng_state(self, state):
            self.state = state.clone()

    fake_torch = FakeTorch()
    snapshot = _capture_rng_state(fake_torch, SimpleNamespace(type="cpu"))
    fake_torch.state = State(2)

    with pytest.raises(RuntimeError, match="recompute failed"):
        with _replay_rng_state(fake_torch, snapshot):
            assert fake_torch.state.value == 1
            raise RuntimeError("recompute failed")

    assert fake_torch.state.value == 2


def test_rng_replay_restores_cpu_state_when_cuda_installation_raises():
    from legal_landscape.training.train import _capture_rng_state, _replay_rng_state

    class State:
        def __init__(self, value):
            self.value = value

        def clone(self):
            return State(self.value)

    class Cuda:
        def __init__(self):
            self.state = State(20)
            self.fail_next_install = False

        def is_available(self):
            return True

        def get_rng_state(self, device):
            return self.state

        def set_rng_state(self, state, device):
            if self.fail_next_install:
                self.fail_next_install = False
                raise RuntimeError("CUDA RNG installation failed")
            self.state = state.clone()

    class FakeTorch:
        def __init__(self):
            self.state = State(10)
            self.cuda = Cuda()

        def get_rng_state(self):
            return self.state

        def set_rng_state(self, state):
            self.state = state.clone()

    fake_torch = FakeTorch()
    device = SimpleNamespace(type="cuda")
    snapshot = _capture_rng_state(fake_torch, device)
    fake_torch.state = State(30)
    fake_torch.cuda.state = State(40)
    fake_torch.cuda.fail_next_install = True

    with pytest.raises(RuntimeError, match="CUDA RNG installation failed"):
        with _replay_rng_state(fake_torch, snapshot):
            pytest.fail("context body must not run")

    assert fake_torch.state.value == 30
    assert fake_torch.cuda.state.value == 40


@pytest.mark.skipif(torch is None, reason="torch is not installed")
def test_rng_replay_reproduces_dropout_and_preserves_future_randomness():
    from legal_landscape.training.train import _capture_rng_state, _replay_rng_state

    torch.manual_seed(17)
    dropout = torch.nn.Dropout(p=0.5)
    values = torch.ones(32)
    snapshot = _capture_rng_state(torch, values.device)
    expected_parent = dropout(values)
    dropout(values)
    post_counterfactual_state = torch.get_rng_state().clone()

    with _replay_rng_state(torch, snapshot):
        recomputed_parent = dropout(values)

    torch.testing.assert_close(recomputed_parent, expected_parent)
    assert torch.equal(torch.get_rng_state(), post_counterfactual_state)


@pytest.mark.skipif(torch is None, reason="torch is not installed")
def test_staged_backward_gradient_equivalence():
    from legal_landscape.models.losses import compute_typed_losses

    def supervised(parameter):
        logits = torch.stack((parameter, -parameter)).unsqueeze(0)
        return compute_typed_losses(
            charge_logits=logits,
            charge_targets=torch.tensor([[1.0, 0.0]]),
        )

    def paired(parameter):
        parent = torch.stack((parameter * 2.0, -parameter)).unsqueeze(0)
        counterfactual = torch.stack((parameter * 3.0, parameter)).unsqueeze(0)
        return compute_typed_losses(
            pair_types=("invariant",),
            parent_charge_logits=parent,
            counterfactual_charge_logits=counterfactual,
        )

    combined_parameter = torch.tensor(0.4, requires_grad=True)
    combined = compute_typed_losses(
        charge_logits=torch.stack(
            (combined_parameter, -combined_parameter)
        ).unsqueeze(0),
        charge_targets=torch.tensor([[1.0, 0.0]]),
        pair_types=("invariant",),
        parent_charge_logits=torch.stack(
            (combined_parameter * 2.0, -combined_parameter)
        ).unsqueeze(0),
        counterfactual_charge_logits=torch.stack(
            (combined_parameter * 3.0, combined_parameter)
        ).unsqueeze(0),
    )
    combined.total.backward()

    staged_parameter = torch.tensor(0.4, requires_grad=True)
    supervised(staged_parameter).total.backward()
    paired(staged_parameter).total.backward()

    torch.testing.assert_close(staged_parameter.grad, combined_parameter.grad)


def test_single_graph_recompute_backward_routes_saved_and_zero_gradients():
    from legal_landscape.training.train import _backward_recomputed_parent

    outputs = {
        name: object()
        for name in (
            "charge_logits",
            "article_logits",
            "penalty_type_logits",
            "sentence_by_charge",
            "sentence_months",
            "factor_logits",
        )
    }
    charge_gradient = object()
    sentence_gradient = object()
    call = {}

    class Autograd:
        @staticmethod
        def backward(tensors, grad_tensors):
            call["tensors"] = tensors
            call["gradients"] = grad_tensors

    fake_torch = SimpleNamespace(
        autograd=Autograd(),
        zeros_like=lambda tensor: ("zero", tensor),
    )

    _backward_recomputed_parent(
        fake_torch,
        outputs,
        charge_gradient=charge_gradient,
        sentence_gradient=sentence_gradient,
    )

    names = (
        "charge_logits",
        "article_logits",
        "penalty_type_logits",
        "sentence_by_charge",
        "sentence_months",
        "factor_logits",
    )
    assert call["tensors"] == tuple(outputs[name] for name in names)
    assert call["gradients"] == (
        charge_gradient,
        ("zero", outputs["article_logits"]),
        ("zero", outputs["penalty_type_logits"]),
        ("zero", outputs["sentence_by_charge"]),
        sentence_gradient,
        ("zero", outputs["factor_logits"]),
    )


@pytest.mark.skipif(torch is None, reason="torch is not installed")
@pytest.mark.parametrize("loss_scale", [1.0, 128.0])
def test_single_graph_recompute_matches_joint_graph_gradients(loss_scale):
    from legal_landscape.training.train import (
        _backward_recomputed_parent,
        _zero_output_anchor,
    )

    class TinyPredictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = torch.nn.Parameter(torch.tensor(0.4))

        def forward(self, value):
            hidden = self.shared * value
            charge = hidden * 2.0
            article = hidden * 3.0
            penalty = hidden * 4.0
            sentence_by_charge = hidden * 5.0
            sentence = sentence_by_charge + charge * 0.25
            factor = hidden * 6.0
            return {
                "charge_logits": charge,
                "article_logits": article,
                "penalty_type_logits": penalty,
                "sentence_by_charge": sentence_by_charge,
                "sentence_months": sentence,
                "factor_logits": factor,
            }

    joint = TinyPredictor()
    parent = joint(torch.tensor(2.0))
    counterfactual = joint(torch.tensor(3.0))
    joint_loss = (
        (parent["charge_logits"] - counterfactual["charge_logits"]).square()
        + (counterfactual["sentence_months"] - parent["sentence_months"]).square()
        + _zero_output_anchor(counterfactual)
    )
    (joint_loss * loss_scale).backward()

    recomputed = TinyPredictor()
    with torch.no_grad():
        graph_free_parent = recomputed(torch.tensor(2.0))
    parent_charge = graph_free_parent["charge_logits"].detach().requires_grad_(True)
    parent_sentence = graph_free_parent["sentence_months"].detach().requires_grad_(True)
    live_counterfactual = recomputed(torch.tensor(3.0))
    split_loss = (
        (parent_charge - live_counterfactual["charge_logits"]).square()
        + (live_counterfactual["sentence_months"] - parent_sentence).square()
        + _zero_output_anchor(live_counterfactual)
    )
    (split_loss * loss_scale).backward()
    recomputed_parent = recomputed(torch.tensor(2.0))
    _backward_recomputed_parent(
        torch,
        recomputed_parent,
        charge_gradient=parent_charge.grad.detach(),
        sentence_gradient=parent_sentence.grad.detach(),
    )

    torch.testing.assert_close(recomputed.shared.grad, joint.shared.grad)


@pytest.mark.skipif(
    torch is None
    or not torch.distributed.is_available()
    or not torch.distributed.is_gloo_available(),
    reason="Torch Gloo distributed runtime is not installed",
)
def test_distributed_parent_requirements_align_heterogeneous_ranks(tmp_path):
    init_path = tmp_path / "gloo-init"
    output_dir = tmp_path / "rank-results"
    output_dir.mkdir()

    torch.multiprocessing.spawn(
        _distributed_parent_requirement_worker,
        args=(2, str(init_path), str(output_dir)),
        nprocs=2,
        join=True,
    )

    assert (output_dir / "rank-0.txt").read_text(encoding="utf-8") == "1,0"
    assert (output_dir / "rank-1.txt").read_text(encoding="utf-8") == "1,0"


@pytest.mark.parametrize(
    ("step", "maximum", "expected"),
    [
        (240, 1000, (240, 1000)),
        (1200, 1000, (1000, 1000)),
        (-5, 1000, (0, 1000)),
        (0, -10, (0, 0)),
        (2, 0, (0, 0)),
    ],
)
def test_training_progress_starts_at_restored_optimizer_step(step, maximum, expected):
    from legal_landscape.training.train import _training_progress_start

    assert _training_progress_start(step, maximum) == expected


def test_worker_progress_factory_is_disabled():
    from legal_landscape.training.train import _progress_enabled_for_process

    assert _progress_enabled_for_process(is_main_process=True) is True
    assert _progress_enabled_for_process(is_main_process=False) is False


def test_training_progress_reports_completed_optimizer_step_and_loss():
    from legal_landscape.training.train import _report_optimizer_step

    events = []
    reporter = RecordingProgress(events)
    with reporter.task("Train M", total=4, completed=2, experiment="M", loss="—") as task:
        # The caller invokes the helper only after gradient accumulation completes.
        for completed, step, loss in [(False, 2, 9.0), (True, 3, 1.25), (True, 4, 0.5)]:
            if completed:
                _report_optimizer_step(task, step, 4, loss)
    updates = [event[2] for event in events if isinstance(event, tuple)]
    assert [item["completed"] for item in updates] == [3, 4]
    assert [item["step"] for item in updates] == [3, 4]
    assert [item["loss"] for item in updates] == [1.25, 0.5]


@pytest.fixture
def training_progress_runtime(monkeypatch, tmp_path):
    """Replace only unavailable model/GPU dependencies; execute the real loop and files."""
    import sys

    from legal_landscape.progress import NullProgressReporter
    from legal_landscape.training import train

    events = []
    grad_enabled = {"value": True}
    gradient_leaves = []
    active_pair_types = {"value": ()}

    class Tensor:
        def __init__(self, value, *, absorbed=False, name=None):
            self.value = value
            self.shape = (1, 1)
            self.absorbed = absorbed
            self.name = name
            self.grad = None
            self.device = SimpleNamespace(type="cpu")

        def detach(self):
            return Tensor(self.value, absorbed=self.absorbed, name=self.name)

        def requires_grad_(self, enabled=True):
            if enabled:
                gradient_leaves.append(self)
            return self

        def sum(self):
            return Tensor(0.0, name=self.name)

        def __getitem__(self, index):
            return Tensor(self.value[index], name=self.name)

        def item(self):
            return self.value

        def __mul__(self, other):
            other_value = other.value if isinstance(other, Tensor) else other
            return Tensor(float(self.value) * float(other_value))

        def __add__(self, other):
            other_value = other.value if isinstance(other, Tensor) else other
            return Tensor(float(self.value) + float(other_value))

        __radd__ = __add__

        def __float__(self):
            return float(self.value)

        def cpu(self):
            return self

        def float(self):
            return self

        def argmax(self, **kwargs):
            return Tensor([0])

        def tolist(self):
            if self.absorbed:
                events.append("absorb")
            return self.value

    class Loader:
        def __init__(self, records, **kwargs):
            self.records = records

        def __len__(self):
            return len(self.records)

        def __iter__(self):
            records = (
                reversed(self.records)
                if self.records and "_prediction_index" in self.records[0]
                else self.records
            )
            for record in records:
                model_batch = {
                    name: Tensor([[1]])
                    for name in (
                        "input_ids",
                        "charge_targets",
                        "article_targets",
                        "article_mask",
                        "penalty_targets",
                        "sentence_targets",
                        "sentence_mask",
                        "factor_targets",
                    )
                }
                if "parent_record" in record:
                    batch = {
                        "parent": model_batch,
                        "counterfactual": model_batch,
                        "pair_types": (record["intervention_type"],),
                        "target_charge_indices": Tensor([0]),
                        "rank_direction": Tensor([0.0]),
                    }
                else:
                    batch = model_batch
                if "_prediction_index" in record:
                    index = Tensor([record["_prediction_index"]])
                    if "parent_record" in record:
                        batch["pair_indices"] = index
                    else:
                        batch["record_indices"] = index
                yield batch

    class Predictor:
        def __init__(self, *args, **kwargs):
            pass

        def parameters(self):
            return [SimpleNamespace(requires_grad=True)]

        def train(self):
            events.append("train")

        def eval(self):
            events.append("eval")

        def __call__(self, *args):
            events.append(("predictor", grad_enabled["value"]))
            return {
                "charge_logits": Tensor([[0.8]], name="charge_logits"),
                "article_logits": Tensor([[0.7]], name="article_logits"),
                "penalty_type_logits": Tensor([[1.0]], name="penalty_type_logits"),
                "sentence_by_charge": Tensor([[12.0]], name="sentence_by_charge"),
                "sentence_months": Tensor([12.0], name="sentence_months"),
                "factor_logits": Tensor([[0.5]], name="factor_logits"),
            }

    class Optimizer:
        def __init__(self, *args, **kwargs):
            pass

        def step(self):
            events.append("optimizer")

        def zero_grad(self):
            pass

    class Writer:
        def __init__(self, *args):
            pass

        def add_scalar(self, name, value, step):
            events.append(("tensorboard", step))

        def close(self):
            events.append("writer:close")

    records = [
        {
            "case_id": f"c{i}",
            "group_id": f"g{i}",
            "charges": ["盗窃"],
            "conviction_articles": ["criminal_law:264"],
            "penalty_type": "fixed_term",
            "imprisonment_months": 12,
        }
        for i in range(4)
    ]
    train_path = tmp_path / "train.jsonl"
    train_path.write_text("".join(json.dumps(row) + "\n" for row in records))
    evaluation_path = tmp_path / "evaluation.jsonl"
    evaluation_path.write_text("".join(json.dumps(row) + "\n" for row in records[:2]))
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text(
        "".join(
            json.dumps(
                {
                    "parent_case_id": row["case_id"],
                    "intervention_type": "invariant",
                    "validation": {"valid": True, "parsed": {"counterfactual_text": "改写事实"}},
                }
            )
            + "\n"
            for row in records[:2]
        )
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "charge_vocabulary": ["盗窃"],
                "article_vocabulary": ["criminal_law:264"],
                "penalty_vocabulary": ["fixed_term"],
            }
        )
    )
    inspected = train.InspectedModelConfig("fake", (), "fake", "fake", (), False, 1)
    monkeypatch.setattr(train, "load_backbone", lambda config: (object(), inspected))
    monkeypatch.setattr(train, "set_deterministic_seed", lambda seed: None)
    monkeypatch.setattr(train, "validate_loss_breakdown", lambda *args, **kwargs: None)
    loss_names = (
        "charge",
        "article",
        "sentence",
        "invariant",
        "boundary",
        "response",
        "factor",
        "total",
    )

    def compute_losses(**kwargs):
        if kwargs.get("pair_types"):
            active_pair_types["value"] = tuple(kwargs["pair_types"])
            values = {name: 0.0 for name in loss_names}
            pair_type = kwargs["pair_types"][0]
            loss_name = {
                "invariant": "invariant",
                "charge_flip": "boundary",
                "sentence_rank": "response",
            }[pair_type]
            values[loss_name] = 0.25
            values["total"] = 0.25
        else:
            values = {name: 0.0 for name in loss_names}
            for name in ("charge", "article", "sentence", "factor"):
                values[name] = 0.125
            values["total"] = 0.5
        return SimpleNamespace(**{name: Tensor(value) for name, value in values.items()})

    @contextmanager
    def no_grad():
        previous = grad_enabled["value"]
        grad_enabled["value"] = False
        try:
            yield
        finally:
            grad_enabled["value"] = previous

    class RngState:
        def clone(self):
            return RngState()

    rng_state = {"value": RngState()}

    class Autograd:
        @staticmethod
        def backward(tensors, grad_tensors):
            events.append(("parent_backward", tuple(grad_tensors)))

    modules = {
        "torch": SimpleNamespace(
            nn=SimpleNamespace(Module=object),
            optim=SimpleNamespace(AdamW=Optimizer),
            no_grad=no_grad,
            sigmoid=lambda value: value,
            get_rng_state=lambda: rng_state["value"],
            set_rng_state=lambda state: rng_state.__setitem__("value", state.clone()),
            cuda=SimpleNamespace(is_available=lambda: False),
            zeros_like=lambda value: Tensor(0.0, name=value.name),
            autograd=Autograd(),
            tensor=lambda value, **kwargs: Tensor(list(value)),
            int64=object(),
        ),
        "torch.utils.data": SimpleNamespace(DataLoader=Loader),
        "torch.utils.tensorboard": SimpleNamespace(SummaryWriter=Writer),
        "transformers": SimpleNamespace(
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: SimpleNamespace(pad_token_id=0)
            )
        ),
        "legal_landscape.models.losses": SimpleNamespace(compute_typed_losses=compute_losses),
        "legal_landscape.models.predictor": SimpleNamespace(LegalLandscapePredictor=Predictor),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    def run(
        *,
        is_main=True,
        restored_step=0,
        max_steps=2,
        fail_gather=False,
        experiment_name="B3",
        gradient_accumulation_steps=2,
        pair_type="invariant",
        remote_parent_requirements=(0, 0),
    ):
        class Accelerator:
            def __init__(self, **kwargs):
                self.is_main_process = is_main
                self.num_processes = 2
                self.sync_gradients = False
                self.microbatches = 0
                self.gradient_accumulation_steps = kwargs["gradient_accumulation_steps"]

            def prepare(self, *args):
                return args if len(args) > 1 else args[0]

            @contextmanager
            def accumulate(self, predictor):
                self.microbatches += 1
                self.sync_gradients = (
                    self.microbatches % self.gradient_accumulation_steps == 0
                )
                yield

            def backward(self, loss):
                events.append(("backward", float(loss)))
                if float(loss) == 0.25:
                    for leaf in gradient_leaves:
                        if (
                            leaf.name == "charge_logits"
                            and "invariant" in active_pair_types["value"]
                        ):
                            leaf.grad = Tensor(0.1, name="charge_gradient")
                        if (
                            leaf.name == "sentence_months"
                            and "sentence_rank" in active_pair_types["value"]
                        ):
                            leaf.grad = Tensor(0.1, name="sentence_gradient")

            def reduce(self, value, reduction):
                assert reduction == "sum"
                events.append(("reduce", tuple(value.value)))
                return Tensor(
                    [
                        value.value[index] + remote_parent_requirements[index]
                        for index in range(2)
                    ]
                )

            def wait_for_everyone(self):
                pass

            def save_state(self, path):
                from pathlib import Path

                Path(path).mkdir(parents=True, exist_ok=True)
                events.append("checkpoint")

            def load_state(self, path):
                events.append("restore")

            def gather_for_metrics(self, values):
                events.append("gather")
                if fail_gather:
                    raise RuntimeError("gather failed")
                values[-1].absorbed = True
                return values

        monkeypatch.setitem(sys.modules, "accelerate", SimpleNamespace(Accelerator=Accelerator))
        reporter = RecordingProgress(events)
        enabled_values = []

        def factory(*, enabled):
            enabled_values.append(enabled)
            return reporter if enabled else NullProgressReporter()

        pair_rows = []
        for row in records[:2]:
            pair_rows.append(
                {
                    "parent_case_id": row["case_id"],
                    "intervention_type": pair_type,
                    "target_charge": "盗窃" if pair_type == "charge_flip" else None,
                    "rank_direction": 1 if pair_type == "sentence_rank" else None,
                    "validation": {
                        "valid": True,
                        "parsed": {"counterfactual_text": "改写事实"},
                    },
                }
            )
        pairs_path.write_text("".join(json.dumps(row) + "\n" for row in pair_rows))
        resume = None
        if restored_step:
            resume = tmp_path / "resume"
            resume.mkdir(exist_ok=True)
            (resume / "training_progress.json").write_text(
                json.dumps(
                    {
                        "steps": restored_step,
                        "completed_microbatches": 0,
                        "seed": 42,
                        "input_identity": {
                            "train": train._file_identity(train_path),
                            "counterfactual": train._file_identity(pairs_path),
                        },
                        "case_sampler": {"epoch": 0, "start_index": 0, "seed": 42},
                        "pair_sampler": None,
                    }
                )
            )
        result = train.run_real_training(
            {
                "model": {"path": "fake"},
                "training": {
                    "max_steps": max_steps,
                    "epochs": 1,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "checkpoint_every": 1,
                },
            },
            train_data=train_path,
            output_dir=tmp_path / "output",
            experiment_name=experiment_name,
            counterfactual_data=pairs_path,
            evaluation_data=evaluation_path,
            resume_from_checkpoint=resume,
            progress_factory=factory,
        )
        return result, reporter, enabled_values

    return run, events, tmp_path


def test_training_progress_factory_tracks_real_loop_optimizer_steps(training_progress_runtime):
    run, events, root = training_progress_runtime
    result, reporter, enabled = run()
    assert enabled == [True]
    training_updates = [
        event[2] for event in events if isinstance(event, tuple) and event[1] == "Train B3"
    ]
    assert [item["completed"] for item in training_updates] == [1, 2]
    assert [item["loss"] for item in training_updates] == [0.5, 0.5]
    assert result["steps"] == 2
    assert reporter.closed
    logs = [json.loads(line) for line in (root / "output/train.jsonl").read_text().splitlines()]
    assert [item["step"] for item in logs] == [1, 2]
    assert events.index(("tensorboard", 1)) < events.index("checkpoint")


def test_m_staged_backward_uses_two_backwards_and_one_optimizer_call(
    training_progress_runtime,
):
    run, events, root = training_progress_runtime

    run(experiment_name="M", max_steps=1, gradient_accumulation_steps=1)

    training_events = []
    for event in events:
        if event == "optimizer":
            training_events.append(event)
            break
        if isinstance(event, tuple) and event[0] in {
            "predictor",
            "backward",
            "parent_backward",
        }:
            training_events.append(
                event[0] if event[0] == "parent_backward" else event
            )
    assert training_events == [
        ("predictor", True),
        ("backward", 0.5),
        ("predictor", False),
        ("predictor", True),
        ("backward", 0.25),
        ("predictor", True),
        "parent_backward",
        "optimizer",
    ]
    log = json.loads((root / "output/train.jsonl").read_text().splitlines()[0])
    assert log == {
        "step": 1,
        "charge": 0.125,
        "article": 0.125,
        "sentence": 0.125,
        "invariant": 0.25,
        "boundary": 0.0,
        "response": 0.0,
        "factor": 0.125,
        "total": 0.75,
    }


def test_boundary_only_pair_skips_parent_forward_and_recomputation(
    training_progress_runtime,
):
    run, events, _root = training_progress_runtime

    run(
        experiment_name="M",
        max_steps=1,
        gradient_accumulation_steps=1,
        pair_type="charge_flip",
    )

    training_events = []
    for event in events:
        if event == "optimizer":
            training_events.append(event)
            break
        if isinstance(event, tuple) and event[0] in {
            "predictor",
            "backward",
            "parent_backward",
        }:
            training_events.append(
                event[0] if event[0] == "parent_backward" else event
            )
    assert training_events == [
        ("predictor", True),
        ("backward", 0.5),
        ("predictor", True),
        ("backward", 0.25),
        "optimizer",
    ]


def test_boundary_only_rank_follows_remote_parent_recompute_schedule(
    training_progress_runtime,
):
    run, events, _root = training_progress_runtime

    run(
        experiment_name="M",
        max_steps=1,
        gradient_accumulation_steps=1,
        pair_type="charge_flip",
        remote_parent_requirements=(1, 0),
    )

    training_events = []
    for event in events:
        if event == "optimizer":
            training_events.append(event)
            break
        if isinstance(event, tuple) and event[0] in {
            "predictor",
            "backward",
            "parent_backward",
        }:
            training_events.append(
                event[0] if event[0] == "parent_backward" else event
            )
    assert training_events == [
        ("predictor", True),
        ("backward", 0.5),
        ("predictor", False),
        ("predictor", True),
        ("backward", 0.25),
        ("predictor", True),
        "parent_backward",
        "optimizer",
    ]


def test_b3_staged_backward_keeps_one_backward_and_one_optimizer_call(
    training_progress_runtime,
):
    run, events, _root = training_progress_runtime

    run(experiment_name="B3", max_steps=1, gradient_accumulation_steps=1)

    training_events = [
        event
        for event in events
        if event == "optimizer" or (isinstance(event, tuple) and event[0] == "backward")
    ]
    assert training_events == [
        ("backward", 0.5),
        "optimizer",
    ]


def test_training_progress_worker_uses_null_reporter(training_progress_runtime):
    run, events, root = training_progress_runtime
    result, reporter, enabled = run(is_main=False)
    assert enabled == [False]
    assert reporter.tasks == []
    assert "reporter:enter" not in events
    assert not (root / "output/train.jsonl").exists()
    assert result["steps"] == 2


@pytest.mark.parametrize(("restored", "expected"), [(1, 2), (5, 5)])
def test_training_progress_resume_clamps_display_only(
    training_progress_runtime, restored, expected
):
    run, events, _root = training_progress_runtime
    result, reporter, _enabled = run(restored_step=restored)
    assert result["steps"] == expected
    initial = reporter.tasks[0]
    assert initial["total"] == 2
    assert initial["completed"] == 2
    assert reporter.started_tasks[0]["completed"] == min(restored, 2)
    if restored > 2:
        assert "optimizer" not in events
    assert reporter.closed


@pytest.mark.parametrize(("loader", "expected"), [([], 0), ([1, 2, 3], 3)])
def test_prediction_progress_total_counts_loader_batches(loader, expected):
    from legal_landscape.training.train import _prediction_progress_total

    assert _prediction_progress_total(loader) == expected


def test_prediction_progress_batches_have_independent_kind_and_counts():
    from legal_landscape.training.train import _prediction_progress_total, _report_prediction_batch

    events = []
    reporter = RecordingProgress(events)
    for kind, loader in [("static", [1, 2, 3]), ("counterfactual", [1, 2])]:
        with reporter.task(
            kind, total=_prediction_progress_total(loader), kind=kind, batches=0
        ) as task:
            for batches, _batch in enumerate(loader, start=1):
                _report_prediction_batch(task, kind, batches)
    updates = [event[2] for event in events if isinstance(event, tuple)]
    assert [(item["kind"], item["batches"], item["completed"]) for item in updates] == [
        ("static", 1, 1),
        ("static", 2, 2),
        ("static", 3, 3),
        ("counterfactual", 1, 1),
        ("counterfactual", 2, 2),
    ]


def test_prediction_progress_advances_after_gather_and_absorption(training_progress_runtime):
    run, events, root = training_progress_runtime
    result, reporter, _enabled = run()
    predictions = [task for task in reporter.tasks if "kind" in task]
    assert [
        (task["kind"], task["total"], task["completed"], task["batches"]) for task in predictions
    ] == [("static", 2, 2, 2), ("counterfactual", 2, 2, 2)]
    transitions = [
        "advance" if isinstance(event, tuple) else event
        for event in events
        if event in ("gather", "absorb") or (isinstance(event, tuple) and event[0] == "advance")
    ]
    assert transitions == ["gather", "absorb", "advance"] * 4
    static_rows = [
        json.loads(row)
        for row in (root / "output/predictions/static.jsonl").read_text().splitlines()
    ]
    pair_rows = [
        json.loads(row)
        for row in (root / "output/predictions/counterfactual.jsonl").read_text().splitlines()
    ]
    assert [row["case_id"] for row in static_rows] == ["c0", "c1"]
    assert [row["case_id"] for row in pair_rows] == ["c0", "c1"]
    assert "static_predictions" in result and "counterfactual_predictions" in result
    assert events.index("task:exit:Predict static") < events.index(
        "task:enter:Predict counterfactual"
    )
    assert events[-1] == "reporter:exit"


def test_prediction_progress_failed_gather_does_not_advance_and_closes(training_progress_runtime):
    run, events, _root = training_progress_runtime
    with pytest.raises(RuntimeError, match="gather failed"):
        run(fail_gather=True)
    assert not any(isinstance(event, tuple) and event[0] == "advance" for event in events)
    assert "task:exit:Predict static" in events
    assert events[-1] == "reporter:exit"


class TrainingPlanTests(unittest.TestCase):
    def test_deterministic_sampler_resume_preserves_remaining_order(self) -> None:
        sampler = DeterministicEpochSampler(6, seed=42)
        full_order = list(sampler)
        self.assertEqual(full_order, [3, 1, 2, 4, 0, 5])

        sampler.advance(2)
        state = sampler.state_dict()
        restored = DeterministicEpochSampler(6, seed=42)
        restored.load_state_dict(state)
        self.assertEqual(list(restored), [2, 4, 0, 5])

        restored.next_epoch()
        self.assertEqual(restored.state_dict(), {"epoch": 1, "start_index": 0, "seed": 42})

    def test_resume_metadata_rejects_seed_or_input_changes(self) -> None:
        progress = {
            "seed": 42,
            "input_identity": {"train": {"sha256": "abc", "bytes": 10}},
        }
        validate_resume_metadata(
            progress,
            expected_seed=42,
            expected_input_identity={"train": {"sha256": "abc", "bytes": 10}},
        )
        with self.assertRaisesRegex(ValueError, "seed"):
            validate_resume_metadata(
                progress,
                expected_seed=2026,
                expected_input_identity=progress["input_identity"],
            )
        with self.assertRaisesRegex(ValueError, "input"):
            validate_resume_metadata(
                progress,
                expected_seed=42,
                expected_input_identity={"train": {"sha256": "changed", "bytes": 10}},
            )

    def test_experiment_matrix_has_all_requested_variants(self) -> None:
        names = ("B0", "B1", "B2", "B3", "B4", "B5", "M", "A1", "A2", "A3", "A4", "A5", "A6")
        experiments = {name: build_experiment(name) for name in names}
        self.assertFalse(experiments["B3"].use_factors)
        self.assertTrue(experiments["B4"].use_factors)
        self.assertTrue(experiments["M"].typed_counterfactuals)
        self.assertFalse(experiments["A1"].use_invariant)
        self.assertTrue(experiments["A5"].hard_charge_sentence)
        self.assertFalse(experiments["A6"].filter_counterfactuals)

    def test_model_loader_choice_is_derived_from_local_config(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                json.dumps({"model_type": "qwen-test", "architectures": ["QwenTestForCausalLM"]}),
                encoding="utf-8",
            )
            inspected = inspect_model_config(root)
        self.assertEqual(inspected.loader_class, "AutoModelForCausalLM")
        self.assertEqual(inspected.architecture_kind, "causal_lm")
        self.assertEqual(inspected.text_backbone_path, ("model",))
        self.assertEqual(inspected.model_type, "qwen-test")

    def test_multimodal_qwen_uses_image_text_loader_and_language_backbone(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        config = {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {
                "architectures": ["Qwen3_5ForCausalLM"],
                "hidden_size": 4096,
            },
            "vision_config": {"hidden_size": 1024},
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            inspected = inspect_model_config(root)
        self.assertEqual(inspected.loader_class, "AutoModelForImageTextToText")
        self.assertEqual(inspected.architecture_kind, "conditional_generation")
        self.assertEqual(inspected.text_backbone_path, ("model", "language_model"))
        self.assertTrue(inspected.has_visual_components)
        self.assertEqual(inspected.hidden_size, 4096)

    def test_training_plan_never_loads_weights(self) -> None:
        plan = training_plan(
            {
                "model": {"path": "/missing/model", "quantization": "nf4"},
                "training": {"seed": 42, "mixed_precision": "bf16"},
            },
            experiment="M",
        )
        self.assertEqual(plan["model_path"], "/missing/model")
        self.assertEqual(plan["mode"], "full")
        self.assertTrue(plan["requires_execute"])

    def test_model_input_has_explicit_dataset_defendant_fact_and_factors(self) -> None:
        text = format_model_input(
            {
                "dataset": "cmdl",
                "target_defendant": "某甲",
                "fact_conservative": "案件事实。",
                "fact_strict": "严格屏蔽后的案件事实。",
                "factors": {"amount": 1000.0, "confession": True},
            },
            use_factors=True,
        )
        self.assertIn("[DATASET]\ncmdl", text)
        self.assertIn("[TARGET_DEFENDANT]\n某甲", text)
        self.assertIn("[FACT]\n严格屏蔽后的案件事实。", text)
        self.assertNotIn("[FACT]\n案件事实。", text)
        self.assertIn("[FACTORS]", text)
        self.assertIn('"amount": 1000.0', text)

    def test_counterfactual_pair_loader_filters_invalid_by_default(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        parent = {
            "case_id": "c1",
            "dataset": "cail",
            "target_defendant": "某甲",
            "fact_conservative": "父事实",
            "factors": {},
            "charges": ["盗窃"],
        }
        valid = {
            "parent_case_id": "c1",
            "intervention_type": "invariant",
            "validation": {"valid": True, "parsed": {"counterfactual_text": "反事实"}},
            "target_charge": None,
            "rank_direction": None,
        }
        invalid = {
            **valid,
            "validation": {"valid": False, "parsed": {"counterfactual_text": "坏样本"}},
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cf.jsonl"
            path.write_text(
                "\n".join(json.dumps(x, ensure_ascii=False) for x in (valid, invalid)),
                encoding="utf-8",
            )
            pairs = load_counterfactual_pairs(path, {"c1": parent}, filter_valid=True)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["parent_text"], "父事实")
        self.assertEqual(pairs[0]["counterfactual_text"], "反事实")

    def test_static_prediction_rows_preserve_labels_and_group_ids(self) -> None:
        records = [
            {
                "case_id": "c1",
                "group_id": "g1",
                "charges": ["盗窃"],
                "conviction_articles": ["criminal_law:264"],
                "penalty_type": "fixed_term",
                "imprisonment_months": 12,
            },
            {
                "case_id": "c2",
                "group_id": "g2",
                "charges": ["诈骗"],
                "conviction_articles": ["criminal_law:266"],
                "penalty_type": "life",
                "imprisonment_months": None,
            },
        ]
        rows = make_static_prediction_rows(
            records,
            charge_probabilities=[[0.8, 0.2], [0.3, 0.7]],
            article_probabilities=[[0.9, 0.1], [0.2, 0.8]],
            penalty_indices=[0, 1],
            sentence_months=[11.5, 99.0],
            charge_vocabulary=["盗窃", "诈骗"],
            article_vocabulary=["criminal_law:264", "criminal_law:266"],
            penalty_vocabulary=["fixed_term", "life"],
        )
        self.assertEqual(rows[0]["group_id"], "g1")
        self.assertEqual(rows[0]["true_charges"], [1, 0])
        self.assertEqual(rows[0]["predicted_charges"], [1, 0])
        self.assertEqual(rows[0]["true_articles"], [1, 0])
        self.assertEqual(rows[0]["predicted_articles"], [1, 0])
        self.assertEqual(rows[0]["predicted_article_labels"], ["criminal_law:264"])
        self.assertTrue(rows[0]["correct"])
        self.assertIsNone(rows[1]["true_months"])

    def test_static_prediction_rows_support_multiple_predicted_charges(self) -> None:
        record = {
            "case_id": "c1",
            "group_id": "g1",
            "charges": ["盗窃", "诈骗"],
            "conviction_articles": ["criminal_law:264", "criminal_law:266"],
            "penalty_type": "fixed_term",
            "imprisonment_months": 12,
        }
        row = make_static_prediction_rows(
            [record],
            charge_probabilities=[[0.8, 0.7]],
            article_probabilities=[[0.8, 0.7]],
            penalty_indices=[0],
            sentence_months=[12.0],
            charge_vocabulary=["盗窃", "诈骗"],
            article_vocabulary=["criminal_law:264", "criminal_law:266"],
            penalty_vocabulary=["fixed_term"],
        )[0]
        self.assertEqual(row["predicted_charges"], [1, 1])
        self.assertTrue(row["correct"])

    def test_counterfactual_prediction_rows_use_charge_index_targets(self) -> None:
        pairs = [
            {
                "parent_record": {"case_id": "c1", "group_id": "g1"},
                "intervention_type": "charge_flip",
                "target_charge": "诈骗",
                "rank_direction": None,
            }
        ]
        rows = make_counterfactual_prediction_rows(
            pairs,
            parent_probabilities=[[0.9, 0.1]],
            counterfactual_probabilities=[[0.2, 0.8]],
            parent_sentences=[12.0],
            counterfactual_sentences=[12.0],
            charge_vocabulary=["盗窃", "诈骗"],
        )
        self.assertEqual(rows[0]["target_charge"], 1)
        self.assertEqual(rows[0]["group_id"], "g1")


@unittest.skipIf(torch is None, "PyTorch is not installed in this interpreter")
class TorchTrainingTests(unittest.TestCase):
    def test_masked_mean_pool_reduces_token_states(self) -> None:
        from legal_landscape.training.train import masked_mean_pool

        hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [99.0, 99.0]]])
        mask = torch.tensor([[1, 1, 0]])
        pooled = masked_mean_pool(hidden, mask)
        self.assertEqual(tuple(pooled.shape), (1, 2))
        self.assertTrue(torch.equal(pooled, torch.tensor([[2.0, 4.0]])))

    def test_dummy_train_step_is_finite(self) -> None:
        from legal_landscape.training.train import run_dummy_train_step

        value = run_dummy_train_step(seed=42)
        self.assertTrue(torch.isfinite(torch.tensor(value)))
        self.assertGreater(value, 0.0)

    def test_nan_and_nonzero_empty_loss_write_diagnostics(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from legal_landscape.models.losses import LossBreakdown
        from legal_landscape.training.train import validate_loss_breakdown

        zero = torch.tensor(0.0)
        bad = LossBreakdown(
            charge=torch.tensor(float("nan")),
            article=zero,
            sentence=zero,
            invariant=zero,
            boundary=zero,
            response=zero,
            factor=torch.tensor(1.0),
            total=zero,
            active_counts={
                "charge": 1,
                "article": 0,
                "sentence": 0,
                "invariant": 0,
                "boundary": 0,
                "response": 0,
                "factor": 0,
            },
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostic.json"
            with self.assertRaises(FloatingPointError):
                validate_loss_breakdown(bad, path, step=3)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["step"], 3)
        self.assertIn("charge", payload["errors"])
        self.assertIn("factor:nonzero_without_samples", payload["errors"])


if __name__ == "__main__":
    unittest.main()
