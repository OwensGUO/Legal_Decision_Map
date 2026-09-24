"""Mock and local-vLLM counterfactual generation with resume support."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import time
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from legal_landscape.counterfactual.prompts import build_messages
from legal_landscape.counterfactual.validators import ValidationResult, validate_generation
from legal_landscape.factors.schema import InterventionSpec


class CounterfactualGenerator(Protocol):
    def generate(self, parent_text: str, spec: InterventionSpec, *, seed: int) -> str: ...


class GenerationTransportError(RuntimeError):
    """The service could not return a response; the request is left for ``--resume``."""


class GenerationRejectedError(RuntimeError):
    """The service refused the request (HTTP 4xx); retrying cannot succeed."""


@dataclass(frozen=True)
class GenerationRequest:
    parent_text: str
    spec: InterventionSpec

    @property
    def generation_id(self) -> str:
        material = f"{self.spec.parent_case_id}\0{self.spec.target_defendant}\0{self.spec.rule_id}"
        return hashlib.sha256(material.encode()).hexdigest()[:24]


@dataclass
class GenerationResult:
    records: list[dict[str, Any]] = field(default_factory=list)
    skipped_completed: int = 0
    skipped_duplicate: int = 0
    failed: list[dict[str, str]] = field(default_factory=list)


class MockGenerator:
    """Deterministic CPU generator that obeys the production JSON contract.

    Like a faithful rewrite, it keeps the parent text and appends the realized change.
    """

    model_revision = "mock-v1"

    def generate(self, parent_text: str, spec: InterventionSpec, *, seed: int) -> str:
        del seed
        fragments = [spec.target_defendant]
        for changed in spec.changed_fields:
            value = getattr(spec.target_factors, changed)
            if changed == "restitution":
                fragments.append("已完成全部退赃" if value else "未退赃")
            elif changed == "conduct":
                fragments.append(str(value))
            elif changed == "amount":
                fragments.append(f"涉案金额调整为{value:.2f}元")
            else:
                fragments.append(f"{changed}调整为{value}")
        payload = {
            "counterfactual_text": parent_text + "，".join(fragments) + "。",
            "target_defendant": spec.target_defendant,
            "changed_fields": list(spec.changed_fields),
            "realized_factors": spec.target_factors.to_dict(),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


class VLLMHTTPGenerator:
    """Thread-safe client for an explicitly configured OpenAI-compatible local vLLM service."""

    def __init__(
        self,
        endpoint: str,
        *,
        model: str,
        timeout: float = 600.0,
        sampling: dict[str, Any] | None = None,
        allow_remote: bool = False,
        max_connections: int = 32,
        request_retries: int = 3,
        backoff_seconds: float = 2.0,
    ) -> None:
        from urllib.parse import urlparse

        hostname = urlparse(endpoint).hostname
        local = hostname == "localhost"
        if hostname:
            try:
                local = local or ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                pass
        if not local and not allow_remote:
            raise ValueError("non-loopback vLLM endpoint requires explicit allow_remote=True")
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.sampling = sampling or {}
        self.max_connections = max_connections
        self.request_retries = request_retries
        self.backoff_seconds = backoff_seconds
        self._client: Any = None

    def _http(self) -> Any:
        if self._client is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover - exercised by requirements checker
                raise RuntimeError("httpx is required for vLLM generation") from exc
            self._client = httpx.Client(
                timeout=self.timeout,
                limits=httpx.Limits(
                    max_connections=self.max_connections,
                    max_keepalive_connections=self.max_connections,
                ),
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def generate(self, parent_text: str, spec: InterventionSpec, *, seed: int) -> str:
        import httpx

        client = self._http()
        payload = {
            "model": self.model,
            "messages": build_messages(parent_text, spec),
            "seed": seed,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
            **self.sampling,
        }
        last_error = ""
        for attempt in range(self.request_retries + 1):
            if attempt:
                time.sleep(self.backoff_seconds * 2 ** (attempt - 1))
            try:
                response = client.post(self.endpoint, json=payload)
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                continue
            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                continue
            if response.status_code >= 400:
                # Client errors such as an over-long prompt will not succeed on retry.
                raise GenerationRejectedError(f"HTTP {response.status_code}: {response.text[:200]}")
            data = response.json()
            return str(data["choices"][0]["message"]["content"])
        raise GenerationTransportError(last_error)


def _completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    result: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                result.add(str(json.loads(line)["generation_id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return result


def _realize(
    request: GenerationRequest,
    generator: CounterfactualGenerator,
    *,
    model_revision: str,
    prompt_version: str,
    sampling: dict[str, Any],
    seed: int,
    retries: int,
    min_similarity: float,
) -> dict[str, Any]:
    """Generate one request, retrying outputs that fail validation."""
    raw = ""
    result = None
    retry_count = 0
    for retry_count in range(retries + 1):
        try:
            raw = generator.generate(request.parent_text, request.spec, seed=seed + retry_count)
        except GenerationRejectedError as exc:
            raw = ""
            result = ValidationResult(False, (f"request_rejected: {exc}",), None, None)
            break
        result = validate_generation(
            request.parent_text, request.spec, raw, min_similarity=min_similarity
        )
        if result.valid:
            break
    assert result is not None
    spec = request.spec
    return {
        "generation_id": request.generation_id,
        "parent_case_id": spec.parent_case_id,
        "target_defendant": spec.target_defendant,
        "intervention_type": spec.intervention_type,
        "source_factors": spec.source_factors.to_dict(),
        "target_factors": spec.target_factors.to_dict(),
        "changed_fields": list(spec.changed_fields),
        "target_charge": spec.target_charge,
        "rank_direction": spec.rank_direction,
        "rule_id": spec.rule_id,
        "model_revision": model_revision,
        "prompt_version": prompt_version,
        "sampling": sampling,
        "seed": seed,
        "raw_response": raw,
        "retry_count": retry_count,
        "validation": result.to_dict(),
    }


def generate_records(
    requests: list[GenerationRequest],
    generator: CounterfactualGenerator,
    output_path: str | Path,
    *,
    model_revision: str,
    prompt_version: str,
    sampling: dict[str, Any],
    seed: int,
    retries: int,
    resume: bool = False,
    min_similarity: float = 0.5,
    concurrency: int = 1,
) -> GenerationResult:
    """Generate validated records concurrently and durably append each completed request.

    Records are written from the calling thread as they finish, so the file order follows
    completion order. Requests that fail at the transport level are not written and are
    retried by the next ``resume`` run.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed(destination) if resume else set()
    outcome = GenerationResult()
    pending: list[GenerationRequest] = []
    seen: set[str] = set()
    for request in requests:
        if request.generation_id in completed:
            outcome.skipped_completed += 1
        elif request.generation_id in seen:
            outcome.skipped_duplicate += 1
        else:
            seen.add(request.generation_id)
            pending.append(request)

    def work(request: GenerationRequest) -> dict[str, Any]:
        return _realize(
            request,
            generator,
            model_revision=model_revision,
            prompt_version=prompt_version,
            sampling=sampling,
            seed=seed,
            retries=retries,
            min_similarity=min_similarity,
        )

    with (
        destination.open("a" if resume else "w", encoding="utf-8") as handle,
        ThreadPoolExecutor(max_workers=concurrency) as executor,
    ):
        queue: Iterator[GenerationRequest] = iter(pending)
        in_flight: dict[Future[dict[str, Any]], GenerationRequest] = {}

        def refill() -> None:
            # Bound the number of queued futures so huge request lists stay cheap.
            while len(in_flight) < concurrency * 2:
                request = next(queue, None)
                if request is None:
                    return
                in_flight[executor.submit(work, request)] = request

        refill()
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                request = in_flight.pop(future)
                try:
                    record = future.result()
                except GenerationTransportError as exc:
                    outcome.failed.append(
                        {"generation_id": request.generation_id, "error": str(exc)}
                    )
                    continue
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                outcome.records.append(record)
            refill()
    return outcome
