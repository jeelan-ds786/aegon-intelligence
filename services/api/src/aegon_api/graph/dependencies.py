"""Dependency-injection contracts and deterministic offline adapters."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from typing import Protocol

from aegon_api.guards import EvidenceGuard, InputGuard, OutputGuard
from aegon_api.retrieval import TextEmbedder


class AgentLlm(Protocol):
    """Bounded generation interface used by graph nodes."""

    async def generate(self, prompt: str, evidence: Sequence[str]) -> str: ...


class NodeSpanHook(Protocol):
    """OTel-compatible lifecycle hook without exposing state payloads."""

    def span(
        self,
        node_name: str,
        *,
        tenant_id: str,
        conversation_id: str,
    ) -> AbstractAsyncContextManager[None]: ...


class FakeLlm:
    """Deterministic fake used by offline tests and CI."""

    async def generate(self, prompt: str, evidence: Sequence[str]) -> str:
        del prompt
        return (
            f"Grounded answer from {len(evidence)} approved evidence item(s)."
            if evidence
            else "No grounded answer is available."
        )


class NoOpSpanHook:
    """No-op span hook retained until OTel instrumentation is enabled."""

    @asynccontextmanager
    async def span(
        self,
        node_name: str,
        *,
        tenant_id: str,
        conversation_id: str,
    ) -> AsyncGenerator[None]:
        del node_name, tenant_id, conversation_id
        yield


@dataclass(frozen=True, slots=True)
class NodeCharge:
    tokens: int = 0
    cost_microusd: int = 0

    def __post_init__(self) -> None:
        if self.tokens < 0 or self.cost_microusd < 0:
            raise ValueError("node charges must be non-negative")


@dataclass(frozen=True, slots=True)
class GraphDependencies:
    """Explicit graph dependencies; credentials belong inside adapters, never state."""

    llm: AgentLlm
    embedder: TextEmbedder
    input_guard: InputGuard = field(default_factory=InputGuard)
    evidence_guard: EvidenceGuard = field(default_factory=EvidenceGuard)
    output_guard: OutputGuard = field(default_factory=OutputGuard)
    span_hook: NodeSpanHook = field(default_factory=NoOpSpanHook)
    node_timeout_seconds: float = 10.0
    node_charges: dict[str, NodeCharge] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        if self.node_timeout_seconds <= 0:
            raise ValueError("node_timeout_seconds must be positive")
