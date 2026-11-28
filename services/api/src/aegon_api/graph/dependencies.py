"""Dependency-injection contracts and deterministic offline adapters."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from typing import Protocol

from aegon_api.guards import EvidenceGuard, InputGuard, OutputGuard
from aegon_api.ir import (
    GradeConfig,
    NoopParentDocumentExpander,
    NoopReranker,
    ParentDocumentExpander,
    RerankConfig,
    Reranker,
    RewriteConfig,
)
from aegon_api.llm import LlmClient
from aegon_api.retrieval import TextEmbedder


class NodeSpanHook(Protocol):
    """OTel-compatible lifecycle hook without exposing state payloads."""

    def span(
        self,
        node_name: str,
        *,
        tenant_id: str,
        conversation_id: str,
    ) -> AbstractAsyncContextManager[None]: ...


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

    llm: LlmClient
    embedder: TextEmbedder
    input_guard: InputGuard = field(default_factory=InputGuard)
    evidence_guard: EvidenceGuard = field(default_factory=EvidenceGuard)
    output_guard: OutputGuard = field(default_factory=OutputGuard)
    reranker: Reranker = field(default_factory=NoopReranker)
    parent_expander: ParentDocumentExpander = field(default_factory=NoopParentDocumentExpander)
    rewrite_config: RewriteConfig = field(default_factory=RewriteConfig)
    rerank_config: RerankConfig = field(default_factory=RerankConfig)
    grade_config: GradeConfig = field(default_factory=GradeConfig)
    span_hook: NodeSpanHook = field(default_factory=NoOpSpanHook)
    node_timeout_seconds: float = 10.0
    generation_max_tokens: int = 1_000
    verification_max_tokens: int = 1_000
    verification_support_threshold: float = 0.8
    node_charges: dict[str, NodeCharge] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        if self.node_timeout_seconds <= 0:
            raise ValueError("node_timeout_seconds must be positive")
        if self.generation_max_tokens < 1 or self.verification_max_tokens < 1:
            raise ValueError("LLM output token bounds must be positive")
        if not 0 < self.verification_support_threshold <= 1:
            raise ValueError("verification_support_threshold must be in (0, 1]")
