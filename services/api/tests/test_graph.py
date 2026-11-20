from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import pytest
from aegon_api.graph import (
    FakeLlm,
    GraphDependencies,
    NodeCharge,
    ServerContext,
    build_graph,
    create_initial_state,
    execution_config,
)
from aegon_api.graph.diagram import render_architecture_page
from aegon_api.graph.routing import (
    NodeDestination,
    after_evidence_guard,
    after_grade,
    after_input_guard,
    after_output_guard,
    after_route,
    after_tools,
    after_verify,
)
from aegon_api.graph.state import AgentState
from aegon_common.embeddings import HashingEmbedder, HashingEmbedderConfig
from aegon_common.models import Citation, Evidence, QuoteSpan
from langgraph.checkpoint.memory import InMemorySaver

type EdgeFunction = Callable[[AgentState], NodeDestination]


class _SlowLlm:
    async def generate(self, prompt: str, evidence: Sequence[str]) -> str:
        del prompt, evidence
        await asyncio.sleep(0.05)
        return "too late"


def _context(conversation_id: str = "conversation-1") -> ServerContext:
    return ServerContext(
        tenant_id="tenant-a",
        scopes=("rag:read",),
        session_id="session-1",
        conversation_id=conversation_id,
    )


def _evidence() -> Evidence:
    text = "The approved retention period is 30 days."
    return Evidence(
        citation=Citation(
            chunk_id="chunk-1",
            source_uri="https://example.test/guide",
            locator="section:1",
            quote_span=QuoteSpan(start=0, end=len(text)),
        ),
        text=text,
        relevance_score=0.9,
    )


def _dependencies(
    *,
    charges: dict[str, NodeCharge] | None = None,
) -> GraphDependencies:
    return GraphDependencies(
        llm=FakeLlm(),
        embedder=HashingEmbedder(HashingEmbedderConfig(dim=8)),
        node_charges=charges or {},
    )


@pytest.mark.parametrize(
    ("edge", "updates", "destination"),
    [
        (after_input_guard, {}, "modality_detect"),
        (after_input_guard, {"refusal_reason": "blocked"}, "refuse"),
        (after_route, {"route": "retrieve"}, "retrieve"),
        (after_route, {"route": "tools"}, "tools"),
        (after_route, {"escalation_reason": "review"}, "escalate"),
        (after_evidence_guard, {}, "rerank"),
        (after_grade, {"grade": "pass"}, "generate"),
        (after_grade, {"grade": "retry", "rewrite_count": 1}, "rewrite"),
        (after_grade, {"grade": "insufficient"}, "refuse"),
        (after_tools, {"action_required": True}, "confirm_action"),
        (after_tools, {"action_required": False}, "generate"),
        (after_verify, {"verification": "pass"}, "output_guard"),
        (after_verify, {"verification": "retry", "rewrite_count": 1}, "rewrite"),
        (after_verify, {"verification": "fail"}, "refuse"),
        (after_output_guard, {}, "__end__"),
        (after_output_guard, {"refusal_reason": "blocked"}, "refuse"),
    ],
)
def test_conditional_edges_are_pure(
    edge: EdgeFunction,
    updates: dict[str, object],
    destination: str,
) -> None:
    state = create_initial_state(context=_context(), question="What is the retention period?")
    state = cast(AgentState, {**state, **updates})
    before = dict(state)

    assert edge(state) == destination
    assert state == before


@pytest.mark.parametrize("edge", [after_grade, after_verify])
def test_max_loop_guard_refuses(edge: EdgeFunction) -> None:
    state = create_initial_state(
        context=_context(),
        question="Retry this.",
        max_rewrites=2,
    )
    state["rewrite_count"] = 2
    if edge is after_grade:
        state["grade"] = "retry"
    else:
        state["verification"] = "retry"

    assert edge(state) == "refuse"


def test_budget_exceeded_routes_to_refusal() -> None:
    graph = build_graph(
        _dependencies(charges={"input_guard": NodeCharge(tokens=2)}),
        checkpointer=InMemorySaver(),
    )
    state = create_initial_state(
        context=_context(),
        question="What is the retention period?",
        max_tokens=1,
    )

    result = asyncio.run(
        graph.ainvoke(  # pyright: ignore[reportUnknownMemberType]
            state, execution_config(_context())
        )
    )

    assert result["refusal_reason"] == "turn_budget_exceeded"
    assert result["final_output"].startswith("I cannot complete")
    assert result["tokens_used"] == 0


def test_checkpoint_resume_after_retrieve_matches_uninterrupted_result() -> None:
    interrupted = build_graph(
        _dependencies(),
        checkpointer=InMemorySaver(),
        interrupt_after=("retrieve",),
    )
    uninterrupted = build_graph(_dependencies(), checkpointer=InMemorySaver())
    initial = create_initial_state(
        context=_context(),
        question="What is the retention period?",
        evidence=(_evidence(),),
    )
    interrupted_config = execution_config(_context())
    uninterrupted_config = execution_config(_context())

    partial = asyncio.run(
        interrupted.ainvoke(  # pyright: ignore[reportUnknownMemberType]
            initial, interrupted_config
        )
    )
    snapshot = asyncio.run(interrupted.aget_state(interrupted_config))
    resumed = asyncio.run(
        interrupted.ainvoke(  # pyright: ignore[reportUnknownMemberType]
            None, interrupted_config
        )
    )
    expected = asyncio.run(
        uninterrupted.ainvoke(  # pyright: ignore[reportUnknownMemberType]
            initial, uninterrupted_config
        )
    )

    assert partial["node_trace"][-1] == "retrieve"
    assert snapshot.next == ("evidence_guard",)
    assert resumed == expected
    assert resumed["final_output"] == "Grounded answer from 1 approved evidence item(s)."


def test_execution_config_composes_checkpoint_thread_id_and_recursion_limit() -> None:
    config = execution_config(_context(), recursion_limit=17)

    assert config.get("configurable", {})["thread_id"] == "9:session-1conversation-1"
    assert config.get("recursion_limit") == 17


def test_per_node_timeout_is_enforced() -> None:
    dependencies = GraphDependencies(
        llm=_SlowLlm(),
        embedder=HashingEmbedder(HashingEmbedderConfig(dim=8)),
        node_timeout_seconds=0.001,
    )
    graph = build_graph(dependencies)
    state = create_initial_state(
        context=_context(),
        question="What is the retention period?",
        evidence=(_evidence(),),
    )

    with pytest.raises(TimeoutError):
        asyncio.run(
            graph.ainvoke(  # pyright: ignore[reportUnknownMemberType]
                state, execution_config(_context())
            )
        )


def test_checked_in_graph_document_is_generated_from_topology() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    document = repository_root / "docs" / "architecture" / "graph.md"

    assert document.read_text(encoding="utf-8") == render_architecture_page()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ServerContext(
            tenant_id="tenant-a",
            scopes=(),
            session_id="session-1",
            conversation_id="conversation-1",
        ),
        lambda: create_initial_state(context=_context(), question=""),
        lambda: create_initial_state(
            context=_context(),
            question="valid",
            attachment_media_types=("",),
        ),
        lambda: create_initial_state(context=_context(), question="valid", max_tokens=0),
        lambda: NodeCharge(tokens=-1),
        lambda: GraphDependencies(
            llm=FakeLlm(),
            embedder=HashingEmbedder(HashingEmbedderConfig(dim=8)),
            node_timeout_seconds=0,
        ),
    ],
)
def test_graph_boundaries_reject_invalid_configuration(factory: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        factory()


def test_execution_config_rejects_unbounded_recursion() -> None:
    with pytest.raises(ValueError, match="recursion_limit"):
        execution_config(_context(), recursion_limit=0)
