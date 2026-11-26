"""LangGraph topology, checkpoint construction, and execution configuration."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph  # pyright: ignore[reportMissingTypeStubs]
from langgraph.graph.state import (  # pyright: ignore[reportMissingTypeStubs]
    CompiledStateGraph,
)

from aegon_api.graph.dependencies import GraphDependencies
from aegon_api.graph.nodes import NODES, NodeName, bind_nodes
from aegon_api.graph.routing import (
    after_evidence_guard,
    after_grade,
    after_input_guard,
    after_output_guard,
    after_route,
    after_tools,
    after_verify,
)
from aegon_api.graph.state import AgentState, ServerContext

DEFAULT_RECURSION_LIMIT = 40


def execution_config(
    context: ServerContext,
    *,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> RunnableConfig:
    """Build the required checkpoint identity and bounded execution config."""

    if recursion_limit < 1 or recursion_limit > 1_000:
        raise ValueError("recursion_limit must be between 1 and 1000")
    thread_id = f"{len(context.session_id)}:{context.session_id}{context.conversation_id}"
    return RunnableConfig(
        configurable={"thread_id": thread_id},
        recursion_limit=recursion_limit,
    )


def build_graph(
    dependencies: GraphDependencies,
    *,
    checkpointer: BaseCheckpointSaver[str] | None = None,
    interrupt_after: Sequence[NodeName] = (),
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Build the versioned agent topology with resumable confirmation."""

    graph = StateGraph(AgentState)
    for node_name, node in bind_nodes(dependencies).items():
        graph.add_node(node_name, node)  # pyright: ignore[reportArgumentType, reportUnknownMemberType]

    graph.add_edge(START, "input_guard")
    graph.add_conditional_edges(
        "input_guard",
        after_input_guard,
        {"modality_detect": "modality_detect", "refuse": "refuse", "escalate": "escalate"},
    )
    graph.add_edge("modality_detect", "rewrite")
    graph.add_edge("rewrite", "route")
    graph.add_conditional_edges(
        "route",
        after_route,
        {
            "retrieve": "retrieve",
            "tools": "tools",
            "refuse": "refuse",
            "escalate": "escalate",
        },
    )
    graph.add_edge("retrieve", "evidence_guard")
    graph.add_conditional_edges(
        "evidence_guard",
        after_evidence_guard,
        {"rerank": "rerank", "refuse": "refuse", "escalate": "escalate"},
    )
    graph.add_edge("rerank", "grade")
    graph.add_conditional_edges(
        "grade",
        after_grade,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "refuse": "refuse",
            "escalate": "escalate",
        },
    )
    graph.add_conditional_edges(
        "tools",
        after_tools,
        {
            "confirm_action": "confirm_action",
            "generate": "generate",
            "refuse": "refuse",
            "escalate": "escalate",
        },
    )
    graph.add_edge("confirm_action", "generate")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges(
        "verify",
        after_verify,
        {
            "output_guard": "output_guard",
            "rewrite": "rewrite",
            "refuse": "refuse",
            "escalate": "escalate",
        },
    )
    graph.add_conditional_edges(
        "output_guard",
        after_output_guard,
        {END: END, "refuse": "refuse", "escalate": "escalate"},
    )
    graph.add_edge("refuse", END)
    graph.add_edge("escalate", END)

    return graph.compile(  # pyright: ignore[reportUnknownMemberType]
        checkpointer=checkpointer,
        interrupt_before=["confirm_action"],
        interrupt_after=list(interrupt_after),
        name="aegon-agent",
    )


@asynccontextmanager
async def postgres_graph(
    connection_string: str,
    dependencies: GraphDependencies,
    *,
    setup: bool = False,
) -> AsyncGenerator[CompiledStateGraph[AgentState, None, AgentState, AgentState]]:
    """Open a graph backed by ``PostgresSaver`` for durable turn checkpoints."""

    if not connection_string:
        raise ValueError("connection_string must be non-empty")
    async with AsyncPostgresSaver.from_conn_string(connection_string) as saver:
        if setup:
            await saver.setup()
        yield build_graph(dependencies, checkpointer=saver)


assert set(NODES) == {
    "input_guard",
    "modality_detect",
    "rewrite",
    "route",
    "retrieve",
    "evidence_guard",
    "rerank",
    "grade",
    "tools",
    "confirm_action",
    "generate",
    "verify",
    "output_guard",
    "refuse",
    "escalate",
}
