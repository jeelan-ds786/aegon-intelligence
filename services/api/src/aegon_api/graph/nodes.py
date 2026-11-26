"""Node registry and uniformly enforced node execution wrapper."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Literal

from aegon_api.graph.dependencies import GraphDependencies, NodeCharge
from aegon_api.graph.state import AgentState
from aegon_api.guards import GuardVerdict

type NodeName = Literal[
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
]
type StateUpdate = dict[str, object]
type NodeImplementation = Callable[[AgentState, GraphDependencies], Awaitable[StateUpdate]]
type BoundNode = Callable[[AgentState], Awaitable[StateUpdate]]

REFUSAL_MESSAGE = "I cannot complete this request with approved evidence and safety controls."
ESCALATION_MESSAGE = "This request requires review by an authorized human operator."


async def _input_guard(state: AgentState, dependencies: GraphDependencies) -> StateUpdate:
    result = await dependencies.input_guard.evaluate(state["question"])
    if result.verdict is GuardVerdict.BLOCK:
        return {"refusal_reason": "input_guard_blocked"}
    return {"sanitized_question": result.redacted_payload}


async def _modality_detect(state: AgentState, _: GraphDependencies) -> StateUpdate:
    modality = "multimodal" if state["attachment_media_types"] else "text"
    return {"modality": modality}


async def _rewrite(state: AgentState, _: GraphDependencies) -> StateUpdate:
    question = state.get("sanitized_question", state["question"])
    return {
        "rewritten_query": question,
        "rewrite_count": state["rewrite_count"] + 1,
    }


async def _route(state: AgentState, _: GraphDependencies) -> StateUpdate:
    if state.get("route") is not None:
        return {}
    return {"route": "retrieve"}


async def _retrieve(state: AgentState, dependencies: GraphDependencies) -> StateUpdate:
    del dependencies
    return {"evidence": state.get("evidence", [])}


async def _evidence_guard(state: AgentState, dependencies: GraphDependencies) -> StateUpdate:
    result = await dependencies.evidence_guard.evaluate(state.get("evidence", []))
    evidence = [item.evidence for item in result.redacted_payload]
    if result.verdict is GuardVerdict.BLOCK:
        return {
            "guarded_evidence": [],
            "refusal_reason": "approved_evidence_unavailable",
        }
    return {"guarded_evidence": evidence}


async def _rerank(state: AgentState, _: GraphDependencies) -> StateUpdate:
    evidence = list(
        sorted(
            state.get("guarded_evidence", ()),
            key=lambda item: item.relevance_score,
            reverse=True,
        )
    )
    return {"guarded_evidence": evidence}


async def _grade(state: AgentState, _: GraphDependencies) -> StateUpdate:
    return {"grade": "pass" if state.get("guarded_evidence") else "insufficient"}


async def _tools(state: AgentState, _: GraphDependencies) -> StateUpdate:
    del state
    return {}


async def _confirm_action(state: AgentState, _: GraphDependencies) -> StateUpdate:
    if state.get("action_confirmed", False):
        return {"action_required": False}
    return {"refusal_reason": "action_not_confirmed"}


async def _generate(state: AgentState, dependencies: GraphDependencies) -> StateUpdate:
    evidence = state.get("guarded_evidence", [])
    answer = await dependencies.llm.generate(
        state.get("rewritten_query", state["question"]),
        tuple(item.text for item in evidence),
    )
    return {"answer": answer}


async def _verify(state: AgentState, _: GraphDependencies) -> StateUpdate:
    verification = "pass" if state.get("answer") and state.get("guarded_evidence") else "fail"
    return {"verification": verification}


async def _output_guard(state: AgentState, dependencies: GraphDependencies) -> StateUpdate:
    answer = state.get("answer", "")
    result = await dependencies.output_guard.evaluate(answer)
    if result.verdict is GuardVerdict.BLOCK:
        return {"refusal_reason": "output_guard_blocked"}
    return {"answer": result.redacted_payload, "final_output": result.redacted_payload}


async def _refuse(state: AgentState, _: GraphDependencies) -> StateUpdate:
    return {
        "refusal_reason": state.get("refusal_reason", "grounding_requirements_not_met"),
        "final_output": REFUSAL_MESSAGE,
    }


async def _escalate(state: AgentState, _: GraphDependencies) -> StateUpdate:
    return {
        "escalation_reason": state.get("escalation_reason", "human_review_required"),
        "final_output": ESCALATION_MESSAGE,
    }


NODES: Mapping[NodeName, NodeImplementation] = {
    "input_guard": _input_guard,
    "modality_detect": _modality_detect,
    "rewrite": _rewrite,
    "route": _route,
    "retrieve": _retrieve,
    "evidence_guard": _evidence_guard,
    "rerank": _rerank,
    "grade": _grade,
    "tools": _tools,
    "confirm_action": _confirm_action,
    "generate": _generate,
    "verify": _verify,
    "output_guard": _output_guard,
    "refuse": _refuse,
    "escalate": _escalate,
}


def _budget_refusal(state: AgentState, charge: NodeCharge) -> StateUpdate | None:
    tokens = state["tokens_used"] + charge.tokens
    cost = state["cost_microusd_used"] + charge.cost_microusd
    if tokens <= state["max_tokens"] and cost <= state["max_cost_microusd"]:
        return None
    return {"refusal_reason": "turn_budget_exceeded"}


def bind_nodes(dependencies: GraphDependencies) -> dict[NodeName, BoundNode]:
    """Bind dependencies and enforce timeout, budget, tracing, and accounting."""

    bound: dict[NodeName, BoundNode] = {}
    for node_name, implementation in NODES.items():
        bound[node_name] = _bind_node(node_name, implementation, dependencies)
    return bound


def _bind_node(
    node_name: NodeName,
    implementation: NodeImplementation,
    dependencies: GraphDependencies,
) -> BoundNode:
    async def execute(state: AgentState) -> StateUpdate:
        charge = dependencies.node_charges.get(node_name, NodeCharge())
        budget_refusal = _budget_refusal(state, charge)
        if budget_refusal is not None and node_name not in {"refuse", "escalate"}:
            return {
                **budget_refusal,
                "node_trace": [*state["node_trace"], node_name],
            }
        async with dependencies.span_hook.span(
            node_name,
            tenant_id=state["tenant_id"],
            conversation_id=state["conversation_id"],
        ):
            async with asyncio.timeout(dependencies.node_timeout_seconds):
                update = await implementation(state, dependencies)
        return {
            **update,
            "tokens_used": state["tokens_used"] + charge.tokens,
            "cost_microusd_used": state["cost_microusd_used"] + charge.cost_microusd,
            "node_trace": [*state["node_trace"], node_name],
        }

    return execute
