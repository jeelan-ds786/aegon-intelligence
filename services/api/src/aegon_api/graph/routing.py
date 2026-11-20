"""Pure conditional-edge functions for the agent graph."""

from __future__ import annotations

from typing import Literal

from aegon_api.graph.state import AgentState

type NodeDestination = Literal[
    "__end__",
    "modality_detect",
    "retrieve",
    "tools",
    "confirm_action",
    "generate",
    "rerank",
    "rewrite",
    "output_guard",
    "refuse",
    "escalate",
]


def _terminal_destination(state: AgentState) -> NodeDestination | None:
    if state.get("refusal_reason"):
        return "refuse"
    if state.get("escalation_reason"):
        return "escalate"
    return None


def after_input_guard(state: AgentState) -> NodeDestination:
    return _terminal_destination(state) or "modality_detect"


def after_route(state: AgentState) -> NodeDestination:
    terminal = _terminal_destination(state)
    if terminal is not None:
        return terminal
    return state.get("route", "retrieve")


def after_evidence_guard(state: AgentState) -> NodeDestination:
    return _terminal_destination(state) or "rerank"


def after_grade(state: AgentState) -> NodeDestination:
    terminal = _terminal_destination(state)
    if terminal is not None:
        return terminal
    grade = state.get("grade", "insufficient")
    if grade == "pass":
        return "generate"
    if grade == "retry" and state["rewrite_count"] < state["max_rewrites"]:
        return "rewrite"
    return "refuse"


def after_tools(state: AgentState) -> NodeDestination:
    terminal = _terminal_destination(state)
    if terminal is not None:
        return terminal
    return "confirm_action" if state.get("action_required", False) else "generate"


def after_verify(state: AgentState) -> NodeDestination:
    terminal = _terminal_destination(state)
    if terminal is not None:
        return terminal
    verification = state.get("verification", "fail")
    if verification == "pass":
        return "output_guard"
    if verification == "retry" and state["rewrite_count"] < state["max_rewrites"]:
        return "rewrite"
    return "refuse"


def after_output_guard(state: AgentState) -> NodeDestination:
    return _terminal_destination(state) or "__end__"
