"""Node registry and uniformly enforced node execution wrapper."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Literal, cast

from aegon_common.models import Citation, Evidence, QuoteSpan

from aegon_api.generation import (
    ClaimForVerification,
    GeneratedAnswer,
    VerificationResult,
    generation_request,
    verification_request,
)
from aegon_api.graph.dependencies import GraphDependencies, NodeCharge
from aegon_api.graph.state import AgentState
from aegon_api.guards import GuardVerdict
from aegon_api.ir import (
    GradeResult,
    RewritePlan,
    grade_by_thresholds,
    grade_request,
    rerank_and_select,
    rewrite_request,
)

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


async def _rewrite(state: AgentState, dependencies: GraphDependencies) -> StateUpdate:
    question = state.get("sanitized_question", state["question"])
    config = dependencies.rewrite_config
    retrying = state.get("grade") == "retry" or state.get("verification") == "retry"
    rewrite_count = state["rewrite_count"] + int(retrying)
    if not config.enabled or not config.use_llm:
        return {
            "rewritten_query": question,
            "rewritten_queries": [question],
            "rewrite_count": rewrite_count,
        }
    remaining_tokens = state["max_tokens"] - state["tokens_used"]
    if remaining_tokens < 1:
        return {"refusal_reason": "turn_budget_exceeded"}
    request, trace = rewrite_request(
        question,
        enable_hyde=config.enable_hyde,
        max_queries=config.max_queries,
        max_output_tokens=min(config.max_output_tokens, remaining_tokens),
        prior_queries=state.get("rewritten_queries", ()) if retrying else (),
    )
    async with asyncio.timeout(config.timeout_seconds):
        response = await dependencies.llm.complete(request)
    plan = RewritePlan.model_validate_json(response.content)
    queries: list[str] = []
    for query in plan.queries[: config.max_queries]:
        guarded = await dependencies.input_guard.evaluate(query)
        if guarded.verdict is GuardVerdict.BLOCK:
            return {"refusal_reason": "rewritten_query_blocked"}
        queries.append(guarded.redacted_payload)
    hypothetical_document = ""
    if config.enable_hyde and plan.hypothetical_document is not None:
        guarded_hyde = await dependencies.input_guard.evaluate(plan.hypothetical_document)
        if guarded_hyde.verdict is GuardVerdict.BLOCK:
            return {"refusal_reason": "hypothetical_document_blocked"}
        hypothetical_document = guarded_hyde.redacted_payload
    return {
        "rewritten_query": queries[0],
        "rewritten_queries": queries,
        "hypothetical_document": hypothetical_document,
        "rewrite_count": rewrite_count,
        "prompt_traces": [*state.get("prompt_traces", []), trace],
        "tokens_used": state["tokens_used"] + response.usage.total_tokens,
        "cost_microusd_used": state["cost_microusd_used"] + response.usage.cost_microusd,
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


async def _rerank(state: AgentState, dependencies: GraphDependencies) -> StateUpdate:
    evidence = await rerank_and_select(
        state.get("rewritten_query", state["question"]),
        state.get("guarded_evidence", ()),
        reranker=dependencies.reranker,
        embedder=dependencies.embedder,
        parent_expander=dependencies.parent_expander,
        config=dependencies.rerank_config,
    )
    guarded = await dependencies.evidence_guard.evaluate(evidence)
    if guarded.verdict is GuardVerdict.BLOCK:
        return {
            "guarded_evidence": [],
            "refusal_reason": "approved_evidence_unavailable",
        }
    return {"guarded_evidence": [item.evidence for item in guarded.redacted_payload]}


async def _grade(state: AgentState, dependencies: GraphDependencies) -> StateUpdate:
    evidence = state.get("guarded_evidence", [])
    config = dependencies.grade_config
    can_retry = state["rewrite_count"] < state["max_rewrites"]
    if not config.enabled:
        return {"grade": "ok" if evidence else ("retry" if can_retry else "refuse")}
    if not grade_by_thresholds(evidence, config):
        update: StateUpdate = {"grade": "retry" if can_retry else "refuse"}
        if not can_retry:
            update["refusal_reason"] = (
                "approved_evidence_unavailable"
                if not evidence
                else "approved_evidence_insufficient"
            )
        return update
    if not config.use_llm_judge:
        return {"grade": "ok"}
    remaining_tokens = state["max_tokens"] - state["tokens_used"]
    if remaining_tokens < 1:
        return {"grade": "refuse", "refusal_reason": "turn_budget_exceeded"}
    request, trace = grade_request(
        state.get("rewritten_query", state["question"]),
        evidence,
        max_output_tokens=min(config.max_output_tokens, remaining_tokens),
    )
    async with asyncio.timeout(config.timeout_seconds):
        response = await dependencies.llm.complete(request)
    result = GradeResult.model_validate_json(response.content)
    decision = result.decision
    if decision == "retry" and not can_retry:
        decision = "refuse"
    update = {
        "grade": decision,
        "prompt_traces": [*state.get("prompt_traces", []), trace],
        "tokens_used": state["tokens_used"] + response.usage.total_tokens,
        "cost_microusd_used": state["cost_microusd_used"] + response.usage.cost_microusd,
    }
    if decision == "refuse":
        update["refusal_reason"] = "approved_evidence_insufficient"
    return update


async def _tools(state: AgentState, _: GraphDependencies) -> StateUpdate:
    del state
    return {}


async def _confirm_action(state: AgentState, _: GraphDependencies) -> StateUpdate:
    if state.get("action_confirmed", False):
        return {"action_required": False}
    return {"refusal_reason": "action_not_confirmed"}


async def _generate(state: AgentState, dependencies: GraphDependencies) -> StateUpdate:
    evidence = state.get("guarded_evidence", [])
    if not evidence:
        return {
            "generated_sentences": [],
            "refusal_reason": "approved_evidence_unavailable",
        }
    remaining_tokens = state["max_tokens"] - state["tokens_used"]
    if remaining_tokens < 1:
        return {"refusal_reason": "turn_budget_exceeded"}
    request, trace = generation_request(
        state.get("rewritten_query", state["question"]),
        tuple((item.citation.chunk_id, item.text) for item in evidence),
        max_output_tokens=min(dependencies.generation_max_tokens, remaining_tokens),
    )
    response = await dependencies.llm.complete(request)
    generated = GeneratedAnswer.model_validate_json(response.content)
    valid_ids = {item.citation.chunk_id for item in evidence}
    sentences = [
        sentence
        for sentence in generated.answer_sentences
        if all(chunk_id in valid_ids for chunk_id in sentence.chunk_ids)
    ]
    update: StateUpdate = {
        "generated_sentences": sentences,
        "prompt_traces": [*state.get("prompt_traces", []), trace],
        "tokens_used": state["tokens_used"] + response.usage.total_tokens,
        "cost_microusd_used": state["cost_microusd_used"] + response.usage.cost_microusd,
    }
    if generated.answer_sentences and not sentences:
        update["refusal_reason"] = "generated_citations_invalid"
    elif not sentences:
        update["refusal_reason"] = "grounding_requirements_not_met"
    return update


async def _verify(state: AgentState, dependencies: GraphDependencies) -> StateUpdate:
    if state.get("refusal_reason"):
        return {"verification": "fail"}
    evidence = state.get("guarded_evidence", [])
    sentences = state.get("generated_sentences", [])
    if not evidence or not sentences:
        return {
            "verification": "fail",
            "refusal_reason": "grounding_requirements_not_met",
        }
    claims = tuple(
        ClaimForVerification(
            claim_id=f"claim-{index}",
            text=sentence.text,
            chunk_ids=sentence.chunk_ids,
        )
        for index, sentence in enumerate(sentences)
    )
    evidence_by_id = {item.citation.chunk_id: item for item in evidence}
    cited_ids = {chunk_id for claim in claims for chunk_id in claim.chunk_ids}
    remaining_tokens = state["max_tokens"] - state["tokens_used"]
    if remaining_tokens < 1:
        return {"verification": "fail", "refusal_reason": "turn_budget_exceeded"}
    request, trace = verification_request(
        claims,
        tuple(
            (chunk_id, evidence_by_id[chunk_id].text)
            for chunk_id in sorted(cited_ids)
            if chunk_id in evidence_by_id
        ),
        max_output_tokens=min(dependencies.verification_max_tokens, remaining_tokens),
    )
    response = await dependencies.llm.complete(request)
    result = VerificationResult.model_validate_json(response.content)
    verdicts = {verdict.claim_id: verdict for verdict in result.claims}
    supported_text: list[str] = []
    citations: list[Citation] = []
    unsupported: list[str] = []
    for claim in claims:
        verdict = verdicts.get(claim.claim_id)
        if (
            verdict is None
            or not verdict.supported
            or verdict.chunk_id is None
            or verdict.quote is None
            or verdict.chunk_id not in claim.chunk_ids
            or verdict.chunk_id not in evidence_by_id
        ):
            unsupported.append(claim.text)
            continue
        item = evidence_by_id[verdict.chunk_id]
        quote_start = item.text.find(verdict.quote)
        if quote_start < 0:
            unsupported.append(claim.text)
            continue
        supported_text.append(claim.text)
        citations.append(
            item.citation.model_copy(
                update={
                    "quote_span": QuoteSpan(
                        start=quote_start,
                        end=quote_start + len(verdict.quote),
                    )
                }
            )
        )
    support_ratio = len(supported_text) / len(claims)
    if support_ratio < dependencies.verification_support_threshold:
        answer, fallback_citation = _extractive_fallback(evidence)
        supported_text = [answer]
        citations = [fallback_citation]
    update: StateUpdate = {
        "answer": " ".join(supported_text),
        "answer_citations": _dedupe_citations(citations),
        "unsupported_claims": unsupported,
        "verification": "pass",
        "prompt_traces": [*state.get("prompt_traces", []), trace],
        "tokens_used": state["tokens_used"] + response.usage.total_tokens,
        "cost_microusd_used": state["cost_microusd_used"] + response.usage.cost_microusd,
    }
    return update


def _extractive_fallback(evidence: list[Evidence]) -> tuple[str, Citation]:
    item = evidence[0]
    match = re.search(r".+?(?:[.!?](?=\s|$)|$)", item.text, flags=re.DOTALL)
    if match is None or not match.group().strip():
        raise ValueError("approved evidence cannot produce an extractive fallback")
    quote = match.group().strip()
    start = item.text.find(quote)
    citation = item.citation.model_copy(
        update={"quote_span": QuoteSpan(start=start, end=start + len(quote))}
    )
    return quote, citation


def _dedupe_citations(citations: list[Citation]) -> list[Citation]:
    unique: dict[tuple[str, int, int], Citation] = {}
    for citation in citations:
        key = (
            citation.chunk_id,
            citation.quote_span.start,
            citation.quote_span.end,
        )
        unique[key] = citation
    return list(unique.values())


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
        tokens_used = cast(int, update.pop("tokens_used", state["tokens_used"])) + charge.tokens
        cost_used = (
            cast(int, update.pop("cost_microusd_used", state["cost_microusd_used"]))
            + charge.cost_microusd
        )
        if (
            tokens_used > state["max_tokens"] or cost_used > state["max_cost_microusd"]
        ) and node_name not in {"refuse", "escalate"}:
            update["refusal_reason"] = "turn_budget_exceeded"
        return {
            **update,
            "tokens_used": tokens_used,
            "cost_microusd_used": cost_used,
            "node_trace": [*state["node_trace"], node_name],
        }

    return execute
