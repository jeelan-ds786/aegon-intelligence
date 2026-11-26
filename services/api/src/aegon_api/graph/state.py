"""Serializable state and server-owned context for the Aegon agent graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Required, TypedDict

from aegon_common.models import Citation, Evidence

from aegon_api.generation import AnswerSentence, PromptTrace

type Route = Literal["retrieve", "tools", "escalate"]
type Grade = Literal["ok", "retry", "refuse"]
type Verification = Literal["pass", "retry", "fail"]
type ModalityName = Literal["text", "multimodal"]


@dataclass(frozen=True, slots=True)
class ServerContext:
    """Identity and scope established by the authenticated server boundary."""

    tenant_id: str
    scopes: tuple[str, ...]
    session_id: str
    conversation_id: str

    def __post_init__(self) -> None:
        values = (self.tenant_id, self.session_id, self.conversation_id, *self.scopes)
        if not self.scopes or any(not value or len(value) > 128 for value in values):
            raise ValueError("server context values must contain 1..128 characters")


class AgentState(TypedDict, total=False):
    """Checkpoint-safe state. Raw credentials must never be added to this schema."""

    tenant_id: Required[str]
    scopes: Required[list[str]]
    session_id: Required[str]
    conversation_id: Required[str]
    question: Required[str]
    attachment_media_types: Required[list[str]]
    max_tokens: Required[int]
    max_cost_microusd: Required[int]
    max_rewrites: Required[int]
    tokens_used: Required[int]
    cost_microusd_used: Required[int]
    rewrite_count: Required[int]
    node_trace: Required[list[str]]
    sanitized_question: str
    rewritten_query: str
    rewritten_queries: list[str]
    hypothetical_document: str
    modality: ModalityName
    route: Route
    evidence: list[Evidence]
    guarded_evidence: list[Evidence]
    grade: Grade
    action_required: bool
    action_confirmed: bool
    generated_sentences: list[AnswerSentence]
    prompt_traces: list[PromptTrace]
    answer_citations: list[Citation]
    unsupported_claims: list[str]
    answer: str
    verification: Verification
    refusal_reason: str
    escalation_reason: str
    final_output: str


def create_initial_state(
    *,
    context: ServerContext,
    question: str,
    attachment_media_types: tuple[str, ...] = (),
    max_tokens: int = 4_000,
    max_cost_microusd: int = 100_000,
    max_rewrites: int = 1,
    evidence: tuple[Evidence, ...] = (),
) -> AgentState:
    """Create state only from a trusted context and bounded turn input."""

    if not question or len(question) > 8_000:
        raise ValueError("question must contain 1..8000 characters")
    if len(attachment_media_types) > 10 or any(
        not media_type or len(media_type) > 255 for media_type in attachment_media_types
    ):
        raise ValueError("attachment media types are invalid")
    if max_tokens < 1 or max_cost_microusd < 0 or not 0 <= max_rewrites <= 1:
        raise ValueError("turn budgets must be valid and max_rewrites must be zero or one")
    return AgentState(
        tenant_id=context.tenant_id,
        scopes=list(context.scopes),
        session_id=context.session_id,
        conversation_id=context.conversation_id,
        question=question,
        attachment_media_types=list(attachment_media_types),
        max_tokens=max_tokens,
        max_cost_microusd=max_cost_microusd,
        max_rewrites=max_rewrites,
        tokens_used=0,
        cost_microusd_used=0,
        rewrite_count=0,
        node_trace=[],
        prompt_traces=[],
        evidence=list(evidence),
    )
