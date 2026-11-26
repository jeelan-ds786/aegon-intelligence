"""Bounded query rewriting, reranking, diversity selection, and sufficiency grading."""

from __future__ import annotations

import asyncio
import html
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Literal, Protocol

from aegon_common.models import ContractModel, Evidence
from pydantic import Field

from aegon_api.generation import PromptTrace, evidence_block, load_prompt
from aegon_api.llm import LlmClient, LlmRequest
from aegon_api.retrieval import TextEmbedder

REWRITE_PROMPT_VERSION = "rewrite-v1"
RERANK_PROMPT_VERSION = "rerank-v1"
GRADE_PROMPT_VERSION = "grade-v1"


class RewritePlan(ContractModel):
    """Schema-constrained decomposition with an optional hypothetical document."""

    queries: tuple[Annotated[str, Field(min_length=1, max_length=2_000)], ...] = Field(
        min_length=1, max_length=3
    )
    hypothetical_document: Annotated[str, Field(min_length=1, max_length=4_000)] | None = None


class RerankScore(ContractModel):
    """One model score tied to a candidate chunk."""

    chunk_id: Annotated[str, Field(min_length=1, max_length=128)]
    score: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class RerankResult(ContractModel):
    """Schema-constrained reranker output."""

    scores: tuple[RerankScore, ...] = Field(max_length=50)


class GradeResult(ContractModel):
    """Schema-constrained lightweight evidence-sufficiency judgment."""

    decision: Literal["ok", "retry", "refuse"]
    confidence: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    reason: Annotated[str, Field(min_length=1, max_length=256)]


@dataclass(frozen=True, slots=True)
class RewriteConfig:
    """Ablation flags and hard bounds for query transformation."""

    enabled: bool = True
    use_llm: bool = True
    enable_hyde: bool = False
    max_queries: int = 3
    max_output_tokens: int = 256
    timeout_seconds: float = 3.0

    def __post_init__(self) -> None:
        if not 1 <= self.max_queries <= 3:
            raise ValueError("max_queries must be between 1 and 3")
        if self.max_output_tokens < 1 or self.timeout_seconds <= 0:
            raise ValueError("rewrite token and timeout bounds must be positive")


@dataclass(frozen=True, slots=True)
class RerankConfig:
    """Ablation flags and hard bounds for reranking and context assembly."""

    enabled: bool = True
    candidate_limit: int = 50
    output_limit: int = 8
    use_mmr: bool = True
    mmr_lambda: float = 0.75
    expand_parents: bool = True
    context_token_budget: int = 3_000
    timeout_seconds: float = 3.0

    def __post_init__(self) -> None:
        if not 1 <= self.output_limit <= self.candidate_limit <= 50:
            raise ValueError("rerank limits must satisfy 1 <= output <= candidates <= 50")
        if not 0 <= self.mmr_lambda <= 1:
            raise ValueError("mmr_lambda must be between zero and one")
        if self.context_token_budget < 1 or self.timeout_seconds <= 0:
            raise ValueError("rerank token and timeout bounds must be positive")


@dataclass(frozen=True, slots=True)
class GradeConfig:
    """Ablation flags and thresholds for evidence sufficiency."""

    enabled: bool = True
    use_llm_judge: bool = True
    minimum_top_score: float = 0.35
    minimum_cumulative_score: float = 0.8
    max_output_tokens: int = 128
    timeout_seconds: float = 2.0

    def __post_init__(self) -> None:
        if not 0 <= self.minimum_top_score <= 1:
            raise ValueError("minimum_top_score must be between zero and one")
        if not 0 <= self.minimum_cumulative_score <= 8:
            raise ValueError("minimum_cumulative_score must be between zero and eight")
        if self.max_output_tokens < 1 or self.timeout_seconds <= 0:
            raise ValueError("grade token and timeout bounds must be positive")


class Reranker(Protocol):
    """Provider-neutral relevance scoring over an already bounded candidate set."""

    async def rerank(self, query: str, candidates: Sequence[Evidence]) -> Sequence[RerankScore]: ...


class CrossEncoderModel(Protocol):
    """Local cross-encoder surface kept independent from a model framework."""

    async def score(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]: ...


@dataclass(frozen=True, slots=True)
class CrossEncoderReranker:
    """Local cross-encoder adapter with strict result validation."""

    model: CrossEncoderModel

    async def rerank(self, query: str, candidates: Sequence[Evidence]) -> Sequence[RerankScore]:
        values = await self.model.score(tuple((query, item.text) for item in candidates))
        if len(values) != len(candidates):
            raise RuntimeError("cross-encoder returned an unexpected score count")
        return tuple(
            RerankScore(chunk_id=item.citation.chunk_id, score=_bounded_score(score))
            for item, score in zip(candidates, values, strict=True)
        )


@dataclass(frozen=True, slots=True)
class GeminiReranker:
    """Schema-constrained reranker using the configured production LLM adapter."""

    llm: LlmClient
    max_output_tokens: int = 512

    def __post_init__(self) -> None:
        if self.max_output_tokens < 1:
            raise ValueError("reranker max_output_tokens must be positive")

    async def rerank(self, query: str, candidates: Sequence[Evidence]) -> Sequence[RerankScore]:
        request, _ = rerank_request(query, candidates, max_output_tokens=self.max_output_tokens)
        response = await self.llm.complete(request)
        return RerankResult.model_validate_json(response.content).scores


class NoopReranker:
    """Deterministic ablation adapter retaining retrieval relevance scores."""

    async def rerank(self, query: str, candidates: Sequence[Evidence]) -> Sequence[RerankScore]:
        del query
        return tuple(
            RerankScore(chunk_id=item.citation.chunk_id, score=item.relevance_score)
            for item in candidates
        )


class ParentDocumentExpander(Protocol):
    """Resolve an approved parent document for an already approved child chunk."""

    async def expand(self, evidence: Evidence) -> Evidence | None: ...


class NoopParentDocumentExpander:
    """Default expander used when no tenant-scoped parent store is configured."""

    async def expand(self, evidence: Evidence) -> Evidence | None:
        del evidence
        return None


def rewrite_request(
    question: str,
    *,
    enable_hyde: bool,
    max_queries: int,
    max_output_tokens: int,
    prior_queries: Sequence[str] = (),
) -> tuple[LlmRequest, PromptTrace]:
    """Build a bounded decomposition request."""

    system_prompt, prompt_hash = load_prompt(REWRITE_PROMPT_VERSION)
    prior = (
        "\n<prior_queries>"
        + "".join(f"\n<query>{html.escape(query)}</query>" for query in prior_queries)
        + "\n</prior_queries>"
        if prior_queries
        else ""
    )
    user_prompt = (
        f"<question>\n{html.escape(question)}\n</question>\n"
        f'<constraints max-queries="{max_queries}" hyde="{str(enable_hyde).lower()}" />'
        f"{prior}"
    )
    return (
        LlmRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=RewritePlan.model_json_schema(),
            temperature=0,
            max_output_tokens=max_output_tokens,
        ),
        PromptTrace(node="rewrite", version=REWRITE_PROMPT_VERSION, sha256=prompt_hash),
    )


def rerank_request(
    query: str,
    candidates: Sequence[Evidence],
    *,
    max_output_tokens: int,
) -> tuple[LlmRequest, PromptTrace]:
    """Build a bounded relevance-scoring request."""

    system_prompt, prompt_hash = load_prompt(RERANK_PROMPT_VERSION)
    blocks = "\n".join(evidence_block(item.citation.chunk_id, item.text) for item in candidates)
    return (
        LlmRequest(
            system_prompt=system_prompt,
            user_prompt=(
                f"<question>\n{html.escape(query)}\n</question>\n"
                f"<candidates>\n{blocks}\n</candidates>"
            ),
            response_schema=RerankResult.model_json_schema(),
            temperature=0,
            max_output_tokens=max_output_tokens,
        ),
        PromptTrace(node="rerank", version=RERANK_PROMPT_VERSION, sha256=prompt_hash),
    )


def grade_request(
    query: str,
    evidence: Sequence[Evidence],
    *,
    max_output_tokens: int,
) -> tuple[LlmRequest, PromptTrace]:
    """Build a bounded evidence-sufficiency request."""

    system_prompt, prompt_hash = load_prompt(GRADE_PROMPT_VERSION)
    blocks = "\n".join(evidence_block(item.citation.chunk_id, item.text) for item in evidence)
    return (
        LlmRequest(
            system_prompt=system_prompt,
            user_prompt=(
                f"<question>\n{html.escape(query)}\n</question>\n"
                f"<evidence_set>\n{blocks}\n</evidence_set>"
            ),
            response_schema=GradeResult.model_json_schema(),
            temperature=0,
            max_output_tokens=max_output_tokens,
        ),
        PromptTrace(node="grade", version=GRADE_PROMPT_VERSION, sha256=prompt_hash),
    )


async def rerank_and_select(
    query: str,
    candidates: Sequence[Evidence],
    *,
    reranker: Reranker,
    embedder: TextEmbedder,
    parent_expander: ParentDocumentExpander,
    config: RerankConfig,
) -> tuple[Evidence, ...]:
    """Score, deduplicate, diversify, expand, and budget a candidate set."""

    deduped = deduplicate_evidence(candidates)[: config.candidate_limit]
    if not deduped:
        return ()
    if config.enabled:
        async with asyncio.timeout(config.timeout_seconds):
            scores = await reranker.rerank(query, deduped)
        expected_ids = {item.citation.chunk_id for item in deduped}
        returned_ids = [item.chunk_id for item in scores]
        if len(returned_ids) != len(expected_ids) or set(returned_ids) != expected_ids:
            raise ValueError("reranker must return exactly one score per candidate")
        score_by_id = {item.chunk_id: item.score for item in scores}
        scored = tuple(
            item.model_copy(
                update={
                    "relevance_score": score_by_id.get(item.citation.chunk_id, item.relevance_score)
                }
            )
            for item in deduped
        )
    else:
        scored = deduped
    ordered = tuple(
        sorted(
            scored,
            key=lambda item: (-item.relevance_score, item.citation.chunk_id),
        )
    )
    if config.enabled and config.use_mmr and len(ordered) > 1:
        async with asyncio.timeout(config.timeout_seconds):
            selected = await maximal_marginal_relevance(
                ordered,
                embedder,
                limit=config.output_limit,
                relevance_weight=config.mmr_lambda,
            )
    else:
        selected = ordered[: config.output_limit]
    return await expand_under_budget(
        selected,
        parent_expander=parent_expander,
        enabled=config.enabled and config.expand_parents,
        token_budget=config.context_token_budget,
        timeout_seconds=config.timeout_seconds,
    )


def deduplicate_evidence(candidates: Sequence[Evidence]) -> tuple[Evidence, ...]:
    """Keep the best item for duplicate chunk IDs or normalized content."""

    unique: dict[str, Evidence] = {}
    for item in candidates:
        content_key = re.sub(r"\s+", " ", item.text).strip().casefold()
        keys = (f"id:{item.citation.chunk_id}", f"text:{content_key}")
        existing = next((unique[key] for key in keys if key in unique), None)
        if existing is not None and existing.relevance_score >= item.relevance_score:
            continue
        if existing is not None:
            for key, value in tuple(unique.items()):
                if value is existing:
                    del unique[key]
        for key in keys:
            unique[key] = item
    return tuple(dict.fromkeys(unique.values()))


async def maximal_marginal_relevance(
    candidates: Sequence[Evidence],
    embedder: TextEmbedder,
    *,
    limit: int,
    relevance_weight: float,
) -> tuple[Evidence, ...]:
    """Select relevant, non-redundant evidence with deterministic tie-breaking."""

    if limit < 1 or not 0 <= relevance_weight <= 1:
        raise ValueError("MMR limit and relevance weight are invalid")
    vectors = await embedder.embed_texts(tuple(item.text for item in candidates))
    if len(vectors) != len(candidates):
        raise RuntimeError("embedder returned an unexpected MMR result count")
    remaining = list(range(len(candidates)))
    selected: list[int] = []
    while remaining and len(selected) < limit:

        def mmr_score(index: int) -> tuple[float, str]:
            diversity = max(
                (_cosine(vectors[index].values, vectors[chosen].values) for chosen in selected),
                default=0.0,
            )
            score = (
                relevance_weight * candidates[index].relevance_score
                - (1 - relevance_weight) * diversity
            )
            return score, candidates[index].citation.chunk_id

        best = min(
            remaining,
            key=lambda index: (-mmr_score(index)[0], mmr_score(index)[1]),
        )
        selected.append(best)
        remaining.remove(best)
    return tuple(candidates[index] for index in selected)


async def expand_under_budget(
    evidence: Sequence[Evidence],
    *,
    parent_expander: ParentDocumentExpander,
    enabled: bool,
    token_budget: int,
    timeout_seconds: float,
) -> tuple[Evidence, ...]:
    """Replace children with approved parents when they fit the context budget."""

    selected: list[Evidence] = []
    used_tokens = 0
    for item in evidence:
        expanded: Evidence | None = None
        if enabled:
            async with asyncio.timeout(timeout_seconds):
                expanded = await parent_expander.expand(item)
        choice = expanded or item
        choice_tokens = _estimated_tokens(choice.text)
        if used_tokens + choice_tokens > token_budget and expanded is not None:
            choice = item
            choice_tokens = _estimated_tokens(item.text)
        if used_tokens + choice_tokens > token_budget:
            continue
        selected.append(choice.model_copy(update={"relevance_score": item.relevance_score}))
        used_tokens += choice_tokens
    return deduplicate_evidence(selected)


def grade_by_thresholds(evidence: Sequence[Evidence], config: GradeConfig) -> bool:
    """Return whether deterministic relevance thresholds admit an LLM judgment."""

    if not evidence:
        return False
    ordered = sorted((item.relevance_score for item in evidence), reverse=True)
    return (
        ordered[0] >= config.minimum_top_score
        and sum(ordered[:3]) >= config.minimum_cumulative_score
    )


def _estimated_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("MMR vectors must have equal dimensions")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _bounded_score(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("reranker scores must be finite")
    return min(max(value, 0.0), 1.0)
