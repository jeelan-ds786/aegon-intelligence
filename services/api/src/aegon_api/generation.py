"""Prompt construction and strict contracts for grounded answer generation."""

from __future__ import annotations

import hashlib
import html
import json
from importlib.resources import files
from typing import Annotated, Literal

from aegon_common.models import ContractModel
from pydantic import Field

from aegon_api.llm import LlmRequest

GENERATE_PROMPT_VERSION = "generate-v1"
VERIFY_PROMPT_VERSION = "verify-v1"


class AnswerSentence(ContractModel):
    """One atomic generated claim and the chunks claimed to support it."""

    text: Annotated[str, Field(min_length=1, max_length=2_000)]
    chunk_ids: tuple[Annotated[str, Field(min_length=1, max_length=128)], ...] = Field(
        min_length=1, max_length=10
    )


class GeneratedAnswer(ContractModel):
    """Schema-constrained output of the generation model."""

    answer_sentences: tuple[AnswerSentence, ...] = Field(max_length=50)


class ClaimForVerification(ContractModel):
    """Stable identifier, text, and claimed evidence supplied to verification."""

    claim_id: Annotated[str, Field(min_length=1, max_length=32)]
    text: Annotated[str, Field(min_length=1, max_length=2_000)]
    chunk_ids: tuple[Annotated[str, Field(min_length=1, max_length=128)], ...] = Field(
        min_length=1, max_length=10
    )


class ClaimVerdict(ContractModel):
    """Strict entailment decision and exact supporting quote."""

    claim_id: Annotated[str, Field(min_length=1, max_length=32)]
    supported: bool
    chunk_id: Annotated[str, Field(min_length=1, max_length=128)] | None
    quote: Annotated[str, Field(min_length=1, max_length=10_000)] | None


class VerificationResult(ContractModel):
    """Schema-constrained output of the verification model."""

    claims: tuple[ClaimVerdict, ...] = Field(max_length=50)


class PromptTrace(ContractModel):
    """Content-free prompt identity recorded in checkpointed state."""

    node: Literal["generate", "verify"]
    version: str
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def load_prompt(version: str) -> tuple[str, str]:
    """Load a packaged prompt and return its content-addressed identity."""

    prompt = files("aegon_api.prompts").joinpath(f"{version}.txt").read_text(encoding="utf-8")
    return prompt, hashlib.sha256(prompt.encode()).hexdigest()


def evidence_block(chunk_id: str, text: str) -> str:
    """Render evidence as escaped, explicitly data-only prompt content."""

    return (
        f'<evidence data-only="true" chunk-id="{html.escape(chunk_id, quote=True)}">\n'
        f"{html.escape(text)}\n</evidence>"
    )


def generation_request(
    question: str,
    evidence: tuple[tuple[str, str], ...],
    *,
    max_output_tokens: int,
) -> tuple[LlmRequest, PromptTrace]:
    """Create the bounded generation request without exposing model tools."""

    system_prompt, prompt_hash = load_prompt(GENERATE_PROMPT_VERSION)
    blocks = "\n".join(evidence_block(chunk_id, text) for chunk_id, text in evidence)
    user_prompt = (
        f"<question>\n{html.escape(question)}\n</question>\n"
        f"<evidence_set>\n{blocks}\n</evidence_set>"
    )
    return (
        LlmRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=GeneratedAnswer.model_json_schema(),
            temperature=0.1,
            max_output_tokens=max_output_tokens,
        ),
        PromptTrace(node="generate", version=GENERATE_PROMPT_VERSION, sha256=prompt_hash),
    )


def verification_request(
    claims: tuple[ClaimForVerification, ...],
    evidence: tuple[tuple[str, str], ...],
    *,
    max_output_tokens: int,
) -> tuple[LlmRequest, PromptTrace]:
    """Create a strict NLI request over only the cited evidence."""

    system_prompt, prompt_hash = load_prompt(VERIFY_PROMPT_VERSION)
    payload = {
        "claims": [claim.model_dump(mode="json") for claim in claims],
        "evidence": [{"chunk_id": chunk_id, "text": text} for chunk_id, text in evidence],
    }
    blocks = "\n".join(evidence_block(chunk_id, text) for chunk_id, text in evidence)
    user_prompt = (
        f"<verification_input>\n{json.dumps(payload, separators=(',', ':'))}\n"
        f"</verification_input>\n<evidence_set>\n{blocks}\n</evidence_set>"
    )
    return (
        LlmRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=VerificationResult.model_json_schema(),
            temperature=0,
            max_output_tokens=max_output_tokens,
        ),
        PromptTrace(node="verify", version=VERIFY_PROMPT_VERSION, sha256=prompt_hash),
    )
