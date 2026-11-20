"""Rule-first safety guards with optional managed screening."""

from __future__ import annotations

import asyncio
import base64
import binascii
import html
import re
import unicodedata
from collections.abc import Iterable, Sequence

from aegon_common.models import Evidence

from aegon_api.guards.models import (
    GuardedEvidence,
    GuardReason,
    GuardResult,
    GuardVerdict,
)
from aegon_api.guards.protocols import ManagedScreeningService

GUARD_VERSION = "rules-1.0.0"
_MAX_TEXT_LENGTH = 100_000
_MAX_ENCODED_LENGTH = 16_384
_MAX_DECODE_PASSES = 3
_MAX_DECODED_VIEWS = 16
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)
_REDACTED = "[REDACTED]"

_INJECTION_PATTERNS = (
    re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?\b"),
    re.compile(r"\b(?:reveal|print|show|repeat|extract)\s+(?:the\s+)?system\s+prompt\b"),
    re.compile(r"\b(?:jailbreak|developer\s+mode|do\s+anything\s+now|dan\s+mode)\b"),
    re.compile(r"\b(?:bypass|disable|override)\s+(?:the\s+)?(?:safety|policy|guardrails?)\b"),
    re.compile(r"\bforget\s+(?:all\s+)?(?:your|the)\s+(?:rules|instructions)\b"),
    re.compile(
        r"\b(?:ignora|ignorez)\s+(?:todas?\s+|toutes\s+|les\s+)?(?:las\s+|les\s+)?"
        r"(?:instrucciones|instructions)\s+(?:anteriores|précédentes)\b"
    ),
    re.compile(r"(?:忽略|无视)(?:之前|以上|所有)(?:的)?(?:指令|指示)"),
    re.compile(r"(?:पिछले|पूर्व)\s+(?:सभी\s+)?निर्देशों\s+को\s+अनदेखा"),
    re.compile(r"\bexecute\s+(?:this\s+)?(?:tool|function)\s+call\b"),
)
_REGULATED_PATTERNS = (
    re.compile(r"\b(?:diagnose|prescribe)\s+(?:me|my|this|a)\b"),
    re.compile(r"\bguarantee(?:d)?\s+(?:an?\s+)?(?:investment|return|profit)\b"),
    re.compile(r"\b(?:evade|bypass)\s+(?:kyc|aml|sanctions)\b"),
    re.compile(r"\b(?:legal|medical|financial)\s+advice\s+without\s+(?:a\s+)?disclaimer\b"),
)
_EVIDENCE_PATTERNS: tuple[tuple[GuardReason, re.Pattern[str]], ...] = (
    *((GuardReason.INSTRUCTION_SHAPED_EVIDENCE, pattern) for pattern in _INJECTION_PATTERNS),
    (
        GuardReason.ROLE_IMPERSONATION,
        re.compile(r"(?:^|\n)\s*(?:system|assistant|developer)\s*:\s*", re.MULTILINE),
    ),
    (
        GuardReason.TOOL_CALL_SYNTAX,
        re.compile(r"<\/?(?:tool_call|function_call)>|(?:^|\n)\s*\{\s*\"(?:tool|function)\""),
    ),
    (
        GuardReason.HIDDEN_TEXT,
        re.compile(
            r"(?:display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0|"
            r"color\s*:\s*(?:white|#fff(?:fff)?)|<!--|\[hidden(?:-text)?\])"
        ),
    ),
)
_POLICY_PATTERNS = (
    re.compile(r"\bI (?:ignored|bypassed|disabled) (?:the )?(?:policy|safety|guardrails?)\b"),
    re.compile(r"\bhere (?:is|are) (?:the )?(?:system prompt|hidden instructions)\b"),
)
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    re.compile(r"\b(?:ghp|github_pat)_[0-9A-Za-z_]{20,}\b"),
    re.compile(r"\bsk-[0-9A-Za-z_-]{20,}\b"),
)
_PII_PATTERNS = (
    re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.-])"),
    re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"),
)


def _bounded(text: str) -> None:
    if not text or len(text) > _MAX_TEXT_LENGTH:
        raise ValueError(f"text must contain 1..{_MAX_TEXT_LENGTH} characters")


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH).casefold()
    return re.sub(r"[ \t]+", " ", normalized)


def _decoded_views(text: str) -> tuple[str, ...]:
    from urllib.parse import unquote

    views = [text]
    seen = {text}
    frontier = [text]
    for _ in range(_MAX_DECODE_PASSES):
        decoded_in_pass: list[str] = []
        for candidate in frontier:
            candidates: list[str] = []
            if "%" in candidate:
                candidates.append(unquote(candidate))

            compact = "".join(candidate.split())
            if 16 <= len(compact) <= _MAX_ENCODED_LENGTH:
                padded = compact + "=" * (-len(compact) % 4)
                for altchars in (None, b"-_"):
                    try:
                        decoded_bytes = base64.b64decode(
                            padded,
                            altchars=altchars,
                            validate=True,
                        )
                        candidates.append(decoded_bytes.decode("utf-8"))
                    except (binascii.Error, UnicodeDecodeError, ValueError):
                        continue

            for decoded in candidates:
                if decoded == candidate or decoded in seen:
                    continue
                seen.add(decoded)
                views.append(decoded)
                decoded_in_pass.append(decoded)
                if len(views) >= _MAX_DECODED_VIEWS:
                    return tuple(views)
        if not decoded_in_pass:
            break
        frontier = decoded_in_pass
    return tuple(views)


def _matches(patterns: Iterable[re.Pattern[str]], text: str) -> bool:
    return any(pattern.search(text) is not None for pattern in patterns)


def _dedupe(reasons: Iterable[GuardReason]) -> tuple[GuardReason, ...]:
    return tuple(dict.fromkeys(reasons))


class PiiRedactor:
    """Deterministically replace common direct identifiers without retaining matches."""

    def redact(self, text: str) -> tuple[str, bool]:
        _bounded(text)
        redacted = text
        for pattern in _PII_PATTERNS:
            redacted = pattern.sub(_REDACTED, redacted)
        return redacted, redacted != text


class InputGuard:
    """Block prompt injection and regulated intent, and redact direct identifiers."""

    def __init__(
        self,
        screening_service: ManagedScreeningService | None = None,
        *,
        screening_timeout_seconds: float = 2.0,
        pii_redactor: PiiRedactor | None = None,
    ) -> None:
        if screening_timeout_seconds <= 0:
            raise ValueError("screening_timeout_seconds must be positive")
        self._screening_service = screening_service
        self._screening_timeout_seconds = screening_timeout_seconds
        self._pii_redactor = pii_redactor or PiiRedactor()

    async def evaluate(self, payload: str) -> GuardResult[str]:
        _bounded(payload)
        views = _decoded_views(payload)
        normalized = tuple(_normalize(view) for view in views)
        injection = any(_matches(_INJECTION_PATTERNS, view) for view in normalized)
        if injection:
            reason = (
                GuardReason.OBFUSCATED_INJECTION if len(views) > 1 else GuardReason.DIRECT_INJECTION
            )
            return GuardResult(GuardVerdict.BLOCK, (reason,), "", GUARD_VERSION)
        if any(_matches(_REGULATED_PATTERNS, view) for view in normalized):
            return GuardResult(
                GuardVerdict.BLOCK, (GuardReason.REGULATED_INTENT,), "", GUARD_VERSION
            )
        if self._screening_service is not None:
            async with asyncio.timeout(self._screening_timeout_seconds):
                decision = await self._screening_service.screen(payload)
            if decision.blocked:
                return GuardResult(
                    GuardVerdict.BLOCK,
                    (GuardReason.MANAGED_SCREENING_BLOCK,),
                    "",
                    GUARD_VERSION,
                )
        redacted, changed = self._pii_redactor.redact(payload)
        if changed:
            return GuardResult(
                GuardVerdict.REDACT,
                (GuardReason.PII_DETECTED,),
                redacted,
                GUARD_VERSION,
            )
        return GuardResult(GuardVerdict.ALLOW, (), payload, GUARD_VERSION)


class EvidenceGuard:
    """Drop instruction-shaped chunks and wrap retained evidence as escaped data."""

    async def evaluate(
        self, payload: Sequence[Evidence]
    ) -> GuardResult[tuple[GuardedEvidence, ...]]:
        reasons: list[GuardReason] = []
        guarded: list[GuardedEvidence] = []
        for evidence in payload:
            normalized_views = tuple(_normalize(view) for view in _decoded_views(evidence.text))
            item_reasons = tuple(
                reason
                for reason, pattern in _EVIDENCE_PATTERNS
                if any(pattern.search(view) for view in normalized_views)
            )
            if item_reasons:
                reasons.extend(item_reasons)
                continue
            escaped = html.escape(evidence.text, quote=True)
            block = (
                f'<evidence data-only="true" chunk-id="{html.escape(evidence.citation.chunk_id)}">'
                f"\n{escaped}\n</evidence>"
            )
            guarded.append(GuardedEvidence(evidence=evidence, prompt_block=block))
        unique_reasons = _dedupe(reasons)
        if not unique_reasons:
            return GuardResult(GuardVerdict.ALLOW, (), tuple(guarded), GUARD_VERSION)
        verdict = GuardVerdict.REDACT if guarded else GuardVerdict.BLOCK
        return GuardResult(verdict, unique_reasons, tuple(guarded), GUARD_VERSION)


class OutputGuard:
    """Prevent policy, secret, canary, and direct-identifier leakage."""

    def __init__(
        self,
        *,
        canary_secrets: Sequence[str] = (),
        pii_redactor: PiiRedactor | None = None,
    ) -> None:
        if any(not canary or len(canary) > 512 for canary in canary_secrets):
            raise ValueError("canary secrets must contain 1..512 characters")
        self._canary_secrets = tuple(canary_secrets)
        self._pii_redactor = pii_redactor or PiiRedactor()

    async def evaluate(self, payload: str) -> GuardResult[str]:
        _bounded(payload)
        normalized = _normalize(payload)
        if any(canary in payload for canary in self._canary_secrets):
            return GuardResult(
                GuardVerdict.BLOCK, (GuardReason.CANARY_DETECTED,), "", GUARD_VERSION
            )
        if _matches(_SECRET_PATTERNS, payload):
            return GuardResult(
                GuardVerdict.BLOCK, (GuardReason.SECRET_DETECTED,), "", GUARD_VERSION
            )
        if _matches(_POLICY_PATTERNS, normalized):
            return GuardResult(
                GuardVerdict.BLOCK, (GuardReason.POLICY_VIOLATION,), "", GUARD_VERSION
            )
        redacted, changed = self._pii_redactor.redact(payload)
        if changed:
            return GuardResult(
                GuardVerdict.REDACT,
                (GuardReason.PII_DETECTED,),
                redacted,
                GUARD_VERSION,
            )
        return GuardResult(GuardVerdict.ALLOW, (), payload, GUARD_VERSION)
