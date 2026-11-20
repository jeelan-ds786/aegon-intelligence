"""Typed decisions shared by all safety guards."""

from dataclasses import dataclass
from enum import StrEnum

from aegon_common.models import Evidence


class GuardVerdict(StrEnum):
    """Stable outcomes emitted by a guard."""

    ALLOW = "allow"
    REDACT = "redact"
    BLOCK = "block"


class GuardReason(StrEnum):
    """Machine-readable reasons that are safe to log without payload content."""

    DIRECT_INJECTION = "direct_injection"
    OBFUSCATED_INJECTION = "obfuscated_injection"
    REGULATED_INTENT = "regulated_intent"
    PII_DETECTED = "pii_detected"
    INSTRUCTION_SHAPED_EVIDENCE = "instruction_shaped_evidence"
    ROLE_IMPERSONATION = "role_impersonation"
    TOOL_CALL_SYNTAX = "tool_call_syntax"
    HIDDEN_TEXT = "hidden_text"
    SECRET_DETECTED = "secret_detected"
    CANARY_DETECTED = "canary_detected"
    POLICY_VIOLATION = "policy_violation"
    MANAGED_SCREENING_BLOCK = "managed_screening_block"


@dataclass(frozen=True, slots=True)
class GuardResult[PayloadT]:
    """One guard decision with a transformed payload and versioned policy."""

    verdict: GuardVerdict
    reasons: tuple[GuardReason, ...]
    redacted_payload: PayloadT
    guard_version: str

    def __post_init__(self) -> None:
        if not self.guard_version:
            raise ValueError("guard_version must be non-empty")
        if self.verdict is GuardVerdict.ALLOW and self.reasons:
            raise ValueError("allow verdicts cannot include reasons")
        if self.verdict is not GuardVerdict.ALLOW and not self.reasons:
            raise ValueError("non-allow verdicts require at least one reason")


@dataclass(frozen=True, slots=True)
class ScreeningDecision:
    """Content-free decision returned by an optional managed screening adapter."""

    blocked: bool
    policy_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.blocked and not self.policy_ids:
            raise ValueError("blocked screening decisions require policy_ids")
        if not self.blocked and self.policy_ids:
            raise ValueError("allowed screening decisions cannot include policy_ids")
        if any(not policy_id or len(policy_id) > 128 for policy_id in self.policy_ids):
            raise ValueError("policy_ids must be non-empty and at most 128 characters")


@dataclass(frozen=True, slots=True)
class GuardedEvidence:
    """Approved evidence plus its escaped, explicitly data-only prompt block."""

    evidence: Evidence
    prompt_block: str
