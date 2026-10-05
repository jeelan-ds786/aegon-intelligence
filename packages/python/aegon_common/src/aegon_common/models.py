"""Validated, immutable contracts shared by Aegon-RAG services."""

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

Identifier = Annotated[str, Field(min_length=1, max_length=128)]
ShortText = Annotated[str, Field(min_length=1, max_length=512)]
UriText = Annotated[str, Field(min_length=1, max_length=2048)]
ContentText = Annotated[str, Field(min_length=1, max_length=100_000)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ContractModel(BaseModel):
    """Base configuration for all wire-level contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Modality(StrEnum):
    """Content modalities supported by retrieval and ingestion."""

    TEXT = "text"
    IMAGE = "image"
    PDF_PAGE = "pdf_page"
    TABLE = "table"
    AUDIO_SEGMENT = "audio_segment"
    VIDEO_SEGMENT = "video_segment"


class ErrorCode(StrEnum):
    """Stable, machine-readable failure categories for public contracts."""

    INVALID_ARGUMENTS = "invalid_arguments"
    UNAUTHORIZED = "unauthorized"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    RATE_LIMITED = "rate_limited"
    INTERNAL_ERROR = "internal_error"
    REFUSED = "refused"


class Timecode(ContractModel):
    """Half-open media interval measured in seconds."""

    start_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    end_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)]

    @model_validator(mode="after")
    def end_follows_start(self) -> Self:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


class QuoteSpan(ContractModel):
    """Half-open character offsets into the cited chunk text."""

    start: Annotated[int, Field(ge=0)]
    end: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def end_follows_start(self) -> Self:
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class Chunk(ContractModel):
    """A tenant-scoped, independently retrievable content unit."""

    id: Identifier
    tenant_id: Identifier
    source_id: Identifier
    modality: Modality
    text: ContentText
    parent_id: Identifier | None = None
    page: Annotated[int, Field(ge=1)] | None = None
    timecode: Timecode | None = None
    embedding_model: ShortText
    embedding_version: ShortText
    content_sha256: Sha256
    signature: Annotated[str, Field(min_length=1, max_length=4096)]
    access_level: Annotated[str, Field(min_length=1, max_length=64)]
    metadata: dict[str, JsonValue] = Field(default_factory=dict, max_length=64)


class Citation(ContractModel):
    """A precise, user-visible reference to supporting source content."""

    chunk_id: Identifier
    source_uri: UriText
    locator: ShortText
    quote_span: QuoteSpan


class Evidence(ContractModel):
    """A bounded excerpt and citation selected to support an answer."""

    citation: Citation
    text: ContentText
    relevance_score: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class PageContext(ContractModel):
    """Bounded host-page context supplied by a trusted server adapter."""

    uri: UriText
    title: Annotated[str, Field(min_length=1, max_length=512)] | None = None
    text: Annotated[str, Field(min_length=1, max_length=20_000)]


class AttachmentRef(ContractModel):
    """Reference to an attachment already accepted by the platform."""

    id: Identifier
    media_type: Annotated[str, Field(min_length=1, max_length=255)]
    name: Annotated[str, Field(min_length=1, max_length=255)] | None = None


class ChatRequest(ContractModel):
    """Tenant-neutral chat input; identity is added by the server."""

    question: Annotated[str, Field(min_length=1, max_length=8_000)]
    page_context: PageContext | None = None
    attachments: tuple[AttachmentRef, ...] = Field(default_factory=tuple, max_length=10)


class ChatResponse(ContractModel):
    """Grounded final answer returned after agent execution."""

    id: Identifier
    answer: ContentText
    evidence: tuple[Evidence, ...] = Field(default_factory=tuple, max_length=100)
    suggestions: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=8)


class StepEvent(ContractModel):
    """Reports a bounded, user-safe agent progress step."""

    type: Literal["step"] = "step"
    step: ShortText


class TokenEvent(ContractModel):
    """Carries an incremental answer token."""

    type: Literal["token"] = "token"
    token: Annotated[str, Field(min_length=1, max_length=8_000)]


class CitationEvent(ContractModel):
    """Publishes a citation as soon as it is available."""

    type: Literal["citation"] = "citation"
    citation: Citation


class SuggestionEvent(ContractModel):
    """Publishes one grounded follow-up suggestion."""

    type: Literal["suggestion"] = "suggestion"
    suggestion: ShortText


class ErrorEvent(ContractModel):
    """Carries a stable error code and safe public message."""

    type: Literal["error"] = "error"
    code: ErrorCode
    message: Annotated[str, Field(min_length=1, max_length=2_000)]
    retryable: bool = False


class DoneEvent(ContractModel):
    """Marks successful stream completion and carries the final response."""

    type: Literal["done"] = "done"
    response: ChatResponse


type StreamEvent = Annotated[
    StepEvent | TokenEvent | CitationEvent | SuggestionEvent | ErrorEvent | DoneEvent,
    Field(discriminator="type"),
]
