"""Shared Aegon-RAG Python contracts."""

from aegon_common.models import (
    AttachmentRef,
    ChatRequest,
    ChatResponse,
    Chunk,
    Citation,
    CitationEvent,
    DoneEvent,
    ErrorCode,
    ErrorEvent,
    Evidence,
    Modality,
    PageContext,
    QuoteSpan,
    StepEvent,
    StreamEvent,
    SuggestionEvent,
    Timecode,
    TokenEvent,
)
from aegon_common.protocols import Embedder, IdGenerator, TextClock

__version__ = "0.0.0"

__all__ = [
    "AttachmentRef",
    "ChatRequest",
    "ChatResponse",
    "Chunk",
    "Citation",
    "CitationEvent",
    "DoneEvent",
    "Embedder",
    "ErrorCode",
    "ErrorEvent",
    "Evidence",
    "IdGenerator",
    "Modality",
    "PageContext",
    "QuoteSpan",
    "StepEvent",
    "StreamEvent",
    "SuggestionEvent",
    "TextClock",
    "Timecode",
    "TokenEvent",
    "__version__",
]
