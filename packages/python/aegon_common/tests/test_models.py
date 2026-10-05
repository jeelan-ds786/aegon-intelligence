from typing import cast

import pytest
from aegon_common import (
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
from hypothesis import given
from hypothesis import strategies as st
from pydantic import TypeAdapter, ValidationError

IDENTIFIERS = st.text(
    alphabet=st.characters(categories=("L", "N"), include_characters="_-"),
    min_size=1,
    max_size=128,
)
CONTENT = st.text(min_size=1, max_size=2_000).filter(lambda value: bool(value.strip()))
STREAM_ADAPTER: TypeAdapter[StreamEvent] = TypeAdapter(StreamEvent)


@given(
    identifier=IDENTIFIERS,
    content=CONTENT,
    modality=st.sampled_from(Modality),
    score=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
)
def test_nested_contracts_round_trip_json(
    identifier: str,
    content: str,
    modality: Modality,
    score: float,
) -> None:
    chunk = Chunk(
        id=identifier,
        tenant_id=identifier,
        source_id=identifier,
        modality=modality,
        text=content,
        timecode=Timecode(start_seconds=0, end_seconds=1),
        embedding_model="hashing-embedder",
        embedding_version="1",
        content_sha256="a" * 64,
        signature="test-signature",
        access_level="private",
        metadata={"synthetic": True},
    )
    citation = Citation(
        chunk_id=identifier,
        source_uri="https://example.test/source",
        locator="page 1",
        quote_span=QuoteSpan(start=0, end=1),
    )
    response = ChatResponse(
        id=identifier,
        answer=content,
        evidence=(Evidence(citation=citation, text=content, relevance_score=score),),
        suggestions=("Ask a follow-up",),
    )

    assert Chunk.model_validate_json(chunk.model_dump_json()) == chunk
    assert ChatResponse.model_validate_json(response.model_dump_json()) == response


@given(question=CONTENT, identifier=IDENTIFIERS)
def test_chat_request_round_trips_without_client_identity(question: str, identifier: str) -> None:
    request = ChatRequest(
        question=question,
        page_context=PageContext(
            uri="https://example.test/current",
            title="Current page",
            text="Approved page context",
        ),
        attachments=(
            AttachmentRef(id=identifier, media_type="application/pdf", name="evidence.pdf"),
        ),
    )

    assert ChatRequest.model_validate_json(request.model_dump_json()) == request
    assert "tenant_id" not in ChatRequest.model_json_schema()["properties"]


@given(event_index=st.integers(min_value=0, max_value=5))
def test_every_stream_event_round_trips_through_discriminator(event_index: int) -> None:
    citation = Citation(
        chunk_id="chunk-1",
        source_uri="https://example.test/source",
        locator="page 1",
        quote_span=QuoteSpan(start=0, end=4),
    )
    response = ChatResponse(
        id="response-1",
        answer="Grounded answer",
        evidence=(Evidence(citation=citation, text="Fact", relevance_score=1),),
    )
    events: tuple[StreamEvent, ...] = (
        StepEvent(step="retrieving"),
        TokenEvent(token="Grounded"),
        CitationEvent(citation=citation),
        SuggestionEvent(suggestion="Read the source"),
        ErrorEvent(code=ErrorCode.TIMEOUT, message="The request timed out", retryable=True),
        DoneEvent(response=response),
    )
    event = events[event_index]

    encoded = STREAM_ADAPTER.dump_json(event)
    assert STREAM_ADAPTER.validate_json(encoded) == event


def test_models_are_frozen_and_reject_extra_fields() -> None:
    request = ChatRequest(question="What is supported?")

    with pytest.raises(ValidationError):
        request.question = "Changed"  # pyright: ignore[reportAttributeAccessIssue]

    with pytest.raises(ValidationError):
        ChatRequest.model_validate({"question": "Valid", "tenant_id": "browser-controlled"})


@pytest.mark.parametrize("length", [0, 8_001])
def test_question_length_is_bounded(length: int) -> None:
    with pytest.raises(ValidationError):
        ChatRequest(question=cast(str, "x" * length))
