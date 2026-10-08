import asyncio
import math
from collections.abc import Sequence
from dataclasses import replace

import pytest
from aegon_common.embeddings import (
    EmbedderError,
    EmbeddingTransport,
    GeminiEmbedder,
    GeminiEmbedderConfig,
    HashingEmbedderConfig,
    get_embedder,
)
from aegon_common.models import ErrorCode


class ProviderFailure(Exception):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class FakeTransport(EmbeddingTransport):
    def __init__(
        self,
        outcomes: list[Sequence[Sequence[float]] | Exception] | None = None,
    ) -> None:
        self.outcomes = outcomes or []
        self.text_calls: list[tuple[str, ...]] = []
        self.image_calls: list[tuple[tuple[bytes, str], ...]] = []

    async def embed_texts(self, texts: Sequence[str], dim: int) -> Sequence[Sequence[float]]:
        self.text_calls.append(tuple(texts))
        return self._outcome(len(texts), dim)

    async def embed_images(
        self, images: Sequence[tuple[bytes, str]], dim: int
    ) -> Sequence[Sequence[float]]:
        self.image_calls.append(tuple(images))
        return self._outcome(len(images), dim)

    def _outcome(self, count: int, dim: int) -> Sequence[Sequence[float]]:
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return [tuple([3.0, 4.0] + [0.0] * (dim - 2)) for _ in range(count)]


class SlowTransport(FakeTransport):
    async def embed_texts(self, texts: Sequence[str], dim: int) -> Sequence[Sequence[float]]:
        await asyncio.sleep(0.05)
        return await super().embed_texts(texts, dim)


def _gemini_config() -> GeminiEmbedderConfig:
    return GeminiEmbedderConfig(
        project="test-project",
        model_id="gemini-test",
        model_version="2026-10-01",
        dim=4,
        max_attempts=1,
    )


def test_hashing_embedder_is_deterministic_normalized_and_versioned() -> None:
    embedder = get_embedder(HashingEmbedderConfig(dim=32, model_version="fixture-v2"))

    first = asyncio.run(embedder.embed_texts(["The quick brown fox"]))[0]
    second = asyncio.run(embedder.embed_texts(["The quick brown fox"]))[0]

    assert first == second
    assert len(first.values) == 32
    assert math.isclose(math.sqrt(sum(value * value for value in first.values)), 1.0)
    assert first.model_id == "signed-feature-hashing"
    assert first.model_version == "fixture-v2"


def test_hashing_image_embedding_hashes_bytes_and_is_normalized() -> None:
    embedder = get_embedder(HashingEmbedderConfig(dim=16))

    vector = asyncio.run(embedder.embed_images([b"not a semantic image"]))[0]

    assert len(vector.values) == 16
    assert math.isclose(math.sqrt(sum(value * value for value in vector.values)), 1.0)


@pytest.mark.parametrize("value", ["", "   "])
def test_hashing_rejects_empty_text(value: str) -> None:
    embedder = get_embedder(HashingEmbedderConfig())

    with pytest.raises(EmbedderError) as caught:
        asyncio.run(embedder.embed_texts([value]))

    assert caught.value.code is ErrorCode.INVALID_ARGUMENTS


def test_hashing_rejects_oversized_inputs() -> None:
    embedder = get_embedder(HashingEmbedderConfig(max_text_chars=3, max_image_bytes=3))

    with pytest.raises(EmbedderError, match="text exceeds"):
        asyncio.run(embedder.embed_texts(["four"]))
    with pytest.raises(EmbedderError, match="image exceeds"):
        asyncio.run(embedder.embed_images([b"four"]))


def test_hashing_validates_config_and_inputs_without_features() -> None:
    with pytest.raises(ValueError, match="dim must be positive"):
        get_embedder(HashingEmbedderConfig(dim=0))
    with pytest.raises(ValueError, match="input limits must be positive"):
        get_embedder(HashingEmbedderConfig(max_image_bytes=0))

    embedder = get_embedder(HashingEmbedderConfig())
    with pytest.raises(EmbedderError, match="image must not be empty"):
        asyncio.run(embedder.embed_images([b""]))
    with pytest.raises(EmbedderError, match="no embeddable features"):
        asyncio.run(embedder.embed_texts(["..."]))


def test_gemini_batches_truncates_normalizes_and_surfaces_version() -> None:
    transport = FakeTransport()
    config = replace(_gemini_config(), max_batch_size=2, max_text_chars=4)
    embedder = GeminiEmbedder(config, transport=transport)

    vectors = asyncio.run(embedder.embed_texts(["abcdef", "two", "three"]))

    assert transport.text_calls == [("abcd", "two"), ("thre",)]
    assert [vector.values for vector in vectors] == [(0.6, 0.8, 0.0, 0.0)] * 3
    assert all(vector.model_id == "gemini-test" for vector in vectors)
    assert all(vector.model_version == "2026-10-01" for vector in vectors)


def test_gemini_validates_and_labels_images() -> None:
    transport = FakeTransport()
    embedder = GeminiEmbedder(_gemini_config(), transport=transport)
    png = b"\x89PNG\r\n\x1a\nfixture"

    asyncio.run(embedder.embed_images([png]))

    assert transport.image_calls == [((png, "image/png"),)]
    with pytest.raises(EmbedderError) as caught:
        asyncio.run(embedder.embed_images([b"unknown-format"]))
    assert caught.value.code is ErrorCode.INVALID_ARGUMENTS


@pytest.mark.parametrize(
    ("header", "mime_type"),
    [
        (b"\xff\xd8\xfffixture", "image/jpeg"),
        (b"GIF89afixture", "image/gif"),
        (b"RIFFxxxxWEBPfixture", "image/webp"),
    ],
)
def test_gemini_detects_supported_image_formats(header: bytes, mime_type: str) -> None:
    transport = FakeTransport()
    embedder = GeminiEmbedder(_gemini_config(), transport=transport)

    asyncio.run(embedder.embed_images([header]))

    assert transport.image_calls == [((header, mime_type),)]


def test_gemini_rejects_disallowed_or_oversized_inputs() -> None:
    config = replace(
        _gemini_config(),
        truncate_text=False,
        max_text_chars=3,
        max_image_bytes=8,
        max_batch_bytes=3,
    )
    embedder = GeminiEmbedder(config, transport=FakeTransport())

    with pytest.raises(EmbedderError, match="text exceeds"):
        asyncio.run(embedder.embed_texts(["four"]))
    with pytest.raises(EmbedderError, match="image must not be empty"):
        asyncio.run(embedder.embed_images([b""]))
    with pytest.raises(EmbedderError, match="image exceeds"):
        asyncio.run(embedder.embed_images([b"\x89PNG\r\n\x1a\nlarge"]))
    with pytest.raises(EmbedderError, match="batch byte limit"):
        asyncio.run(embedder.embed_texts(["\u00e9\u00e9"]))


@pytest.mark.parametrize(
    "config",
    [
        replace(_gemini_config(), model_version=""),
        replace(_gemini_config(), project=""),
        replace(_gemini_config(), max_attempts=0),
    ],
)
def test_gemini_rejects_invalid_config(config: GeminiEmbedderConfig) -> None:
    with pytest.raises(ValueError):
        GeminiEmbedder(config, transport=FakeTransport())


@pytest.mark.parametrize(
    "vectors",
    [
        [],
        [[1.0, 2.0]],
        [[0.0, 0.0, 0.0, 0.0]],
        [[math.inf, 0.0, 0.0, 0.0]],
    ],
)
def test_gemini_rejects_malformed_provider_vectors(
    vectors: Sequence[Sequence[float]],
) -> None:
    embedder = GeminiEmbedder(_gemini_config(), transport=FakeTransport([vectors]))

    with pytest.raises(EmbedderError) as caught:
        asyncio.run(embedder.embed_texts(["provider output fixture"]))

    assert caught.value.code is ErrorCode.DEPENDENCY_UNAVAILABLE


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (429, ErrorCode.RATE_LIMITED),
        (504, ErrorCode.TIMEOUT),
        (503, ErrorCode.DEPENDENCY_UNAVAILABLE),
        (400, ErrorCode.DEPENDENCY_UNAVAILABLE),
    ],
)
def test_gemini_maps_provider_errors(status_code: int, expected: ErrorCode) -> None:
    embedder = GeminiEmbedder(
        _gemini_config(),
        transport=FakeTransport([ProviderFailure(status_code)]),
    )

    with pytest.raises(EmbedderError) as caught:
        asyncio.run(embedder.embed_texts(["safe fixture input"]))

    assert caught.value.code is expected


def test_gemini_enforces_per_call_timeout() -> None:
    config = replace(_gemini_config(), timeout_seconds=0.001)
    embedder = GeminiEmbedder(config, transport=SlowTransport())

    with pytest.raises(EmbedderError) as caught:
        asyncio.run(embedder.embed_texts(["timeout fixture"]))

    assert caught.value.code is ErrorCode.TIMEOUT


def test_gemini_retries_with_exponential_backoff_and_jitter() -> None:
    delays: list[float] = []

    async def record_delay(delay: float) -> None:
        delays.append(delay)

    transport = FakeTransport(
        [
            ProviderFailure(503),
            ProviderFailure(503),
            [[1.0, 0.0, 0.0, 0.0]],
        ]
    )
    config = replace(_gemini_config(), max_attempts=3, backoff_base_seconds=1.0)
    embedder = GeminiEmbedder(
        config,
        transport=transport,
        sleeper=record_delay,
        random_source=lambda: 0.5,
    )

    asyncio.run(embedder.embed_texts(["retry fixture"]))

    assert delays == [1.0, 2.0]
    assert len(transport.text_calls) == 3


def test_gemini_circuit_opens_and_recovers() -> None:
    now = [10.0]
    transport = FakeTransport([ProviderFailure(503), [[1.0, 0.0, 0.0, 0.0]]])
    config = replace(
        _gemini_config(),
        circuit_failure_threshold=1,
        circuit_recovery_seconds=5.0,
    )
    embedder = GeminiEmbedder(config, transport=transport, monotonic=lambda: now[0])

    with pytest.raises(EmbedderError):
        asyncio.run(embedder.embed_texts(["first"]))
    with pytest.raises(EmbedderError, match="circuit is open"):
        asyncio.run(embedder.embed_texts(["blocked"]))
    assert len(transport.text_calls) == 1

    now[0] = 15.0
    vector = asyncio.run(embedder.embed_texts(["recovered"]))[0]
    assert vector.values == (1.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("text", ["", "   "])
def test_all_embedders_map_empty_text_to_invalid_arguments(text: str) -> None:
    embedders = (
        get_embedder(HashingEmbedderConfig()),
        GeminiEmbedder(_gemini_config(), transport=FakeTransport()),
    )

    for embedder in embedders:
        with pytest.raises(EmbedderError) as caught:
            asyncio.run(embedder.embed_texts([text]))
        assert caught.value.code is ErrorCode.INVALID_ARGUMENTS
