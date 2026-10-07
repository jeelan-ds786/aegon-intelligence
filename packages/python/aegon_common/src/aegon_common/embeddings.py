"""Versioned text and image embedding adapters."""

from __future__ import annotations

import hashlib
import math
import random
import re
import time
from asyncio import sleep, wait_for
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, overload

from aegon_common.models import ErrorCode

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


class EmbedderError(Exception):
    """An embedding failure mapped to the public error taxonomy."""

    def __init__(self, code: ErrorCode, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class EmbeddingVector:
    """A normalized vector carrying the model identity needed for migrations."""

    values: tuple[float, ...]
    model_id: str
    model_version: str


@dataclass(frozen=True, slots=True)
class HashingEmbedderConfig:
    """Configuration for the deterministic, non-semantic test embedder."""

    provider: Literal["hashing"] = "hashing"
    dim: int = 256
    model_id: str = "signed-feature-hashing"
    model_version: str = "1"
    max_text_chars: int = 100_000
    max_image_bytes: int = 10_000_000


@dataclass(frozen=True, slots=True)
class GeminiEmbedderConfig:
    """Vertex AI Gemini embedding configuration; authentication always uses ADC."""

    project: str
    model_id: str
    model_version: str
    dim: int
    provider: Literal["gemini"] = "gemini"
    location: str = "us-central1"
    max_batch_size: int = 16
    max_batch_bytes: int = 1_000_000
    max_text_chars: int = 20_000
    max_image_bytes: int = 5_000_000
    truncate_text: bool = True
    timeout_seconds: float = 30.0
    max_attempts: int = 3
    backoff_base_seconds: float = 0.25
    backoff_max_seconds: float = 4.0
    circuit_failure_threshold: int = 5
    circuit_recovery_seconds: float = 30.0


class EmbeddingTransport(Protocol):
    """Narrow provider transport used to keep controls independently testable."""

    async def embed_texts(self, texts: Sequence[str], dim: int) -> Sequence[Sequence[float]]: ...

    async def embed_images(
        self, images: Sequence[tuple[bytes, str]], dim: int
    ) -> Sequence[Sequence[float]]: ...


class _SdkEmbedding(Protocol):
    @property
    def values(self) -> Sequence[float] | None: ...


class HashingEmbedder:
    """Deterministic signed feature hashing for offline development and tests."""

    def __init__(self, config: HashingEmbedderConfig) -> None:
        _validate_common_config(config.dim, config.model_id, config.model_version)
        if config.max_text_chars < 1 or config.max_image_bytes < 1:
            raise ValueError("input limits must be positive")
        self._config = config

    @property
    def model_id(self) -> str:
        return self._config.model_id

    @property
    def version(self) -> str:
        return self._config.model_version

    @property
    def dim(self) -> int:
        return self._config.dim

    async def embed_texts(self, texts: Sequence[str]) -> tuple[EmbeddingVector, ...]:
        return tuple(self._embed_text(text) for text in texts)

    async def embed_images(self, images: Sequence[bytes]) -> tuple[EmbeddingVector, ...]:
        return tuple(self._embed_image(image) for image in images)

    def _embed_text(self, text: str) -> EmbeddingVector:
        if not text.strip():
            raise EmbedderError(ErrorCode.INVALID_ARGUMENTS, "text must not be empty")
        if len(text) > self._config.max_text_chars:
            raise EmbedderError(ErrorCode.INVALID_ARGUMENTS, "text exceeds the configured limit")
        features = (token.encode("utf-8") for token in _TOKEN_PATTERN.findall(text.casefold()))
        return self._build_vector(features)

    def _embed_image(self, image: bytes) -> EmbeddingVector:
        if not image:
            raise EmbedderError(ErrorCode.INVALID_ARGUMENTS, "image must not be empty")
        if len(image) > self._config.max_image_bytes:
            raise EmbedderError(ErrorCode.INVALID_ARGUMENTS, "image exceeds the configured limit")
        features = (image[offset : offset + 64] for offset in range(0, len(image), 64))
        return self._build_vector(features)

    def _build_vector(self, features: Iterable[bytes]) -> EmbeddingVector:
        values = [0.0] * self.dim
        for feature in features:
            digest = hashlib.sha256(feature).digest()
            index = int.from_bytes(digest[:8], "big") % self.dim
            values[index] += 1.0 if digest[8] & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            raise EmbedderError(ErrorCode.INVALID_ARGUMENTS, "input has no embeddable features")
        return EmbeddingVector(
            values=tuple(value / norm for value in values),
            model_id=self.model_id,
            model_version=self.version,
        )


class GeminiEmbedder:
    """Controlled Vertex AI adapter for versioned Gemini multimodal embeddings."""

    def __init__(
        self,
        config: GeminiEmbedderConfig,
        *,
        transport: EmbeddingTransport | None = None,
        sleeper: Callable[[float], Awaitable[None]] = sleep,
        random_source: Callable[[], float] = random.random,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        _validate_gemini_config(config)
        self._config = config
        self._transport = transport or _VertexGeminiTransport(config)
        self._sleep = sleeper
        self._random = random_source
        self._monotonic = monotonic
        self._consecutive_failures = 0
        self._circuit_opened_at: float | None = None

    @property
    def model_id(self) -> str:
        return self._config.model_id

    @property
    def version(self) -> str:
        return self._config.model_version

    @property
    def dim(self) -> int:
        return self._config.dim

    async def embed_texts(self, texts: Sequence[str]) -> tuple[EmbeddingVector, ...]:
        prepared = tuple(self._prepare_text(text) for text in texts)
        batches = _build_batches(
            prepared,
            self._config.max_batch_size,
            self._config.max_batch_bytes,
            lambda text: len(text.encode("utf-8")),
        )
        vectors: list[EmbeddingVector] = []
        for batch in batches:
            raw = await self._call(lambda batch=batch: self._transport.embed_texts(batch, self.dim))
            vectors.extend(self._normalize_batch(raw, len(batch)))
        return tuple(vectors)

    async def embed_images(self, images: Sequence[bytes]) -> tuple[EmbeddingVector, ...]:
        prepared = tuple(self._prepare_image(image) for image in images)
        batches = _build_batches(
            prepared,
            self._config.max_batch_size,
            self._config.max_batch_bytes,
            lambda image: len(image[0]),
        )
        vectors: list[EmbeddingVector] = []
        for batch in batches:
            raw = await self._call(
                lambda batch=batch: self._transport.embed_images(batch, self.dim)
            )
            vectors.extend(self._normalize_batch(raw, len(batch)))
        return tuple(vectors)

    def _prepare_text(self, text: str) -> str:
        if not text.strip():
            raise EmbedderError(ErrorCode.INVALID_ARGUMENTS, "text must not be empty")
        if len(text) <= self._config.max_text_chars:
            return text
        if not self._config.truncate_text:
            raise EmbedderError(ErrorCode.INVALID_ARGUMENTS, "text exceeds the configured limit")
        return text[: self._config.max_text_chars]

    def _prepare_image(self, image: bytes) -> tuple[bytes, str]:
        if not image:
            raise EmbedderError(ErrorCode.INVALID_ARGUMENTS, "image must not be empty")
        if len(image) > self._config.max_image_bytes:
            raise EmbedderError(ErrorCode.INVALID_ARGUMENTS, "image exceeds the configured limit")
        return image, _image_mime_type(image)

    async def _call(
        self, operation: Callable[[], Awaitable[Sequence[Sequence[float]]]]
    ) -> Sequence[Sequence[float]]:
        self._ensure_circuit_available()
        last_error: EmbedderError | None = None
        for attempt in range(self._config.max_attempts):
            try:
                result = await wait_for(operation(), timeout=self._config.timeout_seconds)
            except TimeoutError:
                last_error = EmbedderError(
                    ErrorCode.TIMEOUT,
                    "embedding request timed out",
                    retryable=True,
                )
            except EmbedderError as error:
                last_error = error
            except Exception as error:
                last_error = _map_provider_error(error)
            else:
                self._consecutive_failures = 0
                self._circuit_opened_at = None
                return result

            if not last_error.retryable or attempt + 1 == self._config.max_attempts:
                self._record_failure()
                raise last_error
            delay = min(
                self._config.backoff_base_seconds * (2**attempt),
                self._config.backoff_max_seconds,
            )
            await self._sleep(delay * (0.5 + self._random()))
        raise AssertionError("retry loop exhausted without a result")

    def _ensure_circuit_available(self) -> None:
        if self._circuit_opened_at is None:
            return
        if self._monotonic() - self._circuit_opened_at >= self._config.circuit_recovery_seconds:
            self._circuit_opened_at = None
            self._consecutive_failures = 0
            return
        raise EmbedderError(
            ErrorCode.DEPENDENCY_UNAVAILABLE,
            "embedding provider circuit is open",
            retryable=True,
        )

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._config.circuit_failure_threshold:
            self._circuit_opened_at = self._monotonic()

    def _normalize_batch(
        self, vectors: Sequence[Sequence[float]], expected_count: int
    ) -> tuple[EmbeddingVector, ...]:
        if len(vectors) != expected_count:
            raise EmbedderError(
                ErrorCode.DEPENDENCY_UNAVAILABLE,
                "embedding provider returned an unexpected vector count",
                retryable=True,
            )
        return tuple(self._normalize(vector) for vector in vectors)

    def _normalize(self, vector: Sequence[float]) -> EmbeddingVector:
        if len(vector) != self.dim or not all(math.isfinite(value) for value in vector):
            raise EmbedderError(
                ErrorCode.DEPENDENCY_UNAVAILABLE,
                "embedding provider returned an invalid vector",
                retryable=True,
            )
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            raise EmbedderError(
                ErrorCode.DEPENDENCY_UNAVAILABLE,
                "embedding provider returned a zero vector",
                retryable=True,
            )
        return EmbeddingVector(
            values=tuple(value / norm for value in vector),
            model_id=self.model_id,
            model_version=self.version,
        )


class _VertexGeminiTransport:
    def __init__(self, config: GeminiEmbedderConfig) -> None:
        from google import genai
        from google.genai import types

        self._model_id = config.model_id
        self._types = types
        self._client = genai.Client(
            vertexai=True,
            project=config.project,
            location=config.location,
            http_options=types.HttpOptions(timeout=int(config.timeout_seconds * 1_000)),
        )

    async def embed_texts(self, texts: Sequence[str], dim: int) -> Sequence[Sequence[float]]:
        response = await self._client.aio.models.embed_content(  # pyright: ignore[reportUnknownMemberType]
            model=self._model_id,
            contents=list(texts),
            config=self._types.EmbedContentConfig(output_dimensionality=dim, auto_truncate=False),
        )
        return _extract_sdk_vectors(response.embeddings)

    async def embed_images(
        self, images: Sequence[tuple[bytes, str]], dim: int
    ) -> Sequence[Sequence[float]]:
        contents = [
            self._types.Content(
                role="user",
                parts=[self._types.Part.from_bytes(data=data, mime_type=mime_type)],
            )
            for data, mime_type in images
        ]
        response = await self._client.aio.models.embed_content(  # pyright: ignore[reportUnknownMemberType]
            model=self._model_id,
            contents=contents,
            config=self._types.EmbedContentConfig(output_dimensionality=dim, auto_truncate=False),
        )
        return _extract_sdk_vectors(response.embeddings)


def _validate_common_config(dim: int, model_id: str, model_version: str) -> None:
    if dim < 1:
        raise ValueError("dim must be positive")
    if not model_id or not model_version:
        raise ValueError("model_id and model_version must not be empty")


def _validate_gemini_config(config: GeminiEmbedderConfig) -> None:
    _validate_common_config(config.dim, config.model_id, config.model_version)
    if not config.project or not config.location:
        raise ValueError("project and location must not be empty")
    positive_values = (
        config.max_batch_size,
        config.max_batch_bytes,
        config.max_text_chars,
        config.max_image_bytes,
        config.timeout_seconds,
        config.max_attempts,
        config.backoff_max_seconds,
        config.circuit_failure_threshold,
        config.circuit_recovery_seconds,
    )
    if any(value <= 0 for value in positive_values) or config.backoff_base_seconds < 0:
        raise ValueError("Gemini limits, attempts, and timeouts must be positive")


def _build_batches[T](
    items: Sequence[T],
    max_count: int,
    max_bytes: int,
    size_of: Callable[[T], int],
) -> tuple[tuple[T, ...], ...]:
    batches: list[tuple[T, ...]] = []
    current: list[T] = []
    current_bytes = 0
    for item in items:
        item_bytes = size_of(item)
        if item_bytes > max_bytes:
            raise EmbedderError(
                ErrorCode.INVALID_ARGUMENTS,
                "one input exceeds the batch byte limit",
            )
        if current and (len(current) == max_count or current_bytes + item_bytes > max_bytes):
            batches.append(tuple(current))
            current = []
            current_bytes = 0
        current.append(item)
        current_bytes += item_bytes
    if current:
        batches.append(tuple(current))
    return tuple(batches)


def _image_mime_type(image: bytes) -> str:
    if image.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if image.startswith(b"RIFF") and image[8:12] == b"WEBP":
        return "image/webp"
    raise EmbedderError(ErrorCode.INVALID_ARGUMENTS, "image format is not supported")


def _extract_sdk_vectors(
    embeddings: Sequence[_SdkEmbedding] | None,
) -> tuple[tuple[float, ...], ...]:
    if embeddings is None:
        raise EmbedderError(
            ErrorCode.DEPENDENCY_UNAVAILABLE,
            "embedding provider returned no vectors",
            retryable=True,
        )
    vectors: list[tuple[float, ...]] = []
    for embedding in embeddings:
        values = getattr(embedding, "values", None)
        if values is None:
            raise EmbedderError(
                ErrorCode.DEPENDENCY_UNAVAILABLE,
                "embedding provider returned no vector values",
                retryable=True,
            )
        vectors.append(tuple(values))
    return tuple(vectors)


def _map_provider_error(error: Exception) -> EmbedderError:
    status_code = getattr(error, "status_code", None) or getattr(error, "code", None)
    if status_code == 429:
        return EmbedderError(
            ErrorCode.RATE_LIMITED,
            "embedding provider rate limited",
            retryable=True,
        )
    if status_code in {408, 504}:
        return EmbedderError(ErrorCode.TIMEOUT, "embedding provider timed out", retryable=True)
    if isinstance(status_code, int) and status_code >= 500:
        return EmbedderError(
            ErrorCode.DEPENDENCY_UNAVAILABLE,
            "embedding provider is unavailable",
            retryable=True,
        )
    return EmbedderError(
        ErrorCode.DEPENDENCY_UNAVAILABLE,
        "embedding provider request failed",
        retryable=False,
    )


@overload
def get_embedder(config: HashingEmbedderConfig) -> HashingEmbedder: ...


@overload
def get_embedder(config: GeminiEmbedderConfig) -> GeminiEmbedder: ...


def get_embedder(
    config: HashingEmbedderConfig | GeminiEmbedderConfig,
) -> HashingEmbedder | GeminiEmbedder:
    """Build an embedder without reading credentials or configuration globally."""

    if isinstance(config, HashingEmbedderConfig):
        return HashingEmbedder(config)
    return GeminiEmbedder(config)
