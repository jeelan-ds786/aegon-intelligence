"""Dependency-injection protocols for Aegon-RAG core logic."""

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol


class Embedder(Protocol):
    """Adapter contract for versioned text and image embeddings."""

    @property
    def model_id(self) -> str: ...

    @property
    def dim(self) -> int: ...

    @property
    def version(self) -> str: ...

    async def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...

    async def embed_images(self, images: Sequence[bytes]) -> Sequence[Sequence[float]]: ...


class TextClock(Protocol):
    """Provides an RFC 3339 UTC timestamp for deterministic workflows."""

    def now(self) -> datetime: ...


class IdGenerator(Protocol):
    """Generates opaque identifiers without coupling callers to a format."""

    def new(self) -> str: ...
