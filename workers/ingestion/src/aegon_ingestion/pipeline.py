"""Governed parse-to-publish ingestion orchestration."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from aegon_common.protocols import Embedder
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import BaseModel, ConfigDict, Field

from aegon_ingestion.chunker import Chunker
from aegon_ingestion.documents import ParsedDocument
from aegon_ingestion.signing import (
    SignedChunk,
    SignedManifest,
    Signer,
    sign_chunk,
    sign_manifest,
    verify_chunk,
    verify_manifest,
)


class SourceDisposition(StrEnum):
    """Governance decision for a submitted source."""

    APPROVED = "approved"
    QUARANTINE_NEW = "quarantine_new"
    QUARANTINE_ORIGIN_CHANGED = "quarantine_origin_changed"


class SourceInput(BaseModel):
    """Validated source material supplied after a controlled fetch and parse."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    source_id: str = Field(min_length=1, max_length=128)
    uri: str = Field(min_length=1, max_length=2048)
    kind: str = Field(min_length=1, max_length=64)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    document: ParsedDocument

    @classmethod
    def from_bytes(
        cls, *, source_id: str, uri: str, kind: str, content: bytes, document: ParsedDocument
    ) -> SourceInput:
        """Create an input with a checksum over the exact fetched bytes."""
        return cls(
            source_id=source_id,
            uri=uri,
            kind=kind,
            checksum=hashlib.sha256(content).hexdigest(),
            document=document,
        )


@dataclass(frozen=True, slots=True)
class PublishResult:
    """Observable result of one ingestion attempt."""

    run_id: str
    published_sources: tuple[str, ...]
    quarantined_sources: tuple[str, ...]
    chunk_count: int


class IngestionRepository(Protocol):
    """Persistence boundary; ``publish`` must commit all changes atomically."""

    def source_disposition(self, tenant_id: str, source_id: str, uri: str) -> SourceDisposition: ...

    def quarantine(self, tenant_id: str, source: SourceInput, reason: str) -> None: ...

    def source_checksum(self, tenant_id: str, source_id: str) -> str | None: ...

    def load_chunks(self, tenant_id: str, source_id: str) -> Sequence[SignedChunk]: ...

    def publish(
        self,
        tenant_id: str,
        sources: Sequence[SourceInput],
        chunks: Sequence[SignedChunk],
        manifest: SignedManifest,
    ) -> None: ...


class GovernedIngestionPipeline:
    """Build and atomically publish a complete tenant index from governed sources."""

    def __init__(
        self,
        *,
        repository: IngestionRepository,
        chunker: Chunker,
        embedder: Embedder,
        signer: Signer,
        verification_keys: Mapping[str, Ed25519PublicKey],
    ) -> None:
        self._repository = repository
        self._chunker = chunker
        self._embedder = embedder
        self._signer = signer
        self._verification_keys = verification_keys

    async def run(
        self, *, tenant_id: str, run_id: str, sources: Sequence[SourceInput]
    ) -> PublishResult:
        """Govern, build, verify, and publish one complete source snapshot."""
        _validate_unique_sources(sources)
        approved: list[SourceInput] = []
        quarantined: list[str] = []
        for source in sources:
            if source.document.source_id != source.source_id:
                raise ValueError("parsed document source_id does not match its source input")
            disposition = self._repository.source_disposition(
                tenant_id, source.source_id, source.uri
            )
            if disposition is not SourceDisposition.APPROVED:
                self._repository.quarantine(tenant_id, source, disposition.value)
                quarantined.append(source.source_id)
                continue
            approved.append(source)

        signed_chunks: list[SignedChunk] = []
        for source in approved:
            if self._repository.source_checksum(tenant_id, source.source_id) == source.checksum:
                reused = tuple(self._repository.load_chunks(tenant_id, source.source_id))
                if not all(verify_chunk(chunk, self._verification_keys) for chunk in reused):
                    raise ValueError("stored chunk signature verification failed")
                signed_chunks.extend(reused)
                continue
            chunks = self._chunker.chunk(source.document)
            vectors = await self._embedder.embed_texts(tuple(chunk.text for chunk in chunks))
            if len(vectors) != len(chunks):
                raise ValueError("embedder returned an unexpected vector count")
            for chunk, vector in zip(chunks, vectors, strict=True):
                signed_chunks.append(
                    sign_chunk(
                        tenant_id=tenant_id,
                        source_id=source.source_id,
                        chunk_id=chunk.chunk_id,
                        content=chunk.text,
                        embedding=vector,
                        signer=self._signer,
                    )
                )

        if not all(verify_chunk(chunk, self._verification_keys) for chunk in signed_chunks):
            raise ValueError("generated chunk signature verification failed")
        manifest = sign_manifest(
            run_id,
            tenant_id,
            {source.source_id: source.checksum for source in approved},
            [chunk.chunk_id for chunk in signed_chunks],
            self._signer,
        )
        if not verify_manifest(manifest, self._verification_keys):
            raise ValueError("generated manifest signature verification failed")
        self._repository.publish(tenant_id, approved, signed_chunks, manifest)
        return PublishResult(
            run_id=run_id,
            published_sources=tuple(sorted(source.source_id for source in approved)),
            quarantined_sources=tuple(sorted(quarantined)),
            chunk_count=len(signed_chunks),
        )


def _validate_unique_sources(sources: Sequence[SourceInput]) -> None:
    source_ids = [source.source_id for source in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source ids must be unique within an ingestion run")
