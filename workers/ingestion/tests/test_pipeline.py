import asyncio
from collections.abc import Sequence

import pytest
from aegon_common.embeddings import HashingEmbedder, HashingEmbedderConfig
from aegon_ingestion import (
    Chunker,
    DocumentBlock,
    GovernedIngestionPipeline,
    LocalSigner,
    ParsedDocument,
    SignedChunk,
    SignedManifest,
    SourceDisposition,
    SourceInput,
)


class MemoryRepository:
    def __init__(self) -> None:
        self.origins: dict[str, str] = {"approved": "https://approved.test/doc"}
        self.checksums: dict[str, str] = {}
        self.live: list[SignedChunk] = []
        self.quarantined: list[tuple[str, str]] = []
        self.manifests: list[SignedManifest] = []
        self.fail_publish = False

    def source_disposition(self, tenant_id: str, source_id: str, uri: str) -> SourceDisposition:
        del tenant_id
        approved_uri = self.origins.get(source_id)
        if approved_uri is None:
            return SourceDisposition.QUARANTINE_NEW
        if approved_uri != uri:
            return SourceDisposition.QUARANTINE_ORIGIN_CHANGED
        return SourceDisposition.APPROVED

    def quarantine(self, tenant_id: str, source: SourceInput, reason: str) -> None:
        del tenant_id
        self.quarantined.append((source.source_id, reason))

    def source_checksum(self, tenant_id: str, source_id: str) -> str | None:
        del tenant_id
        return self.checksums.get(source_id)

    def load_chunks(self, tenant_id: str, source_id: str) -> Sequence[SignedChunk]:
        return [
            chunk
            for chunk in self.live
            if chunk.tenant_id == tenant_id and chunk.source_id == source_id
        ]

    def publish(
        self,
        tenant_id: str,
        sources: Sequence[SourceInput],
        chunks: Sequence[SignedChunk],
        manifest: SignedManifest,
    ) -> None:
        del tenant_id
        staged = list(chunks)
        if self.fail_publish:
            raise RuntimeError("simulated crash before swap")
        self.live = staged
        self.checksums = {source.source_id: source.checksum for source in sources}
        self.manifests.append(manifest)


def _source(source_id: str, uri: str, text: str = "approved evidence") -> SourceInput:
    document = ParsedDocument(
        source_id=source_id,
        media_type="text/markdown",
        blocks=(DocumentBlock(kind="paragraph", text=text, locator="line:1"),),
    )
    return SourceInput.from_bytes(
        source_id=source_id,
        uri=uri,
        kind="markdown",
        content=text.encode(),
        document=document,
    )


def _pipeline(repository: MemoryRepository) -> GovernedIngestionPipeline:
    signer = LocalSigner.from_private_bytes("local-1", b"1" * 32)
    return GovernedIngestionPipeline(
        repository=repository,
        chunker=Chunker(),
        embedder=HashingEmbedder(HashingEmbedderConfig(dim=8)),
        signer=signer,
        verification_keys={signer.key_id: signer.public_key},
    )


def test_unapproved_and_changed_origin_sources_are_quarantined() -> None:
    repository = MemoryRepository()
    result = asyncio.run(
        _pipeline(repository).run(
            tenant_id="tenant-a",
            run_id="run-1",
            sources=(
                _source("new", "https://new.test/doc"),
                _source("approved", "https://attacker.test/doc"),
            ),
        )
    )
    assert result.published_sources == ()
    assert result.quarantined_sources == ("approved", "new")
    assert repository.quarantined == [
        ("new", "quarantine_new"),
        ("approved", "quarantine_origin_changed"),
    ]


def test_crash_before_swap_leaves_previous_index_and_manifest() -> None:
    repository = MemoryRepository()
    pipeline = _pipeline(repository)
    asyncio.run(
        pipeline.run(
            tenant_id="tenant-a",
            run_id="run-1",
            sources=(_source("approved", "https://approved.test/doc", "first version"),),
        )
    )
    previous_chunks = list(repository.live)
    previous_manifests = list(repository.manifests)
    repository.fail_publish = True

    with pytest.raises(RuntimeError, match="simulated crash"):
        asyncio.run(
            pipeline.run(
                tenant_id="tenant-a",
                run_id="run-2",
                sources=(_source("approved", "https://approved.test/doc", "second version"),),
            )
        )

    assert repository.live == previous_chunks
    assert repository.manifests == previous_manifests


def test_identical_rerun_reuses_chunks_and_is_idempotent() -> None:
    repository = MemoryRepository()
    pipeline = _pipeline(repository)
    source = _source("approved", "https://approved.test/doc")
    first = asyncio.run(pipeline.run(tenant_id="tenant-a", run_id="run-1", sources=(source,)))
    first_chunks = list(repository.live)
    second = asyncio.run(pipeline.run(tenant_id="tenant-a", run_id="run-2", sources=(source,)))
    assert first.chunk_count == second.chunk_count == 1
    assert repository.live == first_chunks
