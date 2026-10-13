"""Tenant-scoped hybrid retrieval without generation concerns."""

from __future__ import annotations

import asyncio
import hashlib
import struct
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Literal, Protocol, cast

import rfc8785
from aegon_common.embeddings import EmbeddingVector
from aegon_common.models import Citation, Evidence, QuoteSpan
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

type SqlValue = str | int | float | list[str]
type SqlRow = tuple[
    str,
    str,
    str,
    str,
    bytes,
    bytes,
    str,
    str,
    bytes,
    str,
    str,
    str,
    int | None,
    str,
    float | None,
]


class TextEmbedder(Protocol):
    """Embedding behavior required by retrieval."""

    async def embed_texts(self, texts: Sequence[str]) -> Sequence[EmbeddingVector]: ...


class QueryResult(Protocol):
    """Minimal async result surface used by retrieval."""

    async def fetchall(self) -> list[SqlRow]: ...


class AsyncConnection(Protocol):
    """Minimal pooled connection surface used by retrieval."""

    def transaction(self) -> AbstractAsyncContextManager[object]: ...

    async def execute(
        self, query: str, params: Sequence[SqlValue] | None = None
    ) -> QueryResult: ...


class AsyncConnectionPool(Protocol):
    """Connection pool behavior required by retrieval."""

    def connection(self) -> AbstractAsyncContextManager[AsyncConnection]: ...


@dataclass(frozen=True, slots=True)
class TenantContext:
    """Server-derived retrieval scope; never construct from browser identity fields."""

    tenant_id: str
    access_levels: tuple[str, ...]
    current_page_uri: str | None = None

    def __post_init__(self) -> None:
        invalid_access = not self.access_levels or any(not level for level in self.access_levels)
        if not self.tenant_id or invalid_access:
            raise ValueError("tenant_id and access_levels must be non-empty")


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Bounds and compatibility requirements for one retriever instance."""

    candidate_limit: int = 50
    rrf_k: int = 60
    current_page_boost: float = 0.005
    similarity_floor: float = 0.25
    embed_timeout_seconds: float = 5.0
    dense_timeout_seconds: float = 2.0
    sparse_timeout_seconds: float = 2.0
    fusion_timeout_seconds: float = 1.0
    verification_timeout_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.candidate_limit < 1 or self.rrf_k < 1:
            raise ValueError("candidate_limit and rrf_k must be positive")
        if not 0 <= self.current_page_boost <= 1 / (self.rrf_k + 1):
            raise ValueError("current_page_boost exceeds one first-rank RRF contribution")
        if not 0 <= self.similarity_floor <= 1:
            raise ValueError("similarity_floor must be between zero and one")
        timeouts = (
            self.embed_timeout_seconds,
            self.dense_timeout_seconds,
            self.sparse_timeout_seconds,
            self.fusion_timeout_seconds,
            self.verification_timeout_seconds,
        )
        if any(timeout <= 0 for timeout in timeouts):
            raise ValueError("stage timeouts must be positive")


@dataclass(frozen=True, slots=True)
class RetrievalTimings:
    """Wall-clock stage durations in milliseconds."""

    embed_ms: float
    dense_ms: float
    sparse_ms: float
    fusion_ms: float
    verification_ms: float
    total_ms: float


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Evidence plus an explicit groundedness decision and operational timings."""

    evidence: tuple[Evidence, ...]
    signal: Literal["ok", "insufficient_evidence"]
    timings: RetrievalTimings


@dataclass(frozen=True, slots=True)
class SignedCandidate:
    """Database material whose integrity must be established before use."""

    chunk_id: str
    tenant_id: str
    source_id: str
    content: str
    content_sha256: bytes
    signature: bytes
    signer_key_id: str
    embedding: tuple[float, ...]
    embedding_sha256: bytes
    embedding_model: str
    embedding_version: str
    access_level: str
    page: int | None
    source_uri: str
    dense_similarity: float | None


class ChunkVerifier(Protocol):
    """Integrity boundary backed by a tenant-aware public-key resolver."""

    async def verify(self, candidate: SignedCandidate) -> bool: ...


class PublicKeyResolver(Protocol):
    """Resolve tenant-scoped Ed25519 public key bytes without exposing secrets."""

    async def resolve(self, tenant_id: str, signer_key_id: str) -> bytes | None: ...


class _CanonicalDumps(Protocol):
    def __call__(self, value: Mapping[str, str]) -> bytes: ...


class Ed25519ChunkVerifier:
    """Verify chunk hashes and the canonical ingestion signature."""

    def __init__(self, keys: PublicKeyResolver) -> None:
        self._keys = keys

    async def verify(self, candidate: SignedCandidate) -> bool:
        if not chunk_material_matches(candidate):
            return False
        key_bytes = await self._keys.resolve(candidate.tenant_id, candidate.signer_key_id)
        if key_bytes is None:
            return False
        payload = {
            "chunk_id": candidate.chunk_id,
            "content_sha256": candidate.content_sha256.hex(),
            "embedding_model": candidate.embedding_model,
            "embedding_sha256": candidate.embedding_sha256.hex(),
            "embedding_version": candidate.embedding_version,
            "source_id": candidate.source_id,
            "tenant_id": candidate.tenant_id,
        }
        canonicalize = cast(_CanonicalDumps, rfc8785.dumps)
        try:
            Ed25519PublicKey.from_public_bytes(key_bytes).verify(
                candidate.signature, canonicalize(payload)
            )
        except (InvalidSignature, ValueError):
            return False
        return True


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    rrf_k: int = 60,
    boosts: Mapping[str, float] | None = None,
    max_boost: float = 0.0,
) -> dict[str, float]:
    """Fuse ranks only, with an optional independently bounded document boost."""
    if rrf_k < 1 or max_boost < 0:
        raise ValueError("rrf_k must be positive and max_boost non-negative")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(dict.fromkeys(ranking), start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (rrf_k + rank)
    for chunk_id, boost in (boosts or {}).items():
        if chunk_id in scores:
            scores[chunk_id] += min(max(boost, 0.0), max_boost)
    return scores


_DENSE_SQL = """
SELECT c.id, c.tenant_id, c.source_id, c.content, c.content_sha256, c.signature,
       c.signer_key_id, c.embedding::text, c.embedding_sha256,
       c.embedding_model, c.embedding_version,
       c.access_level, c.page, s.uri, 1 - (c.embedding <=> %s::vector) AS similarity
FROM chunks AS c
JOIN sources AS s ON s.tenant_id = c.tenant_id AND s.id = c.source_id
WHERE c.access_level = ANY(%s::text[])
  AND c.embedding_model = %s
  AND c.embedding_version = %s
ORDER BY c.embedding <=> %s::vector
LIMIT %s
"""

_SPARSE_SQL = """
SELECT c.id, c.tenant_id, c.source_id, c.content, c.content_sha256, c.signature,
       c.signer_key_id, c.embedding::text, c.embedding_sha256,
       c.embedding_model, c.embedding_version,
       c.access_level, c.page, s.uri, NULL::double precision AS similarity
FROM chunks AS c
JOIN sources AS s ON s.tenant_id = c.tenant_id AND s.id = c.source_id
WHERE c.access_level = ANY(%s::text[])
  AND c.embedding_model = %s
  AND c.embedding_version = %s
  AND c.tsv @@ websearch_to_tsquery('english', %s)
ORDER BY ts_rank_cd(c.tsv, websearch_to_tsquery('english', %s)) DESC, c.id
LIMIT %s
"""


class HybridRetriever:
    """Retrieve verified evidence with dense/sparse RRF inside one tenant transaction."""

    def __init__(
        self,
        pool: AsyncConnectionPool,
        embedder: TextEmbedder,
        verifier: ChunkVerifier,
        config: RetrievalConfig | None = None,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._pool = pool
        self._embedder = embedder
        self._verifier = verifier
        self._config = config or RetrievalConfig()
        self._monotonic = monotonic

    async def retrieve(self, query: str, tenant_ctx: TenantContext, k: int) -> RetrievalResult:
        """Return at most ``k`` verified evidence items in the server-derived tenant scope."""
        if not query.strip():
            raise ValueError("query must be non-empty")
        if k < 1:
            raise ValueError("k must be positive")

        started = self._monotonic()
        embedding, embed_ms = await self._timed(
            self._embed_query(query), self._config.embed_timeout_seconds
        )
        vector = "[" + ",".join(format(value, ".17g") for value in embedding.values) + "]"

        async with self._pool.connection() as connection, connection.transaction():
            await connection.execute(
                "SELECT set_config('app.tenant_id', %s, true)", (tenant_ctx.tenant_id,)
            )
            dense_rows, dense_ms = await self._timed(
                self._query(
                    connection,
                    _DENSE_SQL,
                    (
                        vector,
                        list(tenant_ctx.access_levels),
                        embedding.model_id,
                        embedding.model_version,
                        vector,
                        self._config.candidate_limit,
                    ),
                ),
                self._config.dense_timeout_seconds,
            )
            sparse_rows, sparse_ms = await self._timed(
                self._query(
                    connection,
                    _SPARSE_SQL,
                    (
                        list(tenant_ctx.access_levels),
                        embedding.model_id,
                        embedding.model_version,
                        query,
                        query,
                        self._config.candidate_limit,
                    ),
                ),
                self._config.sparse_timeout_seconds,
            )

        compatible_sparse = tuple(
            candidate
            for candidate in sparse_rows
            if self._compatible(candidate, tenant_ctx, embedding)
        )
        compatible_dense = tuple(
            candidate
            for candidate in dense_rows
            if self._compatible(candidate, tenant_ctx, embedding)
        )
        candidates = {candidate.chunk_id: candidate for candidate in compatible_sparse}
        candidates.update((candidate.chunk_id, candidate) for candidate in compatible_dense)
        boosts = {
            chunk_id: self._config.current_page_boost
            for chunk_id, candidate in candidates.items()
            if tenant_ctx.current_page_uri is not None
            and candidate.source_uri == tenant_ctx.current_page_uri
        }
        fused, fusion_ms = await self._timed(
            asyncio.to_thread(
                reciprocal_rank_fusion,
                (
                    tuple(candidate.chunk_id for candidate in compatible_dense),
                    tuple(candidate.chunk_id for candidate in compatible_sparse),
                ),
                rrf_k=self._config.rrf_k,
                boosts=boosts,
                max_boost=self._config.current_page_boost,
            ),
            self._config.fusion_timeout_seconds,
        )
        ordered = sorted(fused, key=lambda chunk_id: (-fused[chunk_id], chunk_id))
        eligible = [
            candidates[chunk_id]
            for chunk_id in ordered
            if self._meets_similarity_floor(candidates[chunk_id], self._config.similarity_floor)
        ]
        verified, verification_ms = await self._timed(
            self._verified(eligible, k), self._config.verification_timeout_seconds
        )
        theoretical_max = 2 / (self._config.rrf_k + 1) + self._config.current_page_boost
        evidence = tuple(
            self._to_evidence(candidate, min(fused[candidate.chunk_id] / theoretical_max, 1.0))
            for candidate in verified
        )
        total_ms = (self._monotonic() - started) * 1000
        return RetrievalResult(
            evidence=evidence,
            signal="ok" if evidence else "insufficient_evidence",
            timings=RetrievalTimings(
                embed_ms=embed_ms,
                dense_ms=dense_ms,
                sparse_ms=sparse_ms,
                fusion_ms=fusion_ms,
                verification_ms=verification_ms,
                total_ms=total_ms,
            ),
        )

    async def _embed_query(self, query: str) -> EmbeddingVector:
        embeddings = await self._embedder.embed_texts((query,))
        if len(embeddings) != 1:
            raise RuntimeError("embedder returned an unexpected result count")
        return embeddings[0]

    async def _query(
        self, connection: AsyncConnection, sql: str, params: Sequence[SqlValue]
    ) -> tuple[SignedCandidate, ...]:
        rows = await (await connection.execute(sql, params)).fetchall()
        return tuple(self._candidate(row) for row in rows)

    async def _verified(
        self, candidates: Sequence[SignedCandidate], k: int
    ) -> tuple[SignedCandidate, ...]:
        verified: list[SignedCandidate] = []
        for candidate in candidates:
            if await self._verifier.verify(candidate):
                verified.append(candidate)
                if len(verified) == k:
                    break
        return tuple(verified)

    async def _timed[T](self, operation: Awaitable[T], timeout: float) -> tuple[T, float]:
        started = self._monotonic()
        async with asyncio.timeout(timeout):
            result = await operation
        return result, (self._monotonic() - started) * 1000

    @staticmethod
    def _candidate(row: SqlRow) -> SignedCandidate:
        embedding_text = row[7].strip("[]")
        embedding = tuple(float(value) for value in embedding_text.split(","))
        return SignedCandidate(
            chunk_id=row[0],
            tenant_id=row[1],
            source_id=row[2],
            content=row[3],
            content_sha256=row[4],
            signature=row[5],
            signer_key_id=row[6],
            embedding=embedding,
            embedding_sha256=row[8],
            embedding_model=row[9],
            embedding_version=row[10],
            access_level=row[11],
            page=row[12],
            source_uri=row[13],
            dense_similarity=row[14],
        )

    @staticmethod
    def _to_evidence(candidate: SignedCandidate, score: float) -> Evidence:
        locator = f"page:{candidate.page}" if candidate.page is not None else candidate.source_uri
        return Evidence(
            citation=Citation(
                chunk_id=candidate.chunk_id,
                source_uri=candidate.source_uri,
                locator=locator,
                quote_span=QuoteSpan(start=0, end=len(candidate.content)),
            ),
            text=candidate.content,
            relevance_score=score,
        )

    @staticmethod
    def _meets_similarity_floor(candidate: SignedCandidate, floor: float) -> bool:
        similarity = candidate.dense_similarity
        return similarity is not None and similarity >= floor

    @staticmethod
    def _compatible(
        candidate: SignedCandidate,
        tenant_ctx: TenantContext,
        embedding: EmbeddingVector,
    ) -> bool:
        return (
            candidate.tenant_id == tenant_ctx.tenant_id
            and candidate.access_level in tenant_ctx.access_levels
            and candidate.embedding_model == embedding.model_id
            and candidate.embedding_version == embedding.model_version
        )


def chunk_material_matches(candidate: SignedCandidate) -> bool:
    """Check signed content and binary64 embedding digests before key verification."""
    content_digest = hashlib.sha256(candidate.content.encode("utf-8")).digest()
    embedding_bytes = b"".join(struct.pack(">d", value) for value in candidate.embedding)
    return (
        content_digest == candidate.content_sha256
        and hashlib.sha256(embedding_bytes).digest() == candidate.embedding_sha256
    )
