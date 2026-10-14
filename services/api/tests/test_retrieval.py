from __future__ import annotations

import asyncio
import hashlib
import struct
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from typing import cast

import rfc8785
from aegon_api.retrieval import (
    AsyncConnection,
    ChunkVerifier,
    Ed25519ChunkVerifier,
    HybridRetriever,
    RetrievalConfig,
    RetrievalResult,
    SignedCandidate,
    SqlRow,
    SqlValue,
    TenantContext,
    chunk_material_matches,
    reciprocal_rank_fusion,
)
from aegon_common.embeddings import EmbeddingVector
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


class _Embedder:
    async def embed_texts(self, texts: Sequence[str]) -> tuple[EmbeddingVector, ...]:
        del texts
        return (EmbeddingVector(values=(1.0, 0.0, 0.0), model_id="test", model_version="1"),)


class _Verifier:
    async def verify(self, candidate: SignedCandidate) -> bool:
        return chunk_material_matches(candidate) and candidate.signature != b"tampered"


class _Result:
    def __init__(self, rows: list[SqlRow]) -> None:
        self._rows = rows

    async def fetchall(self) -> list[SqlRow]:
        return self._rows


class _Transaction(AbstractAsyncContextManager[object]):
    async def __aenter__(self) -> object:
        return self

    async def __aexit__(self, *args: object) -> None:
        del args


class _Connection:
    def __init__(self, dense: list[SqlRow], sparse: list[SqlRow]) -> None:
        self.dense = dense
        self.sparse = sparse
        self.calls: list[tuple[str, Sequence[SqlValue] | None]] = []

    def transaction(self) -> AbstractAsyncContextManager[object]:
        return _Transaction()

    async def execute(self, query: str, params: Sequence[SqlValue] | None = None) -> _Result:
        self.calls.append((query, params))
        if "ts_rank_cd" in query:
            return _Result(self.sparse)
        if "embedding <=>" in query:
            return _Result(self.dense)
        return _Result([])


class _PoolContext(AbstractAsyncContextManager[AsyncConnection]):
    def __init__(self, connection: _Connection) -> None:
        self._connection = connection

    async def __aenter__(self) -> AsyncConnection:
        return self._connection

    async def __aexit__(self, *args: object) -> None:
        del args


class _Pool:
    def __init__(self, connection: _Connection) -> None:
        self.connection_value = connection

    def connection(self) -> AbstractAsyncContextManager[AsyncConnection]:
        return _PoolContext(self.connection_value)


class _Keys:
    def __init__(self, public_key: bytes) -> None:
        self._public_key = public_key

    async def resolve(self, tenant_id: str, signer_key_id: str) -> bytes | None:
        if tenant_id == "tenant-a" and signer_key_id == "key-1":
            return self._public_key
        return None


def _embedding_digest(values: tuple[float, ...]) -> bytes:
    material = b"".join(struct.pack(">d", value) for value in values)
    return hashlib.sha256(material).digest()


def _row(
    chunk_id: str,
    *,
    tenant_id: str = "tenant-a",
    content: str = "Aegon keeps approved evidence isolated by tenant.",
    similarity: float | None = 0.9,
    signature: bytes = b"valid",
    access_level: str = "private",
    source_uri: str = "https://a.test/docs",
) -> SqlRow:
    embedding = (1.0, 0.0, 0.0)
    return (
        chunk_id,
        tenant_id,
        "source-1",
        content,
        hashlib.sha256(content.encode()).digest(),
        signature,
        "key-1",
        "[1,0,0]",
        _embedding_digest(embedding),
        "test",
        "1",
        access_level,
        3,
        source_uri,
        similarity,
    )


def _retrieve(
    dense: list[SqlRow],
    sparse: list[SqlRow],
    *,
    floor: float = 0.25,
    verifier: ChunkVerifier | None = None,
) -> tuple[RetrievalResult, _Connection]:
    connection = _Connection(dense, sparse)
    retriever = HybridRetriever(
        _Pool(connection),
        _Embedder(),
        verifier if verifier is not None else _Verifier(),
        RetrievalConfig(similarity_floor=floor),
    )
    result = asyncio.run(
        retriever.retrieve(
            "How is evidence isolated?",
            TenantContext("tenant-a", ("private",), "https://a.test/docs"),
            2,
        )
    )
    return result, connection


def test_rrf_uses_hand_computed_ranks_and_bounds_boost() -> None:
    scores = reciprocal_rank_fusion(
        (("a", "b"), ("b", "c")),
        rrf_k=60,
        boosts={"a": 99.0},
        max_boost=0.005,
    )

    assert scores["a"] == 1 / 61 + 0.005
    assert scores["b"] == 1 / 62 + 1 / 61
    assert scores["c"] == 1 / 62


def test_known_answer_retrieval_is_tenant_scoped_and_access_filtered() -> None:
    result_value, connection = _retrieve(
        [_row("answer", similarity=0.95), _row("distractor", similarity=0.5)],
        [_row("answer", similarity=None)],
    )
    result = result_value

    assert result.signal == "ok"
    assert [item.citation.chunk_id for item in result.evidence] == ["answer", "distractor"]
    assert result.timings.total_ms >= 0
    assert connection.calls[0][1] == ("tenant-a",)
    assert connection.calls[1][1] is not None
    assert connection.calls[1][1][1] == ["private"]


def test_cross_tenant_and_disallowed_access_candidates_never_leak() -> None:
    result_value, _ = _retrieve(
        [_row("other-tenant", tenant_id="tenant-b"), _row("public", access_level="public")],
        [],
    )
    result = result_value

    assert result.signal == "insufficient_evidence"
    assert result.evidence == ()


def test_similarity_floor_returns_explicit_refusal_signal() -> None:
    result_value, _ = _retrieve([_row("weak", similarity=0.19)], [], floor=0.2)
    result = result_value

    assert result.signal == "insufficient_evidence"
    assert result.evidence == ()


def test_tampered_signature_is_rejected() -> None:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    unsigned = _row("tampered", signature=b"")
    payload = {
        "chunk_id": unsigned[0],
        "content_sha256": unsigned[4].hex(),
        "embedding_model": unsigned[9],
        "embedding_sha256": unsigned[8].hex(),
        "embedding_version": unsigned[10],
        "source_id": unsigned[2],
        "tenant_id": unsigned[1],
    }
    valid_signature = private_key.sign(rfc8785.dumps(payload))
    valid = list(unsigned)
    valid[5] = valid_signature
    tampered = cast(SqlRow, tuple(valid[:5] + [b"tampered" * 8] + valid[6:]))
    verifier = Ed25519ChunkVerifier(_Keys(public_key))

    valid_result_value, _ = _retrieve([cast(SqlRow, tuple(valid))], [], verifier=verifier)
    tampered_result_value, _ = _retrieve([tampered], [], verifier=verifier)
    valid_result = valid_result_value
    tampered_result = tampered_result_value

    assert valid_result.signal == "ok"
    assert tampered_result.signal == "insufficient_evidence"
