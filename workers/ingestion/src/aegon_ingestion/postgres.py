"""PostgreSQL persistence for governed ingestion."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg import sql

from aegon_ingestion.pipeline import SourceDisposition, SourceInput
from aegon_ingestion.signing import SignedChunk, SignedManifest, manifest_json


class PostgresIngestionRepository:
    """Tenant-scoped repository with a single-transaction staging swap."""

    def __init__(self, database_url: str) -> None:
        if not database_url:
            raise ValueError("database_url must not be empty")
        self._database_url = database_url

    def source_disposition(self, tenant_id: str, source_id: str, uri: str) -> SourceDisposition:
        with self._connection(tenant_id) as connection:
            row = connection.execute(
                "SELECT uri, status FROM sources WHERE tenant_id = %s AND id = %s",
                (tenant_id, source_id),
            ).fetchone()
        if row is None or row[1] != "approved":
            return SourceDisposition.QUARANTINE_NEW
        if row[0] != uri:
            return SourceDisposition.QUARANTINE_ORIGIN_CHANGED
        return SourceDisposition.APPROVED

    def quarantine(self, tenant_id: str, source: SourceInput, reason: str) -> None:
        payload = {
            "checksum": source.checksum,
            "kind": source.kind,
            "proposed_source_id": source.source_id,
            "uri": source.uri,
        }
        with self._connection(tenant_id) as connection:
            existing = connection.execute(
                "SELECT 1 FROM sources WHERE tenant_id = %s AND id = %s",
                (tenant_id, source.source_id),
            ).fetchone()
            connection.execute(
                """
                INSERT INTO quarantine (tenant_id, source_id, reason, payload, content_sha256)
                VALUES (%s, %s, %s, %s::jsonb, decode(%s, 'hex'))
                """,
                (
                    tenant_id,
                    source.source_id if existing else None,
                    reason,
                    json.dumps(payload),
                    source.checksum,
                ),
            )

    def source_checksum(self, tenant_id: str, source_id: str) -> str | None:
        with self._connection(tenant_id) as connection:
            row = connection.execute(
                "SELECT encode(checksum, 'hex') FROM sources WHERE tenant_id = %s AND id = %s",
                (tenant_id, source_id),
            ).fetchone()
        return None if row is None else str(row[0])

    def load_chunks(self, tenant_id: str, source_id: str) -> Sequence[SignedChunk]:
        with self._connection(tenant_id) as connection:
            rows = connection.execute(
                """
                SELECT id, content, encode(content_sha256, 'hex'), embedding::text,
                       encode(embedding_sha256, 'hex'), embedding_model, embedding_version,
                       signer_key_id, signature
                FROM chunks
                WHERE tenant_id = %s AND source_id = %s
                ORDER BY id
                """,
                (tenant_id, source_id),
            ).fetchall()
        return tuple(
            SignedChunk(
                tenant_id=tenant_id,
                source_id=source_id,
                chunk_id=row[0],
                content=row[1],
                content_sha256=row[2],
                embedding=_parse_vector(row[3]),
                embedding_sha256=row[4],
                embedding_model=row[5],
                embedding_version=row[6],
                signer_key_id=row[7],
                signature=bytes(row[8]),
            )
            for row in rows
        )

    def publish(
        self,
        tenant_id: str,
        sources: Sequence[SourceInput],
        chunks: Sequence[SignedChunk],
        manifest: SignedManifest,
    ) -> None:
        if manifest.tenant_id != tenant_id or any(chunk.tenant_id != tenant_id for chunk in chunks):
            raise ValueError("publish material must belong to the selected tenant")
        source_ids = [source.source_id for source in sources]
        with self._connection(tenant_id) as connection:
            connection.execute(
                "DELETE FROM chunk_staging WHERE tenant_id = %s AND run_id = %s",
                (tenant_id, manifest.run_id),
            )
            for source in sources:
                connection.execute(
                    """
                    UPDATE sources SET checksum = decode(%s, 'hex'), kind = %s
                    WHERE tenant_id = %s AND id = %s AND uri = %s AND status = 'approved'
                    """,
                    (source.checksum, source.kind, tenant_id, source.source_id, source.uri),
                )
            for chunk in chunks:
                connection.execute(
                    """
                    INSERT INTO chunk_staging (
                        run_id, id, tenant_id, source_id, content, content_sha256,
                        embedding, embedding_sha256, embedding_model, embedding_version,
                        signer_key_id, signature
                    ) VALUES (
                        %s, %s, %s, %s, %s, decode(%s, 'hex'), %s::vector,
                        decode(%s, 'hex'), %s, %s, %s, %s
                    )
                    """,
                    (
                        manifest.run_id,
                        chunk.chunk_id,
                        tenant_id,
                        chunk.source_id,
                        chunk.content,
                        chunk.content_sha256,
                        _vector_text(chunk.embedding),
                        chunk.embedding_sha256,
                        chunk.embedding_model,
                        chunk.embedding_version,
                        chunk.signer_key_id,
                        chunk.signature,
                    ),
                )
            connection.execute("DELETE FROM chunks WHERE tenant_id = %s", (tenant_id,))
            connection.execute(
                """
                INSERT INTO chunks (
                    id, tenant_id, source_id, modality, content, content_sha256,
                    signature, embedding, embedding_model, embedding_version, access_level,
                    embedding_sha256, signer_key_id
                )
                SELECT id, tenant_id, source_id, 'text', content, content_sha256,
                       signature, embedding, embedding_model, embedding_version, 'private',
                       embedding_sha256, signer_key_id
                FROM chunk_staging WHERE tenant_id = %s AND run_id = %s
                """,
                (tenant_id, manifest.run_id),
            )
            self._mark_deleted_sources(connection, tenant_id, source_ids)
            connection.execute(
                """
                INSERT INTO ingestion_manifests (
                    run_id, tenant_id, manifest, signer_key_id, signature
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (tenant_id, run_id) DO UPDATE
                    SET manifest = EXCLUDED.manifest,
                        signer_key_id = EXCLUDED.signer_key_id,
                        signature = EXCLUDED.signature
                """,
                (
                    manifest.run_id,
                    tenant_id,
                    manifest_json(manifest),
                    manifest.signer_key_id,
                    manifest.signature,
                ),
            )
            connection.execute(
                "DELETE FROM chunk_staging WHERE tenant_id = %s AND run_id = %s",
                (tenant_id, manifest.run_id),
            )

    @staticmethod
    def _mark_deleted_sources(
        connection: psycopg.Connection[Any], tenant_id: str, source_ids: Sequence[str]
    ) -> None:
        if source_ids:
            placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in source_ids)
            query = sql.SQL(
                "UPDATE sources SET status = 'deleted' "
                "WHERE tenant_id = %s AND status = 'approved' AND id NOT IN ({})"
            ).format(placeholders)
            connection.execute(query, (tenant_id, *source_ids))
        else:
            connection.execute(
                "UPDATE sources SET status = 'deleted' "
                "WHERE tenant_id = %s AND status = 'approved'",
                (tenant_id,),
            )

    @contextmanager
    def _connection(self, tenant_id: str) -> Iterator[psycopg.Connection[Any]]:
        with psycopg.connect(self._database_url) as connection:
            connection.execute("SELECT set_config('app.tenant_id', %s, true)", (tenant_id,))
            yield connection


def _vector_text(values: Sequence[float]) -> str:
    return "[" + ",".join(repr(value) for value in values) + "]"


def _parse_vector(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.removeprefix("[").removesuffix("]").split(","))
