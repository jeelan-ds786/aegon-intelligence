from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import psycopg
import pytest
from aegon_common.embeddings import EmbeddingVector
from aegon_ingestion import (
    DocumentBlock,
    LocalSigner,
    ParsedDocument,
    SourceDisposition,
    SourceInput,
)
from aegon_ingestion.postgres import PostgresIngestionRepository
from aegon_ingestion.signing import SignedChunk, SignedManifest, sign_chunk, sign_manifest

from infra.migrations.migrate import discover_migrations, migrate, parse_args, render_sql

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest_docker.plugin import Services

TEST_DIR = Path(__file__).parent
MIGRATIONS_DIR = TEST_DIR.parent
SNAPSHOT_PATH = TEST_DIR / "snapshots" / "hnsw_explain.txt"


@pytest.fixture(scope="session")
def docker_compose_file() -> Path:
    return TEST_DIR / "docker-compose.yml"


@pytest.fixture(scope="session")
def database_url(docker_services: Services) -> str:
    port = docker_services.port_for("postgres", 5432)
    url = f"postgresql://postgres:postgres@127.0.0.1:{port}/aegon_test"

    def responsive() -> bool:
        try:
            with psycopg.connect(url):
                return True
        except Exception:
            return False

    docker_services.wait_until_responsive(timeout=30, pause=0.2, check=responsive)
    return url


@pytest.fixture
def migrated_database(database_url: str) -> Iterator[str]:
    migrate(database_url, "up", embedding_dimension=3)
    yield database_url
    migrate(database_url, "down", embedding_dimension=3, steps=3)


def test_migrations_apply_down_and_reapply(database_url: str) -> None:
    migrate(database_url, "up", embedding_dimension=3)
    with psycopg.connect(database_url) as connection:
        versions = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert versions == [(1,), (2,), (3,)]
        assert connection.execute("SELECT to_regclass('public.chunks')").fetchone() == ("chunks",)

    migrate(database_url, "down", embedding_dimension=3, steps=3)
    with psycopg.connect(database_url) as connection:
        assert connection.execute("SELECT count(*) FROM schema_migrations").fetchone() == (0,)
        assert connection.execute("SELECT to_regclass('public.chunks')").fetchone() == (None,)

    migrate(database_url, "up", embedding_dimension=3)
    with psycopg.connect(database_url) as connection:
        assert connection.execute("SELECT count(*) FROM schema_migrations").fetchone() == (3,)
    migrate(database_url, "down", embedding_dimension=3, steps=3)


def test_app_role_cannot_bypass_tenant_rls(migrated_database: str) -> None:
    with psycopg.connect(migrated_database) as connection:
        connection.execute(
            "INSERT INTO tenants (id, name) VALUES ('tenant-a', 'A'), ('tenant-b', 'B')"
        )
        connection.execute(
            """
            INSERT INTO sources (id, tenant_id, uri, kind, checksum, status)
            VALUES
                ('source-a', 'tenant-a', 'https://a.test', 'html',
                 decode(repeat('aa', 32), 'hex'), 'approved'),
                ('source-b', 'tenant-b', 'https://b.test', 'html',
                 decode(repeat('bb', 32), 'hex'), 'approved')
            """
        )
        connection.execute(
            """
            INSERT INTO chunks (
                id, tenant_id, source_id, modality, content, content_sha256,
                signature, embedding, embedding_model, embedding_version, access_level,
                embedding_sha256, signer_key_id
            ) VALUES
                ('chunk-a', 'tenant-a', 'source-a', 'text', 'A', decode(repeat('aa', 32), 'hex'),
                 decode('01', 'hex'), '[1,0,0]', 'test', '1', 'private',
                 decode(repeat('01', 32), 'hex'), 'test-key'),
                ('chunk-b', 'tenant-b', 'source-b', 'text', 'B', decode(repeat('bb', 32), 'hex'),
                 decode('02', 'hex'), '[0,1,0]', 'test', '1', 'private',
                 decode(repeat('02', 32), 'hex'), 'test-key')
            """
        )
        connection.execute("SET ROLE aegon_app")
        connection.execute("SELECT set_config('app.tenant_id', 'tenant-a', false)")

        rows = connection.execute(
            "SELECT id, tenant_id FROM chunks WHERE tenant_id = 'tenant-b' OR true ORDER BY id"
        ).fetchall()
        role = connection.execute(
            "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
        ).fetchone()

        assert rows == [("chunk-a", "tenant-a")]
        assert role == (False, False)


def test_vector_query_plan_uses_hnsw_index(migrated_database: str) -> None:
    with psycopg.connect(migrated_database) as connection:
        connection.execute("INSERT INTO tenants (id, name) VALUES ('tenant-a', 'A')")
        connection.execute(
            """
            INSERT INTO sources (id, tenant_id, uri, kind, checksum, status)
            VALUES ('source-a', 'tenant-a', 'https://a.test', 'html',
                    decode(repeat('aa', 32), 'hex'), 'approved')
            """
        )
        connection.execute(
            """
            INSERT INTO chunks (
                id, tenant_id, source_id, modality, content, content_sha256,
                signature, embedding, embedding_model, embedding_version, access_level,
                embedding_sha256, signer_key_id
            ) VALUES ('chunk-a', 'tenant-a', 'source-a', 'text', 'A',
                      decode(repeat('aa', 32), 'hex'), decode('01', 'hex'),
                      '[1,0,0]', 'test', '1', 'private',
                      decode(repeat('01', 32), 'hex'), 'test-key')
            """
        )
        connection.execute("SET ROLE aegon_app")
        connection.execute("SELECT set_config('app.tenant_id', 'tenant-a', false)")
        connection.execute("SET LOCAL enable_seqscan = off")
        connection.execute("SET LOCAL enable_sort = off")
        plan_rows = connection.execute(
            """
            EXPLAIN (COSTS OFF)
            SELECT id FROM chunks
            ORDER BY embedding <=> '[1,0,0]'::vector
            LIMIT 10
            """
        ).fetchall()

    plan = "\n".join(str(row[0]) for row in plan_rows)
    normalized = re.sub(r"\s+", " ", plan).strip()
    expected = SNAPSHOT_PATH.read_text(encoding="utf-8").strip()
    assert "chunks_embedding_hnsw_idx" in normalized
    assert normalized == expected


def test_failed_staging_swap_preserves_live_index(migrated_database: str) -> None:
    signer = LocalSigner.from_private_bytes("local-1", b"1" * 32)
    repository = PostgresIngestionRepository(migrated_database)
    with psycopg.connect(migrated_database) as connection:
        connection.execute("INSERT INTO tenants (id, name) VALUES ('tenant-a', 'A')")
        connection.execute(
            """
            INSERT INTO sources (id, tenant_id, uri, kind, checksum, status)
            VALUES ('source-a', 'tenant-a', 'https://a.test', 'markdown',
                    decode(repeat('00', 32), 'hex'), 'approved')
            """
        )

    def material(text: str, run_id: str) -> tuple[SourceInput, SignedChunk, SignedManifest]:
        document = ParsedDocument(
            source_id="source-a",
            media_type="text/markdown",
            blocks=(DocumentBlock(kind="paragraph", text=text, locator="line:1"),),
        )
        source = SourceInput.from_bytes(
            source_id="source-a",
            uri="https://a.test",
            kind="markdown",
            content=text.encode(),
            document=document,
        )
        chunk = sign_chunk(
            tenant_id="tenant-a",
            source_id="source-a",
            chunk_id=f"chunk-{run_id}",
            content=text,
            embedding=EmbeddingVector(values=(1.0, 0.0, 0.0), model_id="test", model_version="1"),
            signer=signer,
        )
        manifest = sign_manifest(
            run_id, "tenant-a", {source.source_id: source.checksum}, [chunk.chunk_id], signer
        )
        return source, chunk, manifest

    source, chunk, manifest = material("first version", "run-1")
    repository.publish("tenant-a", [source], [chunk], manifest)
    with psycopg.connect(migrated_database) as connection:
        connection.execute(
            """
            CREATE FUNCTION reject_second_version() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN
                IF NEW.content = 'second version' THEN
                    RAISE EXCEPTION 'injected swap failure';
                END IF;
                RETURN NEW;
            END;
            $$
            """
        )
        connection.execute(
            """
            CREATE TRIGGER reject_second_version
            BEFORE INSERT ON chunks FOR EACH ROW EXECUTE FUNCTION reject_second_version()
            """
        )

    source, chunk, manifest = material("second version", "run-2")
    with pytest.raises(psycopg.errors.RaiseException, match="injected swap failure"):
        repository.publish("tenant-a", [source], [chunk], manifest)

    with psycopg.connect(migrated_database) as connection:
        assert connection.execute(
            "SELECT id, content FROM chunks WHERE tenant_id = 'tenant-a'"
        ).fetchall() == [("chunk-run-1", "first version")]
        assert connection.execute(
            "SELECT run_id FROM ingestion_manifests WHERE tenant_id = 'tenant-a' ORDER BY run_id"
        ).fetchall() == [("run-1",)]


def test_postgres_repository_governance_and_chunk_round_trip(migrated_database: str) -> None:
    repository = PostgresIngestionRepository(migrated_database)
    signer = LocalSigner.from_private_bytes("local-1", b"1" * 32)
    with psycopg.connect(migrated_database) as connection:
        connection.execute("INSERT INTO tenants (id, name) VALUES ('tenant-a', 'A')")
        connection.execute(
            """
            INSERT INTO sources (id, tenant_id, uri, kind, checksum, status)
            VALUES ('source-a', 'tenant-a', 'https://a.test', 'markdown',
                    decode(repeat('00', 32), 'hex'), 'approved')
            """
        )

    def source(source_id: str, uri: str) -> SourceInput:
        document = ParsedDocument(
            source_id=source_id,
            media_type="text/markdown",
            blocks=(DocumentBlock(kind="paragraph", text="evidence", locator="line:1"),),
        )
        return SourceInput.from_bytes(
            source_id=source_id,
            uri=uri,
            kind="markdown",
            content=b"evidence",
            document=document,
        )

    approved = source("source-a", "https://a.test")
    unknown = source("source-new", "https://new.test")
    assert (
        repository.source_disposition("tenant-a", "source-a", "https://a.test")
        is SourceDisposition.APPROVED
    )
    assert (
        repository.source_disposition("tenant-a", "source-a", "https://changed.test")
        is SourceDisposition.QUARANTINE_ORIGIN_CHANGED
    )
    assert (
        repository.source_disposition("tenant-a", "source-new", "https://new.test")
        is SourceDisposition.QUARANTINE_NEW
    )
    assert repository.source_checksum("tenant-a", "source-a") == "00" * 32
    assert repository.source_checksum("tenant-a", "missing") is None

    repository.quarantine("tenant-a", approved, "quarantine_origin_changed")
    repository.quarantine("tenant-a", unknown, "quarantine_new")
    with psycopg.connect(migrated_database) as connection:
        assert connection.execute(
            "SELECT source_id, reason FROM quarantine ORDER BY id"
        ).fetchall() == [
            ("source-a", "quarantine_origin_changed"),
            (None, "quarantine_new"),
        ]

    chunk = sign_chunk(
        tenant_id="tenant-a",
        source_id="source-a",
        chunk_id="chunk-a",
        content="evidence",
        embedding=EmbeddingVector(values=(1.0, 0.0, 0.0), model_id="test", model_version="1"),
        signer=signer,
    )
    manifest = sign_manifest(
        "run-1", "tenant-a", {approved.source_id: approved.checksum}, [chunk.chunk_id], signer
    )
    repository.publish("tenant-a", [approved], [chunk], manifest)
    assert repository.load_chunks("tenant-a", "source-a") == (chunk,)


def test_runner_rejects_unsafe_embedding_dimension() -> None:
    migration = discover_migrations(MIGRATIONS_DIR)[1]
    with pytest.raises(ValueError, match="embedding dimension"):
        render_sql(migration.up_path, 0)


def test_runner_rejects_invalid_and_incomplete_migration_sets(tmp_path: Path) -> None:
    (tmp_path / "invalid.up.sql").write_text("SELECT 1;", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid migration filename"):
        discover_migrations(tmp_path)

    (tmp_path / "invalid.up.sql").unlink()
    (tmp_path / "0001_incomplete.up.sql").write_text("SELECT 1;", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing down migration"):
        discover_migrations(tmp_path)


def test_runner_rejects_duplicate_versions(tmp_path: Path) -> None:
    for name in ("0001_first", "0001_second"):
        (tmp_path / f"{name}.up.sql").write_text("SELECT 1;", encoding="utf-8")
        (tmp_path / f"{name}.down.sql").write_text("SELECT 1;", encoding="utf-8")

    with pytest.raises(ValueError, match="versions must be unique"):
        discover_migrations(tmp_path)


def test_runner_rejects_applied_checksum_drift(database_url: str, tmp_path: Path) -> None:
    up_path = tmp_path / "0099_probe.up.sql"
    down_path = tmp_path / "0099_probe.down.sql"
    original_sql = "CREATE TABLE migration_probe (id integer);"
    up_path.write_text(original_sql, encoding="utf-8")
    down_path.write_text("DROP TABLE migration_probe;", encoding="utf-8")
    migrate(database_url, "up", embedding_dimension=3, directory=tmp_path)

    up_path.write_text("CREATE TABLE migration_probe (id bigint);", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from disk"):
        migrate(database_url, "up", embedding_dimension=3, directory=tmp_path)

    up_path.write_text(original_sql, encoding="utf-8")
    migrate(database_url, "down", embedding_dimension=3, directory=tmp_path)


def test_cli_arguments_are_parsed() -> None:
    args = parse_args(
        [
            "up",
            "--database-url",
            "postgresql://example.test/db",
            "--embedding-dimension",
            "1536",
            "--steps",
            "2",
        ]
    )
    assert args.direction == "up"
    assert args.database_url == "postgresql://example.test/db"
    assert args.embedding_dimension == 1536
    assert args.steps == 2
