#!/usr/bin/env python3
"""Apply the reversible Aegon PostgreSQL migrations."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

import psycopg
from psycopg import sql

if TYPE_CHECKING:
    from collections.abc import Sequence

MIGRATIONS_DIR: Final = Path(__file__).resolve().parent
MIGRATION_PATTERN: Final = re.compile(r"^(?P<version>\d{4})_(?P<name>[a-z0-9_]+)\.up\.sql$")
DIMENSION_TOKEN: Final = "{{embedding_dimension}}"
ADVISORY_LOCK_ID: Final = 6_164_691_831_379_745_103


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    up_path: Path
    down_path: Path


def discover_migrations(directory: Path = MIGRATIONS_DIR) -> tuple[Migration, ...]:
    migrations: list[Migration] = []
    for up_path in directory.glob("*.up.sql"):
        match = MIGRATION_PATTERN.fullmatch(up_path.name)
        if match is None:
            raise ValueError(f"Invalid migration filename: {up_path.name}")
        down_path = up_path.with_name(up_path.name.replace(".up.sql", ".down.sql"))
        if not down_path.is_file():
            raise ValueError(f"Missing down migration: {down_path.name}")
        migrations.append(
            Migration(
                version=int(match.group("version")),
                name=match.group("name"),
                up_path=up_path,
                down_path=down_path,
            )
        )

    migrations.sort(key=lambda migration: migration.version)
    versions = [migration.version for migration in migrations]
    if len(versions) != len(set(versions)):
        raise ValueError("Migration versions must be unique")
    return tuple(migrations)


def render_sql(path: Path, embedding_dimension: int) -> str:
    if not 1 <= embedding_dimension <= 16_000:
        raise ValueError("embedding dimension must be between 1 and 16000")
    sql = path.read_text(encoding="utf-8")
    return sql.replace(DIMENSION_TOKEN, str(embedding_dimension))


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_history_table(connection: psycopg.Connection[tuple[object, ...]]) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS public.schema_migrations (
            version integer PRIMARY KEY,
            name text NOT NULL,
            checksum text NOT NULL,
            embedding_dimension integer NOT NULL,
            applied_at timestamptz NOT NULL DEFAULT clock_timestamp()
        )
        """
    )
    connection.commit()


def migrate(
    database_url: str,
    direction: Literal["up", "down"],
    embedding_dimension: int,
    steps: int | None = None,
    directory: Path = MIGRATIONS_DIR,
) -> None:
    migrations = discover_migrations(directory)
    by_version = {migration.version: migration for migration in migrations}

    with psycopg.connect(database_url) as connection:
        ensure_history_table(connection)
        with connection.transaction():
            connection.execute("SELECT pg_advisory_xact_lock(%s)", (ADVISORY_LOCK_ID,))
            applied_rows = connection.execute(
                "SELECT version, checksum, embedding_dimension "
                "FROM public.schema_migrations ORDER BY version"
            ).fetchall()
            applied = {int(row[0]): (str(row[1]), int(row[2])) for row in applied_rows}

            for migration in migrations:
                if migration.version in applied:
                    expected = (checksum(migration.up_path), embedding_dimension)
                    if applied[migration.version] != expected:
                        raise RuntimeError(
                            f"Applied migration {migration.version:04d} "
                            "differs from disk or dimension"
                        )

            if direction == "up":
                pending = [
                    migration for migration in migrations if migration.version not in applied
                ]
                selected = pending if steps is None else pending[:steps]
                for migration in selected:
                    connection.execute(sql.SQL(render_sql(migration.up_path, embedding_dimension)))
                    connection.execute(
                        "INSERT INTO public.schema_migrations "
                        "(version, name, checksum, embedding_dimension) VALUES (%s, %s, %s, %s)",
                        (
                            migration.version,
                            migration.name,
                            checksum(migration.up_path),
                            embedding_dimension,
                        ),
                    )
                return

            applied_migrations = [by_version[version] for version in reversed(applied)]
            selected = applied_migrations[: steps if steps is not None else 1]
            for migration in selected:
                connection.execute(sql.SQL(render_sql(migration.down_path, embedding_dimension)))
                connection.execute(
                    "DELETE FROM public.schema_migrations WHERE version = %s",
                    (migration.version,),
                )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direction", choices=("up", "down"))
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="PostgreSQL connection URL; defaults to DATABASE_URL",
    )
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=int(os.environ.get("EMBEDDING_DIMENSION", "768")),
    )
    parser.add_argument("--steps", type=int, help="number of migrations to apply or revert")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if not args.database_url:
        raise SystemExit("--database-url or DATABASE_URL is required")
    if args.steps is not None and args.steps < 1:
        raise SystemExit("--steps must be positive")
    migrate(args.database_url, args.direction, args.embedding_dimension, args.steps)


if __name__ == "__main__":
    main()
