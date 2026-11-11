"""Opaque session issuance, verification, and persistence."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from aegon_api.settings import TenantKeyGrant

TOKEN_BYTES = 32


def sha256(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


@dataclass(frozen=True, slots=True)
class SessionRecord:
    token_hash: bytes
    csrf_hash: bytes
    tenant_id: str
    scopes: tuple[str, ...]
    expires_at: datetime


class SessionStore(Protocol):
    async def create(self, session: SessionRecord) -> None: ...

    async def get(self, token_hash: bytes) -> SessionRecord | None: ...

    async def revoke(self, token_hash: bytes) -> None: ...

    async def ready(self) -> bool: ...


class InMemorySessionStore:
    """Deterministic test store that retains hashes, never raw credentials."""

    def __init__(self) -> None:
        self._sessions: list[SessionRecord] = []
        self.available = True

    @property
    def records(self) -> tuple[SessionRecord, ...]:
        return tuple(self._sessions)

    async def create(self, session: SessionRecord) -> None:
        self._sessions.append(session)

    async def get(self, token_hash: bytes) -> SessionRecord | None:
        for session in self._sessions:
            if hmac.compare_digest(session.token_hash, token_hash):
                return session
        return None

    async def revoke(self, token_hash: bytes) -> None:
        self._sessions = [
            session
            for session in self._sessions
            if not hmac.compare_digest(session.token_hash, token_hash)
        ]

    async def ready(self) -> bool:
        return self.available


type SessionRow = tuple[bytes, str, list[str], datetime]


class QueryResult(Protocol):
    async def fetchone(self) -> SessionRow | None: ...


class DatabaseConnection(Protocol):
    async def execute(self, query: str, params: Sequence[object] | None = None) -> QueryResult: ...


class DatabasePool(Protocol):
    def connection(self) -> AbstractAsyncContextManager[DatabaseConnection]: ...


class PostgresSessionStore:
    """PostgreSQL store using the migration-defined, privilege-contained functions."""

    def __init__(self, pool: DatabasePool) -> None:
        self._pool = pool

    async def create(self, session: SessionRecord) -> None:
        async with self._pool.connection() as connection:
            await connection.execute(
                "SELECT api_session_create(%s, %s, %s, %s, %s)",
                (
                    session.token_hash,
                    session.csrf_hash,
                    session.tenant_id,
                    list(session.scopes),
                    session.expires_at,
                ),
            )

    async def get(self, token_hash: bytes) -> SessionRecord | None:
        async with self._pool.connection() as connection:
            result = await connection.execute("SELECT * FROM api_session_get(%s)", (token_hash,))
            row = await result.fetchone()
        if row is None:
            return None
        return SessionRecord(
            token_hash=token_hash,
            csrf_hash=row[0],
            tenant_id=row[1],
            scopes=tuple(row[2]),
            expires_at=row[3],
        )

    async def revoke(self, token_hash: bytes) -> None:
        async with self._pool.connection() as connection:
            await connection.execute("SELECT api_session_revoke(%s)", (token_hash,))

    async def ready(self) -> bool:
        async with self._pool.connection() as connection:
            result = await connection.execute("SELECT 1")
            return await result.fetchone() is not None


class SessionService:
    def __init__(
        self,
        store: SessionStore,
        tenant_keys: tuple[TenantKeyGrant, ...],
        ttl_seconds: int,
    ) -> None:
        self._store = store
        self._tenant_keys = tenant_keys
        self._ttl = timedelta(seconds=ttl_seconds)

    def resolve_tenant_key(self, raw_key: str) -> TenantKeyGrant | None:
        supplied_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        matched: TenantKeyGrant | None = None
        for grant in self._tenant_keys:
            if hmac.compare_digest(grant.key_sha256, supplied_hash):
                matched = grant
        return matched

    async def issue(self, grant: TenantKeyGrant) -> tuple[str, str, SessionRecord]:
        token = secrets.token_urlsafe(TOKEN_BYTES)
        csrf_token = secrets.token_urlsafe(TOKEN_BYTES)
        record = SessionRecord(
            token_hash=sha256(token),
            csrf_hash=sha256(csrf_token),
            tenant_id=grant.tenant_id,
            scopes=grant.scopes,
            expires_at=datetime.now(UTC) + self._ttl,
        )
        await self._store.create(record)
        return token, csrf_token, record

    async def authenticate(self, token: str) -> SessionRecord | None:
        token_hash = sha256(token)
        session = await self._store.get(token_hash)
        if session is None or not hmac.compare_digest(session.token_hash, token_hash):
            return None
        if session.expires_at <= datetime.now(UTC):
            await self._store.revoke(token_hash)
            return None
        return session

    async def revoke(self, token: str) -> None:
        await self._store.revoke(sha256(token))

    @staticmethod
    def csrf_matches(session: SessionRecord, cookie_token: str, header_token: str) -> bool:
        return hmac.compare_digest(cookie_token, header_token) and hmac.compare_digest(
            session.csrf_hash, sha256(header_token)
        )
