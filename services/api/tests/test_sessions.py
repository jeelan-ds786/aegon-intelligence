from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType

import pytest
from aegon_api.app import create_app
from aegon_api.main import run
from aegon_api.sessions import (
    InMemorySessionStore,
    PostgresSessionStore,
    SessionRecord,
    SessionRow,
    sha256,
)
from aegon_api.settings import ApiSettings, Environment, TenantKeyGrant
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

TENANT_KEY = "test-tenant-key"
ORIGIN = "https://host.example"
SNAPSHOT = Path(__file__).parent / "snapshots" / "openapi.json"


def _settings(**overrides: object) -> ApiSettings:
    values: dict[str, object] = {
        "environment": Environment.TEST,
        "allowed_origins": (ORIGIN,),
        "tenant_keys": (
            TenantKeyGrant(
                tenant_id="tenant-a",
                key_sha256=hashlib.sha256(TENANT_KEY.encode()).hexdigest(),
                scopes=("chat:read",),
            ),
        ),
        "max_request_bytes": 1_024,
    }
    values.update(overrides)
    return ApiSettings(**values)  # pyright: ignore[reportArgumentType]


def _client(
    *,
    settings: ApiSettings | None = None,
    store: InMemorySessionStore | None = None,
) -> tuple[TestClient, InMemorySessionStore]:
    resolved_store = store or InMemorySessionStore()
    return (
        TestClient(
            create_app(settings or _settings(), resolved_store),
            base_url="https://api.example",
        ),
        resolved_store,
    )


def _exchange(client: TestClient) -> tuple[str, str]:
    response = client.post(
        "/v1/sessions",
        headers={
            "Authorization": f"AegonKey {TENANT_KEY}",
            "Origin": ORIGIN,
        },
    )
    assert response.status_code == 201
    session_token = client.cookies.get("__Host-aegon_session")
    csrf_token = response.json()["csrf_token"]
    assert session_token is not None
    return session_token, csrf_token


def test_session_cookies_enforce_browser_security_attributes() -> None:
    client, _ = _client()

    response = client.post(
        "/v1/sessions",
        headers={"Authorization": f"AegonKey {TENANT_KEY}", "Origin": ORIGIN},
    )
    cookies = response.headers.get_list("set-cookie")

    session_cookie = next(value for value in cookies if value.startswith("__Host-aegon_session="))
    csrf_cookie = next(value for value in cookies if value.startswith("__Host-aegon_csrf="))
    assert "HttpOnly" in session_cookie
    assert "Secure" in session_cookie
    assert "SameSite=strict" in session_cookie
    assert "Path=/" in session_cookie
    assert "Domain=" not in session_cookie
    assert "HttpOnly" not in csrf_cookie
    assert "Secure" in csrf_cookie
    assert "SameSite=strict" in csrf_cookie
    assert "Path=/" in csrf_cookie
    assert "Domain=" not in csrf_cookie


def test_health_readiness_and_request_id_propagation() -> None:
    client, store = _client()

    health = client.get("/health", headers={"X-Request-ID": "request-123"})
    ready = client.get("/ready")
    store.available = False
    unavailable = client.get("/ready")

    assert health.json() == {"status": "ok"}
    assert health.headers["x-request-id"] == "request-123"
    assert ready.json() == {"status": "ready"}
    assert unavailable.status_code == 503
    assert unavailable.json()["error"]["code"] == "dependency_unavailable"
    assert unavailable.json()["error"]["request_id"] == unavailable.headers["x-request-id"]


def test_session_exchange_stores_only_hashes_and_ignores_client_scope_claims() -> None:
    client, store = _client()

    token, csrf_token = _exchange(client)
    identity = client.get(
        "/v1/sessions/current",
        headers={"X-Aegon-Scopes": "admin:*", "X-Tenant-ID": "tenant-b"},
    )

    assert identity.status_code == 200
    assert identity.json()["tenant_id"] == "tenant-a"
    assert identity.json()["scopes"] == ["chat:read"]
    assert all(record.token_hash != token.encode() for record in store.records)
    assert all(record.csrf_hash != csrf_token.encode() for record in store.records)
    assert store.records[0].token_hash == sha256(token)
    assert store.records[0].csrf_hash == sha256(csrf_token)


def test_forged_and_expired_tokens_are_rejected() -> None:
    client, store = _client()
    client.cookies.set("__Host-aegon_session", "forged")
    forged = client.get("/v1/sessions/current")

    expired_token = "expired-token"
    asyncio.run(
        store.create(
            SessionRecord(
                token_hash=sha256(expired_token),
                csrf_hash=sha256("csrf"),
                tenant_id="tenant-a",
                scopes=("chat:read",),
                expires_at=datetime.now(UTC) - timedelta(seconds=1),
            )
        )
    )

    client.cookies.set("__Host-aegon_session", expired_token)
    expired = client.get("/v1/sessions/current")

    assert forged.status_code == 401
    assert expired.status_code == 401
    assert all(record.token_hash != sha256(expired_token) for record in store.records)


def test_missing_csrf_and_bad_origin_are_rejected() -> None:
    client, _ = _client()
    _, csrf_token = _exchange(client)

    missing = client.delete(
        "/v1/sessions/current",
        headers={"Origin": ORIGIN},
    )
    bad_origin = client.delete(
        "/v1/sessions/current",
        headers={"Origin": "https://attacker.example", "X-CSRF-Token": csrf_token},
    )
    missing_origin = client.delete(
        "/v1/sessions/current",
        headers={"X-CSRF-Token": csrf_token},
    )

    assert missing.status_code == 403
    assert missing.json()["error"]["message"] == "CSRF validation failed."
    assert bad_origin.status_code == 403
    assert bad_origin.json()["error"]["message"] == "Origin is not allowed."
    assert missing_origin.status_code == 403
    assert missing_origin.json()["error"]["message"] == "A trusted Origin is required."


def test_revoked_token_cannot_be_replayed() -> None:
    client, _ = _client()
    token, csrf_token = _exchange(client)

    revoked = client.delete(
        "/v1/sessions/current",
        headers={"Origin": ORIGIN, "X-CSRF-Token": csrf_token},
    )
    client.cookies.set("__Host-aegon_session", token)
    replay = client.get("/v1/sessions/current")

    assert revoked.status_code == 204
    assert replay.status_code == 401


def test_oversized_body_is_rejected_before_route_processing() -> None:
    client, store = _client()

    response = client.post(
        "/v1/sessions",
        content=b"x" * 1_025,
        headers={"Authorization": f"AegonKey {TENANT_KEY}", "Origin": ORIGIN},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "invalid_arguments"
    assert store.records == ()


def test_chunked_oversized_body_is_rejected_without_content_length() -> None:
    client, store = _client()

    def chunks() -> Sequence[bytes]:
        return (b"x" * 600, b"y" * 600)

    response = client.post(
        "/v1/sessions",
        content=iter(chunks()),
        headers={
            "Authorization": f"AegonKey {TENANT_KEY}",
            "Origin": ORIGIN,
            "Transfer-Encoding": "chunked",
        },
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "invalid_arguments"
    assert store.records == ()


def test_production_refuses_demo_authentication() -> None:
    with pytest.raises(ValidationError, match="cannot be enabled in production"):
        ApiSettings(
            environment=Environment.PRODUCTION,
            database_url=SecretStr("postgresql://database.example/aegon"),
            allowed_origins=(ORIGIN,),
            tenant_keys=_settings().tenant_keys,
            enable_demo_bearer_token=True,
            demo_bearer_token=SecretStr("not-used"),
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"environment": Environment.PRODUCTION}, "database_url is required"),
        (
            {
                "environment": Environment.PRODUCTION,
                "database_url": "postgresql://database.example/aegon",
                "allowed_origins": ("http://host.example",),
            },
            "production origins must use https",
        ),
        (
            {"enable_demo_bearer_token": True},
            "demo_bearer_token is required",
        ),
        (
            {"tenant_keys": (_settings().tenant_keys[0], _settings().tenant_keys[0])},
            "tenant key hashes must be unique",
        ),
        (
            {"allowed_origins": ("https://host.example/path",)},
            "canonical HTTP origins",
        ),
        (
            {"allowed_origins": ("https://host.example:99999",)},
            "canonical HTTP origins",
        ),
        (
            {"allowed_origins": (ORIGIN, ORIGIN)},
            "allowed origins must be unique",
        ),
        (
            {"session_cookie_name": "aegon_session"},
            "__Host- prefix",
        ),
        (
            {"csrf_cookie_name": "__Host-aegon_session"},
            "cookie names must differ",
        ),
    ],
)
def test_settings_fail_fast(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _settings(**overrides)


def test_invalid_authentication_and_request_ids_are_safe() -> None:
    client, _ = _client()

    missing = client.post("/v1/sessions", headers={"Origin": ORIGIN})
    invalid = client.post(
        "/v1/sessions",
        headers={"Origin": ORIGIN, "Authorization": "AegonKey wrong"},
    )
    generated_id = client.get("/health", headers={"X-Request-ID": "contains spaces"})

    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert generated_id.headers["x-request-id"] != "contains spaces"


def test_framework_errors_use_the_public_envelope() -> None:
    client, _ = _client()

    not_found = client.get("/missing")
    method_not_allowed = client.put("/health")

    assert not_found.status_code == 404
    assert not_found.json()["error"]["code"] == "not_found"
    assert method_not_allowed.status_code == 405
    assert method_not_allowed.json()["error"]["code"] == "invalid_arguments"


def test_tenant_scopes_are_validated() -> None:
    with pytest.raises(ValidationError, match="tenant scopes must be unique"):
        TenantKeyGrant(
            tenant_id="tenant-a",
            key_sha256="0" * 64,
            scopes=("chat:read", "chat:read"),
        )
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        TenantKeyGrant(
            tenant_id="tenant-a",
            key_sha256="0" * 64,
            scopes=("INVALID SCOPE",),
        )


class _Result:
    def __init__(self, row: SessionRow | None) -> None:
        self._row = row

    async def fetchone(self) -> SessionRow | None:
        return self._row


class _Connection:
    def __init__(self, row: SessionRow) -> None:
        self.row = row
        self.queries: list[str] = []

    async def execute(self, query: str, params: Sequence[object] | None = None) -> _Result:
        del params
        self.queries.append(query)
        if "api_session_get" in query:
            return _Result(self.row)
        if query == "SELECT 1":
            return _Result(self.row)
        return _Result(None)


class _ConnectionContext(AbstractAsyncContextManager[_Connection]):
    def __init__(self, connection: _Connection) -> None:
        self._connection = connection

    async def __aenter__(self) -> _Connection:
        return self._connection

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback


class _Pool:
    def __init__(self, connection: _Connection) -> None:
        self.connection_value = connection

    def connection(self) -> AbstractAsyncContextManager[_Connection]:
        return _ConnectionContext(self.connection_value)


def test_postgres_store_uses_privilege_contained_functions() -> None:
    expires_at = datetime.now(UTC) + timedelta(minutes=30)
    connection = _Connection((sha256("csrf"), "tenant-a", ["chat:read"], expires_at))
    store = PostgresSessionStore(_Pool(connection))
    record = SessionRecord(
        token_hash=sha256("token"),
        csrf_hash=sha256("csrf"),
        tenant_id="tenant-a",
        scopes=("chat:read",),
        expires_at=expires_at,
    )

    async def exercise_store() -> tuple[SessionRecord | None, bool]:
        await store.create(record)
        loaded_record = await store.get(record.token_hash)
        await store.revoke(record.token_hash)
        return loaded_record, await store.ready()

    loaded, ready = asyncio.run(exercise_store())

    assert loaded == record
    assert ready is True
    assert connection.queries == [
        "SELECT api_session_create(%s, %s, %s, %s, %s)",
        "SELECT * FROM api_session_get(%s)",
        "SELECT api_session_revoke(%s)",
        "SELECT 1",
    ]


def test_production_factory_builds_postgres_store() -> None:
    settings = ApiSettings(
        environment=Environment.PRODUCTION,
        database_url=SecretStr("postgresql://database.example/aegon"),
        allowed_origins=(ORIGIN,),
        tenant_keys=_settings().tenant_keys,
    )

    app = create_app(settings)

    assert app.title == "Aegon-RAG API"


def test_production_launcher_uses_cloud_run_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def capture_run(app: str, **kwargs: object) -> None:
        calls.append((app, kwargs))

    monkeypatch.setenv("PORT", "9090")
    monkeypatch.setattr("aegon_api.main.uvicorn.run", capture_run)

    run()

    assert calls == [
        (
            "aegon_api:create_app",
            {
                "factory": True,
                "host": "0.0.0.0",
                "port": 9090,
                "proxy_headers": True,
            },
        )
    ]


def test_production_launcher_rejects_invalid_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PORT", "70000")

    with pytest.raises(ValidationError):
        run()


def test_openapi_snapshot() -> None:
    client, _ = _client()
    app = client.app
    assert isinstance(app, FastAPI)
    schema = app.openapi()
    schemes = schema["components"]["securitySchemes"]

    assert schemes["TenantKey"] == {
        "type": "apiKey",
        "description": "Tenant exchange credential formatted as `AegonKey <key>`.",
        "in": "header",
        "name": "Authorization",
    }
    assert schemes["SessionCookie"]["in"] == "cookie"
    assert schema["paths"]["/v1/sessions"]["post"]["security"] == [{"TenantKey": []}]
    assert schema["paths"]["/v1/sessions/current"]["get"]["security"] == [{"SessionCookie": []}]
    actual = json.dumps(schema, indent=2, sort_keys=True) + "\n"

    assert actual == SNAPSHOT.read_text()
