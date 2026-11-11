"""FastAPI application factory and session routes."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, cast

from aegon_common.models import ContractModel, ErrorCode
from fastapi import Cookie, Depends, FastAPI, Header, Request, Response, Security
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyCookie, APIKeyHeader
from psycopg import AsyncConnection
from psycopg import Error as PsycopgError
from psycopg_pool import AsyncConnectionPool
from starlette.exceptions import HTTPException as StarletteHTTPException

from aegon_api.errors import ApiError, ErrorDetail, ErrorEnvelope
from aegon_api.middleware import OriginMiddleware, RequestIdMiddleware, RequestSizeLimitMiddleware
from aegon_api.sessions import (
    DatabasePool,
    PostgresSessionStore,
    SessionRecord,
    SessionService,
    SessionStore,
)
from aegon_api.settings import ApiSettings

logger = logging.getLogger(__name__)


class HealthResponse(ContractModel):
    status: str


class SessionResponse(ContractModel):
    tenant_id: str
    scopes: tuple[str, ...]
    csrf_token: str
    expires_at: datetime


class SessionIdentityResponse(ContractModel):
    tenant_id: str
    scopes: tuple[str, ...]
    expires_at: datetime


def _request_id(request: Request) -> str:
    return cast(str, request.state.request_id)


def _error_response(
    request: Request,
    status_code: int,
    code: ErrorCode,
    message: str,
    *,
    retryable: bool = False,
) -> JSONResponse:
    envelope = ErrorEnvelope(
        error=ErrorDetail(
            code=code,
            message=message,
            retryable=retryable,
            request_id=_request_id(request),
        )
    )
    return JSONResponse(status_code=status_code, content=envelope.model_dump(mode="json"))


def create_app(
    settings: ApiSettings | None = None,
    session_store: SessionStore | None = None,
) -> FastAPI:
    """Create an API instance with explicit dependencies for tests and deployments."""

    resolved_settings = settings or ApiSettings()  # pyright: ignore[reportCallIssue]
    pool: AsyncConnectionPool[AsyncConnection[tuple[object, ...]]] | None = None
    if session_store is None:
        database_url = resolved_settings.database_url
        if database_url is None:
            raise ValueError("database_url is required when no session store is injected")
        pool = AsyncConnectionPool(
            conninfo=database_url.get_secret_value(),
            open=False,
            min_size=1,
            max_size=10,
        )
        session_store = PostgresSessionStore(cast(DatabasePool, pool))

    service = SessionService(
        session_store,
        resolved_settings.tenant_keys,
        resolved_settings.session_ttl_seconds,
    )
    tenant_key_header = APIKeyHeader(
        name="Authorization",
        scheme_name="TenantKey",
        description="Tenant exchange credential formatted as `AegonKey <key>`.",
        auto_error=False,
    )
    session_cookie = APIKeyCookie(
        name=resolved_settings.session_cookie_name,
        scheme_name="SessionCookie",
        description="Opaque Aegon browser session.",
        auto_error=False,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
        if pool is not None:
            await pool.open()
        try:
            yield
        finally:
            if pool is not None:
                await pool.close()

    app = FastAPI(
        title="Aegon-RAG API",
        version="0.1.0",
        lifespan=lifespan,
        responses={
            400: {"model": ErrorEnvelope},
            401: {"model": ErrorEnvelope},
            403: {"model": ErrorEnvelope},
            413: {"model": ErrorEnvelope},
            422: {"model": ErrorEnvelope},
            500: {"model": ErrorEnvelope},
            503: {"model": ErrorEnvelope},
        },
    )

    @app.exception_handler(ApiError)
    async def handle_api_error(  # pyright: ignore[reportUnusedFunction]
        request: Request, exception: ApiError
    ) -> JSONResponse:
        return _error_response(
            request,
            exception.status_code,
            exception.code,
            exception.message,
            retryable=exception.retryable,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(  # pyright: ignore[reportUnusedFunction]
        request: Request, _: RequestValidationError
    ) -> JSONResponse:
        return _error_response(
            request, 422, ErrorCode.INVALID_ARGUMENTS, "Request validation failed."
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(  # pyright: ignore[reportUnusedFunction]
        request: Request, exception: StarletteHTTPException
    ) -> JSONResponse:
        code = ErrorCode.NOT_FOUND if exception.status_code == 404 else ErrorCode.INVALID_ARGUMENTS
        message = (
            "Resource not found." if exception.status_code == 404 else "Request is not allowed."
        )
        return _error_response(request, exception.status_code, code, message)

    @app.exception_handler(Exception)
    async def handle_unexpected_error(  # pyright: ignore[reportUnusedFunction]
        request: Request, exception: Exception
    ) -> JSONResponse:
        del exception
        logger.error("Unhandled API error", extra={"request_id": _request_id(request)})
        return _error_response(
            request, 500, ErrorCode.INTERNAL_ERROR, "An internal error occurred."
        )

    async def current_session(
        token: Annotated[str | None, Security(session_cookie)],
    ) -> SessionRecord:
        if token is None:
            raise ApiError(401, ErrorCode.UNAUTHORIZED, "Authentication is required.")
        session = await service.authenticate(token)
        if session is None:
            raise ApiError(401, ErrorCode.UNAUTHORIZED, "Session is invalid or expired.")
        return session

    async def csrf_session(
        session: Annotated[SessionRecord, Depends(current_session)],
        origin: Annotated[str | None, Header()] = None,
        csrf_cookie: Annotated[str | None, Cookie(alias=resolved_settings.csrf_cookie_name)] = None,
        csrf_header: Annotated[str | None, Header(alias="X-CSRF-Token")] = None,
    ) -> SessionRecord:
        if origin is None or origin not in resolved_settings.allowed_origins:
            raise ApiError(403, ErrorCode.UNAUTHORIZED, "A trusted Origin is required.")
        if (
            csrf_cookie is None
            or csrf_header is None
            or not service.csrf_matches(session, csrf_cookie, csrf_header)
        ):
            raise ApiError(403, ErrorCode.UNAUTHORIZED, "CSRF validation failed.")
        return session

    @app.get("/health", response_model=HealthResponse, tags=["operations"])
    async def health() -> HealthResponse:  # pyright: ignore[reportUnusedFunction]
        return HealthResponse(status="ok")

    @app.get(
        "/ready",
        response_model=HealthResponse,
        responses={503: {"model": ErrorEnvelope}},
        tags=["operations"],
    )
    async def ready() -> HealthResponse:  # pyright: ignore[reportUnusedFunction]
        try:
            is_ready = await session_store.ready()
        except PsycopgError:
            logger.warning("Database readiness check failed")
            is_ready = False
        if not is_ready:
            raise ApiError(
                503,
                ErrorCode.DEPENDENCY_UNAVAILABLE,
                "Database is not ready.",
                retryable=True,
            )
        return HealthResponse(status="ready")

    @app.post(
        "/v1/sessions",
        response_model=SessionResponse,
        status_code=201,
        tags=["sessions"],
    )
    async def exchange_session(  # pyright: ignore[reportUnusedFunction]
        response: Response,
        authorization: Annotated[str | None, Security(tenant_key_header)],
    ) -> SessionResponse:
        authorization = authorization or ""
        scheme, separator, raw_key = authorization.partition(" ")
        if separator != " " or scheme.lower() != "aegonkey" or not raw_key:
            raise ApiError(401, ErrorCode.UNAUTHORIZED, "A valid tenant key is required.")
        grant = service.resolve_tenant_key(raw_key)
        if grant is None:
            raise ApiError(401, ErrorCode.UNAUTHORIZED, "A valid tenant key is required.")
        token, csrf_token, session = await service.issue(grant)
        response.set_cookie(
            resolved_settings.session_cookie_name,
            token,
            httponly=True,
            secure=True,
            samesite="strict",
            path="/",
            max_age=resolved_settings.session_ttl_seconds,
        )
        response.set_cookie(
            resolved_settings.csrf_cookie_name,
            csrf_token,
            httponly=False,
            secure=True,
            samesite="strict",
            path="/",
            max_age=resolved_settings.session_ttl_seconds,
        )
        return SessionResponse(
            tenant_id=session.tenant_id,
            scopes=session.scopes,
            csrf_token=csrf_token,
            expires_at=session.expires_at,
        )

    @app.get(
        "/v1/sessions/current",
        response_model=SessionIdentityResponse,
        tags=["sessions"],
    )
    async def inspect_session(  # pyright: ignore[reportUnusedFunction]
        session: Annotated[SessionRecord, Depends(current_session)],
    ) -> SessionIdentityResponse:
        return SessionIdentityResponse(
            tenant_id=session.tenant_id,
            scopes=session.scopes,
            expires_at=session.expires_at,
        )

    @app.delete("/v1/sessions/current", status_code=204, tags=["sessions"])
    async def revoke_session(  # pyright: ignore[reportUnusedFunction]
        request: Request,
        response: Response,
        _: Annotated[SessionRecord, Depends(csrf_session)],
    ) -> None:
        token = request.cookies[resolved_settings.session_cookie_name]
        await service.revoke(token)
        response.delete_cookie(
            resolved_settings.session_cookie_name, secure=True, httponly=True, samesite="strict"
        )
        response.delete_cookie(
            resolved_settings.csrf_cookie_name, secure=True, httponly=False, samesite="strict"
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.allowed_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-CSRF-Token", "X-Request-ID"],
    )
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=resolved_settings.max_request_bytes)
    app.add_middleware(OriginMiddleware, allowed_origins=resolved_settings.allowed_origins)
    app.add_middleware(RequestIdMiddleware)
    return app
