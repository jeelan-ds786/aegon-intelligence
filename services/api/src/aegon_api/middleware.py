"""Security middleware for request identity, origins, and body limits."""

from __future__ import annotations

import json
import re
import uuid

from aegon_common.models import ErrorCode
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def _error_body(code: ErrorCode, message: str, request_id: str) -> bytes:
    return json.dumps(
        {
            "error": {
                "code": code.value,
                "message": message,
                "retryable": False,
                "request_id": request_id,
            }
        },
        separators=(",", ":"),
    ).encode()


async def _send_error(
    send: Send,
    *,
    status: int,
    code: ErrorCode,
    message: str,
    request_id: str,
) -> None:
    body = _error_body(code, message, request_id)
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class RequestIdMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        candidate = headers.get(b"x-request-id", b"").decode("ascii", errors="ignore")
        request_id = candidate if REQUEST_ID_PATTERN.fullmatch(candidate) else str(uuid.uuid4())
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                message["headers"].append((b"x-request-id", request_id.encode()))
            await send(message)

        await self._app(scope, receive, send_with_request_id)


class OriginMiddleware:
    def __init__(self, app: ASGIApp, allowed_origins: tuple[str, ...]) -> None:
        self._app = app
        self._allowed_origins = frozenset(origin.encode() for origin in allowed_origins)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            origin = dict(scope["headers"]).get(b"origin")
            if origin is not None and origin not in self._allowed_origins:
                await _send_error(
                    send,
                    status=403,
                    code=ErrorCode.UNAUTHORIZED,
                    message="Origin is not allowed.",
                    request_id=scope["state"]["request_id"],
                )
                return
        await self._app(scope, receive, send)


class RequestSizeLimitMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self._app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        content_length = dict(scope["headers"]).get(b"content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                declared = self._max_bytes + 1
            if declared > self._max_bytes:
                await self._reject(scope, send)
                return

        received = 0
        messages: list[Message] = []
        while True:
            message = await receive()
            messages.append(message)
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self._max_bytes:
                    await self._reject(scope, send)
                    return
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                break

        async def replay_receive() -> Message:
            if messages:
                return messages.pop(0)
            return {"type": "http.disconnect"}

        await self._app(scope, replay_receive, send)

    async def _reject(self, scope: Scope, send: Send) -> None:
        await _send_error(
            send,
            status=413,
            code=ErrorCode.INVALID_ARGUMENTS,
            message="Request body exceeds the configured limit.",
            request_id=scope["state"]["request_id"],
        )
