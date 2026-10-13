"""SSRF-resistant, robots-aware HTTPS acquisition."""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import socket
import ssl
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from email.message import Message
from typing import Protocol
from urllib.parse import SplitResult, unquote, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aegon_ingestion.parsers import parse_html
from aegon_ingestion.pipeline import SourceInput

USER_AGENT = "Aegon-RAG-Ingestion/1.0 (+https://github.com/jeelan-ds786/aegon-intelligence)"
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_BLOCKED_PATH_PARTS = {"account", "accounts", "checkout", "claim", "claims", "sign-in", "signin"}


class CrawlError(RuntimeError):
    """Base error for a rejected or failed crawl."""


class SsrfBlockedError(CrawlError):
    """Raised when a URL resolves to a non-public address."""


class RobotsDeniedError(CrawlError):
    """Raised when robots.txt disallows a requested URL."""


class ResponseRejectedError(CrawlError):
    """Raised when a response violates size, type, or quality policy."""


class CrawlConfig(BaseModel):
    """Validated policy for one approved-host crawl."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    approved_hosts: frozenset[str] = Field(min_length=1)
    request_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    minimum_interval_seconds: float = Field(default=0.5, ge=0, le=60)
    maximum_retries: int = Field(default=2, ge=0, le=5)
    maximum_redirects: int = Field(default=5, ge=0, le=10)
    maximum_response_bytes: int = Field(default=2_000_000, ge=1_024, le=20_000_000)
    minimum_content_characters: int = Field(default=80, ge=1, le=10_000)

    @field_validator("approved_hosts")
    @classmethod
    def normalize_hosts(cls, hosts: frozenset[str]) -> frozenset[str]:
        normalized = frozenset(host.rstrip(".").lower() for host in hosts)
        if any(not host or ":" in host or "/" in host for host in normalized):
            raise ValueError("approved_hosts must contain DNS hostnames without ports")
        return normalized


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """Bounded transport response."""

    status: int
    headers: Mapping[str, str]
    body: bytes


class Resolver(Protocol):
    """DNS resolution boundary used for SSRF validation."""

    def resolve(self, hostname: str, port: int) -> Sequence[str]: ...


class Transport(Protocol):
    """HTTPS transport that connects to an already validated address."""

    def get(
        self,
        *,
        url: str,
        resolved_ip: str,
        headers: Mapping[str, str],
        timeout_seconds: float,
        maximum_bytes: int,
    ) -> HttpResponse: ...


class SystemResolver:
    """Resolve all stream addresses through the operating system."""

    def resolve(self, hostname: str, port: int) -> Sequence[str]:
        results = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
        return tuple(sorted({str(result[4][0]) for result in results}))


class PinnedHttpsTransport:
    """HTTPS transport pinned to the address approved by the resolver check."""

    def __init__(
        self,
        *,
        ssl_context: ssl.SSLContext | None = None,
        connection_port: int | None = None,
        socket_factory: Callable[[tuple[str, int], float], socket.socket] = (
            socket.create_connection
        ),
    ) -> None:
        self._ssl_context = ssl_context or ssl.create_default_context()
        if self._ssl_context.minimum_version < ssl.TLSVersion.TLSv1_2:
            self._ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        self._connection_port = connection_port
        self._socket_factory = socket_factory

    def get(
        self,
        *,
        url: str,
        resolved_ip: str,
        headers: Mapping[str, str],
        timeout_seconds: float,
        maximum_bytes: int,
    ) -> HttpResponse:
        parsed = urlsplit(url)
        hostname = parsed.hostname or ""
        target = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        request_headers = {"Connection": "close", "Host": hostname, **headers}
        request = (
            f"GET {target} HTTP/1.1\r\n"
            + "".join(f"{name}: {value}\r\n" for name, value in request_headers.items())
            + "\r\n"
        )
        raw_socket = self._socket_factory(
            (resolved_ip, self._connection_port or parsed.port or 443), timeout_seconds
        )
        with self._ssl_context.wrap_socket(raw_socket, server_hostname=hostname) as tls_socket:
            tls_socket.sendall(request.encode("ascii"))
            response = http.client.HTTPResponse(tls_socket)
            response.begin()
            content_length = response.getheader("Content-Length")
            if content_length is not None:
                try:
                    declared_length = int(content_length)
                except ValueError as error:
                    raise ResponseRejectedError("response Content-Length is invalid") from error
                if declared_length > maximum_bytes:
                    raise ResponseRejectedError(
                        "response Content-Length exceeds configured maximum"
                    )
            body = response.read(maximum_bytes + 1)
            if len(body) > maximum_bytes:
                raise ResponseRejectedError("response body exceeds configured maximum")
            return HttpResponse(
                status=response.status,
                headers={name.lower(): value for name, value in response.getheaders()},
                body=body,
            )


def _is_public_address(value: str) -> bool:
    address = ipaddress.ip_address(value)
    return address.is_global and not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def validate_public_addresses(addresses: Sequence[str]) -> tuple[str, ...]:
    """Reject empty, malformed, or mixed public/private DNS answers."""
    if not addresses:
        raise SsrfBlockedError("hostname did not resolve")
    try:
        normalized = tuple(sorted({str(ipaddress.ip_address(value)) for value in addresses}))
    except ValueError as error:
        raise SsrfBlockedError("hostname returned an invalid IP address") from error
    if not all(_is_public_address(value) for value in normalized):
        raise SsrfBlockedError("hostname resolved to a non-public IP address")
    return normalized


class SafeCrawler:
    """Fetch approved public HTTPS pages and emit pipeline-ready inputs."""

    def __init__(
        self,
        config: CrawlConfig,
        *,
        resolver: Resolver | None = None,
        transport: Transport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._resolver = resolver or SystemResolver()
        self._transport = transport or PinnedHttpsTransport()
        self._clock = clock
        self._sleep = sleep
        self._last_request: dict[str, float] = {}
        self._robots: dict[str, RobotFileParser] = {}
        self._robots_interval: dict[str, float] = {}

    def fetch(self, url: str) -> SourceInput:
        """Fetch and parse one page after all acquisition policy checks pass."""
        final_url, response = self._request(url, enforce_robots=True)
        media_type = _media_type(response.headers)
        if media_type not in {"text/html", "application/xhtml+xml"}:
            raise ResponseRejectedError(f"unsupported content type: {media_type or 'missing'}")
        charset = _charset(response.headers)
        try:
            text = response.body.decode(charset, errors="strict")
        except (LookupError, UnicodeDecodeError) as error:
            raise ResponseRejectedError("response body is not valid declared text") from error
        source_id = f"web:{hashlib.sha256(final_url.encode()).hexdigest()}"
        document = parse_html(source_id, text)
        quality_text = " ".join(block.text for block in document.blocks)
        if len(quality_text) < self._config.minimum_content_characters:
            raise ResponseRejectedError("extracted content is below the quality threshold")
        return SourceInput.from_bytes(
            source_id=source_id,
            uri=final_url,
            kind="html",
            content=response.body,
            document=document,
        )

    def fetch_bytes(self, url: str, *, accepted_types: frozenset[str]) -> tuple[str, bytes]:
        """Fetch a bounded non-page resource such as a sitemap."""
        final_url, response = self._request(url, enforce_robots=True)
        media_type = _media_type(response.headers)
        if media_type not in accepted_types:
            raise ResponseRejectedError(f"unsupported content type: {media_type or 'missing'}")
        return final_url, response.body

    def _request(self, initial_url: str, *, enforce_robots: bool) -> tuple[str, HttpResponse]:
        current_url = initial_url
        for redirect_count in range(self._config.maximum_redirects + 1):
            parsed = self._validate_url(current_url)
            if enforce_robots:
                self._check_robots(current_url)
            response = self._request_with_retries(current_url, parsed.hostname or "")
            if response.status not in _REDIRECT_STATUSES:
                if not 200 <= response.status < 300:
                    raise CrawlError(f"request failed with HTTP {response.status}")
                return current_url, response
            if redirect_count == self._config.maximum_redirects:
                raise CrawlError("maximum redirects exceeded")
            location = _header(response.headers, "location")
            if not location:
                raise CrawlError("redirect response omitted Location")
            current_url = urljoin(current_url, location)
        raise AssertionError("redirect loop must return or raise")

    def _request_with_retries(self, url: str, hostname: str) -> HttpResponse:
        for attempt in range(self._config.maximum_retries + 1):
            addresses = validate_public_addresses(self._resolver.resolve(hostname, 443))
            self._pace(hostname)
            try:
                response = self._transport.get(
                    url=url,
                    resolved_ip=addresses[0],
                    headers={
                        "Accept": "text/html,application/xhtml+xml,application/xml,text/xml",
                        "User-Agent": USER_AGENT,
                    },
                    timeout_seconds=self._config.request_timeout_seconds,
                    maximum_bytes=self._config.maximum_response_bytes,
                )
            except (OSError, TimeoutError) as error:
                if attempt == self._config.maximum_retries:
                    raise CrawlError("request failed after retries") from error
                self._sleep(2.0**attempt)
                continue
            if not 100 <= response.status <= 599:
                raise ResponseRejectedError("transport returned an invalid HTTP status")
            if len(response.body) > self._config.maximum_response_bytes:
                raise ResponseRejectedError("response body exceeds configured maximum")
            if response.status not in _RETRY_STATUSES or attempt == self._config.maximum_retries:
                return response
            retry_after = _header(response.headers, "retry-after") or ""
            delay = min(float(retry_after), 60.0) if retry_after.isdigit() else 2.0**attempt
            self._sleep(delay)
        raise AssertionError("retry loop must return")

    def _validate_url(self, url: str) -> SplitResult:
        parsed = urlsplit(url)
        hostname = (parsed.hostname or "").rstrip(".").lower()
        if parsed.scheme != "https" or not hostname or parsed.username or parsed.password:
            raise CrawlError("only credential-free absolute HTTPS URLs are allowed")
        if parsed.port not in {None, 443}:
            raise CrawlError("only the default HTTPS port is allowed")
        if hostname not in self._config.approved_hosts:
            raise CrawlError("URL host is not approved")
        decoded_path = unquote(parsed.path)
        if "\\" in decoded_path or any(ord(character) < 32 for character in decoded_path):
            raise CrawlError("URL path contains unsafe characters")
        path_parts = {part.lower() for part in decoded_path.split("/") if part}
        if path_parts & _BLOCKED_PATH_PARTS:
            raise CrawlError("URL path is blocked by acquisition policy")
        return parsed

    def _check_robots(self, url: str) -> None:
        hostname = (urlsplit(url).hostname or "").lower()
        parser = self._robots.get(hostname)
        if parser is None:
            robots_url = f"https://{hostname}/robots.txt"
            _, response = self._request(robots_url, enforce_robots=False)
            parser = RobotFileParser(robots_url)
            parser.parse(response.body.decode("utf-8", errors="replace").splitlines())
            self._robots[hostname] = parser
            crawl_delay = parser.crawl_delay(USER_AGENT) or 0
            request_rate = parser.request_rate(USER_AGENT)
            rate_interval = (
                request_rate.seconds / request_rate.requests if request_rate is not None else 0
            )
            self._robots_interval[hostname] = max(float(crawl_delay), rate_interval)
        if not parser.can_fetch(USER_AGENT, url):
            raise RobotsDeniedError(f"robots.txt disallows {url}")

    def _pace(self, hostname: str) -> None:
        now = self._clock()
        last_request = self._last_request.get(hostname)
        if last_request is not None:
            minimum_interval = max(
                self._config.minimum_interval_seconds,
                self._robots_interval.get(hostname, 0),
            )
            delay = minimum_interval - (now - last_request)
            if delay > 0:
                self._sleep(delay)
                now = self._clock()
        self._last_request[hostname] = now


def _media_type(headers: Mapping[str, str]) -> str:
    return (_header(headers, "content-type") or "").split(";", 1)[0].strip().lower()


def _charset(headers: Mapping[str, str]) -> str:
    message = Message()
    message["content-type"] = _header(headers, "content-type") or "text/html"
    return message.get_content_charset() or "utf-8"


def _header(headers: Mapping[str, str], name: str) -> str | None:
    expected = name.lower()
    return next((value for key, value in headers.items() if key.lower() == expected), None)
