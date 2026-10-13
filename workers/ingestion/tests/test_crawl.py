import socket
import ssl
import threading
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from aegon_ingestion.crawl import (
    USER_AGENT,
    CrawlConfig,
    CrawlError,
    HttpResponse,
    PinnedHttpsTransport,
    ResponseRejectedError,
    RobotsDeniedError,
    SafeCrawler,
    SsrfBlockedError,
    validate_public_addresses,
)
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from pydantic import ValidationError

PUBLIC_IP = "93.184.216.34"


class FakeResolver:
    def __init__(self, answers: Sequence[Sequence[str]]) -> None:
        self.answers = list(answers)

    def resolve(self, hostname: str, port: int) -> Sequence[str]:
        del hostname, port
        return self.answers.pop(0)


class FakeTransport:
    def __init__(self, responses: Mapping[str, Sequence[HttpResponse | OSError]]) -> None:
        self.responses = {url: list(values) for url, values in responses.items()}
        self.requests: list[tuple[str, str, Mapping[str, str]]] = []

    def get(
        self,
        *,
        url: str,
        resolved_ip: str,
        headers: Mapping[str, str],
        timeout_seconds: float,
        maximum_bytes: int,
    ) -> HttpResponse:
        del timeout_seconds, maximum_bytes
        self.requests.append((url, resolved_ip, headers))
        result = self.responses[url].pop(0)
        if isinstance(result, OSError):
            raise result
        return result


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _response(
    body: bytes,
    *,
    status: int = 200,
    content_type: str = "text/html; charset=utf-8",
    headers: Mapping[str, str] | None = None,
) -> HttpResponse:
    return HttpResponse(
        status=status,
        headers={"content-type": content_type, **(headers or {})},
        body=body,
    )


def _crawler(
    transport: FakeTransport,
    resolver: FakeResolver,
    *,
    clock: FakeClock | None = None,
    approved_hosts: frozenset[str] = frozenset({"docs.example.test"}),
) -> SafeCrawler:
    fake_clock = clock or FakeClock()
    return SafeCrawler(
        CrawlConfig(
            approved_hosts=approved_hosts,
            minimum_interval_seconds=1,
            minimum_content_characters=1,
        ),
        resolver=resolver,
        transport=transport,
        clock=fake_clock,
        sleep=fake_clock.sleep,
    )


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "169.254.169.254",
        "::1",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.0.1",
        "0.0.0.0",
        "224.0.0.1",
    ],
)
def test_ssrf_matrix_rejects_non_public_addresses(address: str) -> None:
    with pytest.raises(SsrfBlockedError):
        validate_public_addresses((address,))


@pytest.mark.parametrize("addresses", [(), ("not-an-ip",)])
def test_ssrf_rejects_empty_and_malformed_dns_answers(addresses: tuple[str, ...]) -> None:
    with pytest.raises(SsrfBlockedError):
        validate_public_addresses(addresses)


def test_crawl_config_rejects_non_hostname_allowlist_values() -> None:
    with pytest.raises(ValidationError):
        CrawlConfig(approved_hosts=frozenset({"docs.example.test:443"}))


def test_mixed_dns_answer_and_rebinding_are_rejected() -> None:
    with pytest.raises(SsrfBlockedError):
        validate_public_addresses((PUBLIC_IP, "127.0.0.1"))

    robots_url = "https://docs.example.test/robots.txt"
    page_url = "https://docs.example.test/guide"
    transport = FakeTransport({robots_url: [_response(b"User-agent: *\nAllow: /")]})
    crawler = _crawler(
        transport,
        FakeResolver(((PUBLIC_IP,), ("169.254.169.254",))),
    )

    with pytest.raises(SsrfBlockedError):
        crawler.fetch(page_url)
    assert [request[0] for request in transport.requests] == [robots_url]


def test_redirect_hop_is_revalidated_before_following() -> None:
    robots_url = "https://docs.example.test/robots.txt"
    start_url = "https://docs.example.test/start"
    transport = FakeTransport(
        {
            robots_url: [_response(b"User-agent: *\nAllow: /")],
            start_url: [_response(b"", status=302, headers={"location": "https://evil.test/x"})],
        }
    )
    crawler = _crawler(transport, FakeResolver(((PUBLIC_IP,), (PUBLIC_IP,))))

    with pytest.raises(CrawlError, match="host is not approved"):
        crawler.fetch(start_url)


def test_robots_denial_prevents_page_request() -> None:
    robots_url = "https://docs.example.test/robots.txt"
    page_url = "https://docs.example.test/private"
    transport = FakeTransport({robots_url: [_response(b"User-agent: *\nDisallow: /private")]})
    crawler = _crawler(transport, FakeResolver(((PUBLIC_IP,),)))

    with pytest.raises(RobotsDeniedError):
        crawler.fetch(page_url)
    assert [request[0] for request in transport.requests] == [robots_url]


def test_rate_limit_retry_pacing_and_user_agent() -> None:
    robots_url = "https://docs.example.test/robots.txt"
    page_url = "https://docs.example.test/guide"
    transport = FakeTransport(
        {
            robots_url: [_response(b"User-agent: *\nAllow: /")],
            page_url: [
                _response(b"busy", status=429, headers={"retry-after": "2"}),
                _response(b"<main><h1>Guide</h1><p>Approved text.</p></main>"),
            ],
        }
    )
    clock = FakeClock()
    crawler = _crawler(
        transport,
        FakeResolver(((PUBLIC_IP,), (PUBLIC_IP,), (PUBLIC_IP,))),
        clock=clock,
    )

    source = crawler.fetch(page_url)

    assert source.uri == page_url
    assert source.source_id.startswith("web:")
    assert clock.sleeps == [1.0, 2.0]
    assert all(request[2]["User-Agent"] == USER_AGENT for request in transport.requests)


def test_transient_network_failure_is_retried() -> None:
    robots_url = "https://docs.example.test/robots.txt"
    page_url = "https://docs.example.test/guide"
    transport = FakeTransport(
        {
            robots_url: [_response(b"User-agent: *\nAllow: /")],
            page_url: [
                OSError("connection reset"),
                _response(b"<main><p>Recovered content.</p></main>"),
            ],
        }
    )
    crawler = _crawler(
        transport,
        FakeResolver(((PUBLIC_IP,), (PUBLIC_IP,), (PUBLIC_IP,))),
    )

    assert crawler.fetch(page_url).uri == page_url


def test_robots_crawl_delay_strengthens_pacing() -> None:
    robots_url = "https://docs.example.test/robots.txt"
    page_url = "https://docs.example.test/guide"
    transport = FakeTransport(
        {
            robots_url: [_response(b"User-agent: *\nAllow: /\nCrawl-delay: 3")],
            page_url: [_response(b"<main><p>Approved content.</p></main>")],
        }
    )
    clock = FakeClock()
    crawler = _crawler(
        transport,
        FakeResolver(((PUBLIC_IP,), (PUBLIC_IP,))),
        clock=clock,
    )

    crawler.fetch(page_url)

    assert clock.sleeps == [3.0]


def test_local_fake_server_exercises_redirects_robots_rate_and_pacing(
    tmp_path: Path,
) -> None:
    final_requests = 0

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            nonlocal final_requests
            if self.path == "/robots.txt":
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                body = b"User-agent: *\nAllow: /"
            elif self.path == "/start":
                self.send_response(302)
                self.send_header("Location", "/final")
                body = b""
            elif self.path == "/final" and final_requests == 0:
                final_requests += 1
                self.send_response(429)
                self.send_header("Retry-After", "1")
                body = b"busy"
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                body = b"<main><h1>Final</h1><p>Approved local server content.</p></main>"
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "docs.example.test")])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(UTC) - timedelta(minutes=1))
        .not_valid_after(datetime.now(UTC) + timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("docs.example.test")]), critical=False
        )
        .sign(private_key, hashes.SHA256())
    )
    certificate_pem = certificate.public_bytes(serialization.Encoding.PEM)
    certificate_path = tmp_path / "certificate.pem"
    key_path = tmp_path / "key.pem"
    certificate_path.write_bytes(certificate_pem)
    key_path.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.minimum_version = ssl.TLSVersion.TLSv1_2
    server_context.load_cert_chain(certificate_path, key_path)
    server.socket = server_context.wrap_socket(server.socket, server_side=True)
    client_context = ssl.create_default_context(cadata=certificate_pem.decode())
    client_context.minimum_version = ssl.TLSVersion.TLSv1_2
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        clock = FakeClock()
        crawler = SafeCrawler(
            CrawlConfig(
                approved_hosts=frozenset({"docs.example.test"}),
                minimum_interval_seconds=1,
                minimum_content_characters=1,
            ),
            resolver=FakeResolver(((PUBLIC_IP,),) * 4),
            transport=PinnedHttpsTransport(
                ssl_context=client_context,
                connection_port=server.server_port,
                socket_factory=lambda address, timeout: socket.create_connection(
                    ("127.0.0.1", address[1]), timeout
                ),
            ),
            clock=clock,
            sleep=clock.sleep,
        )
        source = crawler.fetch("https://docs.example.test/start")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert source.uri == "https://docs.example.test/final"
    assert final_requests == 1
    assert clock.sleeps == [1.0, 1.0, 1.0]


@pytest.mark.parametrize(
    "path",
    ["/account", "/sign-in/reset", "/checkout", "/claims/1", "/account%2Fsettings"],
)
def test_sensitive_paths_are_blocked_before_network(path: str) -> None:
    crawler = _crawler(FakeTransport({}), FakeResolver(()))
    with pytest.raises(CrawlError, match="path is blocked"):
        crawler.fetch(f"https://docs.example.test{path}")


@pytest.mark.parametrize(
    "url,error",
    [
        ("http://docs.example.test/guide", "credential-free absolute HTTPS"),
        ("https://user@docs.example.test/guide", "credential-free absolute HTTPS"),
        ("https://docs.example.test:444/guide", "default HTTPS port"),
        ("https://other.example.test/guide", "host is not approved"),
        ("https://docs.example.test/a%5Cb", "unsafe characters"),
    ],
)
def test_invalid_url_forms_are_rejected_before_network(url: str, error: str) -> None:
    crawler = _crawler(FakeTransport({}), FakeResolver(()))
    with pytest.raises(CrawlError, match=error):
        crawler.fetch(url)


def test_low_quality_and_wrong_content_type_are_rejected() -> None:
    robots_url = "https://docs.example.test/robots.txt"
    page_url = "https://docs.example.test/guide"
    transport = FakeTransport(
        {
            robots_url: [_response(b"User-agent: *\nAllow: /")],
            page_url: [_response(b"{}", content_type="application/json")],
        }
    )
    crawler = _crawler(transport, FakeResolver(((PUBLIC_IP,), (PUBLIC_IP,))))
    with pytest.raises(ResponseRejectedError, match="content type"):
        crawler.fetch(page_url)


@pytest.mark.parametrize(
    ("body", "content_type", "error"),
    [
        (b"\xff", "text/html; charset=utf-8", "declared text"),
        (b"<main>x</main>", "text/html", "quality threshold"),
    ],
)
def test_invalid_text_and_low_quality_content_are_rejected(
    body: bytes, content_type: str, error: str
) -> None:
    robots_url = "https://docs.example.test/robots.txt"
    page_url = "https://docs.example.test/guide"
    transport = FakeTransport(
        {
            robots_url: [_response(b"User-agent: *\nAllow: /")],
            page_url: [_response(body, content_type=content_type)],
        }
    )
    clock = FakeClock()
    crawler = SafeCrawler(
        CrawlConfig(
            approved_hosts=frozenset({"docs.example.test"}),
            minimum_interval_seconds=0,
            minimum_content_characters=10,
        ),
        resolver=FakeResolver(((PUBLIC_IP,), (PUBLIC_IP,))),
        transport=transport,
        clock=clock,
        sleep=clock.sleep,
    )
    with pytest.raises(ResponseRejectedError, match=error):
        crawler.fetch(page_url)


def test_transport_cannot_bypass_response_size_limit() -> None:
    robots_url = "https://docs.example.test/robots.txt"
    page_url = "https://docs.example.test/guide"
    transport = FakeTransport(
        {
            robots_url: [_response(b"User-agent: *\nAllow: /")],
            page_url: [_response(b"x" * 2_000_001)],
        }
    )
    crawler = _crawler(transport, FakeResolver(((PUBLIC_IP,), (PUBLIC_IP,))))
    with pytest.raises(ResponseRejectedError, match="exceeds configured maximum"):
        crawler.fetch(page_url)


def test_pinned_transport_hardens_injected_tls_context() -> None:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = ssl.TLSVersion.MINIMUM_SUPPORTED

    PinnedHttpsTransport(ssl_context=context)

    assert context.minimum_version is ssl.TLSVersion.TLSv1_2
