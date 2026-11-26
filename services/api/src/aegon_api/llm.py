"""Bounded structured-output adapters for generation and entailment checks."""

from __future__ import annotations

import asyncio
import html
import json
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, JsonValue, SecretStr


class LlmUsage(BaseModel):
    """Provider-reported usage converted to the graph's budget units."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_microusd: int = Field(ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LlmRequest(BaseModel):
    """A single bounded, schema-constrained model request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    system_prompt: str = Field(min_length=1, max_length=20_000)
    user_prompt: str = Field(min_length=1, max_length=200_000)
    response_schema: dict[str, JsonValue]
    temperature: float = Field(ge=0, le=1)
    max_output_tokens: int = Field(ge=1, le=16_384)


class LlmResponse(BaseModel):
    """Raw structured model output plus metered usage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str = Field(min_length=1, max_length=500_000)
    usage: LlmUsage


class LlmClient(Protocol):
    """Provider-neutral structured completion interface."""

    async def complete(self, request: LlmRequest) -> LlmResponse: ...


class JsonTransport(Protocol):
    """Minimal injectable HTTP boundary used by the Vertex adapter."""

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, object],
        timeout_seconds: float,
    ) -> Mapping[str, object]: ...


class UrllibJsonTransport:
    """Standard-library JSON transport that keeps blocking I/O off the event loop."""

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, object],
        timeout_seconds: float,
    ) -> Mapping[str, object]:
        body = json.dumps(payload, separators=(",", ":")).encode()

        def send() -> Mapping[str, object]:
            request = Request(url, data=body, headers=dict(headers), method="POST")
            with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                decoded = json.loads(response.read())
            if not isinstance(decoded, dict):
                raise ValueError("Gemini returned a non-object response")
            return cast(dict[str, object], decoded)

        return await asyncio.to_thread(send)


def _token_count(metadata: Mapping[str, object], key: str) -> int:
    value = metadata.get(key, 0)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"LLM {key} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class GeminiClient:
    """Vertex AI Gemini adapter with bounded retries and usage-based cost metering."""

    project_id: str
    location: str
    model: str
    access_token: SecretStr
    transport: JsonTransport = field(default_factory=UrllibJsonTransport)
    timeout_seconds: float = 15.0
    max_attempts: int = 3
    initial_backoff_seconds: float = 0.25
    input_microusd_per_million_tokens: int = 0
    output_microusd_per_million_tokens: int = 0

    def __post_init__(self) -> None:
        identifiers = (self.project_id, self.location, self.model)
        if any(not value or len(value) > 256 for value in identifiers):
            raise ValueError("Gemini resource identifiers must contain 1..256 characters")
        if self.timeout_seconds <= 0 or self.max_attempts < 1:
            raise ValueError("Gemini timeout and retry bounds must be positive")
        if self.initial_backoff_seconds < 0:
            raise ValueError("Gemini backoff must be non-negative")
        if (
            self.input_microusd_per_million_tokens < 0
            or self.output_microusd_per_million_tokens < 0
        ):
            raise ValueError("Gemini token prices must be non-negative")

    async def complete(self, request: LlmRequest) -> LlmResponse:
        endpoint = (
            f"https://{self.location}-aiplatform.googleapis.com/v1/projects/"
            f"{self.project_id}/locations/{self.location}/publishers/google/models/"
            f"{self.model}:generateContent"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": request.system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": request.user_prompt}]}],
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_output_tokens,
                "responseMimeType": "application/json",
                "responseSchema": request.response_schema,
            },
        }
        response: Mapping[str, object] | None = None
        for attempt in range(self.max_attempts):
            try:
                response = await self.transport.post_json(
                    endpoint,
                    headers={
                        "Authorization": f"Bearer {self.access_token.get_secret_value()}",
                        "Content-Type": "application/json",
                    },
                    payload=payload,
                    timeout_seconds=self.timeout_seconds,
                )
                break
            except (HTTPError, URLError, TimeoutError, OSError):
                if attempt + 1 == self.max_attempts:
                    raise
                delay = self.initial_backoff_seconds * (2**attempt)
                await asyncio.sleep(delay + random.uniform(0, delay / 4 if delay else 0))
        if response is None:
            raise RuntimeError("Gemini request completed without a response")
        return self._parse_response(response)

    def _parse_response(self, response: Mapping[str, object]) -> LlmResponse:
        try:
            candidates = cast(Sequence[Mapping[str, object]], response["candidates"])
            content = cast(Mapping[str, object], candidates[0]["content"])
            parts = cast(Sequence[Mapping[str, object]], content["parts"])
            text = cast(str, parts[0]["text"])
            metadata = cast(Mapping[str, object], response.get("usageMetadata", {}))
            input_tokens = _token_count(metadata, "promptTokenCount")
            output_tokens = _token_count(metadata, "candidatesTokenCount")
        except (KeyError, IndexError, TypeError, ValueError) as exception:
            raise ValueError("Gemini response is missing required structured fields") from exception
        cost = (
            input_tokens * self.input_microusd_per_million_tokens
            + output_tokens * self.output_microusd_per_million_tokens
        ) // 1_000_000
        return LlmResponse(
            content=text,
            usage=LlmUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_microusd=cost,
            ),
        )


@dataclass(frozen=True, slots=True)
class OllamaClient:
    """Loopback-only Ollama adapter for local model integration testing."""

    model: str
    base_url: str = "http://127.0.0.1:11434"
    transport: JsonTransport = field(default_factory=UrllibJsonTransport)
    timeout_seconds: float = 30.0
    max_attempts: int = 2
    initial_backoff_seconds: float = 0.25

    def __post_init__(self) -> None:
        if not self.model or len(self.model) > 256:
            raise ValueError("Ollama model must contain 1..256 characters")
        parsed = urlsplit(self.base_url)
        try:
            _ = parsed.port
        except ValueError as exception:
            raise ValueError("Ollama base URL must be a valid loopback HTTP origin") from exception
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Ollama base URL must be a loopback HTTP origin")
        if self.timeout_seconds <= 0 or self.max_attempts < 1:
            raise ValueError("Ollama timeout and retry bounds must be positive")
        if self.initial_backoff_seconds < 0:
            raise ValueError("Ollama backoff must be non-negative")

    async def complete(self, request: LlmRequest) -> LlmResponse:
        payload = {
            "model": self.model,
            "stream": False,
            "format": request.response_schema,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_output_tokens,
            },
        }
        response: Mapping[str, object] | None = None
        for attempt in range(self.max_attempts):
            try:
                response = await self.transport.post_json(
                    f"{self.base_url.rstrip('/')}/api/chat",
                    headers={"Content-Type": "application/json"},
                    payload=payload,
                    timeout_seconds=self.timeout_seconds,
                )
                break
            except (HTTPError, URLError, TimeoutError, OSError):
                if attempt + 1 == self.max_attempts:
                    raise
                delay = self.initial_backoff_seconds * (2**attempt)
                await asyncio.sleep(delay + random.uniform(0, delay / 4 if delay else 0))
        if response is None:
            raise RuntimeError("Ollama request completed without a response")
        return self._parse_response(response)

    @staticmethod
    def _parse_response(response: Mapping[str, object]) -> LlmResponse:
        try:
            message = cast(Mapping[str, object], response["message"])
            content = cast(str, message["content"])
            input_tokens = _token_count(response, "prompt_eval_count")
            output_tokens = _token_count(response, "eval_count")
        except (KeyError, TypeError, ValueError) as exception:
            raise ValueError("Ollama response is missing required structured fields") from exception
        return LlmResponse(
            content=content,
            usage=LlmUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_microusd=0,
            ),
        )


@dataclass(slots=True)
class FakeLlm:
    """Deterministic schema-aware fake used by offline tests and CI."""

    responses: list[str] = field(default_factory=lambda: [])
    input_tokens_per_call: int = 1
    output_tokens_per_call: int = 1
    cost_microusd_per_call: int = 0
    requests: list[LlmRequest] = field(default_factory=lambda: [], init=False)

    async def complete(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        if self.responses:
            content = self.responses.pop(0)
        elif "answer_sentences" in cast(
            dict[str, JsonValue], request.response_schema.get("properties", {})
        ):
            content = self._generate(request.user_prompt)
        else:
            content = self._verify(request.user_prompt)
        return LlmResponse(
            content=content,
            usage=LlmUsage(
                input_tokens=self.input_tokens_per_call,
                output_tokens=self.output_tokens_per_call,
                cost_microusd=self.cost_microusd_per_call,
            ),
        )

    @staticmethod
    def _generate(prompt: str) -> str:
        matches = re.findall(
            r'<evidence data-only="true" chunk-id="([^"]+)">\n(.*?)\n</evidence>',
            prompt,
            flags=re.DOTALL,
        )
        if not matches:
            return '{"answer_sentences":[]}'
        chunk_id, text = matches[0]
        text = html.unescape(text)
        sentence = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
        return json.dumps(
            {"answer_sentences": [{"text": sentence, "chunk_ids": [chunk_id]}]},
            separators=(",", ":"),
        )

    @staticmethod
    def _verify(prompt: str) -> str:
        payload_match = re.search(
            r"<verification_input>\n(.*?)\n</verification_input>", prompt, flags=re.DOTALL
        )
        if payload_match is None:
            return '{"claims":[]}'
        payload = json.loads(payload_match.group(1))
        evidence = {item["chunk_id"]: item["text"] for item in payload["evidence"]}
        claims: list[dict[str, object]] = []
        for item in payload["claims"]:
            text = item["text"]
            supporting_id = next(
                (
                    chunk_id
                    for chunk_id in item["chunk_ids"]
                    if text.casefold().rstrip(".") in evidence.get(chunk_id, "").casefold()
                ),
                None,
            )
            claims.append(
                {
                    "claim_id": item["claim_id"],
                    "supported": supporting_id is not None,
                    "chunk_id": supporting_id,
                    "quote": text if supporting_id is not None else None,
                }
            )
        return json.dumps({"claims": claims}, separators=(",", ":"))
