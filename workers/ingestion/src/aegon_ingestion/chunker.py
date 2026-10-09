import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from aegon_ingestion.documents import DocumentBlock, ParsedDocument


class Tokenizer(Protocol):
    """Tokenizer boundary used for sizing and reversible prose splitting."""

    def encode(self, text: str) -> Sequence[str]: ...

    def decode(self, tokens: Sequence[str]) -> str: ...


class PiiRedactor(Protocol):
    """Optional policy hook; implementations live outside ingestion chunking."""

    def redact(self, text: str, *, source_id: str, locator: str) -> str: ...


class WhitespaceTokenizer:
    """Deterministic offline tokenizer suitable for tests and local ingestion."""

    def encode(self, text: str) -> Sequence[str]:
        return tuple(text.split())

    def decode(self, tokens: Sequence[str]) -> str:
        return " ".join(tokens)


class ChunkingConfig(BaseModel):
    """Validated sizing policy for retrieval chunks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_tokens: int = Field(default=400, ge=300, le=500)
    hard_max_tokens: int = Field(default=800, ge=500, le=800)
    overlap_ratio: float = Field(default=0.12, ge=0.10, le=0.15)


class Chunk(BaseModel):
    """A deterministic retrieval unit with source and section provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_id: str
    text: str = Field(min_length=1)
    heading_path: tuple[str, ...]
    locator: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    token_count: int = Field(gt=0, le=800)
    overlap_tokens: int = Field(ge=0)


class OversizedAtomicBlockError(ValueError):
    """Raised when preserving a code or table block would exceed the hard maximum."""


@dataclass(frozen=True)
class _Section:
    heading_path: tuple[str, ...]
    locator: str
    blocks: tuple[DocumentBlock, ...]


@dataclass(frozen=True)
class _Unit:
    text: str
    locator: str
    kind: str
    overlap: bool = False


def _digest(*parts: str) -> str:
    value = "\0".join(parts).encode()
    return hashlib.sha256(value).hexdigest()


def _sections(document: ParsedDocument) -> tuple[_Section, ...]:
    sections: list[_Section] = []
    heading_stack: list[str] = []
    current_blocks: list[DocumentBlock] = []
    current_path: tuple[str, ...] = ()
    current_locator = "document"

    for block in document.blocks:
        if block.kind == "heading":
            if current_blocks:
                sections.append(_Section(current_path, current_locator, tuple(current_blocks)))
            level = block.heading_level or 1
            heading_stack[level - 1 :] = []
            while len(heading_stack) < level - 1:
                heading_stack.append("")
            heading_stack.append(block.text)
            current_path = tuple(item for item in heading_stack if item)
            current_locator = block.locator
            current_blocks = [block]
        else:
            current_blocks.append(block)
    if current_blocks:
        sections.append(_Section(current_path, current_locator, tuple(current_blocks)))
    return tuple(sections)


class Chunker:
    """Split parsed documents into deterministic, source-aligned retrieval chunks."""

    def __init__(
        self,
        tokenizer: Tokenizer | None = None,
        *,
        config: ChunkingConfig | None = None,
        pii_redactor: PiiRedactor | None = None,
    ) -> None:
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._config = config or ChunkingConfig()
        self._pii_redactor = pii_redactor

    def chunk(self, document: ParsedDocument) -> tuple[Chunk, ...]:
        """Chunk one document without network access or external state."""
        chunks: list[Chunk] = []
        for section in _sections(document):
            chunks.extend(self._chunk_section(document.source_id, section))
        return tuple(chunks)

    def _count(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    def _chunk_section(self, source_id: str, section: _Section) -> list[Chunk]:
        chunks: list[Chunk] = []
        current: list[_Unit] = []
        overlap_budget = round(self._config.target_tokens * self._config.overlap_ratio)

        def current_count() -> int:
            return self._count("\n\n".join(unit.text for unit in current))

        def emit() -> None:
            nonlocal current
            if not current:
                return
            chunks.append(self._build_chunk(source_id, section, current))
            eligible = [unit for unit in current if unit.kind == "paragraph" and not unit.overlap]
            if not eligible:
                current = []
                return
            tail = self._tokenizer.encode(eligible[-1].text)[-overlap_budget:]
            current = [
                _Unit(
                    text=self._tokenizer.decode(tail),
                    locator=eligible[-1].locator,
                    kind="paragraph",
                    overlap=True,
                )
            ]

        for block in section.blocks:
            block_tokens = self._tokenizer.encode(block.text)
            if block.kind in {"code", "table"}:
                if len(block_tokens) > self._config.hard_max_tokens:
                    raise OversizedAtomicBlockError(
                        f"{block.kind} block at {block.locator} has {len(block_tokens)} tokens; "
                        f"maximum is {self._config.hard_max_tokens}"
                    )
                if current and current_count() + len(block_tokens) > self._config.target_tokens:
                    if all(unit.overlap for unit in current):
                        current = []
                    elif all(unit.kind == "heading" for unit in current):
                        pass
                    else:
                        emit()
                current.append(_Unit(block.text, block.locator, block.kind))
                if current_count() >= self._config.target_tokens:
                    emit()
                continue
            if block.kind == "heading":
                current.append(_Unit(block.text, block.locator, block.kind))
                continue

            remaining = block_tokens
            while remaining:
                available = self._config.target_tokens - current_count()
                if available <= 0:
                    emit()
                    available = self._config.target_tokens - current_count()
                take = min(len(remaining), available)
                current.append(
                    _Unit(
                        self._tokenizer.decode(remaining[:take]),
                        block.locator,
                        block.kind,
                    )
                )
                remaining = remaining[take:]
                if remaining:
                    emit()
        emit()
        return chunks

    def _build_chunk(self, source_id: str, section: _Section, units: list[_Unit]) -> Chunk:
        text = "\n\n".join(unit.text for unit in units)
        source_units = [unit for unit in units if not unit.overlap] or units
        locator = f"{source_units[0].locator}..{source_units[-1].locator}"
        if self._pii_redactor is not None:
            text = self._pii_redactor.redact(text, source_id=source_id, locator=locator)
        token_count = self._count(text)
        if not text.strip() or token_count > self._config.hard_max_tokens:
            raise ValueError(
                "PII redaction must return non-empty text within the hard token maximum"
            )
        content_sha256 = _digest(text)
        return Chunk(
            chunk_id=_digest(source_id, locator, content_sha256),
            parent_id=_digest(source_id, section.locator),
            source_id=source_id,
            text=text,
            heading_path=section.heading_path,
            locator=locator,
            content_sha256=content_sha256,
            token_count=token_count,
            overlap_tokens=sum(self._count(unit.text) for unit in units if unit.overlap),
        )
