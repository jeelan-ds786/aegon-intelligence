"""Host-owned source configuration and complete snapshot loading."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import urlsplit
from xml.etree import ElementTree

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aegon_ingestion.documents import ParsedDocument
from aegon_ingestion.parsers import WeightedField, parse_json_export, parse_markdown
from aegon_ingestion.pipeline import SourceInput

_FRONTMATTER_KEY = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*):(?:\s|$)")


class MarkdownGlob(BaseModel):
    """A root-confined Markdown glob and its mandatory metadata keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pattern: str = Field(min_length=1, max_length=512)
    required_frontmatter: tuple[str, ...] = ()

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, pattern: str) -> str:
        path = Path(pattern)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("markdown patterns must be relative and cannot traverse parents")
        return pattern

    @field_validator("required_frontmatter")
    @classmethod
    def validate_frontmatter_keys(cls, keys: tuple[str, ...]) -> tuple[str, ...]:
        if len(keys) != len(set(keys)):
            raise ValueError("required_frontmatter keys must be unique")
        if any(not _FRONTMATTER_KEY.match(f"{key}:") for key in keys):
            raise ValueError("required_frontmatter contains an invalid key")
        return keys


class JsonFieldConfig(BaseModel):
    """An allowlisted JSON export field and retrieval weight."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=128)
    weight: float = Field(gt=0, le=100)


class JsonExportConfig(BaseModel):
    """A root-confined JSON export and its allowlisted fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1, max_length=512)
    fields: tuple[JsonFieldConfig, ...] = Field(min_length=1)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("JSON export path must be relative and cannot traverse parents")
        return value

    @model_validator(mode="after")
    def validate_unique_fields(self) -> JsonExportConfig:
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("JSON export fields must be unique")
        return self


class AegonConfig(BaseModel):
    """Strict schema for host-controlled ingestion sources."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_uri: str | None = Field(default=None, alias="$schema", max_length=2048)
    schema_version: Literal["1"]
    markdown: tuple[MarkdownGlob, ...] = ()
    json_export: JsonExportConfig | None = None
    sitemap_url: str | None = Field(default=None, max_length=2048)
    maximum_sitemap_urls: int = Field(default=1_000, ge=1, le=10_000)

    @field_validator("sitemap_url")
    @classmethod
    def validate_sitemap_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("sitemap_url must be an absolute credential-free HTTPS URL")
        return value

    @model_validator(mode="after")
    def require_source(self) -> AegonConfig:
        if not self.markdown and self.json_export is None and self.sitemap_url is None:
            raise ValueError("configuration must declare at least one source")
        return self

    @classmethod
    def from_json(cls, content: str | bytes) -> AegonConfig:
        """Validate an ``aegon.config.json`` payload."""
        return cls.model_validate_json(content)


class AcquisitionPlan(BaseModel):
    """Complete source snapshot and its incremental change classification."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    sources: tuple[SourceInput, ...]
    changed_source_ids: tuple[str, ...]
    unchanged_source_ids: tuple[str, ...]
    deleted_source_ids: tuple[str, ...]


class SourceCrawler(Protocol):
    """Crawler operations required by host sitemap loading."""

    def fetch(self, url: str) -> SourceInput: ...

    def fetch_bytes(self, url: str, *, accepted_types: frozenset[str]) -> tuple[str, bytes]: ...


class HostSourceLoader:
    """Build a complete pipeline snapshot from host files and an optional sitemap."""

    def __init__(self, root: Path, *, crawler: SourceCrawler | None = None) -> None:
        self._root = root.resolve(strict=True)
        if not self._root.is_dir():
            raise ValueError("host source root must be a directory")
        self._crawler = crawler

    def load(
        self,
        config: AegonConfig,
        *,
        previous_sources: Mapping[str, SourceInput] | None = None,
    ) -> AcquisitionPlan:
        """Load every configured source; any failure prevents a plan from being returned."""
        previous = previous_sources or {}
        loaded: dict[str, SourceInput] = {}
        changed: set[str] = set()
        unchanged: set[str] = set()

        for markdown_config in config.markdown:
            paths = [
                path for path in sorted(self._root.glob(markdown_config.pattern)) if path.is_file()
            ]
            if not paths:
                raise ValueError(f"markdown pattern matched no files: {markdown_config.pattern}")
            for path in paths:
                resolved = self._confined(path)
                if resolved.suffix.lower() not in {".md", ".mdx"}:
                    raise ValueError(f"markdown pattern matched unsupported file: {path}")
                relative = resolved.relative_to(self._root).as_posix()
                content = resolved.read_bytes()
                text = content.decode("utf-8", errors="strict")
                _validate_frontmatter(relative, text, markdown_config.required_frontmatter)
                is_mdx = resolved.suffix.lower() == ".mdx"
                source = self._reuse_or_build(
                    source_id=relative,
                    uri=resolved.as_uri(),
                    kind="mdx" if is_mdx else "markdown",
                    content=content,
                    previous=previous,
                    build=lambda relative=relative, text=text, is_mdx=is_mdx: parse_markdown(
                        relative,
                        text,
                        mdx=is_mdx,
                    ),
                )
                self._add_source(loaded, source)
                (unchanged if source is previous.get(relative) else changed).add(relative)

        if config.json_export is not None:
            resolved = self._confined(self._root / config.json_export.path)
            content = resolved.read_bytes()
            source_id = resolved.relative_to(self._root).as_posix()
            fields = tuple(
                WeightedField(name=field.name, weight=field.weight)
                for field in config.json_export.fields
            )
            source = self._reuse_or_build(
                source_id=source_id,
                uri=resolved.as_uri(),
                kind="json_export",
                content=content,
                previous=previous,
                build=lambda: parse_json_export(source_id, content, fields=fields),
            )
            self._add_source(loaded, source)
            (unchanged if source is previous.get(source_id) else changed).add(source_id)

        if config.sitemap_url is not None:
            if self._crawler is None:
                raise ValueError("a crawler is required when sitemap_url is configured")
            _, sitemap = self._crawler.fetch_bytes(
                config.sitemap_url,
                accepted_types=frozenset({"application/xml", "text/xml", "application/rss+xml"}),
            )
            for url in _sitemap_urls(sitemap, maximum_urls=config.maximum_sitemap_urls):
                source = self._crawler.fetch(url)
                prior = previous.get(source.source_id)
                if prior is not None and prior.checksum == source.checksum:
                    source = prior
                    unchanged.add(source.source_id)
                else:
                    changed.add(source.source_id)
                self._add_source(loaded, source)

        deleted = set(previous) - set(loaded)
        return AcquisitionPlan(
            sources=tuple(loaded[source_id] for source_id in sorted(loaded)),
            changed_source_ids=tuple(sorted(changed)),
            unchanged_source_ids=tuple(sorted(unchanged)),
            deleted_source_ids=tuple(sorted(deleted)),
        )

    def _confined(self, path: Path) -> Path:
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(self._root):
            raise ValueError(f"source path escapes configured root: {path}")
        return resolved

    @staticmethod
    def _reuse_or_build(
        *,
        source_id: str,
        uri: str,
        kind: str,
        content: bytes,
        previous: Mapping[str, SourceInput],
        build: Callable[[], ParsedDocument],
    ) -> SourceInput:
        checksum = hashlib.sha256(content).hexdigest()
        prior = previous.get(source_id)
        if prior is not None and prior.checksum == checksum and prior.uri == uri:
            return prior
        document = build()
        return SourceInput.from_bytes(
            source_id=source_id,
            uri=uri,
            kind=kind,
            content=content,
            document=document,
        )

    @staticmethod
    def _add_source(loaded: dict[str, SourceInput], source: SourceInput) -> None:
        if source.source_id in loaded:
            raise ValueError(f"duplicate configured source id: {source.source_id}")
        loaded[source.source_id] = source


def _validate_frontmatter(source_id: str, text: str, required: tuple[str, ...]) -> None:
    if not required:
        return
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{source_id} is missing required frontmatter")
    keys: set[str] = set()
    for line in lines[1:]:
        if line.strip() == "---":
            missing = set(required) - keys
            if missing:
                raise ValueError(f"{source_id} is missing frontmatter keys: {sorted(missing)}")
            return
        match = _FRONTMATTER_KEY.match(line)
        if match:
            keys.add(match.group(1))
    raise ValueError(f"{source_id} has unterminated frontmatter")


def _sitemap_urls(content: bytes, *, maximum_urls: int) -> tuple[str, ...]:
    if b"<!DOCTYPE" in content.upper() or b"<!ENTITY" in content.upper():
        raise ValueError("sitemap DTDs and entities are not allowed")
    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError as error:
        raise ValueError("sitemap is not valid XML") from error
    if root.tag.rsplit("}", 1)[-1] != "urlset":
        raise ValueError("only sitemap urlset documents are supported")
    urls = tuple(
        (element.text or "").strip()
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1] == "loc" and (element.text or "").strip()
    )
    if len(urls) > maximum_urls:
        raise ValueError("sitemap exceeds configured URL maximum")
    if len(urls) != len(set(urls)):
        raise ValueError("sitemap contains duplicate URLs")
    return urls


def config_json_schema() -> dict[str, object]:
    """Return the authoritative JSON Schema for ``aegon.config.json``."""
    return json.loads(json.dumps(AegonConfig.model_json_schema()))
