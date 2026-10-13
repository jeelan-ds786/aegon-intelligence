import json
from pathlib import Path

import pytest
from aegon_ingestion import DocumentBlock, ParsedDocument, SourceInput
from aegon_ingestion.host_sources import AegonConfig, HostSourceLoader, config_json_schema
from pydantic import ValidationError


def _config(**values: object) -> AegonConfig:
    return AegonConfig.model_validate({"schema_version": "1", **values})


def _web_source(url: str, text: str) -> SourceInput:
    source_id = f"web-{url.rsplit('/', 1)[-1]}"
    document = ParsedDocument(
        source_id=source_id,
        media_type="text/html",
        blocks=(DocumentBlock(kind="paragraph", text=text, locator="p:1"),),
    )
    return SourceInput.from_bytes(
        source_id=source_id,
        uri=url,
        kind="html",
        content=text.encode(),
        document=document,
    )


class FakeCrawler:
    def __init__(self, sitemap: bytes, pages: dict[str, SourceInput]) -> None:
        self.sitemap = sitemap
        self.pages = pages

    def fetch_bytes(self, url: str, *, accepted_types: frozenset[str]) -> tuple[str, bytes]:
        assert accepted_types == frozenset({"application/xml", "text/xml", "application/rss+xml"})
        return url, self.sitemap

    def fetch(self, url: str) -> SourceInput:
        return self.pages[url]


def test_config_schema_snapshot_and_strict_validation() -> None:
    schema_path = Path(__file__).parents[1] / "schemas" / "aegon.config.schema.json"
    assert config_json_schema() == json.loads(schema_path.read_text())
    schema_uri = "./workers/ingestion/schemas/aegon.config.schema.json"
    assert (
        AegonConfig.model_validate(
            {
                "$schema": schema_uri,
                "schema_version": "1",
                "markdown": [{"pattern": "docs/*.md"}],
            }
        ).schema_uri
        == schema_uri
    )

    with pytest.raises(ValidationError):
        _config(markdown=[{"pattern": "../**/*.md"}])
    with pytest.raises(ValidationError):
        _config(markdown=[{"pattern": "docs/*.md"}], unknown=True)
    with pytest.raises(ValidationError):
        AegonConfig.model_validate({"schema_version": "2", "markdown": [{"pattern": "*.md"}]})
    with pytest.raises(ValidationError, match="at least one source"):
        AegonConfig(schema_version="1")


def test_markdown_frontmatter_incremental_reuse_and_deletion(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "kept.md").write_text("---\ntitle: Kept\n---\n# Kept\n\nApproved text.")
    config = _config(markdown=[{"pattern": "docs/*.md", "required_frontmatter": ["title"]}])
    loader = HostSourceLoader(tmp_path)
    first = loader.load(config)
    deleted = _web_source("https://docs.example.test/deleted", "old")

    second = loader.load(
        config,
        previous_sources={first.sources[0].source_id: first.sources[0], deleted.source_id: deleted},
    )

    assert second.sources[0] is first.sources[0]
    assert second.changed_source_ids == ()
    assert second.unchanged_source_ids == ("docs/kept.md",)
    assert second.deleted_source_ids == (deleted.source_id,)


def test_changed_checksum_rebuilds_source(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text("# Guide\n\nFirst version")
    config = _config(markdown=[{"pattern": "*.md"}])
    loader = HostSourceLoader(tmp_path)
    first = loader.load(config)
    path.write_text("# Guide\n\nSecond version")

    second = loader.load(config, previous_sources={"guide.md": first.sources[0]})

    assert second.sources[0].checksum != first.sources[0].checksum
    assert second.changed_source_ids == ("guide.md",)


def test_required_frontmatter_failure_returns_no_plan(tmp_path: Path) -> None:
    (tmp_path / "good.md").write_text("---\ntitle: Good\n---\nGood")
    (tmp_path / "bad.md").write_text("Missing metadata")
    loader = HostSourceLoader(tmp_path)
    config = _config(markdown=[{"pattern": "*.md", "required_frontmatter": ["title"]}])

    with pytest.raises(ValueError, match="bad.md is missing required frontmatter"):
        loader.load(config)


def test_empty_or_non_markdown_glob_fails_closed(tmp_path: Path) -> None:
    loader = HostSourceLoader(tmp_path)
    with pytest.raises(ValueError, match="matched no files"):
        loader.load(_config(markdown=[{"pattern": "*.md"}]))

    (tmp_path / "notes.txt").write_text("not markdown")
    with pytest.raises(ValueError, match="unsupported file"):
        loader.load(_config(markdown=[{"pattern": "*.txt"}]))


def test_json_export_is_validated_and_weighted(tmp_path: Path) -> None:
    (tmp_path / "export.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "documents": [
                    {"id": "1", "title": "Runbook", "fields": {"body": "Recover safely"}}
                ],
            }
        )
    )
    config = _config(json_export={"path": "export.json", "fields": [{"name": "body", "weight": 2}]})

    plan = HostSourceLoader(tmp_path).load(config)

    assert plan.sources[0].document.blocks[1].weight == 2


def test_sitemap_feeds_complete_web_snapshot(tmp_path: Path) -> None:
    first_url = "https://docs.example.test/one"
    second_url = "https://docs.example.test/two"
    sitemap = (
        '<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"<url><loc>{first_url}</loc></url><url><loc>{second_url}</loc></url></urlset>"
    ).encode()
    crawler = FakeCrawler(
        sitemap,
        {first_url: _web_source(first_url, "one"), second_url: _web_source(second_url, "two")},
    )
    config = _config(sitemap_url="https://docs.example.test/sitemap.xml")

    plan = HostSourceLoader(tmp_path, crawler=crawler).load(config)

    assert len(plan.sources) == 2
    assert plan.changed_source_ids == ("web-one", "web-two")


def test_sitemap_rejects_entities_duplicates_and_url_overflow(tmp_path: Path) -> None:
    config = _config(sitemap_url="https://docs.example.test/sitemap.xml", maximum_sitemap_urls=1)
    bad_sitemaps = (
        b'<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><urlset/>',
        b"<urlset><url><loc>https://docs.example.test/a</loc></url>"
        b"<url><loc>https://docs.example.test/a</loc></url></urlset>",
        b"<urlset><url><loc>https://docs.example.test/a</loc></url>"
        b"<url><loc>https://docs.example.test/b</loc></url></urlset>",
    )
    for sitemap in bad_sitemaps:
        with pytest.raises(ValueError):
            HostSourceLoader(tmp_path, crawler=FakeCrawler(sitemap, {})).load(config)
