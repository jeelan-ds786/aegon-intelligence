# Crawling and host sources

The ingestion acquisition layer creates a complete, validated `SourceInput` snapshot for the
governed ingestion pipeline. It does not embed, sign, or write data. Callers must construct an
`AcquisitionPlan` successfully before invoking `GovernedIngestionPipeline.run`; an acquisition
exception therefore leaves the currently published index untouched.

## Safe crawling

`SafeCrawler` accepts only credential-free HTTPS URLs on an explicit `approved_hosts` allowlist.
Every initial URL and redirect hop is validated independently. Paths containing `account`,
`sign-in`, `signin`, `checkout`, `claim`, or `claims` as complete path segments are rejected.
Authenticated crawling and JavaScript execution are not supported.

Before every request and retry, the crawler resolves the hostname and rejects the complete answer
set if any address is private, loopback, link-local, multicast, reserved, unspecified, or otherwise
non-global. The production transport connects directly to the validated IP while retaining the
approved hostname for TLS certificate verification and the HTTP `Host` header. Re-resolution on
each request prevents a DNS rebinding answer from bypassing the initial check.

The crawler fetches and caches each host's `robots.txt` before page acquisition and fails closed if
robots acquisition fails or a path is disallowed. Requests use:

```text
Aegon-RAG-Ingestion/1.0 (+https://github.com/jeelan-ds786/aegon-intelligence)
```

`CrawlConfig` controls per-host pacing, retry count, redirect count, timeout, response-byte limit,
and minimum extracted-content length. Retries cover HTTP 429 and transient 5xx responses plus
socket and timeout failures. `Retry-After` is honored up to 60 seconds. Page responses must be HTML
or XHTML, decode with their declared charset, remain within the byte limit, and pass main-content
extraction quality checks. Sitemap fetches use the same URL, robots, SSRF, pacing, retry, redirect,
and size policies.

## Host configuration

Hosts declare sources in `aegon.config.json`. The authoritative JSON Schema is
`workers/ingestion/schemas/aegon.config.schema.json`; the Pydantic model rejects unknown fields.

```json
{
  "$schema": "./workers/ingestion/schemas/aegon.config.schema.json",
  "schema_version": "1",
  "markdown": [
    {
      "pattern": "docs/**/*.md",
      "required_frontmatter": ["title", "visibility"]
    },
    {
      "pattern": "docs/**/*.mdx",
      "required_frontmatter": ["title", "visibility"]
    }
  ],
  "json_export": {
    "path": "exports/help-center.json",
    "fields": [
      {"name": "body", "weight": 2.0},
      {"name": "keywords", "weight": 0.5}
    ]
  },
  "sitemap_url": "https://docs.example.com/sitemap.xml",
  "maximum_sitemap_urls": 1000
}
```

Paths and globs are relative to the configured host root. Parent traversal, absolute paths,
symlink escapes, empty glob results, non-Markdown glob matches, malformed UTF-8, and missing
required frontmatter fail the entire load. Frontmatter validation checks top-level key presence;
the Markdown parser continues to omit frontmatter from retrieval text. JSON exports retain their
versioned parser schema and configured field weights.

Only sitemap `urlset` documents are accepted. DTDs, entities, duplicate URLs, malformed XML, and
URL counts above `maximum_sitemap_urls` are rejected. Every listed page still passes normal crawler
policy; sitemap membership is not approval by itself.

## Incremental replacement

Pass the prior complete `source_id -> SourceInput` snapshot to `HostSourceLoader.load`. Local files
whose exact-byte SHA-256 checksum and URI are unchanged reuse the prior validated input. Web pages
are classified after bounded acquisition. `AcquisitionPlan` contains:

- `sources`: the complete new snapshot to pass to `GovernedIngestionPipeline.run`
- `changed_source_ids`: new or changed sources
- `unchanged_source_ids`: checksum-identical reused sources
- `deleted_source_ids`: prior source IDs absent from the new snapshot

The governed pipeline receives `plan.sources`, reuses signed chunks for unchanged checksums, and
publishes atomically. The PostgreSQL repository marks prior sources omitted from that complete
snapshot as deleted during the same replacement transaction. `deleted_source_ids` is provided for
audit and telemetry; callers must not perform a separate pre-publish delete.

## Security tests

Tests cover robots denial, redirects, 429 and network retries, pacing, response policy, and a local
fake server. The SSRF matrix includes loopback, RFC 1918, link-local metadata
`169.254.169.254`, IPv6 loopback, mixed DNS answers, and a rebinding simulation where the first
resolution is public and the next is private.