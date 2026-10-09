# Parsing and chunking

The ingestion worker converts supported source formats into a common sequence of immutable,
source-aligned blocks before chunking. Parsing and chunking are deterministic and perform no
network access.

## Parser behavior

- Markdown preserves heading levels, fenced code, and pipe tables. YAML front matter is metadata
  rather than retrieval text and is omitted.
- MDX uses the Markdown parser after removing imports, executable exports, JSX tags, and JSX
  expressions. String-valued `export const` declarations remain as retrieval text.
- HTML prefers `<main>`, then `<article>`, then `<body>`. Navigation, headers, footers, forms,
  asides, scripts, styles, and elements whose class, ID, or role identifies common page chrome are
  omitted. `<pre>` whitespace and table rows remain intact.
- JSON exports must match schema version `1`. Callers explicitly allowlist fields and assign each a
  positive weight; unknown record properties fail validation and unconfigured fields are omitted.

Every block includes a source locator. Line ranges locate Markdown and MDX, semantic element
positions locate HTML, and JSON Pointers locate export fields.

## Chunking policy

Headings open sections and build a hierarchical `heading_path`. Chunks never overlap across a
section boundary. The default policy targets 400 tokens, permits a configurable 300-500 token
target, applies 10-15% prose overlap (12% by default), and enforces an 800-token hard maximum.
Short section tails may be below the target.

Code blocks and tables are atomic. A block over 800 tokens raises `OversizedAtomicBlockError`
instead of silently damaging executable or tabular structure. Callers should reject the source or
apply a format-aware preprocessing policy before retrying.

`Tokenizer` is a pluggable encode/decode protocol. `WhitespaceTokenizer` is the deterministic,
offline default; deployments should inject the tokenizer paired with their retrieval model.
`PiiRedactor` is an optional policy hook invoked before content hashing. This package intentionally
defines no redaction policy.

IDs use SHA-256 with NUL-delimited inputs:

- `parent_id = hash(source_id, section_locator)`
- `content_sha256 = hash(final_chunk_text)`
- `chunk_id = hash(source_id, chunk_locator, content_sha256)`

The same source, parser, tokenizer, redactor, and chunking configuration therefore produce the same
IDs. Changing redacted content changes the content and chunk IDs.

## Verification

Five golden fixtures live in `workers/ingestion/tests/fixtures`. Property tests remove each chunk's
declared overlap and verify that the remaining token stream exactly reconstructs all parsed source
text in order. Stability tests run the same document twice and compare complete chunk values.

Run the synthetic 10,000-document parse-and-chunk benchmark from the repository root:

```bash
.venv/bin/python workers/ingestion/benchmarks/chunk_10k.py
```

The benchmark reports elapsed time, generated chunks, and documents per second. It is diagnostic,
not a fixed CI threshold, because shared runner performance is not stable enough for a meaningful
absolute gate.