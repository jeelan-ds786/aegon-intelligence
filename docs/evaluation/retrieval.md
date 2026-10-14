# Retrieval evaluation

The versioned evaluation set is a small, public-corpus benchmark derived from the Python 3
tutorial and standard-library documentation. It contains 100 questions (80 answerable and 20
unanswerable), gold chunk/source IDs, difficulty labels, and a checksum manifest:
[dataset](../../evaluations/retrieval/dataset/), [schema](../../evaluations/retrieval/schema.json),
and [labeling guide](../../evaluations/retrieval/LABELING.md).

## Schema and sample rows

The machine-readable contract is [schema.json](../../evaluations/retrieval/schema.json). Rows
are JSON Lines, for example:

```json
{"id":"q-001","query":"What does the documentation say about indentation and compound statements?","gold_chunk_ids":["python-docs-01"],"gold_source_ids":["python-docs-tutorial"],"difficulty":"easy","answerable":true}
{"id":"q-081","query":"Which Python documentation rule covers this unrelated topic 0?","gold_chunk_ids":[],"gold_source_ids":[],"difficulty":"hard","answerable":false}
```

## Run locally

```bash
make evaluate-retrieval
```

This runs the deterministic `signed-feature-hashing` `HashingEmbedder` without network access
or cloud credentials. JSON reports and the generated ablation table are written to
`evaluations/results/` and `docs/evaluation/retrieval-results.md`. The runner records the git
SHA, dataset manifest hash, embedding model and version, Recall@1/5/10, MRR, nDCG@10,
unanswerable-query false-accept rate, and p50/p95 latency. The dense-only, sparse-only, and
hybrid modes are all run in one invocation.

The command compares hybrid metrics to the committed baseline and exits non-zero when Recall@5
drops by more than two percentage points or MRR drops by more than 0.02. The committed baseline
is intentionally offline and reproducible; an optional Gemini job may use the same runner with
a Vertex adapter, but its result must be stored separately and must never replace the CI
baseline.

An operator-enabled job can run the optional provider without changing the deterministic gate:

```yaml
- name: Optional Gemini retrieval evaluation
  if: ${{ vars.ENABLE_GEMINI_EVAL == 'true' }}
  env:
    GEMINI_PROJECT: ${{ vars.GEMINI_PROJECT }}
    GOOGLE_APPLICATION_CREDENTIALS: ${{ secrets.GEMINI_WIF_CREDENTIALS }}
  run: |
    uv run python -m evaluations.retrieval.runner \
      --provider gemini \
      --dataset evaluations/retrieval/dataset \
      --output-dir evaluations/results/gemini
```

The optional job requires the repository's normal workload-identity setup and must not run on
untrusted pull requests. Gemini output is an observation report, not a regression baseline.

## Dataset changes

Update the JSONL rows and then recompute `queries_sha256` in `MANIFEST.json`. Dataset changes
are versioned and require a fresh baseline report plus a labeling-guide update. Never add
private, tenant-scoped, or synthetic documents while claiming public-corpus coverage.
