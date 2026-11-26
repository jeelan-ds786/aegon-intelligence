# ADR 0003: Provider-neutral reranking with bounded MMR

## Status

Accepted

## Context

Hybrid retrieval returns up to 50 verified, tenant-scoped candidates. Passing all candidates
to generation wastes context, repeats near-duplicate chunks, and makes provider choice part of
graph logic. A hosted reranker is useful in the production GCP profile, while offline CI and
local development must not require cloud credentials. Parent-document expansion also needs to
remain behind the same tenant and integrity boundary as retrieval.

## Decision

Use a provider-neutral asynchronous `Reranker` protocol that only receives an already guarded
query and evidence values. Provide:

1. `CrossEncoderReranker`, wrapping an injected local cross-encoder model for data-local
   scoring and framework-independent tests.
2. `GeminiReranker`, using the global schema-constrained `LlmClient` adapter and therefore the
   configured Gemini production profile without fallback.
3. `NoopReranker`, preserving retrieval scores for deterministic CI and ablation.

The graph deduplicates before scoring, caps input at 50, validates scores in `[0, 1]`, applies
MMR for diversity, optionally replaces selected children with approved parents under a token
budget, and emits at most 8 items. Each operation has an explicit timeout. Configuration can
disable reranking, MMR, and parent expansion independently.

## Alternatives considered

- **Always use a hosted reranker:** rejected because CI would require network credentials and
  local outages could become an implicit provider fallback problem.
- **Always use a bundled sentence-transformers dependency:** rejected because model/runtime
  selection and accelerator packaging should remain a deployment-profile concern.
- **Sort only by retrieval score:** retained as `NoopReranker`, but rejected as the production
  default because it cannot model cross-encoder relevance or reduce semantic redundancy.
- **Ask the generation model to select evidence:** rejected because it couples context
  selection to answer generation and sends all 50 candidates into the most expensive stage.

## Consequences

Application composition must explicitly choose and initialize a reranker; the graph does not
fall back between providers. Cross-encoder deployments must supply a model implementation,
and Gemini deployments inherit the global adapter's authentication, timeout, and metering
requirements. MMR adds one bounded embedding call, while the smaller and more diverse context
reduces generation input. New reranker adapters must satisfy the shared deterministic contract
tests and retrieval ablation gate.
