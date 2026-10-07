# Embeddings

`aegon_common.embeddings` provides two adapters behind the shared `Embedder` protocol.
Every returned `EmbeddingVector` includes `model_id` and `model_version`; ingestion must persist
both fields so a configured version change can select stale rows for re-embedding.

## Hashing adapter

`HashingEmbedder` is deterministic signed feature hashing for offline development and CI. Text is
case-folded and tokenized before projection. Images are projected from byte chunks. Both outputs
are L2-normalized.

The image output is not semantic and must never be presented as a production image embedding.
The adapter rejects empty and oversized inputs rather than silently changing them.

```python
from aegon_common.embeddings import HashingEmbedderConfig, get_embedder

embedder = get_embedder(HashingEmbedderConfig(dim=768, model_version="ci-v1"))
```

## Vertex AI Gemini adapter

`GeminiEmbedder` uses the official Google Gen AI SDK in Vertex AI mode. It accepts only an
explicit Google Cloud project, location, model ID, model version, and output dimension. The SDK is
not given an API key or credential object: production authentication therefore uses Application
Default Credentials, normally supplied through Workload Identity on Cloud Run.

```python
from aegon_common.embeddings import GeminiEmbedderConfig, get_embedder

embedder = get_embedder(
    GeminiEmbedderConfig(
        project="my-project",
        location="us-central1",
        model_id="configured-vertex-model",
        model_version="published-version-or-deployment-revision",
        dim=768,
    )
)
```

The adapter enforces count and byte limits before each batch, truncates oversized text to
`max_text_chars` only when `truncate_text=True`, and rejects oversized images because truncating
encoded image bytes corrupts the media. Supported image containers are PNG, JPEG, GIF, and WebP.
Provider-side automatic truncation is disabled.

Each call has an application timeout in addition to the SDK HTTP timeout. Retryable timeout,
rate-limit, and provider failures use capped exponential backoff with jitter. Repeated failed
operations open an in-process circuit breaker; one operation is admitted after the recovery
interval. Errors map to the shared `ErrorCode` taxonomy. The implementation does not log prompts,
images, vectors, SDK exceptions, or credentials.

## Latency and cost

Batch limits are deliberately configuration values because Vertex quotas, supported dimensions,
and pricing vary by model and region. Before production rollout:

1. Confirm the selected model's current regional quota, media support, input limits, and billing
   unit in the official Vertex AI model documentation.
2. Load-test representative text and image size distributions. Record p50, p95, and p99 latency,
   retry rate, truncation rate, circuit-open count, and inputs per batch without recording content.
3. Estimate cost from the provider's current billing unit and measured production input volume;
   add retry overhead and the expected re-embedding volume for a model-version migration.
4. Set `max_batch_size`, byte limits, timeout, and retry count from those measurements. Alert on
   latency and error-budget regressions rather than relying on the defaults as an SLA.

Unit tests use injected fakes and make no network calls. A credentialed staging smoke test may be
run separately, but recordings must not contain source content, authorization headers, or signed
URLs.