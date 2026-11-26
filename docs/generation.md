# Grounded generation and verification

The `generate` and `verify` graph nodes form a fail-closed faithfulness boundary.
Generation receives only evidence retained by `EvidenceGuard`; each chunk is HTML
escaped and enclosed in a `data-only` delimiter. The generator has no tool
interface and must return schema-constrained atomic answer sentences with
retrieved chunk IDs.

Prompts are packaged under `services/api/src/aegon_api/prompts/` and versioned in
their filenames. Each node records the prompt version and SHA-256 digest in
`AgentState.prompt_traces`, without recording prompt or evidence content in trace
metadata.

## LLM adapters

`LlmClient` is the provider-neutral asynchronous protocol. `GeminiClient` calls
Vertex AI's schema-constrained `generateContent` endpoint, applies bounded
exponential-backoff retries to transient transport failures, and converts
provider token usage into tokens and micro-US dollars. Credentials remain inside
the adapter and are never added to graph state.

`FakeLlm` is deterministic and requires no network or cloud credentials. It
implements the same structured contract for offline development and CI.

`OllamaClient` is an opt-in local-development adapter. It calls Ollama's
non-streaming `/api/chat` endpoint, passes the required JSON schema through
Ollama's `format` field, maps `prompt_eval_count` and `eval_count` into the
shared token budget, and reports zero monetary cost. To prevent accidental
network exposure or production use, its base URL must be a loopback HTTP
origin.

Install and start Ollama on macOS, then pull a local model:

```shell
brew install ollama
ollama serve
ollama pull qwen3:8b
```

Construct `GraphDependencies` with `OllamaClient(model="qwen3:8b")` for local
model testing. CI must continue to inject `FakeLlm`, and production must inject
`GeminiClient`; there is no automatic provider fallback. Ollama integration
depends on the selected model, quantization, local hardware, and Ollama version,
so real-daemon checks must remain opt-in rather than part of deterministic CI.

The opt-in rewrite/grade schema check records the exact Ollama tag and digest in pytest's
test properties:

```shell
AEGON_OLLAMA_MODEL=qwen3:8b \
  pytest services/api/tests/test_ir.py::test_ollama_rewrite_and_grade_schema_compatibility \
  --junitxml=ollama-ir.xml
```

## Verification

The verifier evaluates each generated atomic claim only against its cited
chunks. A claim is retained only when the verifier returns:

1. the expected claim ID;
2. a retrieved chunk ID cited by that claim; and
3. an exact quote present in that chunk.

The exact quote is converted to a half-open `QuoteSpan`. Unsupported claims are
removed and recorded in `unsupported_claims`. If the supported fraction is below
the configured threshold, the model-authored answer is discarded and replaced
with a sentence copied directly from the highest-ranked approved evidence.
Empty evidence, unknown citation IDs, invalid model output, exhausted budgets,
and unavailable grounding all fail closed.

Both model calls use low or zero temperature and bounded output tokens.
Provider-reported usage is added to `tokens_used` and `cost_microusd_used`; a
turn that exceeds either server-owned budget routes to refusal.
