# Safety guards

The API safety package provides deterministic, versioned controls for the three content
boundaries that will surround the agent graph. The graph itself is intentionally implemented in
a later task. Callers must run input, retrieved evidence, and generated output through the
corresponding guard before the content crosses its boundary.

These controls reduce known risks; they do not prove that content is safe. The checked-in
evaluation corpus is synthetic and is not a production threat benchmark.

## Contract

Every guard returns:

- `verdict`: `allow`, `redact`, or `block`
- `reasons`: stable machine-readable reason codes, with no matched content
- `redacted_payload`: the safe transformed value, or an empty value for a block
- `guard_version`: the policy/rules version used for the decision

Guards validate payload bounds, never invoke agent tools, and do not log payloads. A blocked
decision may be logged using only the guard version, reason codes, and tenant-safe opaque request
metadata.

## Input

`InputGuard` normalizes Unicode and zero-width characters and inspects plain text plus bounded,
nested URL-encoded and standard or URL-safe Base64 views. Missing Base64 padding is restored
before strict decoding. Rule matches cover direct instruction override, prompt extraction,
jailbreak phrases, policy bypasses, tool-call hijacking, and selected multilingual forms. The
regulated-intent screen blocks narrow high-risk requests such as diagnosis/prescription,
guaranteed investment returns, and KYC/AML evasion.

Direct identifiers (email addresses, US Social Security numbers, and phone-number-shaped values)
are replaced with `[REDACTED]`. This is a deliberately bounded pattern redactor, not a claim of
complete PII discovery.

An optional `ManagedScreeningService` can add provider screening. Rule blocks short-circuit that
dependency. Provider policy identifiers do not enter the public guard reasons. Adapter errors and
timeouts propagate so callers cannot mistake a screening failure for an allow decision.

## Evidence

`EvidenceGuard` removes complete evidence items containing instruction override text,
role-impersonation markers, tool/function-call syntax, or common hidden-text markers. It never
partially edits cited text because doing so would invalidate quote offsets. Retained evidence is
HTML-escaped and wrapped in an explicit `data-only` block for prompt assembly:

```text
<evidence data-only="true" chunk-id="opaque-id">
escaped source text
</evidence>
```

Callers must use `GuardedEvidence.prompt_block` when constructing prompts and retain the original
`Evidence` only for citations. If every item is removed, the verdict is `block`; the agent must
return an insufficient-evidence response rather than answer from model memory.

## Output

`OutputGuard` blocks configured canary values, common provider/repository/private-key shapes, and
explicit claims that policy or hidden prompts were bypassed or disclosed. It redacts the same
bounded direct-identifier patterns as the input guard. Canary values must come from deployment
configuration or secret storage and must never be committed or logged.

## Evaluation

`evaluations/safety` contains schema-versioned, checksummed JSONL with 200 synthetic attack cases
and 200 synthetic benign cases. It includes multilingual and encoded variants. Run:

```bash
make evaluate-safety
```

The command writes `evaluations/results/safety.json` and fails unless all gates pass:

- injection block rate is at least `0.98`
- benign block false-positive rate is at most `0.02`
- deterministic rule p99 latency is below `10 ms`

Add a regression case before changing a rule for a newly observed unsafe behavior. Never add
customer content, credentials, real personal data, or undisclosed provider responses to the
corpus.
