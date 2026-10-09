# ADR 0002: RFC 8785 canonical signed ingestion records

## Status

Accepted

## Context

Chunk and manifest signatures must verify across local Python, Cloud KMS, future retrieval services,
and key rotations. Ordinary JSON serialization does not define object-key order, number rendering,
or escaping tightly enough to be a cryptographic wire format. Signing content alone would also
leave tenant provenance, model identity, and embedding vectors mutable.

## Decision

Use RFC 8785 JSON Canonicalization Scheme (JCS), implemented by the pinned `rfc8785` package, as the
bytes signed with Ed25519. Do not maintain a project-specific JSON canonicalizer.

The chunk payload contains:

```json
{
  "chunk_id": "...",
  "content_sha256": "...",
  "embedding_model": "...",
  "embedding_sha256": "...",
  "embedding_version": "...",
  "source_id": "...",
  "tenant_id": "..."
}
```

`embedding_sha256` extends the minimum metadata payload because the acceptance policy requires an
embedding bit flip to fail verification. Vector components are encoded as ordered IEEE-754 binary64
values in network byte order before SHA-256, avoiding dependence on JSON floating-point rendering.
The content itself and vector itself are stored outside the signed payload and are bound by their
digests.

Manifest payloads use JCS and bind the immutable key-version ID in addition to run, tenant, source
checksum, and chunk membership data. Signatures and key IDs are stored alongside each record to
support rotation. Canonical test vectors and tamper tests are versioned with the implementation.

## Consequences

- Every producer and verifier must implement RFC 8785 exactly and preserve the defined field names.
- Adding, removing, or changing a signed field is a wire-format change and requires a new signature
  format version or a coordinated migration.
- Non-finite vector values cannot be represented safely and must be rejected by the embedding
  boundary.
- Old signatures remain verifiable while their public key-version IDs remain in the keyring.
- Ed25519 signatures prove possession of a signing key, not source approval or tenant authorization;
  those controls remain separate.