# Ingestion signing

The ingestion worker signs every chunk and every run manifest with Ed25519. Signatures provide
tamper evidence; they do not grant source approval. Tenant isolation and source approval remain
database policy decisions enforced by RLS and the governed pipeline.

## Chunk envelope

`sign_chunk()` computes:

- `content_sha256 = SHA-256(UTF-8(content))`
- `embedding_sha256 = SHA-256(concat(binary64-big-endian(component)))`
- an RFC 8785 JSON Canonicalization Scheme (JCS) payload containing `tenant_id`, `source_id`,
  `chunk_id`, both digests, `embedding_model`, and `embedding_version`
- an Ed25519 signature over the canonical payload

The stored envelope includes `signer_key_id`. Retrieval must call `verify_chunk()` before using a
chunk. Verification selects the public key by that exact key-version identifier, recomputes both
digests, and verifies the signature. Missing key IDs, changed content, changed vectors, changed
metadata, and changed signatures fail closed.

## Keys

`LocalSigner` accepts a 32-byte Ed25519 private seed for development and CI. Load that seed from a
local secret source; never commit it or log it. It is not a production control.

`KmsSigner` delegates to a Cloud KMS asymmetric signing key version. Configure an Ed25519 key and
use its immutable version resource name as `key_id`, not a key alias. The adapter sends and checks
CRC32C values for both request data and the returned signature. Authentication uses the client
library's Application Default Credentials; no credential enters the signer API.

For rotation, add the new public key to the verification keyring before changing the active signer.
Keep old public keys while any old chunk or manifest can be retrieved. Removing an old key makes
records signed by that key unverifiable by design.

## Manifests

Each successful run stores canonical manifest bytes, its signature, and key ID in
`ingestion_manifests`. The signed payload binds the run ID, tenant ID, sorted source checksums,
sorted chunk IDs, and signer key ID. Use `verify_manifest()` when reading or exporting a manifest.
Object storage replication may retain the same canonical bytes as an immutable audit copy; the
database row is the publication record.

## Governance and publication

An allowlisted source is an existing `sources` row in `approved` status whose URI still matches.
Unknown sources and changed URIs are written to `quarantine`; their proposed source ID and URI are
stored in the payload and they are excluded from publication. A reviewer must explicitly create or
update the source and set `status = 'approved'` before a later run can publish it.

For approved sources, unchanged checksums reuse live chunks only after signature verification.
Changed checksums run through chunking, configured PII redaction, embedding, signing, and local
verification. Omitting a formerly approved source from a complete run marks it deleted and removes
its chunks.

`PostgresIngestionRepository.publish()` performs these operations in one transaction:

1. Populate `chunk_staging` for the run.
2. Update approved source checksums.
3. Delete and replace the tenant's live chunks from staging.
4. Mark omitted approved sources deleted.
5. Store the signed manifest and clear staging rows.

Any parse, validation, embedding, signing, verification, constraint, or database failure prevents
publication. A transaction failure rolls back source changes, the live index, and the manifest.
Quarantine decisions are independent governance records and can persist without publishing an
index.

Never log source content, embeddings, signatures, database URLs, private key bytes, or provider
responses. Operational logs should contain only tenant-safe opaque IDs, run IDs, counts, status,
and stable error codes.