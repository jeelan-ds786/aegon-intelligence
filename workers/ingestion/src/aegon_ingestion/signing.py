"""Canonical Ed25519 signatures for chunks and ingestion manifests."""

from __future__ import annotations

import base64
import hashlib
import importlib
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import rfc8785
from aegon_common.embeddings import EmbeddingVector
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from pydantic import BaseModel, ConfigDict, Field

type JsonValue = None | bool | int | float | str | Sequence["JsonValue"] | Mapping[str, "JsonValue"]


class Signer(Protocol):
    """Signing boundary implemented locally or by a managed key service."""

    @property
    def key_id(self) -> str: ...

    def sign(self, message: bytes) -> bytes: ...


class KmsSigningClient(Protocol):
    """Minimal synchronous Cloud KMS client surface used by ``KmsSigner``."""

    def asymmetric_sign(self, *, request: Mapping[str, object]) -> object: ...


class _Crc32cModule(Protocol):
    def value(self, data: bytes) -> int: ...


class _CanonicalDumps(Protocol):
    def __call__(self, value: JsonValue) -> bytes: ...


class _SigningMaterial(Protocol):
    def sign(self, data: bytes) -> bytes: ...

    def public_key(self) -> Ed25519PublicKey: ...


@dataclass(frozen=True, slots=True)
class LocalSigner:
    """In-process Ed25519 signer for local development and CI."""

    key_id: str
    _material: _SigningMaterial

    @classmethod
    def from_private_bytes(cls, key_id: str, seed_bytes: bytes) -> LocalSigner:
        """Create a signer from a 32-byte Ed25519 private key seed."""
        return cls(key_id=key_id, _material=Ed25519PrivateKey.from_private_bytes(seed_bytes))

    def sign(self, message: bytes) -> bytes:
        return self._material.sign(message)

    @property
    def public_key(self) -> Ed25519PublicKey:
        """Return the verification key without exposing private material."""
        return self._material.public_key()


@dataclass(frozen=True, slots=True)
class KmsSigner:
    """Ed25519 signer backed by a Cloud KMS asymmetric signing key version."""

    key_id: str
    client: KmsSigningClient

    def sign(self, message: bytes) -> bytes:
        checksum = _crc32c(message)
        response = self.client.asymmetric_sign(
            request={"name": self.key_id, "data": message, "data_crc32c": checksum}
        )
        signature = getattr(response, "signature", None)
        if not isinstance(signature, bytes):
            raise RuntimeError("KMS returned an invalid signature")
        if getattr(response, "verified_data_crc32c", False) is not True:
            raise RuntimeError("KMS did not verify the request checksum")
        if getattr(response, "signature_crc32c", None) != _crc32c(signature):
            raise RuntimeError("KMS returned a signature checksum mismatch")
        return signature


class SignedChunk(BaseModel):
    """Chunk material whose content, vector, provenance, and model identity are signed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(min_length=1, max_length=128)
    source_id: str = Field(min_length=1, max_length=128)
    chunk_id: str = Field(min_length=1, max_length=128)
    content: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding: tuple[float, ...] = Field(min_length=1)
    embedding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding_model: str = Field(min_length=1)
    embedding_version: str = Field(min_length=1)
    signer_key_id: str = Field(min_length=1)
    signature: bytes = Field(min_length=64, max_length=64)


class SignedManifest(BaseModel):
    """Signed, immutable summary of one ingestion run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1, max_length=128)
    tenant_id: str = Field(min_length=1, max_length=128)
    source_checksums: dict[str, str]
    chunk_ids: tuple[str, ...]
    signer_key_id: str = Field(min_length=1)
    signature: bytes = Field(min_length=64, max_length=64)


def canonicalize(value: JsonValue) -> bytes:
    """Serialize a JSON-compatible value using RFC 8785 JCS."""
    dumps = cast(_CanonicalDumps, rfc8785.dumps)
    return dumps(value)


def embedding_sha256(values: Sequence[float]) -> str:
    """Hash IEEE-754 binary64 values with an explicit network byte order."""
    encoded = b"".join(struct.pack(">d", value) for value in values)
    return hashlib.sha256(encoded).hexdigest()


def sign_chunk(
    *,
    tenant_id: str,
    source_id: str,
    chunk_id: str,
    content: str,
    embedding: EmbeddingVector,
    signer: Signer,
) -> SignedChunk:
    """Validate and sign one embedded chunk."""
    content_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    vector_digest = embedding_sha256(embedding.values)
    payload = _chunk_payload(
        tenant_id=tenant_id,
        source_id=source_id,
        chunk_id=chunk_id,
        content_sha256=content_digest,
        embedding_sha256=vector_digest,
        embedding_model=embedding.model_id,
        embedding_version=embedding.model_version,
    )
    return SignedChunk(
        tenant_id=tenant_id,
        source_id=source_id,
        chunk_id=chunk_id,
        content=content,
        content_sha256=content_digest,
        embedding=embedding.values,
        embedding_sha256=vector_digest,
        embedding_model=embedding.model_id,
        embedding_version=embedding.model_version,
        signer_key_id=signer.key_id,
        signature=signer.sign(canonicalize(payload)),
    )


def verify_chunk(chunk: SignedChunk, keys: Mapping[str, Ed25519PublicKey]) -> bool:
    """Verify all chunk material using the key selected by its embedded key id."""
    key = keys.get(chunk.signer_key_id)
    if key is None:
        return False
    if hashlib.sha256(chunk.content.encode("utf-8")).hexdigest() != chunk.content_sha256:
        return False
    if embedding_sha256(chunk.embedding) != chunk.embedding_sha256:
        return False
    payload = _chunk_payload(
        tenant_id=chunk.tenant_id,
        source_id=chunk.source_id,
        chunk_id=chunk.chunk_id,
        content_sha256=chunk.content_sha256,
        embedding_sha256=chunk.embedding_sha256,
        embedding_model=chunk.embedding_model,
        embedding_version=chunk.embedding_version,
    )
    try:
        key.verify(chunk.signature, canonicalize(payload))
    except InvalidSignature:
        return False
    return True


def sign_manifest(
    run_id: str,
    tenant_id: str,
    source_checksums: Mapping[str, str],
    chunk_ids: Sequence[str],
    signer: Signer,
) -> SignedManifest:
    """Create a deterministic signed run manifest."""
    checksums = dict(sorted(source_checksums.items()))
    ordered_chunk_ids = tuple(sorted(chunk_ids))
    payload = {
        "chunk_ids": list(ordered_chunk_ids),
        "run_id": run_id,
        "signer_key_id": signer.key_id,
        "source_checksums": checksums,
        "tenant_id": tenant_id,
    }
    return SignedManifest(
        run_id=run_id,
        tenant_id=tenant_id,
        source_checksums=checksums,
        chunk_ids=ordered_chunk_ids,
        signer_key_id=signer.key_id,
        signature=signer.sign(canonicalize(payload)),
    )


def verify_manifest(manifest: SignedManifest, keys: Mapping[str, Ed25519PublicKey]) -> bool:
    """Verify a run manifest using its embedded rotation key id."""
    key = keys.get(manifest.signer_key_id)
    if key is None:
        return False
    payload = {
        "chunk_ids": list(manifest.chunk_ids),
        "run_id": manifest.run_id,
        "signer_key_id": manifest.signer_key_id,
        "source_checksums": manifest.source_checksums,
        "tenant_id": manifest.tenant_id,
    }
    try:
        key.verify(manifest.signature, canonicalize(payload))
    except InvalidSignature:
        return False
    return True


def manifest_json(manifest: SignedManifest) -> bytes:
    """Serialize a signed manifest as canonical JSON for immutable storage."""
    value = {
        "chunk_ids": list(manifest.chunk_ids),
        "run_id": manifest.run_id,
        "signature": base64.b64encode(manifest.signature).decode("ascii"),
        "signer_key_id": manifest.signer_key_id,
        "source_checksums": manifest.source_checksums,
        "tenant_id": manifest.tenant_id,
    }
    return canonicalize(value)


def _chunk_payload(
    *,
    tenant_id: str,
    source_id: str,
    chunk_id: str,
    content_sha256: str,
    embedding_sha256: str,
    embedding_model: str,
    embedding_version: str,
) -> dict[str, str]:
    return {
        "chunk_id": chunk_id,
        "content_sha256": content_sha256,
        "embedding_model": embedding_model,
        "embedding_sha256": embedding_sha256,
        "embedding_version": embedding_version,
        "source_id": source_id,
        "tenant_id": tenant_id,
    }


def _crc32c(data: bytes) -> int:
    try:
        module = importlib.import_module("google_crc32c")
    except ImportError as error:
        raise RuntimeError("KmsSigner requires google-crc32c") from error
    return cast(_Crc32cModule, module).value(data)
