import pytest
from aegon_common.embeddings import EmbeddingVector
from aegon_ingestion.signing import (
    KmsSigner,
    LocalSigner,
    SignedChunk,
    canonicalize,
    sign_chunk,
    sign_manifest,
    verify_chunk,
    verify_manifest,
)
from pydantic import ValidationError


def _signed_chunk(seed: int = 1, key_id: str = "local-1") -> tuple[SignedChunk, LocalSigner]:
    signer = LocalSigner.from_private_bytes(key_id, bytes([seed]) * 32)
    chunk = sign_chunk(
        tenant_id="tenant-a",
        source_id="source-a",
        chunk_id="chunk-a",
        content="approved evidence",
        embedding=EmbeddingVector(values=(0.25, -0.5, 0.75), model_id="hashing", model_version="1"),
        signer=signer,
    )
    return chunk, signer


def test_rfc8785_canonicalization_vector() -> None:
    value = {"string": '€$\u000f\nA\'B"\\"/', "numbers": [333333333.33333329, 1e30, 4.5]}
    assert canonicalize(value) == (
        b'{"numbers":[333333333.3333333,1e+30,4.5],'
        b'"string":"\xe2\x82\xac$\\u000f\\nA\'B\\"\\\\\\"/"}'
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("content", "approved evidencf"),
        ("embedding", (0.25, -0.5, 0.7500000000000001)),
        ("source_id", "source-b"),
        ("embedding_model", "other-model"),
    ],
)
def test_bit_flip_in_signed_material_is_rejected(field: str, value: object) -> None:
    chunk, signer = _signed_chunk()
    tampered = chunk.model_copy(update={field: value})
    assert not verify_chunk(tampered, {signer.key_id: signer.public_key})


def test_key_rotation_keeps_old_signatures_verifiable_by_key_id() -> None:
    old_chunk, old_signer = _signed_chunk(1, "key-2025")
    new_chunk, new_signer = _signed_chunk(2, "key-2026")
    keys = {
        old_signer.key_id: old_signer.public_key,
        new_signer.key_id: new_signer.public_key,
    }
    assert verify_chunk(old_chunk, keys)
    assert verify_chunk(new_chunk, keys)
    assert not verify_chunk(old_chunk, {new_signer.key_id: new_signer.public_key})


def test_manifest_tampering_and_key_id_substitution_are_rejected() -> None:
    signer = LocalSigner.from_private_bytes("key-1", b"1" * 32)
    manifest = sign_manifest("run-1", "tenant-a", {"source-a": "ab" * 32}, ["chunk-a"], signer)
    assert verify_manifest(manifest, {signer.key_id: signer.public_key})
    assert not verify_manifest(
        manifest.model_copy(update={"chunk_ids": ("chunk-b",)}),
        {signer.key_id: signer.public_key},
    )
    assert not verify_manifest(
        manifest.model_copy(update={"signer_key_id": "key-alias"}),
        {"key-alias": signer.public_key},
    )


def test_kms_signer_rejects_unverified_transport_response() -> None:
    class Response:
        signature = b"x" * 64
        signature_crc32c = 0
        verified_data_crc32c = False

    class Client:
        def asymmetric_sign(self, *, request: object) -> object:
            del request
            return Response()

    with pytest.raises(RuntimeError, match="did not verify"):
        KmsSigner("kms-key-version", Client()).sign(b"message")


def test_signature_must_be_ed25519_length() -> None:
    chunk, _ = _signed_chunk()
    with pytest.raises(ValidationError):
        chunk.model_copy(update={"signature": b"short"}).model_validate(
            {**chunk.model_dump(), "signature": b"short"}
        )
