"""Aegon-RAG ingestion worker."""

__version__ = "0.0.0"

from aegon_ingestion.chunker import (
    Chunk,
    Chunker,
    ChunkingConfig,
    OversizedAtomicBlockError,
    PiiRedactor,
    Tokenizer,
    WhitespaceTokenizer,
)
from aegon_ingestion.documents import DocumentBlock, ParsedDocument
from aegon_ingestion.pipeline import (
    GovernedIngestionPipeline,
    IngestionRepository,
    PublishResult,
    SourceDisposition,
    SourceInput,
)
from aegon_ingestion.postgres import PostgresIngestionRepository
from aegon_ingestion.signing import (
    KmsSigner,
    LocalSigner,
    SignedChunk,
    SignedManifest,
    Signer,
    sign_chunk,
    sign_manifest,
    verify_chunk,
    verify_manifest,
)

__all__ = [
    "Chunk",
    "Chunker",
    "ChunkingConfig",
    "DocumentBlock",
    "GovernedIngestionPipeline",
    "IngestionRepository",
    "KmsSigner",
    "LocalSigner",
    "OversizedAtomicBlockError",
    "ParsedDocument",
    "PiiRedactor",
    "PostgresIngestionRepository",
    "PublishResult",
    "SignedChunk",
    "SignedManifest",
    "Signer",
    "SourceDisposition",
    "SourceInput",
    "Tokenizer",
    "WhitespaceTokenizer",
    "__version__",
    "sign_chunk",
    "sign_manifest",
    "verify_chunk",
    "verify_manifest",
]
