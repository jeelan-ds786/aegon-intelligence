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
from aegon_ingestion.crawl import (
    USER_AGENT,
    CrawlConfig,
    CrawlError,
    ResponseRejectedError,
    RobotsDeniedError,
    SafeCrawler,
    SsrfBlockedError,
)
from aegon_ingestion.documents import DocumentBlock, ParsedDocument
from aegon_ingestion.host_sources import (
    AcquisitionPlan,
    AegonConfig,
    HostSourceLoader,
    JsonExportConfig,
    JsonFieldConfig,
    MarkdownGlob,
    config_json_schema,
)
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
    "AcquisitionPlan",
    "AegonConfig",
    "Chunk",
    "Chunker",
    "ChunkingConfig",
    "CrawlConfig",
    "CrawlError",
    "DocumentBlock",
    "GovernedIngestionPipeline",
    "HostSourceLoader",
    "IngestionRepository",
    "JsonExportConfig",
    "JsonFieldConfig",
    "KmsSigner",
    "LocalSigner",
    "MarkdownGlob",
    "OversizedAtomicBlockError",
    "ParsedDocument",
    "PiiRedactor",
    "PostgresIngestionRepository",
    "PublishResult",
    "ResponseRejectedError",
    "RobotsDeniedError",
    "SafeCrawler",
    "SignedChunk",
    "SignedManifest",
    "Signer",
    "SourceDisposition",
    "SourceInput",
    "SsrfBlockedError",
    "Tokenizer",
    "USER_AGENT",
    "WhitespaceTokenizer",
    "__version__",
    "sign_chunk",
    "sign_manifest",
    "config_json_schema",
    "verify_chunk",
    "verify_manifest",
]
