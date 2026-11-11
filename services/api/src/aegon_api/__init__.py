"""Aegon-RAG API service."""

from aegon_api.app import create_app
from aegon_api.retrieval import (
    Ed25519ChunkVerifier,
    HybridRetriever,
    RetrievalConfig,
    RetrievalResult,
    RetrievalTimings,
    TenantContext,
    reciprocal_rank_fusion,
)

__version__ = "0.0.0"

__all__ = [
    "Ed25519ChunkVerifier",
    "HybridRetriever",
    "RetrievalConfig",
    "RetrievalResult",
    "RetrievalTimings",
    "TenantContext",
    "__version__",
    "create_app",
    "reciprocal_rank_fusion",
]
