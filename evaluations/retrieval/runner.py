"""Deterministic retrieval evaluation and regression comparison CLI."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

from aegon_common.embeddings import (
    EmbeddingVector,
    GeminiEmbedderConfig,
    HashingEmbedder,
    HashingEmbedderConfig,
    get_embedder,
)

Mode = Literal["dense", "sparse", "hybrid"]
Provider = Literal["hashing", "gemini"]


class EvaluatorEmbedder(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def version(self) -> str: ...

    async def embed_texts(self, texts: Sequence[str]) -> Sequence[EmbeddingVector]: ...


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    source_id: str
    text: str


@dataclass(frozen=True, slots=True)
class Row:
    id: str
    query: str
    gold_chunk_ids: frozenset[str]
    gold_source_ids: frozenset[str]
    difficulty: str
    answerable: bool


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_dataset(directory: Path) -> tuple[list[Chunk], list[Row], str]:
    chunks = [
        Chunk(str(item["chunk_id"]), str(item["source_id"]), str(item["text"]))
        for item in _read_jsonl(directory / "corpus.jsonl")
    ]
    rows = [
        Row(
            id=str(item["id"]),
            query=str(item["query"]),
            gold_chunk_ids=frozenset(str(value) for value in item["gold_chunk_ids"]),  # type: ignore[union-attr]
            gold_source_ids=frozenset(str(value) for value in item["gold_source_ids"]),  # type: ignore[union-attr]
            difficulty=str(item["difficulty"]),
            answerable=bool(item["answerable"]),
        )
        for item in _read_jsonl(directory / "queries.jsonl")
    ]
    if len(rows) < 100:
        raise ValueError("retrieval dataset must contain at least 100 questions")
    manifest = (directory / "MANIFEST.json").read_text(encoding="utf-8")
    expected = str(json.loads(manifest)["queries_sha256"])
    actual = hashlib.sha256((directory / "queries.jsonl").read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("queries.jsonl checksum does not match MANIFEST.json")
    return chunks, rows, hashlib.sha256((directory / "MANIFEST.json").read_bytes()).hexdigest()


def _tokens(text: str) -> set[str]:
    normalized = "".join(char.lower() if char.isalnum() else " " for char in text)
    return set(normalized.split())


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


async def _rank(
    query: str, chunks: list[Chunk], mode: Mode, embedder: EvaluatorEmbedder
) -> list[str]:
    query_vector = (await embedder.embed_texts((query,)))[0].values
    chunk_vectors = await embedder.embed_texts(tuple(chunk.text for chunk in chunks))
    query_tokens = _tokens(query)
    scored: list[tuple[float, str]] = []
    for chunk, vector in zip(chunks, chunk_vectors, strict=True):
        dense = _cosine(query_vector, vector.values)
        overlap = len(query_tokens & _tokens(chunk.text)) / max(len(query_tokens), 1)
        score = dense if mode == "dense" else overlap if mode == "sparse" else dense + overlap
        scored.append((score, chunk.chunk_id))
    return [chunk_id for _, chunk_id in sorted(scored, key=lambda item: (-item[0], item[1]))]


def _percentile(values: list[float], percentile: float) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(percentile) - 1]


def _metrics(
    rows: list[Row], rankings: list[list[str]], latencies: list[float]
) -> dict[str, float]:
    recalls: dict[str, float] = {}
    for k in (1, 5, 10):
        recalls[f"recall_at_{k}"] = (
            100
            * sum(
                bool(row.gold_chunk_ids & set(ranking[:k]))
                for row, ranking in zip(rows, rankings, strict=True)
            )
            / len(rows)
        )
    reciprocal: list[float] = []
    ndcgs: list[float] = []
    for row, ranking in zip(rows, rankings, strict=True):
        rank = next(
            (index for index, chunk_id in enumerate(ranking, 1) if chunk_id in row.gold_chunk_ids),
            None,
        )
        reciprocal.append(1 / rank if rank else 0.0)
        gains = [1 if chunk_id in row.gold_chunk_ids else 0 for chunk_id in ranking[:10]]
        dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
        ideal = sum(1 / math.log2(index + 2) for index in range(min(len(row.gold_chunk_ids), 10)))
        ndcgs.append(dcg / ideal if ideal else 0.0)
    unanswerable = [index for index, row in enumerate(rows) if not row.answerable]
    false_accept = sum(bool(rankings[index][:10]) for index in unanswerable) / max(
        len(unanswerable), 1
    )
    return {
        **recalls,
        "mrr": sum(reciprocal) / len(reciprocal),
        "ndcg_at_10": sum(ndcgs) / len(ndcgs),
        "unanswerable_false_accept_rate": false_accept,
        "p50_latency_ms": _percentile(latencies, 50),
        "p95_latency_ms": _percentile(latencies, 95),
    }


def _build_embedder(provider: Provider) -> EvaluatorEmbedder:
    if provider == "hashing":
        return HashingEmbedder(HashingEmbedderConfig(dim=256, model_version="1"))
    project = os.environ.get("GEMINI_PROJECT")
    if not project:
        raise ValueError("GEMINI_PROJECT is required with --provider gemini")
    return get_embedder(
        GeminiEmbedderConfig(
            project=project,
            model_id=os.environ.get("GEMINI_MODEL", "gemini-embedding-001"),
            model_version=os.environ.get("GEMINI_MODEL_VERSION", "configured"),
            dim=int(os.environ.get("GEMINI_DIM", "3072")),
        )
    )


def evaluate(directory: Path, mode: Mode, provider: Provider) -> dict[str, object]:
    chunks, rows, dataset_hash = load_dataset(directory)
    embedder = _build_embedder(provider)
    rankings: list[list[str]] = []
    latencies: list[float] = []
    for row in rows:
        started = time.perf_counter()
        rankings.append(asyncio.run(_rank(row.query, chunks, mode, embedder)))
        latencies.append((time.perf_counter() - started) * 1000)
    return {
        "schema_version": "1.0",
        "git_sha": _git_sha(),
        "dataset_hash": dataset_hash,
        "dataset_questions": len(rows),
        "embedding_model": embedder.model_id,
        "embedding_version": embedder.version,
        "mode": mode,
        "metrics": _metrics(rows, rankings, latencies),
    }


def compare(current: dict[str, object], baseline: dict[str, object]) -> list[str]:
    current_metrics = cast(dict[str, float], current["metrics"])
    baseline_metrics = cast(dict[str, float], baseline["metrics"])
    failures: list[str] = []
    if float(baseline_metrics["recall_at_5"]) - float(current_metrics["recall_at_5"]) > 2:
        failures.append("Recall@5 dropped by more than 2 percentage points")
    if float(baseline_metrics["mrr"]) - float(current_metrics["mrr"]) > 0.02:
        failures.append("MRR dropped by more than 0.02")
    return failures


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _markdown(reports: list[dict[str, object]]) -> str:
    lines = [
        "# Retrieval evaluation",
        "",
        "| Ablation | R@1 | R@5 | R@10 | MRR | nDCG@10 | False accept | p50 ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        metrics = cast(dict[str, float], report["metrics"])
        lines.append(
            f"| {report['mode']} | {metrics['recall_at_1']:.2f} | "
            f"{metrics['recall_at_5']:.2f} | {metrics['recall_at_10']:.2f} | "
            f"{metrics['mrr']:.4f} | {metrics['ndcg_at_10']:.4f} | "
            f"{metrics['unanswerable_false_accept_rate']:.4f} | "
            f"{metrics['p50_latency_ms']:.2f} | {metrics['p95_latency_ms']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("evaluations/retrieval/dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluations/results"))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--provider", choices=("hashing", "gemini"), default="hashing")
    args = parser.parse_args()
    provider = cast(Provider, args.provider)
    reports = [evaluate(args.dataset, mode, provider) for mode in ("dense", "sparse", "hybrid")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for report in reports:
        (args.output_dir / f"retrieval-{report['mode']}.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    markdown = _markdown(reports)
    (args.output_dir / "retrieval.md").write_text(markdown, encoding="utf-8")
    docs_report = Path("docs/evaluation/retrieval-results.md")
    if docs_report.parent.is_dir():
        docs_report.write_text(markdown, encoding="utf-8")
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        failures = compare(reports[2], baseline)
        if failures:
            print("\n".join(f"FAIL: {failure}" for failure in failures))
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
