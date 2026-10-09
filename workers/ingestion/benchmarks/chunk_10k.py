import argparse
import json
from time import perf_counter

from aegon_ingestion import Chunker
from aegon_ingestion.parsers import parse_markdown

DOCUMENT = """# Operations guide

Confirm the tenant scope and inspect the current service health before making changes.

## Recovery

Restore the latest verified backup, validate the result, and record the evidence.

| Check | Expected |
| --- | --- |
| Health | Ready |
"""


def run(document_count: int) -> dict[str, float | int]:
    """Measure deterministic parse-and-chunk throughput for synthetic documents."""
    chunker = Chunker()
    chunk_count = 0
    started = perf_counter()
    for index in range(document_count):
        document = parse_markdown(f"benchmark-{index}", DOCUMENT)
        chunk_count += len(chunker.chunk(document))
    elapsed_seconds = perf_counter() - started
    return {
        "documents": document_count,
        "chunks": chunk_count,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "documents_per_second": round(document_count / elapsed_seconds, 2),
    }


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--documents", type=int, default=10_000)
    arguments = argument_parser.parse_args()
    print(json.dumps(run(arguments.documents), sort_keys=True))
