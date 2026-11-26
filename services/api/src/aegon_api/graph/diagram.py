"""Generate the checked-in architecture page from the compiled graph."""

from __future__ import annotations

from pathlib import Path

from aegon_common.embeddings import HashingEmbedder, HashingEmbedderConfig

from aegon_api.graph.builder import build_graph
from aegon_api.graph.dependencies import GraphDependencies
from aegon_api.llm import FakeLlm

_INTRODUCTION = """# Agent graph

The API agent is a checkpointed LangGraph state machine. Identity and retrieval
scope are copied into `AgentState` only by the authenticated server boundary;
browser input cannot provide `tenant_id` or scopes. Raw credentials are never
part of graph state.
"""

_INVARIANTS = """
## Runtime invariants

- The checkpoint `thread_id` is a collision-safe composition of the authenticated
  session ID and conversation ID.
- Production construction uses `AsyncPostgresSaver`; checkpoint schema setup is
  an explicit deployment operation.
- The graph interrupts before `confirm_action`, so a side effect cannot proceed
  without a persisted confirmation boundary.
- Every node runs through the same timeout, budget-accounting, and span-hook
  wrapper. The span hook is intentionally a no-op until OTel instrumentation is
  enabled.
- `recursion_limit` and `max_rewrites` are independent: the first bounds all
  graph steps, while the second is restricted to zero or one retrieval retry.
- Offline and CI construction injects `FakeLlm` and `HashingEmbedder`; it does
  not require cloud credentials.
- Rewrite, rerank, MMR, parent expansion, and sufficiency grading are bounded,
  toggleable stages. Every rewritten query passes `InputGuard`; reranking caps
  50 candidates to 8 evidence items. Retrieval and tools remain topology stubs.
- Generation and verification use schema-constrained LLM calls, strict citation
  checks, and extractive fallback. Guards, refusal, and escalation are executable
  safety behavior.

Regenerate this page with:

```shell
python -m aegon_api.graph.diagram
```
"""


def render_architecture_page() -> str:
    """Render documentation directly from the compiled graph topology."""

    dependencies = GraphDependencies(
        llm=FakeLlm(),
        embedder=HashingEmbedder(HashingEmbedderConfig(dim=8)),
    )
    graph = build_graph(dependencies)
    mermaid = graph.get_graph().draw_mermaid()  # pyright: ignore[reportUnknownMemberType]
    return f"{_INTRODUCTION}\n```mermaid\n{mermaid}```\n{_INVARIANTS}"


def main() -> None:
    repository_root = Path(__file__).resolve().parents[5]
    output = repository_root / "docs" / "architecture" / "graph.md"
    output.write_text(render_architecture_page(), encoding="utf-8")


if __name__ == "__main__":
    main()
