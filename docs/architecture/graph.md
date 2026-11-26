# Agent graph

The API agent is a checkpointed LangGraph state machine. Identity and retrieval
scope are copied into `AgentState` only by the authenticated server boundary;
browser input cannot provide `tenant_id` or scopes. Raw credentials are never
part of graph state.

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	input_guard(input_guard)
	modality_detect(modality_detect)
	rewrite(rewrite)
	route(route)
	retrieve(retrieve)
	evidence_guard(evidence_guard)
	rerank(rerank)
	grade(grade)
	tools(tools)
	confirm_action(confirm_action<hr/><small><em>__interrupt = before</em></small>)
	generate(generate)
	verify(verify)
	output_guard(output_guard)
	refuse(refuse)
	escalate(escalate)
	__end__([<p>__end__</p>]):::last
	__start__ --> input_guard;
	confirm_action --> generate;
	evidence_guard -.-> escalate;
	evidence_guard -.-> refuse;
	evidence_guard -.-> rerank;
	generate --> verify;
	grade -.-> escalate;
	grade -.-> generate;
	grade -.-> refuse;
	grade -.-> rewrite;
	input_guard -.-> escalate;
	input_guard -.-> modality_detect;
	input_guard -.-> refuse;
	modality_detect --> rewrite;
	output_guard -.-> __end__;
	output_guard -.-> escalate;
	output_guard -.-> refuse;
	rerank --> grade;
	retrieve --> evidence_guard;
	rewrite --> route;
	route -.-> escalate;
	route -.-> refuse;
	route -.-> retrieve;
	route -.-> tools;
	tools -.-> confirm_action;
	tools -.-> escalate;
	tools -.-> generate;
	tools -.-> refuse;
	verify -.-> escalate;
	verify -.-> output_guard;
	verify -.-> refuse;
	verify -.-> rewrite;
	escalate --> __end__;
	refuse --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

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
  graph steps, while the second explicitly bounds rewrite loops.
- Offline and CI construction injects `FakeLlm` and `HashingEmbedder`; it does
  not require cloud credentials.
- Retrieval and tools remain topology stubs. Generation and verification use
  schema-constrained LLM calls, strict citation checks, and extractive fallback.
  Guards, refusal, and escalation are executable safety behavior.

Regenerate this page with:

```shell
python -m aegon_api.graph.diagram
```
