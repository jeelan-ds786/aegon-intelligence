# ADR 0001: pgvector ANN under forced tenant RLS

- Status: Accepted
- Date: 2026-10-06

## Context

Retrieval needs cosine approximate nearest-neighbor search over tenant-scoped chunks. Tenant
identity is server-derived and stored in the transaction or session setting `app.tenant_id`.
PostgreSQL row-level security is the final isolation boundary, including when a caller submits a
crafted predicate.

pgvector's HNSW index is global: it orders candidate vectors before ordinary SQL filters,
including RLS predicates, are applied. Selective tenant or approved-source filters can therefore
remove ANN candidates and return fewer than the requested limit. A btree index cannot be combined
with HNSW to make one tenant-partitioned ANN index.

## Decision

All tenant-bearing tables use `ENABLE ROW LEVEL SECURITY` and `FORCE ROW LEVEL SECURITY`.
Policies compare their tenant column with `current_setting('app.tenant_id')`; a missing setting
fails closed with an error. The application role can read only approved sources and their chunks.
The ingestion role can mutate rows only for its current tenant. Both roles are explicitly created
with `NOBYPASSRLS`.

Chunks use one cosine HNSW index with `m = 16` and `ef_construction = 64`. These pgvector defaults
are the initial balance between graph memory, build cost, and recall. They must be changed only
after representative recall and latency benchmarks. Exact access paths lead with `tenant_id`, and
partial btree indexes cover approved sources and pending quarantine work.

Retrieval transactions on pgvector 0.8 or later should enable iterative scans and tune search
breadth locally:

```sql
BEGIN;
SET LOCAL app.tenant_id = 'server-derived-tenant';
SET LOCAL hnsw.iterative_scan = 'strict_order';
SET LOCAL hnsw.ef_search = 100;

SELECT id, content, embedding <=> $1::vector AS distance
FROM chunks
ORDER BY embedding <=> $1::vector
LIMIT $2;
COMMIT;
```

`strict_order` preserves exact distance ordering while expanding the HNSW scan until enough rows
survive RLS and other filters or scan limits are reached. `relaxed_order` may improve recall and
latency but requires an outer materialized CTE sorted by distance when strict ordering matters.
Production tuning must monitor filtered result counts, recall, latency, and
`hnsw.max_scan_tuples`; raising `ef_search` alone does not guarantee enough post-filter rows.

## Consequences

- Database policy protects tenant isolation independently of caller-provided SQL predicates.
- The schema owner remains subject to RLS because policies are forced, but runtime roles still may
  not own tables or receive elevated role membership.
- Highly selective tenants can require more ANN traversal and have variable latency.
- If benchmarks show unacceptable cross-tenant HNSW filtering, a future migration may partition
  chunks by a bounded tenant grouping. Per-tenant partitions are not the default because tenant
  cardinality is unbounded and operational overhead would grow with it.
- Query-plan tests disable sequential scans and explicit sorts only to prove that the HNSW path is
  available on a tiny fixture. Production planning remains cost based.