# Migrations

Plain SQL migrations are paired as `NNNN_name.up.sql` and `NNNN_name.down.sql`.
The Python runner applies each file in version order inside a transaction, serializes runners
with an advisory lock, and records the source checksum and embedding dimension in
`schema_migrations`.

## Run

Use an administrative deployment identity only for bootstrap. Runtime services connect as
`aegon_app` or `aegon_ingest`; neither role is a superuser nor has `BYPASSRLS`.

```bash
export DATABASE_URL='postgresql://...'
export EMBEDDING_DIMENSION=768
.venv/bin/python infra/migrations/migrate.py up
.venv/bin/python infra/migrations/migrate.py down --steps 1
```

The dimension is validated as an integer from 1 through 16,000 before replacing the
`{{embedding_dimension}}` SQL token. It cannot change after a migration is applied. Changing
the embedding dimension requires a new expand/backfill/contract migration.

The database administrator supplies role passwords or identity-based login separately through
Secret Manager. Migrations intentionally create no credentials. `aegon_migrator` owns schema
objects and may be assumed by the deployment identity; it is not a runtime role.

## Test

Docker must be available. The integration test uses PostgreSQL 16 with pgvector, applies all
migrations, rolls all migrations back, reapplies them, probes tenant isolation, and snapshots an
HNSW query plan.

```bash
.venv/bin/pytest infra/migrations/tests
```

See [ADR 0001](../../docs/adr/0001-pgvector-rls.md) for ANN and RLS behavior.