# Aegon-RAG contracts

This directory contains the committed JSON Schema snapshots for the public models in
`aegon_common`. Consumers may use them for validation and client generation. The Python
models remain the source of truth.

## Compatibility

Schema changes are API changes. A normal test run compares Pydantic's generated schemas with
these snapshots and fails when they differ. After reviewing an intentional contract change,
regenerate the snapshots from the repository root:

```shell
UPDATE_CONTRACT_SCHEMAS=1 .venv/bin/pytest \
  packages/python/aegon_common/tests/test_schema_snapshots.py
```

Run the same test again without `UPDATE_CONTRACT_SCHEMAS` before committing. Do not hand-edit
the generated schema files.

## Stable error codes

Error codes are machine-readable and stable. Public messages may change and must not be parsed
for control flow.

| Code | Meaning |
| --- | --- |
| `invalid_arguments` | One or more request values failed validation. |
| `unauthorized` | Authentication is missing or the server-derived identity lacks access. |
| `not_found` | The requested resource does not exist in the caller's authorized scope. |
| `timeout` | Processing exceeded its bounded deadline. |
| `dependency_unavailable` | A required downstream service is temporarily unavailable. |
| `rate_limited` | The caller exceeded an enforced usage limit. |
| `internal_error` | An unexpected internal failure occurred. |
| `refused` | The platform cannot answer safely from approved evidence. |

Adding a code is an additive schema change. Renaming, removing, or changing the meaning of an
existing code is a breaking change.