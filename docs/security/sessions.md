# API sessions

The API exchanges a server-held tenant key for a 30-minute opaque browser session. Call
`POST /v1/sessions` with `Authorization: AegonKey <key>` from an allowed origin. The response
sets two `Secure`, `SameSite=Strict`, host-only cookies:

- `__Host-aegon_session` is `HttpOnly` and carries a random 256-bit opaque token.
- `__Host-aegon_csrf` is browser-readable and carries a separate random 256-bit CSRF token.

The response also returns the CSRF token so a client can send it as `X-CSRF-Token`. Mutating
cookie-authenticated requests must present the CSRF cookie, matching header, and an allowlisted
`Origin`. The API compares credentials in constant time. `DELETE /v1/sessions/current` revokes
the session and expires both cookies.

## Storage and identity

Only SHA-256 hashes of the session and CSRF tokens are stored. Raw tokens, tenant keys, and
database URLs must not be logged. PostgreSQL access is confined to `SECURITY DEFINER` functions;
the runtime `aegon_app` role has no direct table privileges.

Tenant identity and scopes come from the server's `AEGON_TENANT_KEYS` configuration. Browser
headers or bodies cannot select a tenant or add scopes. Each configured entry contains a tenant
ID, the lowercase SHA-256 digest of its key, and its allowed scopes. Rotate a tenant key by
deploying a new digest and then removing the old digest after the overlap period.
Tenant keys must be generated from at least 256 bits of cryptographically secure randomness;
human-chosen keys are vulnerable to offline guessing if configuration is exposed.

Example environment values (the key digest is illustrative, not a usable credential):

```text
AEGON_ENVIRONMENT=production
AEGON_DATABASE_URL=postgresql://...
AEGON_ALLOWED_ORIGINS=["https://app.example.com"]
AEGON_TENANT_KEYS=[{"tenant_id":"tenant-a","key_sha256":"<64 lowercase hex>","scopes":["chat:read"]}]
```

Production startup fails if the database URL is absent, an origin is not HTTPS, or development
demo bearer authentication is enabled. Demo bearer authentication is disabled by default.
Origins must be canonical `http://host[:port]` or `https://host[:port]` values without paths,
queries, fragments, or credentials. Production permits HTTPS only. Scope values are bounded,
lowercase server identifiers such as `chat:read`.

## Run

After configuring the required `AEGON_*` variables and applying migrations, start the API with:

```bash
make serve
```

The `aegon-api` console command starts Uvicorn on `0.0.0.0` and reads the Cloud Run `PORT`
variable, defaulting to `8080`.

The generated OpenAPI contract defines `TenantKey` for session exchange and `SessionCookie` for
cookie-authenticated routes. These declarations are documentation for tooling; runtime checks
continue to validate the `AegonKey` scheme, opaque cookie, origin, and CSRF values.

## Boundary behavior

- `/health` reports process health without touching dependencies.
- `/ready` executes a database check and returns `dependency_unavailable` with HTTP 503 on
  failure.
- Requests larger than `AEGON_MAX_REQUEST_BYTES` are rejected with HTTP 413 before routing.
- Every response carries `X-Request-ID`. A valid caller-provided ID is propagated; invalid IDs
  are replaced.
- Public failures use `{ "error": { "code", "message", "retryable", "request_id" } }`.
- Any supplied `Origin` must match the exact allowlist. Mutating session requests require it.

The in-memory store exists only for dependency-injected tests. Deployments use PostgreSQL and
must apply migration `0004_api_sessions` before starting the API.
