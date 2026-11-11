SET LOCAL ROLE aegon_migrator;

CREATE TABLE api_sessions (
    token_hash bytea PRIMARY KEY CHECK (octet_length(token_hash) = 32),
    csrf_hash bytea NOT NULL CHECK (octet_length(csrf_hash) = 32),
    tenant_id text NOT NULL REFERENCES tenants (id) ON DELETE CASCADE,
    scopes text[] NOT NULL CHECK (cardinality(scopes) BETWEEN 1 AND 32),
    expires_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

CREATE INDEX api_sessions_expiry_idx ON api_sessions (expires_at);

REVOKE ALL ON api_sessions FROM PUBLIC;
REVOKE ALL ON api_sessions FROM aegon_app;

CREATE FUNCTION api_session_create(
    session_token_hash bytea,
    session_csrf_hash bytea,
    session_tenant_id text,
    session_scopes text[],
    session_expires_at timestamptz
) RETURNS void
LANGUAGE sql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
    DELETE FROM public.api_sessions WHERE expires_at <= clock_timestamp();
    INSERT INTO public.api_sessions (token_hash, csrf_hash, tenant_id, scopes, expires_at)
    VALUES (
        session_token_hash,
        session_csrf_hash,
        session_tenant_id,
        session_scopes,
        session_expires_at
    );
$$;

CREATE FUNCTION api_session_get(session_token_hash bytea)
RETURNS TABLE (csrf_hash bytea, tenant_id text, scopes text[], expires_at timestamptz)
LANGUAGE sql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
    SELECT sessions.csrf_hash, sessions.tenant_id, sessions.scopes, sessions.expires_at
    FROM public.api_sessions AS sessions
    WHERE sessions.token_hash = session_token_hash
      AND sessions.expires_at > clock_timestamp();
$$;

CREATE FUNCTION api_session_revoke(session_token_hash bytea) RETURNS void
LANGUAGE sql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
    DELETE FROM public.api_sessions WHERE token_hash = session_token_hash;
$$;

REVOKE ALL ON FUNCTION api_session_create(bytea, bytea, text, text[], timestamptz) FROM PUBLIC;
REVOKE ALL ON FUNCTION api_session_get(bytea) FROM PUBLIC;
REVOKE ALL ON FUNCTION api_session_revoke(bytea) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION api_session_create(bytea, bytea, text, text[], timestamptz) TO aegon_app;
GRANT EXECUTE ON FUNCTION api_session_get(bytea) TO aegon_app;
GRANT EXECUTE ON FUNCTION api_session_revoke(bytea) TO aegon_app;

RESET ROLE;
