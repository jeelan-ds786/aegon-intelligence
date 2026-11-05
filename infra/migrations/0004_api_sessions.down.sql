SET LOCAL ROLE aegon_migrator;

DROP FUNCTION api_session_revoke(bytea);
DROP FUNCTION api_session_get(bytea);
DROP FUNCTION api_session_create(bytea, bytea, text, text[], timestamptz);
DROP TABLE api_sessions;

RESET ROLE;
