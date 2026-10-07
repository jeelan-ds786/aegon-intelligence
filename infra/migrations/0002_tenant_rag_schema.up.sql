SET LOCAL ROLE aegon_migrator;

CREATE TABLE tenants (
    id text PRIMARY KEY CHECK (length(id) BETWEEN 1 AND 128),
    name text NOT NULL CHECK (length(name) BETWEEN 1 AND 512),
    status text NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'suspended', 'disabled')),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE tenant_keys (
    key_hash bytea PRIMARY KEY CHECK (octet_length(key_hash) = 32),
    tenant_id text NOT NULL REFERENCES tenants (id) ON DELETE CASCADE,
    scopes text[] NOT NULL DEFAULT '{}',
    expires_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    CHECK (array_position(scopes, NULL) IS NULL)
);

CREATE TABLE sources (
    id text NOT NULL CHECK (length(id) BETWEEN 1 AND 128),
    tenant_id text NOT NULL REFERENCES tenants (id) ON DELETE CASCADE,
    uri text NOT NULL CHECK (length(uri) BETWEEN 1 AND 2048),
    kind text NOT NULL CHECK (length(kind) BETWEEN 1 AND 64),
    checksum bytea NOT NULL CHECK (octet_length(checksum) = 32),
    status text NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'approved', 'rejected', 'deleted')),
    approved_by text,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (id),
    UNIQUE (tenant_id, id),
    UNIQUE (tenant_id, uri)
);

CREATE TABLE chunks (
    id text NOT NULL CHECK (length(id) BETWEEN 1 AND 128),
    tenant_id text NOT NULL REFERENCES tenants (id) ON DELETE CASCADE,
    source_id text NOT NULL,
    modality text NOT NULL
        CHECK (modality IN ('text', 'image', 'pdf_page', 'table',
                            'audio_segment', 'video_segment')),
    content text NOT NULL CHECK (length(content) BETWEEN 1 AND 100000),
    content_sha256 bytea NOT NULL CHECK (octet_length(content_sha256) = 32),
    signature bytea NOT NULL CHECK (octet_length(signature) > 0),
    embedding vector({{embedding_dimension}}) NOT NULL,
    embedding_model text NOT NULL CHECK (length(embedding_model) BETWEEN 1 AND 512),
    embedding_version text NOT NULL CHECK (length(embedding_version) BETWEEN 1 AND 512),
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    access_level text NOT NULL CHECK (length(access_level) BETWEEN 1 AND 64),
    parent_id text,
    page integer CHECK (page >= 1),
    timecode jsonb,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (id),
    UNIQUE (tenant_id, id),
    FOREIGN KEY (tenant_id, source_id) REFERENCES sources (tenant_id, id)
        ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, parent_id) REFERENCES chunks (tenant_id, id)
        ON DELETE SET NULL,
    CHECK (
        timecode IS NULL OR (
            jsonb_typeof(timecode) = 'object'
            AND jsonb_typeof(timecode -> 'start_seconds') = 'number'
            AND jsonb_typeof(timecode -> 'end_seconds') = 'number'
            AND (timecode ->> 'start_seconds')::double precision >= 0
            AND (timecode ->> 'end_seconds')::double precision
                > (timecode ->> 'start_seconds')::double precision
        )
    )
);

CREATE TABLE quarantine (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant_id text NOT NULL REFERENCES tenants (id) ON DELETE CASCADE,
    source_id text,
    reason text NOT NULL CHECK (length(reason) BETWEEN 1 AND 512),
    payload jsonb NOT NULL,
    content_sha256 bytea CHECK (octet_length(content_sha256) = 32),
    disposition text NOT NULL DEFAULT 'pending'
        CHECK (disposition IN ('pending', 'released', 'rejected')),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    reviewed_at timestamptz,
    FOREIGN KEY (tenant_id, source_id) REFERENCES sources (tenant_id, id)
        ON DELETE SET NULL
);

CREATE INDEX tenant_keys_tenant_id_idx ON tenant_keys (tenant_id);
CREATE INDEX tenant_keys_unexpired_idx ON tenant_keys (tenant_id, expires_at)
    WHERE expires_at IS NOT NULL;
CREATE INDEX sources_tenant_id_id_idx ON sources (tenant_id, id);
CREATE INDEX sources_approved_idx ON sources (tenant_id, id) WHERE status = 'approved';
CREATE INDEX chunks_tenant_source_idx ON chunks (tenant_id, source_id);
CREATE INDEX chunks_parent_idx ON chunks (tenant_id, parent_id) WHERE parent_id IS NOT NULL;
CREATE INDEX chunks_tsv_gin_idx ON chunks USING gin (tsv);
CREATE INDEX chunks_embedding_hnsw_idx ON chunks
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX quarantine_tenant_source_idx ON quarantine (tenant_id, source_id);
CREATE INDEX quarantine_pending_idx ON quarantine (tenant_id, created_at)
    WHERE disposition = 'pending';

COMMENT ON INDEX chunks_embedding_hnsw_idx IS
    'Cosine HNSW baseline: m=16 balances graph size/recall; ef_construction=64 '
    'improves build recall at moderate ingestion cost. Benchmark before changing.';

ALTER TABLE tenants ENABLE ROW LEVEL SECURITY;
ALTER TABLE tenants FORCE ROW LEVEL SECURITY;
ALTER TABLE tenant_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE tenant_keys FORCE ROW LEVEL SECURITY;
ALTER TABLE sources ENABLE ROW LEVEL SECURITY;
ALTER TABLE sources FORCE ROW LEVEL SECURITY;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks FORCE ROW LEVEL SECURITY;
ALTER TABLE quarantine ENABLE ROW LEVEL SECURITY;
ALTER TABLE quarantine FORCE ROW LEVEL SECURITY;

CREATE POLICY tenants_app_select ON tenants FOR SELECT TO aegon_app
    USING (id = current_setting('app.tenant_id'));
CREATE POLICY tenants_ingest_select ON tenants FOR SELECT TO aegon_ingest
    USING (id = current_setting('app.tenant_id'));

CREATE POLICY tenant_keys_app_select ON tenant_keys FOR SELECT TO aegon_app
    USING (tenant_id = current_setting('app.tenant_id'));
CREATE POLICY tenant_keys_ingest_all ON tenant_keys FOR ALL TO aegon_ingest
    USING (tenant_id = current_setting('app.tenant_id'))
    WITH CHECK (tenant_id = current_setting('app.tenant_id'));

CREATE POLICY sources_app_select ON sources FOR SELECT TO aegon_app
    USING (tenant_id = current_setting('app.tenant_id') AND status = 'approved');
CREATE POLICY sources_ingest_all ON sources FOR ALL TO aegon_ingest
    USING (tenant_id = current_setting('app.tenant_id'))
    WITH CHECK (tenant_id = current_setting('app.tenant_id'));

CREATE POLICY chunks_app_select ON chunks FOR SELECT TO aegon_app
    USING (
        tenant_id = current_setting('app.tenant_id')
        AND EXISTS (
            SELECT 1 FROM sources
            WHERE sources.tenant_id = chunks.tenant_id
              AND sources.id = chunks.source_id
              AND sources.status = 'approved'
        )
    );
CREATE POLICY chunks_ingest_all ON chunks FOR ALL TO aegon_ingest
    USING (tenant_id = current_setting('app.tenant_id'))
    WITH CHECK (tenant_id = current_setting('app.tenant_id'));

CREATE POLICY quarantine_ingest_all ON quarantine FOR ALL TO aegon_ingest
    USING (tenant_id = current_setting('app.tenant_id'))
    WITH CHECK (tenant_id = current_setting('app.tenant_id'));

GRANT USAGE ON SCHEMA public TO aegon_app, aegon_ingest;
GRANT SELECT ON tenants, tenant_keys, sources, chunks TO aegon_app;
GRANT SELECT ON tenants TO aegon_ingest;
GRANT SELECT, INSERT, UPDATE, DELETE ON tenant_keys, sources, chunks, quarantine
    TO aegon_ingest;
GRANT USAGE, SELECT ON SEQUENCE quarantine_id_seq TO aegon_ingest;

RESET ROLE;
