SET LOCAL ROLE aegon_migrator;

ALTER TABLE chunks
    ADD COLUMN embedding_sha256 bytea NOT NULL
        CHECK (octet_length(embedding_sha256) = 32),
    ADD COLUMN signer_key_id text NOT NULL
        CHECK (length(signer_key_id) BETWEEN 1 AND 1024);

CREATE TABLE chunk_staging (
    run_id text NOT NULL CHECK (length(run_id) BETWEEN 1 AND 128),
    id text NOT NULL CHECK (length(id) BETWEEN 1 AND 128),
    tenant_id text NOT NULL REFERENCES tenants (id) ON DELETE CASCADE,
    source_id text NOT NULL,
    content text NOT NULL CHECK (length(content) BETWEEN 1 AND 100000),
    content_sha256 bytea NOT NULL CHECK (octet_length(content_sha256) = 32),
    embedding vector({{embedding_dimension}}) NOT NULL,
    embedding_sha256 bytea NOT NULL CHECK (octet_length(embedding_sha256) = 32),
    embedding_model text NOT NULL CHECK (length(embedding_model) BETWEEN 1 AND 512),
    embedding_version text NOT NULL CHECK (length(embedding_version) BETWEEN 1 AND 512),
    signer_key_id text NOT NULL CHECK (length(signer_key_id) BETWEEN 1 AND 1024),
    signature bytea NOT NULL CHECK (octet_length(signature) = 64),
    PRIMARY KEY (tenant_id, run_id, id),
    FOREIGN KEY (tenant_id, source_id) REFERENCES sources (tenant_id, id)
        ON DELETE CASCADE
);

CREATE TABLE ingestion_manifests (
    run_id text NOT NULL CHECK (length(run_id) BETWEEN 1 AND 128),
    tenant_id text NOT NULL REFERENCES tenants (id) ON DELETE CASCADE,
    manifest bytea NOT NULL CHECK (octet_length(manifest) > 0),
    signer_key_id text NOT NULL CHECK (length(signer_key_id) BETWEEN 1 AND 1024),
    signature bytea NOT NULL CHECK (octet_length(signature) = 64),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (tenant_id, run_id)
);

CREATE INDEX chunk_staging_run_idx ON chunk_staging (tenant_id, run_id);
CREATE INDEX ingestion_manifests_created_idx
    ON ingestion_manifests (tenant_id, created_at DESC);

ALTER TABLE chunk_staging ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunk_staging FORCE ROW LEVEL SECURITY;
ALTER TABLE ingestion_manifests ENABLE ROW LEVEL SECURITY;
ALTER TABLE ingestion_manifests FORCE ROW LEVEL SECURITY;

CREATE POLICY chunk_staging_ingest_all ON chunk_staging FOR ALL TO aegon_ingest
    USING (tenant_id = current_setting('app.tenant_id'))
    WITH CHECK (tenant_id = current_setting('app.tenant_id'));
CREATE POLICY ingestion_manifests_ingest_all ON ingestion_manifests FOR ALL TO aegon_ingest
    USING (tenant_id = current_setting('app.tenant_id'))
    WITH CHECK (tenant_id = current_setting('app.tenant_id'));
CREATE POLICY ingestion_manifests_app_select ON ingestion_manifests FOR SELECT TO aegon_app
    USING (tenant_id = current_setting('app.tenant_id'));

GRANT SELECT, INSERT, UPDATE, DELETE ON chunk_staging TO aegon_ingest;
GRANT SELECT, INSERT ON ingestion_manifests TO aegon_ingest;
GRANT SELECT ON ingestion_manifests TO aegon_app;

RESET ROLE;