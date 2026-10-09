SET LOCAL ROLE aegon_migrator;

DROP TABLE ingestion_manifests;
DROP TABLE chunk_staging;
ALTER TABLE chunks DROP COLUMN signer_key_id, DROP COLUMN embedding_sha256;

RESET ROLE;