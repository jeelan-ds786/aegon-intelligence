SET LOCAL ROLE aegon_migrator;

DROP TABLE quarantine;
DROP TABLE chunks;
DROP TABLE sources;
DROP TABLE tenant_keys;
DROP TABLE tenants;

RESET ROLE;