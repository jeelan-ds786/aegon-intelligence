REVOKE aegon_migrator FROM CURRENT_USER;
REVOKE ALL ON SCHEMA public FROM aegon_app, aegon_ingest, aegon_migrator;

DROP ROLE aegon_app;
DROP ROLE aegon_ingest;
DROP ROLE aegon_migrator;

DROP EXTENSION vector;
