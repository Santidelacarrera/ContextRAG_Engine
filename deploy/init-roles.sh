#!/bin/bash
# Runs once, on the first start of the postgres container. Creates the plain role the API connects as: not a
# superuser, not BYPASSRLS and not the owner of answer_audit, so the tenant_isolation policy applies to it.
set -euo pipefail
psql -v ON_ERROR_STOP=1 -v app_pw="$APP_DB_PASSWORD" --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<'EOSQL'
CREATE ROLE contextrag_app LOGIN PASSWORD :'app_pw';
EOSQL
