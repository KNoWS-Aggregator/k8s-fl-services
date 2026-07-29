# Data preparation service

This continuously running service starts the Kvasir RDF-to-Parquet pipeline
when the HTTP server starts. Mount a persistent volume at `/app/data` and poll
`GET /status` for completion. `POST /prepare` starts a new conversion later;
it returns `409` while one is already running. `GET /healthz` is a liveness
endpoint and `GET /readyz` validates configuration and persistent-volume write
access. `GET /results` downloads a ZIP containing only the aggregated files
`accel.parquet` and `gt.parquet`; raw intermediate data is not exposed.

The service persists a participant/distribution manifest and polls the case
slice on a cron schedule. New and changed participants are converted
incrementally. Removed participants are deleted transactionally from the
canonical DuckDB aggregate. A failed poll leaves the previous published
download intact.

Required environment variables:

- `SOURCES`: full case-slice URL without `/query`, for example
  `https://kvasir.example/hospital1/slices/case-pacsoi-fl`
- `DATASET`: participant dataset slice ID, for example `accellero`
- `AUTHN`: Keycloak realm URL

Optional environment variables:

- `DATA_DIR` (`/app/data`)
- `PORT` (`8080`)
- `LOG_LEVEL` (`INFO`)
- `DOWNLOAD_CHUNK_SIZE` (`104857600`)
- `WRITE_BATCH_SIZE` (`500000`)
- `MAX_CONCURRENT_PARTICIPANTS` (`1`)
- `MAX_RETRIES` (`5`)
- `RETRY_BACKOFF_BASE` (`2.0`)
- `REQUEST_TIMEOUT_SECONDS` (`60`)
- `DOWNLOAD_TIMEOUT_SECONDS` (`3600`)
- `AUTH_CLIENT_ID_TEMPLATE` (`{participant_id}_client`)
- `AUTH_CLIENT_SECRET_TEMPLATE` (`{participant_id}`)
- `RESULT_ARCHIVE_NAME` (`prepared-data.zip`)
- `POL_ENABLED` (`true`): enables or disables scheduled polling
- `POLL_INTERVAL` (`@hourly`): a five-field cron expression or cron alias such
  as `@hourly`, `@daily`, or `@weekly`

Build from the repository root:

```sh
docker build -f services/data-preparation/Dockerfile -t data-preparation .
```

The volume contains `aggregate.duckdb`, `manifest.json`, internal raw/staging
data, `timing_report.json`, and `prepared-data.zip`. The ZIP exposes exactly
two non-partitioned files: `accel.parquet` and `gt.parquet`.

Run one Kubernetes replica for this service. Multiple replicas must not write
to the same DuckDB database and persistent volume concurrently.

## Aggregator platform

[`aggregator-platform.yaml`](aggregator-platform.yaml) defines the
`FnoDescription` and `ServiceConfiguration`. Its required function inputs map
to `SOURCES`, `DATASET`, and `AUTHN`. The optional `pollEnabled` and
`pollInterval` inputs map to `POL_ENABLED` and `POLL_INTERVAL`. The
`preparedData` output is exposed as a `downloadURL` at `/results`.

The Deployment references the fixed `data-preparation-data` PVC mounted at
`/app/data`. Kubernetes Deployments cannot create PVCs inside their pod
specification, so [`pvc.yaml`](pvc.yaml) must be created in the same namespace
before the aggregator creates the Deployment:

```sh
kubectl apply -f services/data-preparation/pvc.yaml
```

The requested `100Gi` capacity is an initial placeholder and should be sized
for the raw RDF, staging Parquet, DuckDB database, and exported results. The
cluster's default StorageClass is used. The Deployment pulls
`ghcr.io/knows-aggregator/data-preparation:latest`; publish that tag or change
the configuration to a versioned image before deployment.
