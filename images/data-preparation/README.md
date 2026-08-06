# Data-preparation image

This continuously running process starts the Kvasir RDF-to-Parquet pipeline
when the HTTP server starts. Mount a persistent volume at `/app/data` and poll
`GET /status` for completion. Scheduled polling keeps the prepared data aligned
with the hospital case slice. `GET /healthz` is a liveness endpoint,
`GET /readyz` validates configuration and persistent-volume write access, and
`GET /results` downloads a ZIP containing only the aggregated files
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
- `RESULT_ARCHIVE_NAME` (`prepared-data.zip`)
- `POLL_ENABLED` (`true`): enables or disables scheduled polling
- `POLL_INTERVAL` (`@hourly`): a five-field cron expression or cron alias such
  as `@hourly`, `@daily`, or `@weekly`

Build from the repository root:

```sh
docker build -f images/data-preparation/Dockerfile -t data-preparation .
```

The volume contains `aggregate.duckdb`, `manifest.json`, internal raw/staging
data, `dataset-state.json`, `timing_report.json`, and `prepared-data.zip`. The
ZIP exposes exactly two non-partitioned files: `accel.parquet` and
`gt.parquet`.

`dataset-state.json` coordinates readers of the shared volume. It is published
as `updating` before aggregate replacement and as `ready` with the new
generation identifier only after both Parquet exports and `manifest.json` are
complete.

Run one Kubernetes replica for this service. Multiple replicas must not write
to the same DuckDB database and persistent volume concurrently.

## Logical service role

This image is one implementation component of the
[`federated-training-client`](../../service-descriptions/federated-training-client/)
service. It is deployed in the same Pod as the model-training image and shares
the `/app/data` volume with it.

The combined example workload and PVC are under
[`deployments/federated-training-client/`](../../deployments/federated-training-client/).
