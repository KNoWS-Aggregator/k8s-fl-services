# Federated training client

One instance is deployed per hospital. It is one logical aggregator service
implemented by `data-preparation` and `model-training` containers sharing a
persistent volume at `/app/data`.

Data preparation converts the hospital case slice into aggregated Parquet
data. Model training consumes the latest valid generation, trains local
weights, and evaluates global weights. The public `/preparation/status` route
targets the preparation container; all other operations target model training.

The examples use this deployed service URL:

```text
https://aggregator.example/research/services/hospital-client
```

Follow the `dcat:endpointURL`, `dcat:accessURL`, and `dcat:downloadURL` values
in the deployed service description rather than assuming these example URLs.
Protected requests require an UMA bearer token with the role shown below.

## Deploy the service

The deployment inputs are:

| Parameter | Container variable | Example |
| --- | --- | --- |
| `caseSlice` | `SOURCES` | `https://hospital.example/slices/case-example` |
| `datasetId` | `DATASET` | `accellero` |
| `pollEnabled` | `POLL_ENABLED` | `true` |
| `pollInterval` | `POLL_INTERVAL` | `@hourly` |

Retrieve `/deployments/fl-client-training` first and use the exact IRIs it
advertises. An example creation request is:

```http
POST /research/services HTTP/1.1
Host: aggregator.example
Authorization: Bearer <access-token>
Content-Type: text/turtle
Accept: text/turtle

@prefix aggr: <https://w3id.org/aggregator#> .

<https://aggregator.example/research/services/hospital-client>
  a aggr:ServiceRequest ;
  aggr:deploymentFunction
    <https://aggregator.example/deployments/fl-client-training> ;
  <https://aggregator.example/deployments/fl-client-training#case-slice>
    <https://hospital.example/slices/case-example> ;
  <https://aggregator.example/deployments/fl-client-training#dataset-id>
    "accellero" ;
  <https://aggregator.example/deployments/fl-client-training#poll-enabled>
    true ;
  <https://aggregator.example/deployments/fl-client-training#poll-interval>
    "@hourly" .
```

Success returns `201 Created`, a `Location` header, and the Turtle service
description. Kubernetes deployment continues asynchronously.

## Endpoints intended for service users

The user-facing operation is the preparation status endpoint. It uses the
`prepared-data-reader` role, which may also read the prepared-data
distribution described below.

```http
GET /research/services/hospital-client/preparation/status HTTP/1.1
Host: aggregator.example
Authorization: Bearer <prepared-data-reader-token>
Accept: application/json
```

It reports `idle`, `running`, `succeeded`, or `failed`, timestamps, polling
information, and a public preparation result or error when available.

## Endpoints managed by weight aggregation

The following endpoints are part of the federated protocol and require the
`coordinator` role. The weight-aggregation service discovers this client and
calls them automatically. A user normally starts and monitors the federated
session through the weight-aggregation service instead of calling these
endpoints directly.

| Method and path | Purpose |
| --- | --- |
| `POST /session/start` | Assign this client to the coordinator's session and exchange its model signature. |
| `POST /session/end` | Release that session assignment. |
| `POST /train` | Supply global weights and start one local training round. |
| `POST /evaluate` | Supply aggregated weights for local evaluation. |
| `GET /status` | Let the coordinator determine preparation validity, assignment, and trainability. |

Training and evaluation results are sent asynchronously to the coordinator's
`/training-results` and `/evaluation-results` endpoints. These protocol routes
are documented here so users can distinguish them from the endpoints they are
expected to call, not as a manual integration interface.

## Outputs and distributions

Outputs are persisted datasets, distinct from operational endpoints.

| Dataset/distribution | Role | URL property | Media type | Availability |
| --- | --- | --- | --- | --- |
| `prepared-data/zip` | `prepared-data-reader` | `dcat:downloadURL` | `application/zip` | Latest prepared archive; `404` before publication. |
| `client-weights/binary` | `training-results-reader` | `dcat:downloadURL` | `application/octet-stream` | Latest local `weights.npz`; `404` before a round completes. |
| `training-metrics/json` | `training-results-reader` | `dcat:accessURL` | `application/json` | Latest local training metrics; `404` before a round completes. |
| `evaluation-metrics/json` | `training-results-reader` | `dcat:accessURL` | `application/json` | Latest local evaluation result; `404` before evaluation completes. |

Example JSON distribution request:

```http
GET /research/services/hospital-client/metrics HTTP/1.1
Host: aggregator.example
Authorization: Bearer <training-results-reader-token>
Accept: application/json
```

```json
{
  "session_id": "session-123",
  "round_id": 1,
  "client_id": "client-7",
  "metrics": {
    "num_examples": 840,
    "train_loss": 0.31,
    "train_accuracy": 0.91,
    "train_f1_weighted": 0.90,
    "val_loss": 0.39,
    "val_accuracy": 0.87,
    "val_f1_weighted": 0.86,
    "setup_seconds": 180.2,
    "training_seconds": 900.5,
    "total_seconds": 1090.9,
    "peak_ram_bytes": 4294967296
  }
}
```

The timing fields measure model and data setup, model fitting, and total local
round duration. `peak_ram_bytes` reports the process peak resident memory.
Use `GET /metrics?include_history=true` to include per-epoch history for all
stored rounds, and `GET /weights?include_in_progress=true` to retrieve the
latest checkpoint from a round that may still be training.

Fetch downloads with `GET`, accept their advertised media type, and save the
response bytes using the `Content-Disposition` filename. Do not parse ZIP or
NPZ downloads as JSON.

## Definition files

- [`profile.yaml`](profile.yaml) defines the service interface and datasets.
- [`deployment-function.yaml`](deployment-function.yaml) binds that interface
  and the deployment inputs to the two-container workload.
