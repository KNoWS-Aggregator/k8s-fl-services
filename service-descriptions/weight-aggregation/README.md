# Weight aggregation

One central service discovers federated-training clients from the configured
researcher case slice. It refreshes membership at startup, on its polling
schedule, or on demand. Membership changes found during an active round are
applied between rounds.

A session assigns random client IDs, dispatches global weights, performs
example-weighted FedAvg, dispatches the aggregate for evaluation, aggregates
evaluation metrics, and repeats until completion. The coordinator has its own
persistent volume and never mounts a hospital client's volume.

The examples use this deployed service URL:

```text
https://aggregator.example/research/services/coordinator
```

Follow the `dcat:endpointURL`, `dcat:accessURL`, and `dcat:downloadURL` values
in the service description rather than assuming these example URLs. Protected
requests require an UMA bearer token with the role shown below.

## Deploy the service

The deployment inputs are:

| Parameter | Container variable | Example |
| --- | --- | --- |
| `caseSlice` | `CASE_SLICE` | `https://researcher.example/slices/case-example` |
| `pollEnabled` | `POLL_ENABLED` | `true` |
| `pollInterval` | `POLL_INTERVAL` | `@hourly` |

Retrieve `/deployments/weight-aggregation` first and use the exact IRIs it
advertises. An example creation request is:

```http
POST /research/services HTTP/1.1
Host: aggregator.example
Authorization: Bearer <access-token>
Content-Type: text/turtle
Accept: text/turtle

@prefix aggr: <https://w3id.org/aggregator#> .

<https://aggregator.example/research/services/coordinator>
  a aggr:ServiceRequest ;
  aggr:deploymentFunction
    <https://aggregator.example/deployments/weight-aggregation> ;
  <https://aggregator.example/deployments/weight-aggregation#case-slice>
    <https://researcher.example/slices/case-example> ;
  <https://aggregator.example/deployments/weight-aggregation#poll-enabled>
    true ;
  <https://aggregator.example/deployments/weight-aggregation#poll-interval>
    "@hourly" .
```

Success returns `201 Created`, a `Location` header, and the Turtle service
description. Kubernetes deployment continues asynchronously.

Round count, minimum clients, timeout, and training configuration are runtime
values for `/session/start`, not deployment inputs.

## Endpoints intended for service users

The `researcher` role represents a user or application controlling and
observing federated learning. These are the endpoints that consumers of this
service normally call.

| Method and path | Required role | Behavior |
| --- | --- | --- |
| `POST /session/start` | `researcher` | Start an asynchronous session; return `202` and its generated ID. |
| `POST /refresh-clients` | `researcher` | Refresh discovery; return added and removed service URLs. |
| `GET /status` | `researcher` | Return session state plus registered and trainable client counts; `include_round_metrics=true` also returns timing and peak-RAM metrics for every completed round in the latest session. |

### Start a federated session

```http
POST /research/services/coordinator/session/start HTTP/1.1
Host: aggregator.example
Authorization: Bearer <researcher-token>
Content-Type: application/json

{
  "expected_rounds": 10,
  "min_clients": 2,
  "round_timeout_seconds": 3600,
  "training_config": {
    "local_epochs": 1,
    "batch_size": 32,
    "verbose": 0,
    "scaling": false,
    "balance": false,
    "group_size": 0,
    "val_size": 0.2,
    "test_size": 0.2
  }
}
```

Only `expected_rounds` is required. `min_clients` defaults to `1`, timeout to
`3600` seconds, and omitted training fields use their model defaults. The
coordinator overrides `training_config.num_rounds` with `expected_rounds`.

Success returns immediately while client initialization continues:

```http
HTTP/1.1 202 Accepted
Content-Type: application/json

{"status":"initializing","session_id":"<generated-uuid>"}
```

A concurrent session returns `409`. A minimum greater than the number of
currently trainable clients returns `422`. Use `/status` to follow progress;
session creation does not synchronously wait for training to complete.

### Inspect status

```http
GET /research/services/coordinator/status HTTP/1.1
Host: aggregator.example
Authorization: Bearer <researcher-token>
Accept: application/json
```

The response contains the persisted coordinator state and session/round
context, plus:

```json
{
  "status": "running",
  "session_id": "<generated-uuid>",
  "round_id": 1,
  "registered_clients": 3,
  "trainable_clients": 3
}
```

Additional fields depend on the phase or failure. A registered client is
trainable only when it is reachable, has valid prepared data, and is not
assigned to another coordinator session.

### Refresh client discovery

```http
POST /research/services/coordinator/refresh-clients HTTP/1.1
Host: aggregator.example
Authorization: Bearer <researcher-token>
Content-Length: 0
```

```json
{
  "added": ["https://aggregator.example/hospital-a/services/training"],
  "removed": []
}
```

This updates discovery immediately. Changes to an in-progress round remain
buffered until the next round boundary.

## Endpoints managed by training clients

The following callback endpoints are part of the federated service-to-service
protocol and require the `training-client` role. Federated-training-client
services call them automatically after the coordinator dispatches work. A
researcher or other service user should not normally POST to them.

| Method and path | Purpose |
| --- | --- |
| `POST /training-results` | Receive a client's local weights and training metrics, or its training failure. |
| `POST /evaluation-results` | Receive a client's evaluation metrics or evaluation failure. |

Both return `202 Accepted` after validating a callback and record it in the
background. Training successes contain a JSON control message and NPZ weights
as multipart data; failures and evaluation callbacks use JSON. Their detailed
payload contract belongs to the protocol between the two FL services rather
than the user-facing API.

## Outputs and distributions

Outputs are persisted datasets, distinct from the callback and control
endpoints.

| Dataset/distribution | Role | URL property | Media type | Availability |
| --- | --- | --- | --- | --- |
| `aggregated-global-weights/binary` | `researcher` | `dcat:downloadURL` | `application/octet-stream` | Latest successful session's `global-weights.npz`; `404` before one completes. |
| `aggregated-evaluation-metrics/json` | `researcher` | `dcat:accessURL` | `application/json` | Latest aggregate evaluation metrics; `404` before evaluation completes. |

The global model is atomically replaced only after a successful final round.
Failed-session and in-progress weights are never exposed.

Download the model using its advertised distribution URL:

```http
GET /research/services/coordinator/weights HTTP/1.1
Host: aggregator.example
Authorization: Bearer <researcher-token>
Accept: application/octet-stream
```

Save the response bytes using the `Content-Disposition` filename
`global-weights.npz`; do not parse the response as JSON.

Read the aggregate evaluation document with:

```http
GET /research/services/coordinator/evaluation-metrics HTTP/1.1
Host: aggregator.example
Authorization: Bearer <researcher-token>
Accept: application/json
```

The document contains session and round context, total evaluated examples,
weighted loss, accuracy, macro and weighted F1, macro precision and recall,
per-class scores, and the summed confusion matrix. Included optional fields
depend on the completed evaluation.

## Definition files

- [`profile.yaml`](profile.yaml) defines operations, the `training-client` and
  `researcher` roles, and both output datasets.
- [`deployment-function.yaml`](deployment-function.yaml) binds the interface
  and deployment inputs to the coordinator workload.
