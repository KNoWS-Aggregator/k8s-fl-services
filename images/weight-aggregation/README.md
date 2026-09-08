# Weight-aggregation image

This long-running service coordinates federated sessions and performs
example-weighted FedAvg. It has its own persistent volume and does not mount
the data-preparation/model-training shared volume.

## Discovered clients

`CASE_SLICE` is the researcher case-slice URL. The service queries its
`trainingServices` field at startup and on the `POLL_INTERVAL` schedule
(`@hourly` by default). `POLL_ENABLED` controls scheduled polling and defaults
to `true`. When `EGRESS_UMA_URL` is set, all outbound case-slice requests and
client callbacks are sent through `<EGRESS_UMA_URL>/fetch`. Authentication is
intentionally left to the deployment platform.

Membership changes discovered during training or evaluation are buffered.
Immediately before the next training round, departed clients are released and
new clients are initialized. The client set expected by an in-progress round
therefore never changes.

`POST /refresh-clients` immediately queries the case slice and returns the
training-service URLs added to and removed from the registry. This updates
discovery immediately, while the same between-round membership rule still
applies to an active session.

Each session assigns a fresh random client ID to every discovered training
service. Clients whose model signature differs from the canonical model are
excluded.

## Start a session

```http
POST /session/start
Content-Type: application/json
```

```json
{
  "expected_rounds": 10,
  "min_clients": 1,
  "round_timeout_seconds": 3600,
  "training_config": {
    "local_epochs": 1,
    "batch_size": 32,
    "val_size": 0.2,
    "test_size": 0.2,
    "scaling": false,
    "balance": false
  }
}
```

`training_config` and all fields except `expected_rounds` use defaults.
The coordinator sets `training_config.num_rounds` from `expected_rounds`, so
the session endpoint remains the authority for round count.

The callback URL sent to clients is built from `AGG_PUBLIC_URL`, which
defaults to `http://weight-aggregation:8080`.

## Initial and subsequent weights

The service uses the shared `fl_model` package to create a canonical model only
when no successful previous-session weights exist. Every training client
receives exactly those weights.

After a successful final round, weights are atomically promoted to:

```text
/app/data/weight-aggregation/global-weights.npz
```

The next session starts from that file. A failed session never replaces it.

Per-session inputs, client outputs, metrics, and aggregates are retained under
`/app/data/weight-aggregation/sessions/<session-id>/`.

## Client failures

Dispatch failures, explicit training failures, incompatible model signatures,
and round timeouts remove the affected client. Training continues while at
least `min_clients` remain. FedAvg weights only successful clients by their
reported number of examples.

## Endpoints

- `POST /session/start`
- `POST /refresh-clients`
- `POST /training-results`
- `POST /evaluation-results`
- `GET /status`
- `GET /weights`
- `GET /metrics`
- `GET /metrics/history`
- `GET /healthz`
- `GET /readyz`

`GET /status` includes `session_started_at`, `round_started_at`,
`registered_clients`, and `trainable_clients`. `round_started_at` is `null`
until the first round starts. Trainability is determined by probing each
registered training service: it must have a valid prepared dataset and must
either be unassigned or assigned to this coordinator's current session.
Unreachable services and services assigned to another coordinator are not
trainable.

## Logical service description

This image implements the
[`weight-aggregation`](../../service-descriptions/weight-aggregation/) service.
Its semantic definition describes session creation, incremental weight and
evaluation-metric aggregation, endpoints, and the public aggregated metrics
dataset.

After every successful weight aggregation, the coordinator dispatches the
aggregated weights to the remaining clients for evaluation. It waits for the
evaluation quorum, combines loss using each client's evaluated example count,
and sums client confusion matrices. Global accuracy, macro and weighted F1,
macro precision and recall, and per-class scores are derived from that summed
matrix. Both per-client and aggregated metrics are persisted. Only then does it
start the next training round or complete the session.

The latest round's client-training, aggregated evaluation, timing, and memory
metrics are available from `GET /metrics`. The same data for every round in the
latest session is available from `GET /metrics/history` and persisted at
`/app/data/weight-aggregation/round-metrics.json`.
Each round contains client setup, training, total duration, and peak RSS,
plus coordinator dispatch, result collection, messaging-overhead estimate,
weight aggregation duration, and peak RSS. Durations are seconds and memory
values are bytes.

The newest global model is downloadable from `GET /weights`. During a session,
the endpoint returns the latest available aggregate checkpoint, updated after
each client result. Before the active session has produced a checkpoint, it
falls back to the latest successfully completed session. It returns `404` only
when neither is available.

## Container

```sh
make containers-build CONTAINER=weight-aggregation
make containers-push CONTAINER=weight-aggregation TAG=0.1.0
```
