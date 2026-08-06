# Weight-aggregation image

This long-running service coordinates federated sessions and performs
example-weighted FedAvg. It has its own persistent volume and does not mount
the data-preparation/model-training shared volume.

## Discovered clients

`CASE_SLICE` is the researcher case-slice URL. The service queries its
`trainingServices` field at startup and on the `POLL_INTERVAL` schedule
(`@hourly` by default). `POLL_ENABLED` controls scheduled polling and defaults
to `true`. Authentication is intentionally left to the deployment platform.

Membership changes discovered during training or evaluation are buffered.
Immediately before the next training round, departed clients are released and
new clients are initialized. The client set expected by an in-progress round
therefore never changes.

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

The callback URL sent to clients is built from `PUBLIC_BASE_URL`, which
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
- `POST /training-results`
- `POST /evaluation-results`
- `GET /status`
- `GET /evaluation-metrics`
- `GET /healthz`
- `GET /readyz`

`GET /status` includes `registered_clients` and `trainable_clients`. The latter
is determined by probing each registered training service: it must have a valid
prepared dataset and must either be unassigned or assigned to this coordinator's
current session. Unreachable services and services assigned to another
coordinator are not trainable.

## Logical service description

This image implements the
[`weight-aggregation`](../../service-descriptions/weight-aggregation/) service.
Its semantic definition describes session creation, incremental weight and
evaluation-metric aggregation, endpoints, and the public aggregated metrics
dataset.

After every successful weight aggregation, the coordinator dispatches the
aggregated weights to the remaining clients for evaluation. It waits for the
evaluation quorum, combines loss and accuracy using each client's evaluated
example count, and persists both per-client and aggregated metrics. Only then
does it start the next training round or complete the session.

The most recent aggregate is available from `GET /evaluation-metrics` and at:

```text
/app/data/weight-aggregation/evaluation-metrics.json
```

## Container

```sh
make containers-build CONTAINER=weight-aggregation
make containers-push CONTAINER=weight-aggregation TAG=0.1.0
```
