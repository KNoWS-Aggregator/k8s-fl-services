# Aggregator service configuration

This directory contains the current aggregator configuration for the two
logical federated-learning services:

- `federated-training-client/profile.yaml` and `deployment-function.yaml`;
- `weight-aggregation/profile.yaml` and `deployment-function.yaml`.

Profiles describe public operations, roles, functions, and datasets.
Deployment functions bind those profile routes and deployment inputs to the
containers, ports, environment variables, storage, and Kubernetes workloads.
The files use the same `profiles:` and `deploymentFunctions:` format as the
aggregator platform configuration.

The older TTL files and `service-definition.yaml` are retained as legacy
semantic-design references. They are not examples of the current aggregator
configuration format.

## Federated training client

One client is implemented by `data-preparation` and `model-training` containers
sharing `/app/data`. Its deployment inputs are:

| Parameter | Container variable | Example |
| --- | --- | --- |
| `caseSlice` | `SOURCES` | `https://hospital.example/slices/case-example` |
| `datasetId` | `DATASET` | `accellero` |
| `pollEnabled` | `POLL_ENABLED` | `true` |
| `pollInterval` | `POLL_INTERVAL` | `@hourly` |

The public `/status` route is the model-training status used by the coordinator.
The public `/preparation/status` route is bound to the data-preparation
container's internal `/status` endpoint. They are deliberately separate because
both containers implement `/status` on different Pod ports.

With the aggregator CLI, inspect both independently:

```sh
agg list-endpoints --svc hospital-client
agg get-endpoint status --svc hospital-client
agg get-endpoint preparation/status --svc hospital-client
```

## Weight aggregation

The coordinator deployment inputs are:

| Parameter | Container variable | Example |
| --- | --- | --- |
| `caseSlice` | `CASE_SLICE` | `https://researcher.example/slices/case-example` |
| `pollEnabled` | `POLL_ENABLED` | `true` |
| `pollInterval` | `POLL_INTERVAL` | `@hourly` |

Runtime values such as round count, minimum clients, timeout, and training
configuration are supplied to `/session/start`; they are not deployment inputs.
