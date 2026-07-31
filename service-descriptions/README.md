# Logical service descriptions

This directory describes aggregator services, not container images.

There are currently two logical services:

- [`federated-training-client/`](federated-training-client/) combines the
  data-preparation and model-training images in one service definition.
- [`weight-aggregation/`](weight-aggregation/) describes the central
  coordinator.

Each service directory contains:

- `functions.ttl`: FnO functions, parameters, and outputs;
- `service-definition.ttl`: reusable endpoints and abstract compositions;
- `service-request.example.ttl`: deployment-time function bindings; and
- `service.example.ttl`: an example deployed service using `aggr:instanceOf`.

[`service-definition-vocabulary.ttl`](service-definition-vocabulary.ttl)
declares the shared extension terms.

The example IRIs use `https://example.org/k8s-fl-services/`. A real platform
must replace them with stable, dereferenceable identifiers.

## Service and image boundaries

The federated-training-client is one semantic service implemented by two
containers:

```text
aggr:Service federated-training-client
├── data-preparation image
├── model-training image
└── shared /app/data volume
```

Container ports, volume mounts, and process health probes are orchestration
details. Functions, API operations, data flow, datasets, and distributions
belong to the service definition.

## Discovery

A deployed service links to its blueprint with `aggr:instanceOf`. Operations
and functions are discovered through:

```text
Service
  -> aggr:instanceOf
  -> ServiceDefinition
  -> aggr:supportsEndpoint
  -> Endpoint
  -> hydra:supportedOperation
  -> Operation
  -> aggr:executes
  -> Function
```

`aggr:path` is relative to the deployed service's `dcat:endpointURL`.

## Abstract data flow

`aggr:composition` links a service definition to its FnO composition. The
federated-training-client declares:

```text
prepare.preparedData
          -> train.preparedData

train.evaluationData
          -> evaluate.evaluationData
```

The composition is independent of the shared-volume implementation.

## Datasets and distributions

An FnO output does not automatically require public access:

- `dcat:servesDataset` makes the logical dataset discoverable;
- `dcat:distribution` is added only when an access or download representation
  exists.

The internal evaluation dataset is described without a distribution. Prepared
data, weights, training metrics, and evaluation metrics have distributions.

