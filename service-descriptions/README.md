# Logical service descriptions

This directory describes aggregator services, not container images.

There are currently two logical services:

- [`federated-training-client/`](federated-training-client/) combines the
  data-preparation and model-training images in one service profile.
- [`weight-aggregation/`](weight-aggregation/) describes the central
  coordinator.

Each service directory contains:

- `functions.ttl`: the deployment function and the runtime FnO functions,
  parameters, and outputs;
- `service-definition.ttl`: the reusable `aggr:ServiceProfile`, endpoints,
  and abstract compositions;
- `service-request.example.ttl`: deployment-function inputs; and
- `service.example.ttl`: an example service produced by the deployment
  function.

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
belong to the service profile.

## Deployment

Deployment and runtime behavior are distinct FnO functions. Each deployment
function declares the values a platform must receive in an
`aggr:ServiceRequest`. Its output has `fno:type aggr:Service`, and the output
parameter links to the service profile with `dct:conformsTo`.

A request selects the function with `aggr:deploymentFunction` and supplies
inputs directly with the predicates declared by its `fno:Parameter` values:

```text
ServiceRequest
  -> aggr:deploymentFunction
  -> deployment Function
       -> fno:expects -> deployment Parameters
       -> fno:returns -> Output
                          -> fno:type aggr:Service
                          -> dct:conformsTo ServiceProfile
```

The resulting service records the same `aggr:deploymentFunction`.

## Discovery

A deployed service's profile is discovered through its deployment function's
output. Runtime operations and functions are then discovered through:

```text
Service
  -> aggr:deploymentFunction
  -> Function
  -> fno:returns
  -> Output
  -> dct:conformsTo
  -> ServiceProfile
  -> aggr:supportsEndpoint
  -> Endpoint
  -> hydra:supportedOperation
  -> Operation
  -> aggr:executes
  -> Function
```

`aggr:path` is relative to the deployed service's `dcat:endpointURL`.

## Abstract data flow

`aggr:composition` links a service profile to its FnO composition. The
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
