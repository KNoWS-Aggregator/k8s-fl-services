# Aggregator service descriptions

This directory contains the current aggregator definitions for two logical
federated-learning services:

- [`federated-training-client`](federated-training-client/README.md) prepares
  hospital data, trains local weights, and evaluates global weights.
- [`weight-aggregation`](weight-aggregation/README.md) discovers clients,
  coordinates federated sessions, and publishes aggregate results.

Each directory contains a `profile.yaml` describing public operations, roles,
functions, datasets, and distributions, and a `deployment-function.yaml` that
binds them to Kubernetes resources and container routes. Its README documents
the deployment request, every endpoint, and every output distribution for that
service.

## Common aggregator interaction model

Examples in the service READMEs use actual HTTP requests and these placeholders:

```text
https://aggregator.example       aggregator server
research                         aggregator instance ID
```

Protected requests need a bearer token accepted by the aggregator's UMA
ingress. The token must grant the role documented for the operation or
distribution.

Before deploying, retrieve the relevant catalog document and use the exact
deployment and parameter predicate IRIs it advertises:

```http
GET /deployments/<deployment-function> HTTP/1.1
Host: aggregator.example
Accept: text/turtle
```

After deployment, discover public URLs from the service description:

```http
GET /research/services/<service-name> HTTP/1.1
Host: aggregator.example
Authorization: Bearer <access-token>
Accept: text/turtle
```

The response advertises two distinct types of URL:

- `dcat:endpointURL` identifies an operational endpoint that triggers work or
  reports live state.
- `dcat:servesDataset` leads to datasets and distributions whose
  `dcat:accessURL` or `dcat:downloadURL` reads persisted output.

Consumers should follow the advertised URLs rather than construct them from
profile paths. Access to an operation does not automatically grant access to a
distribution.

The service READMEs also distinguish user-facing endpoints from federated
service-to-service endpoints. Operations requiring the `coordinator` role are
managed by weight aggregation and operations requiring the `training-client`
role are callbacks managed by federated-training clients. They are listed for
orientation, while detailed interaction examples focus on the `researcher`,
`prepared-data-reader`, and `training-results-reader` roles used by service
consumers.
