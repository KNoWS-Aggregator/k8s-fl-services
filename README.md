# Kubernetes Federated Learning Services

This repository separates deployable container images from logical aggregator
services.

## Logical services

The platform exposes two services:

| Service | Container images |
| --- | --- |
| `federated-training-client` | `data-preparation`, `model-training` |
| `weight-aggregation` | `weight-aggregation` |

Each hospital receives one federated-training-client service. Its two
containers run in the same Pod and mount the same persistent volume at
`/app/data`. One central weight-aggregation service coordinates all deployed
training clients and uses its own volume.

Semantic FnO, endpoint, composition, dataset, and distribution descriptions
are under [`service-descriptions/`](service-descriptions/README.md). Container
implementation details are under `images/`.

## Repository structure

```text
.
├── deployments/
│   └── federated-training-client/
│       ├── deployment.example.yaml
│       └── pvc.yaml
├── images/
│   ├── data-preparation/
│   ├── model-training/
│   └── weight-aggregation/
├── libs/
│   ├── common/
│   └── fl-model/
├── service-descriptions/
│   ├── federated-training-client/
│   ├── weight-aggregation/
│   ├── README.md
│   └── service-definition-vocabulary.ttl
└── Makefile
```

The repository root is the Docker build context so images can copy shared
packages from `libs/`.

## Container images

Images are published to the KNoWS-Aggregator organization:

| Image package | Registry image |
| --- | --- |
| `data-preparation` | `ghcr.io/knows-aggregator/data-preparation` |
| `model-training` | `ghcr.io/knows-aggregator/model-training` |
| `weight-aggregation` | `ghcr.io/knows-aggregator/weight-aggregation` |

The separate data-preparation and model-training images do not imply separate
aggregator services. They are implementation components of the single
federated-training-client service.

## Registry login

Prerequisites:

- Docker with access to its daemon.
- A GitHub personal access token (classic) with `write:packages`.
- SSO authorization when required by the organization.
- Permission to publish packages in `KNoWS-Aggregator`.

Load the token without putting it directly in shell history:

```sh
read -s CR_PAT
export CR_PAT
export GHCR_USER="your-github-username"
make containers-login
```

Do not commit the token to Git or store it in the Makefile.

## Build images

List all enabled images:

```sh
make containers-list
```

Build every image:

```sh
make containers-build
```

Build one versioned image:

```sh
make containers-build CONTAINER=model-training TAG=0.1.0
```

All image names are listed explicitly in `IMAGES` in the root Makefile.

## Push images

Push one versioned image:

```sh
make containers-push CONTAINER=model-training TAG=0.1.0
```

Push every enabled image with the default `latest` tag:

```sh
make containers-push
```

The push target builds before pushing. Every Dockerfile includes the OCI source
label linking its GitHub package to this repository.

## Example deployment

[`deployment.example.yaml`](deployments/federated-training-client/deployment.example.yaml)
shows data preparation and model training as two containers in one Deployment.
[`pvc.yaml`](deployments/federated-training-client/pvc.yaml) provides their
shared volume.

The example demonstrates container placement and storage only. The aggregator
platform remains responsible for substituting function inputs, publishing the
service endpoints, creating the final workload, and injecting
`EGRESS_UMA_URL` when outbound traffic must traverse the aggregator UMA egress
proxy.

## Adding an image

Add an image package under `images/<name>/`, then add its name to `IMAGES` in
the Makefile. Adding an image does not automatically create a logical service;
add or update a service profile and deployment function under
`service-descriptions/` separately.
