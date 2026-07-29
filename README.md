# Kubernetes Federated Learning Services

This repository is a container-image monorepo. Each directory under
`services/` represents one service image. Images are published to the
KNoWS-Aggregator organization in GitHub Container Registry and linked back to
this source repository through OCI image metadata.

Currently publishable:

| Service | Image |
| --- | --- |
| `data-preparation` | `ghcr.io/knows-aggregator/data-preparation` |

The model-training and weight-aggregation directories are not included in the
build list until their Dockerfiles are ready.

## Repository structure

```text
.
├── Makefile
├── libs/
└── services/
    ├── data-preparation/
    │   ├── Dockerfile
    │   ├── aggregator-platform.yaml
    │   ├── pvc.yaml
    │   └── src/
    ├── model-training/
    └── weight-aggregation/
```

All images use the repository root as their Docker build context. This permits
future services to copy shared packages from `libs/`. The `.dockerignore` file
keeps Git metadata, virtual environments, tests, and local data out of that
context.

## Prerequisites

- Docker with access to the Docker daemon.
- A GitHub personal access token (classic) with `write:packages`.
- SSO authorization for the token when required by the organization.
- Permission to publish packages in `KNoWS-Aggregator`.

Do not add a PAT to the Makefile, a `.env` file, or Git. Load it into the
current shell without placing it directly in shell history:

```sh
read -s CR_PAT
export CR_PAT
export GHCR_USER="your-github-username"
```

Authenticate to GHCR:

```sh
make containers-login
```

The login target passes the token to Docker through standard input.

## Build images

List the images currently enabled for publication:

```sh
make containers-list
```

Build all enabled services:

```sh
make containers-build
```

Build only data-preparation:

```sh
make containers-build CONTAINER=data-preparation
```

Build a versioned image:

```sh
make containers-build CONTAINER=data-preparation TAG=0.1.0
```

This produces:

```text
ghcr.io/knows-aggregator/data-preparation:0.1.0
```

## Push images

The push target builds before pushing, ensuring the requested local tag exists
and reflects the current source:

```sh
make containers-push CONTAINER=data-preparation TAG=0.1.0
```

Push every enabled service using `latest`:

```sh
make containers-push
```

For releases, prefer an immutable version tag and optionally publish `latest`
afterward:

```sh
make containers-push CONTAINER=data-preparation TAG=0.1.0
make containers-push CONTAINER=data-preparation TAG=latest
```

The data-preparation Dockerfile includes
`org.opencontainers.image.source=https://github.com/KNoWS-Aggregator/k8s-fl-services`,
which allows GitHub to associate the package with this repository.

## Adding another service

Complete and test `services/<service>/Dockerfile`, then add its directory name
to the explicit list in the root Makefile:

```make
SERVICES := data-preparation model-training
```

After that, the existing list, build, and push targets automatically include
the new image. Keeping this list explicit prevents unfinished service
directories from breaking build-all and push-all operations.

## Data preparation

Service-specific API, polling, persistent-volume, and aggregator-platform
documentation is available in
[`services/data-preparation/README.md`](services/data-preparation/README.md).
