# Federated training client

One instance is deployed per hospital. It is one logical aggregator service
implemented by two containers:

```text
federated-training-client
├── data-preparation
├── model-training
└── shared persistent volume at /app/data
```

Data preparation continuously converts the hospital case slice into aggregated
Parquet data. Model training consumes that prepared generation, trains local
weights, and evaluates aggregated weights.

The service profile performs three runtime functions:

```text
prepare -> train -> evaluate
```

Deployment receives the hospital case-slice URL, participant dataset ID, and
polling configuration. These values are bound to `SOURCES`, `DATASET`,
`POLL_ENABLED`, and `POLL_INTERVAL` in the data-preparation container.

`aggr:composition` describes the internal prepared-data and evaluation-data
connections without exposing the persistent-volume paths.

Prepared data, client weights, training metrics, and evaluation metrics are
served as datasets with distributions. The internal evaluation dataset is
described without a distribution.

The current aggregator examples are `profile.yaml` and
`deployment-function.yaml`. The deployment function routes `/results` and
`/preparation/status` to data preparation on port `preparation`. It routes
`/status`, session, training, weights, metrics, and evaluation operations to
model training on port `training`.

`/status` therefore reports model-training/client readiness, while
`/preparation/status` exposes detailed progress from the preparation process.

Example deployment with the current aggregator CLI:

```sh
agg create-service \
  --name hospital-client \
  --deployment-function fl-client-training \
  --param caseSlice=https://hospital.example/slices/case-example \
  --param datasetId=accellero \
  --param pollEnabled=true \
  --param pollInterval=@hourly
```

After deployment, `agg get-endpoint preparation/status --svc hospital-client`
retrieves the preparation status through the service's authenticated public
route.
