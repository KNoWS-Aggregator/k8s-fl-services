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

`aggr:composition` describes the internal prepared-data and evaluation-data
connections without exposing the persistent-volume paths.

Prepared data, client weights, training metrics, and evaluation metrics are
served as datasets with distributions. The internal evaluation dataset is
described without a distribution.

The two container APIs currently listen on separate Pod ports. The platform
must publish the relative paths in `service-definition.ttl` under one service
root, routing preparation operations to the data-preparation container and
training/session/evaluation operations to the model-training container.
