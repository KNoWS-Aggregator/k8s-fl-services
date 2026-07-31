# Weight aggregation

One central instance coordinates the deployed federated-training-client
services.

It:

- starts and ends federated sessions;
- assigns a random client identifier to each available client service;
- dispatches global weights for local training;
- incrementally aggregates successful client weights;
- dispatches aggregated weights for local evaluation;
- aggregates evaluation metrics; and
- persists successful global weights for the next session.

It has its own persistent volume and never mounts a hospital client's shared
volume. Aggregated evaluation metrics are exposed as a dataset with a JSON
distribution; global and intermediate weights remain private state.

