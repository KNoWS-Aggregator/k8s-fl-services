# Weight aggregation

One central instance coordinates the deployed federated-training-client
services. At deployment it receives the researcher case-slice URL. That slice
exposes the participating training-service base URLs through
`trainingServices`; membership is polled and changes are applied between
rounds.

Its status endpoint reports the total number of registered training services
and how many are currently trainable. A service is trainable when it has a
valid prepared generation and is not assigned to a different aggregation
session.

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
