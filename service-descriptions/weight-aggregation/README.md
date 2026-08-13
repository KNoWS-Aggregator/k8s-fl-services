# Weight aggregation

One central instance coordinates the deployed federated-training-client
services. At deployment it receives the researcher case-slice URL. That slice
exposes the participating training-service base URLs through
`trainingServices`; membership is polled and changes are applied between
rounds. Researchers can also trigger discovery immediately through
`POST /refresh-clients`.

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
distribution. The final global weights from the latest successfully completed
session are exposed as a researcher-only download; intermediate weights remain
private state.

The current aggregator configuration is in `profile.yaml` and
`deployment-function.yaml`. For example:

```sh
agg create-service \
  --name coordinator \
  --deployment-function weight-aggregation \
  --param caseSlice=https://researcher.example/slices/case-example \
  --param pollEnabled=true \
  --param pollInterval=@hourly
```
