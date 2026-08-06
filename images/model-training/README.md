# Model-training image

The model-training process is the training component of one federated client.
It shares `/app/data` with the data-preparation container and trains on all
participants registered locally. A federated `client_id` identifies the
combined client service to weight aggregation; it is not a participant
identifier.

## Prepared-data contract

By default, the loader reads:

```text
/app/data/downloads/accel.parquet
/app/data/downloads/gt.parquet
```

`DATA_DIR` changes the shared volume root. `ACCEL_PARQUET_PATH` and
`GT_PARQUET_PATH` can override the individual files.

The loader:

1. Finds participant IDs present in both files.
2. Loads and processes each participant independently.
3. Detects `OnTable` activity from one-second accelerometer statistics.
4. Converts acceleration from m/s² to g.
5. Produces six-second windows at 32 Hz (192 samples) with 50% overlap.
6. Aligns each window with the nearest ground-truth label within six seconds.
7. Removes incomplete windows and unsupported labels.
8. Uses the fixed class order `Lying`, `OnTable`, `Sitting`, `Standing`,
   `Walking`.
9. Concatenates completed participant windows into one client-local dataset.

Processing participants separately ensures that no model window crosses a
participant boundary.

## Participant-level splits

Complete participants, rather than individual windows, are assigned to
training, validation, and test sets. Assignments are persisted by default at:

```text
/app/data/model-training/participant-splits.json
```

`SPLIT_REGISTRY_PATH` overrides that location and `SPLIT_SEED` controls the
initial deterministic assignment (default `42`). Existing assignments remain
stable when participants are added or removed. New participants fill the
largest split deficit.

`val_size` and `test_size` in `TrainingConfig` accept fractions (`0.2`) or
percentages (`20`). Once a registry exists, those values and `SPLIT_SEED`
cannot change silently because moving a participant between splits would
introduce evaluation leakage. To intentionally repartition, remove the
registry before the next training run.

Both split values default to `0.2`. Federated evaluation uses the test split
and fails explicitly when too few local participants produce a test partition.
The validation split is passed to `model.fit` during local training to calculate
validation loss and accuracy after each epoch. It is not used for the separate
post-aggregation evaluation phase.

Training histories are written under
`/app/data/model-training/results` by default. `RESULTS_DIR` overrides this
location.

## Generation-aware preprocessing cache

Data-preparation writes `/app/data/dataset-state.json` as `updating` before it
replaces aggregate files and as `ready` after both Parquet files are published.
Model-training refuses to preprocess a non-ready generation and verifies that
the generation did not change while reading.

Processed train/validation/test arrays are cached under:

```text
/app/data/model-training/cache/
```

The cache key covers the dataset generation, participant assignments, split
configuration, activity order, window settings, and preprocessing version.
Later rounds reuse the cache. A new data generation or preprocessing change
creates a new cache automatically. `MODEL_DATA_CACHE_DIR` and
`DATASET_STATE_PATH` override the default locations.

## Container image

The CPU-oriented image installs the shared `common` package and the
`model_training` package, runs as non-root user `65532`, and mounts
`/app/data`. Build it from the repository root:

```sh
make containers-build CONTAINER=model-training
```

Publish a versioned image after logging in to GHCR:

```sh
make containers-push CONTAINER=model-training TAG=0.1.0
```

## HTTP lifecycle

- `POST /train` accepts a multipart `TrainingInitMessage` and global weights.
- `POST /evaluate` evaluates aggregated weights without modifying them.
- `POST /session/start` assigns an idle service to a federated session.
- `POST /session/end` releases that assignment.
- `GET /weights` retrieves the weights from the latest completed local round.
- `GET /metrics` retrieves the metrics from the latest completed local round.
- `GET /evaluation-metrics` retrieves client evaluation metrics.
- `GET /healthz` provides a process liveness check.
- `GET /readyz` verifies prepared Parquet inputs and shared-volume write access.
- `GET /status` returns the persistent state of the most recent round together
  with `prepared_data.valid`, the current session assignment, and `trainable`.
  An updating preparation remains valid when a previously published generation
  and both prepared Parquet files still exist. The first preparation is not
  trainable until its initial generation has been published.

Only one round runs at a time. A different concurrent round receives `409`.
Repeating the currently running round is idempotent. Successful result messages
and weights are cached under `/app/data/model-training/rounds`; repeating a
completed round re-sends that result without retraining.

Weight aggregation randomly assigns `session_id` and `client_id` values.
Model-training persists that assignment and rejects `/train` messages from
another session or client until the matching `/session/end` arrives. Training
configuration is stored once by `/session/start` and reused for every round.
Training failures are POSTed to the supplied `reply_url` as a
`TrainingFailureMessage`, allowing weight aggregation to remove a failed client
without waiting for the round timeout.

## Logical service role

This image is deployed with the data-preparation image as the single
[`federated-training-client`](../../service-descriptions/federated-training-client/)
service. The combined service definition contains preparation, training, and
evaluation functions plus their abstract data-flow composition.

The shared `/app/data` cache is the physical implementation of the composed
prepared-data and evaluation-data flows.
