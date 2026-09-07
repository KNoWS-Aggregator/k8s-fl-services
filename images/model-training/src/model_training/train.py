"""Training logic for the Training Job container."""
import gc
import json
import resource
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from common.messages import (
    EvaluationInitMessage,
    EvaluationMetrics,
    EvaluationResultMessage,
    TrainingConfig,
    TrainingInitMessage,
    TrainingMetrics,
    TrainingResultMessage,
)
from common.weight_io import bytes_to_weights, weights_to_bytes

from fl_model import load_model
from .data_pipeline import ACTIVITIES
from .support import get_class_weights, load_data, round_dir

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import keras


class TrainingProgressCallback(keras.callbacks.Callback):
    """Persist epoch metrics and weights while a round is running."""

    def __init__(self, output_dir: Path):
        super().__init__()
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        history_path = self.output_dir / "history.json"
        history = (
            json.loads(history_path.read_text(encoding="utf-8"))
            if history_path.is_file()
            else []
        )
        history.append({name: float(value) for name, value in (logs or {}).items()})
        temporary_history = self.output_dir / "history.json.tmp"
        temporary_history.write_text(json.dumps(history), encoding="utf-8")
        temporary_history.replace(history_path)

        weights = weights_to_bytes(self.model.get_weights())
        epoch_path = self.output_dir / f"weights_epoch_{epoch + 1:03d}.npz"
        temporary_epoch = self.output_dir / f"{epoch_path.name}.tmp"
        temporary_epoch.write_bytes(weights)
        temporary_epoch.replace(epoch_path)
        temporary_latest = self.output_dir / "weights.npz.tmp"
        temporary_latest.write_bytes(weights)
        temporary_latest.replace(self.output_dir / "weights.npz")


def run_training(
    msg: TrainingInitMessage,
    config: TrainingConfig,
    global_weights: bytes,
) -> tuple[TrainingResultMessage, bytes]:
    """Handle TrainingInitMessage, run local training, and return TrainingResultMessage and local weights."""
    started = time.perf_counter()
    keras.backend.clear_session()
    model = load_model()
    model.set_weights(bytes_to_weights(global_weights))

    cfg = config
    datasets = load_data(cfg)
    train = datasets["train"]
    validation = datasets["val"]
    X_train, y_train = train.inputs, train.labels
    X_val = validation.inputs if validation.num_examples else None
    y_val = validation.labels if validation.num_examples else None

    class_weights = get_class_weights(y_train) if cfg.balance else None

    setup_seconds = time.perf_counter() - started
    training_started = time.perf_counter()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if validation.num_examples else None,
        epochs=cfg.local_epochs,
        batch_size=cfg.batch_size,
        shuffle=True,
        class_weight=class_weights,
        verbose=cfg.verbose,
        callbacks=[
            TrainingProgressCallback(
                round_dir(msg.session_id, msg.client_id, msg.round_id)
            )
        ],
    )
    training_seconds = time.perf_counter() - training_started

    def final_history_value(name: str) -> float | None:
        values = history.history.get(name)
        return float(values[-1]) if values else None

    train_accuracy = final_history_value("accuracy")

    local_weights = weights_to_bytes(model.get_weights())

    result = TrainingResultMessage(
        session_id=msg.session_id,
        round_id=msg.round_id,
        client_id=msg.client_id,
        metrics=TrainingMetrics(
            num_examples=len(X_train),
            train_loss=final_history_value("loss"),
            train_accuracy=train_accuracy,
            train_f1_weighted=final_history_value("f1_weighted"),
            val_loss=final_history_value("val_loss"),
            val_accuracy=final_history_value("val_accuracy"),
            val_f1_weighted=final_history_value("val_f1_weighted"),
            train_acc=train_accuracy,
            setup_seconds=setup_seconds,
            training_seconds=training_seconds,
            total_seconds=time.perf_counter() - started,
            peak_ram_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        ),
    )

    del history, datasets, X_train, y_train, model
    keras.backend.clear_session()
    gc.collect()
    return result, local_weights

def run_evaluation(
    msg: EvaluationInitMessage,
    config: TrainingConfig,
    global_weights: bytes,
) -> EvaluationResultMessage:
    """Evaluate aggregated weights on this client's local test split."""
    keras.backend.clear_session()
    model = load_model()
    model.set_weights(bytes_to_weights(global_weights))
    datasets = load_data(config)
    test = datasets["test"]
    if not test.num_examples:
        raise RuntimeError(
            "The local test split is empty; configure a non-zero test_size "
            "with enough participants for evaluation"
        )
    results = model.evaluate(
        test.inputs,
        test.labels,
        batch_size=config.batch_size,
        verbose=config.verbose,
        return_dict=True,
    )
    loss = results.get("loss")
    probabilities = model.predict(
        test.inputs,
        batch_size=config.batch_size,
        verbose=config.verbose,
    )
    truth = np.argmax(test.labels, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    labels = np.arange(len(ACTIVITIES))
    matrix = confusion_matrix(truth, predictions, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        predictions,
        labels=labels,
        zero_division=0,
    )
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else None
    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))
    weighted_f1 = float(np.average(f1, weights=support)) if support.sum() else None
    per_class = {
        activity: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, activity in enumerate(ACTIVITIES)
    }
    result = EvaluationResultMessage(
        session_id=msg.session_id,
        round_id=msg.round_id,
        client_id=msg.client_id,
        metrics=EvaluationMetrics(
            num_examples=test.num_examples,
            eval_loss=float(loss) if loss is not None else None,
            eval_accuracy=accuracy,
            eval_f1_macro=macro_f1,
            eval_f1_weighted=weighted_f1,
            eval_precision_macro=macro_precision,
            eval_recall_macro=macro_recall,
            confusion_matrix=matrix.astype(int).tolist(),
            per_class=per_class,
        ),
    )
    del results, probabilities, datasets, test, model
    keras.backend.clear_session()
    gc.collect()
    return result
