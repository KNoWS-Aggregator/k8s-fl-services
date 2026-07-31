"""Training logic for the Training Job container."""
import gc
import os
import warnings
from pathlib import Path

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
from .support import get_save_name, save_training_history, load_data, get_class_weights

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import keras


def run_training(
    msg: TrainingInitMessage,
    config: TrainingConfig,
    global_weights: bytes,
) -> tuple[TrainingResultMessage, bytes]:
    """Handle TrainingInitMessage, run local training, and return TrainingResultMessage and local weights."""
    keras.backend.clear_session()
    model = load_model()
    model.set_weights(bytes_to_weights(global_weights))

    cfg = config
    save_name = get_save_name(cfg)
    datasets = load_data(cfg)
    train = datasets["train"]
    validation = datasets["val"]
    X_train, y_train = train.inputs, train.labels
    X_val = validation.inputs if validation.num_examples else None
    y_val = validation.labels if validation.num_examples else None

    class_weights = get_class_weights(y_train) if cfg.balance else None

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if validation.num_examples else None,
        epochs=cfg.local_epochs,
        batch_size=cfg.batch_size,
        shuffle=True,
        class_weight=class_weights,
        verbose=cfg.verbose,
    )

    train_loss = history.history.get("loss", [None])[-1]
    train_acc = history.history.get("accuracy")
    train_acc = train_acc[-1] if train_acc else None

    save_training_history(
        n_round=msg.round_id,
        client_name=f"client_{msg.client_id}",
        history=history.history,
        base_path=Path(os.getenv("RESULTS_DIR", "/app/data/model-training/results")) / save_name,
    )

    result = TrainingResultMessage(
        session_id=msg.session_id,
        round_id=msg.round_id,
        client_id=msg.client_id,
        metrics=TrainingMetrics(num_examples=len(X_train), train_loss=train_loss, train_acc=train_acc),
    )
    local_weights = weights_to_bytes(model.get_weights())

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
    accuracy = results.get("accuracy")
    result = EvaluationResultMessage(
        session_id=msg.session_id,
        round_id=msg.round_id,
        client_id=msg.client_id,
        metrics=EvaluationMetrics(
            num_examples=test.num_examples,
            eval_loss=float(loss) if loss is not None else None,
            eval_accuracy=float(accuracy) if accuracy is not None else None,
        ),
    )
    del results, datasets, test, model
    keras.backend.clear_session()
    gc.collect()
    return result
