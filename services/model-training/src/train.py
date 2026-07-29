"""Training logic for the Training Job container."""
import gc
import warnings
from pathlib import Path

from common.messages import TrainingInitMessage, TrainingMetrics, TrainingResultMessage
from common.weight_io import bytes_to_weights, weights_to_bytes

from model import load_model
from support import get_save_name, save_training_history, load_data, get_class_weights

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import keras


def run_training(msg: TrainingInitMessage, global_weights: bytes) -> tuple[TrainingResultMessage, bytes]:
    """Handle TrainingInitMessage, run local training, and return TrainingResultMessage and local weights."""
    keras.backend.clear_session()
    model = load_model()
    model.set_weights(bytes_to_weights(global_weights))

    cfg = msg.config
    save_name = get_save_name(cfg)

    X_train, y_train, _ = load_data(d_set="train", scaling=cfg.scaling)
    X_val, y_val = (None, None)
    if cfg.val_size != 0:
        X_val, y_val, _ = load_data(d_set="val", scaling=cfg.scaling)

    class_weights = get_class_weights(y_train) if cfg.balance else None

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if cfg.val_size != 0 else None,
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
        client_name=f"participant_{msg.participant_id}",
        history=history.history,
        base_path=save_name,
    )

    # TODO: Save weights to PV instead of sending them back in the message
    # out_path = Path(msg.results_path) / "weights" / f"round_{msg.round_id}_partition_{msg.partition_id}.npz"
    # out_path.parent.mkdir(parents=True, exist_ok=True)
    # save_weights(str(out_path), model.get_weights())

    result = TrainingResultMessage(
        round_id=msg.round_id,
        participant_id=msg.participant_id,
        metrics=TrainingMetrics(num_examples=len(X_train), train_loss=train_loss, train_acc=train_acc),
    )
    local_weights = weights_to_bytes(model.get_weights())

    del history, X_train, y_train, _, model
    keras.backend.clear_session()
    gc.collect()
    return result, local_weights