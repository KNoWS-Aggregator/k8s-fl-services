import numpy as np

from common.messages import TrainingConfig, TrainingInitMessage, TrainingMetrics
from common.weight_io import bytes_to_weights, weights_to_bytes

from weight_aggregation.aggregate import federated_average
from weight_aggregation.round_state import RoundTracker


def test_training_message_uses_federated_client_identity():
    message = TrainingInitMessage(
        session_id="session-1",
        round_id=4,
        client_id="hospital-a",
        reply_url="http://weight-aggregation/results",
    )

    assert message.client_id == "hospital-a"
    assert "participant_id" not in message.model_dump()


def test_round_tracking_and_fedavg_are_keyed_by_client():
    tracker = RoundTracker()
    tracker.start_round(1, {"hospital-a", "hospital-b"})
    metrics = TrainingMetrics(num_examples=1)
    tracker.record_result(1, "hospital-a", b"a", metrics)
    state = tracker.record_result(1, "hospital-b", b"b", metrics)

    assert state.is_complete
    assert set(state.weights_by_client) == {"hospital-a", "hospital-b"}

    averaged = federated_average(
        {
            "hospital-a": weights_to_bytes([np.array([1.0])]),
            "hospital-b": weights_to_bytes([np.array([3.0])]),
        },
        {
            "hospital-a": TrainingMetrics(num_examples=1),
            "hospital-b": TrainingMetrics(num_examples=3),
        },
    )
    assert np.allclose(bytes_to_weights(averaged)[0], np.array([2.5]))
