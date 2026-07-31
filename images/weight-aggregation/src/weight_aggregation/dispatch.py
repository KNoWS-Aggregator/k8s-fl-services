"""Dispatches a TrainingInitMessage + global weights to each known
participant's Model Training Service."""
import logging

from common.http_client import post_message
from common.messages import EvaluationInitMessage, TrainingInitMessage

logger = logging.getLogger(__name__)


def dispatch_round(session_id: str, round_id: int, client_urls: dict[str, str],
                   global_weights: bytes, reply_url: str) -> set[str]:
    """Send one round to each federated client's model-training endpoint."""
    failed = set()
    for client_id, url in client_urls.items():
        msg = TrainingInitMessage(
            session_id=session_id,
            round_id=round_id,
            client_id=client_id,
            reply_url=reply_url,
        )
        try:
            post_message(url, msg, weights=global_weights)
            logger.info("Dispatched round %d to client '%s' at %s", round_id, client_id, url)
        except Exception:
            logger.exception("Failed to dispatch round %d to client '%s'", round_id, client_id)
            failed.add(client_id)
    return failed


def dispatch_evaluation(
    session_id: str,
    round_id: int,
    client_urls: dict[str, str],
    global_weights: bytes,
    reply_url: str,
) -> set[str]:
    """Send aggregated round weights to each client for local evaluation."""
    failed = set()
    for client_id, url in client_urls.items():
        msg = EvaluationInitMessage(
            session_id=session_id,
            round_id=round_id,
            client_id=client_id,
            reply_url=reply_url,
        )
        try:
            post_message(url, msg, weights=global_weights)
            logger.info(
                "Dispatched evaluation for round %d to client '%s' at %s",
                round_id,
                client_id,
                url,
            )
        except Exception:
            logger.exception(
                "Failed to dispatch evaluation for round %d to client '%s'",
                round_id,
                client_id,
            )
            failed.add(client_id)
    return failed
