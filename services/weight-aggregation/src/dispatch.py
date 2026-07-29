"""Dispatches a TrainingInitMessage + global weights to each known
participant's Model Training Service."""
import logging

from common.http_client import post_message
from common.messages import TrainingConfig, TrainingInitMessage

logger = logging.getLogger(__name__)


def dispatch_round(round_id: int, participant_urls: dict[str, str], global_weights: bytes,
                    config: TrainingConfig, results_path: str, reply_url: str) -> None:
    """participant_urls maps participant_id -> that participant's /train
    URL (from static config - see PARTICIPANTS_JSON)."""
    for participant_id, url in participant_urls.items():
        msg = TrainingInitMessage(
            round_id=round_id,
            participant_id=participant_id,
            results_path=results_path,
            config=config,
            reply_url=reply_url,
        )
        try:
            post_message(url, msg, weights=global_weights)
            logger.info("Dispatched round %d to '%s' at %s", round_id, participant_id, url)
        except Exception:
            logger.exception("Failed to dispatch round %d to '%s'", round_id, participant_id)