"""Entrypoint for the Model Training Service.

A single long-running Deployment that listens for TrainingInitMessage
requests and runs training itself - no separate Job container. Training
runs as a background task so the HTTP response isn't blocked for the
duration of a round; the actual result is POSTed to reply_url once done.
"""
import logging
import os

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile

from common.http_client import post_message
from common.messages import TrainingInitMessage
from train import run_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


@app.post("/train")
async def train_endpoint(
    background_tasks: BackgroundTasks,
    message: str = Form(...),
    weights: UploadFile = File(...),
):
    msg = TrainingInitMessage.model_validate_json(message)

    global_weights = await weights.read()
    background_tasks.add_task(_run_and_report, msg, global_weights)
    return {"status": "accepted", "round_id": msg.round_id}


def _run_and_report(msg: TrainingInitMessage, global_weights: bytes) -> None:
    try:
        result, local_weights = run_training(msg, global_weights)
        post_message(msg.reply_url, result, weights=local_weights)
        logger.info("Reported result for round %d to %s", msg.round_id, msg.reply_url)
    except Exception:
        logger.exception("Training failed for round %d", msg.round_id)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))