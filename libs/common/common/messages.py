"""Shared message schemas for inter-service and Job communication.

Weight arrays are NOT embedded in these JSON models - they're sent as a
separate binary part (.npz bytes) alongside the JSON control message in a
multipart HTTP request. See http_client.py for the send/receive helpers.
"""
from __future__ import annotations

from enum import Enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    TRAINING_INIT = "training_init"
    TRAINING_RESULT = "training_result"
    EVALUATION_INIT = "evaluation_init"
    EVALUATION_RESULT = "evaluation_result"

class TrainingConfig(BaseModel):
    """Hyperparameters/config forwarded to a Training Job."""
    num_rounds: int = 1
    local_epochs: int = 1
    batch_size: int = 32
    verbose: int = 0
    scaling: Optional[str] = None
    balance: bool = False
    group_size: int = 0
    val_size: float = 0.0
    test_size: float = 0.0
    gap: int = 0
    model_prefix: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d"))


class TrainingInitMessage(BaseModel):
    """Weight Aggregation Service -> Model Training Service.
    Global weights travel as a separate binary part, not a field here.
    """
    type: MessageType = MessageType.TRAINING_INIT
    round_id: int
    participant_id: int
    config: TrainingConfig
    reply_url: str


class TrainingMetrics(BaseModel):
    num_examples: int
    train_loss: Optional[float] = None
    train_acc: Optional[float] = None


class TrainingResultMessage(BaseModel):
    """Training Job -> Weight Aggregation Service. Diagram arrow 5.
    Local weights travel as a separate binary part, not a field here.
    """
    type: MessageType = MessageType.TRAINING_RESULT
    round_id: int
    participant_id: int
    metrics: TrainingMetrics


class EvaluationInitMessage(BaseModel):
    type: MessageType = MessageType.EVALUATION_INIT
    round_id: int
    participant_id: int
    config: TrainingConfig
    reply_url: str


class EvaluationMetrics(BaseModel):
    num_examples: int
    eval_loss: Optional[float] = None
    eval_accuracy: Optional[float] = None


class EvaluationResultMessage(BaseModel):
    type: MessageType = MessageType.EVALUATION_RESULT
    round_id: int
    participant_id: int
    metrics: EvaluationMetrics