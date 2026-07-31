"""Shared message schemas for inter-service and Job communication.

Weight arrays are NOT embedded in these JSON models - they're sent as a
separate binary part (.npz bytes) alongside the JSON control message in a
multipart HTTP request. See http_client.py for the send/receive helpers.
"""
from __future__ import annotations

from enum import Enum
from datetime import datetime
from typing import Annotated, Optional

from pydantic import BaseModel, Field

Identifier = Annotated[
    str,
    Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"),
]


class MessageType(str, Enum):
    TRAINING_INIT = "training_init"
    TRAINING_RESULT = "training_result"
    TRAINING_FAILURE = "training_failure"
    EVALUATION_INIT = "evaluation_init"
    EVALUATION_RESULT = "evaluation_result"
    EVALUATION_FAILURE = "evaluation_failure"

class TrainingConfig(BaseModel):
    """Hyperparameters/config forwarded to a Training Job."""
    num_rounds: int = 1
    local_epochs: int = 1
    batch_size: int = 32
    verbose: int = 0
    scaling: bool = False
    balance: bool = False
    group_size: int = 0
    val_size: float = 0.2
    test_size: float = 0.2
    gap: int = 0
    model_prefix: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d"))


class TrainingInitMessage(BaseModel):
    """Weight Aggregation Service -> Model Training Service.
    Global weights travel as a separate binary part, not a field here.
    """
    type: MessageType = MessageType.TRAINING_INIT
    session_id: Identifier
    round_id: int
    client_id: Identifier
    reply_url: str


class TrainingMetrics(BaseModel):
    num_examples: int = Field(gt=0)
    train_loss: Optional[float] = None
    train_acc: Optional[float] = None


class TrainingResultMessage(BaseModel):
    """Training Job -> Weight Aggregation Service. Diagram arrow 5.
    Local weights travel as a separate binary part, not a field here.
    """
    type: MessageType = MessageType.TRAINING_RESULT
    session_id: Identifier
    round_id: int
    client_id: Identifier
    metrics: TrainingMetrics


class TrainingFailureMessage(BaseModel):
    """Model Training Service -> Weight Aggregation Service on failure."""
    type: MessageType = MessageType.TRAINING_FAILURE
    session_id: Identifier
    round_id: int
    client_id: Identifier
    error: str


class ClientSessionStart(BaseModel):
    session_id: Identifier
    client_id: Identifier
    expected_rounds: int = Field(ge=1)
    training_config: TrainingConfig = Field(default_factory=TrainingConfig)


class ClientSessionEnd(BaseModel):
    session_id: Identifier
    client_id: Identifier


class EvaluationInitMessage(BaseModel):
    type: MessageType = MessageType.EVALUATION_INIT
    session_id: Identifier
    round_id: int
    client_id: Identifier
    reply_url: str


class EvaluationMetrics(BaseModel):
    num_examples: int = Field(gt=0)
    eval_loss: Optional[float] = None
    eval_accuracy: Optional[float] = None


class EvaluationResultMessage(BaseModel):
    type: MessageType = MessageType.EVALUATION_RESULT
    session_id: Identifier
    round_id: int
    client_id: Identifier
    metrics: EvaluationMetrics


class EvaluationFailureMessage(BaseModel):
    type: MessageType = MessageType.EVALUATION_FAILURE
    session_id: Identifier
    round_id: int
    client_id: Identifier
    error: str
