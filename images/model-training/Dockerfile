FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/KNoWS-Aggregator/k8s-fl-services"
LABEL org.opencontainers.image.description="Federated TensorFlow model-training service"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    DATA_DIR=/app/data \
    RESULTS_DIR=/app/data/model-training/results \
    PORT=8080

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends --yes libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY libs/common /build/common
RUN pip install --no-cache-dir /build/common

COPY libs/fl-model /build/fl-model
RUN pip install --no-cache-dir /build/fl-model

COPY images/model-training/pyproject.toml /build/model-training/pyproject.toml
COPY images/model-training/src /build/model-training/src
RUN pip install --no-cache-dir /build/model-training \
    && rm -rf /build

RUN mkdir -p /app/data/model-training && chown -R 65532:65532 /app
USER 65532:65532

EXPOSE 8080
VOLUME ["/app/data"]

CMD ["python", "-m", "model_training.main"]
