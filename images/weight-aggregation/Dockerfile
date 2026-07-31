FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/KNoWS-Aggregator/k8s-fl-services"
LABEL org.opencontainers.image.description="Federated session coordinator and weight aggregation service"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    DATA_DIR=/app/data \
    PORT=8080

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends --yes libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY libs/common /build/common
RUN pip install --no-cache-dir /build/common

COPY libs/fl-model /build/fl-model
RUN pip install --no-cache-dir /build/fl-model

COPY images/weight-aggregation/pyproject.toml /build/weight-aggregation/pyproject.toml
COPY images/weight-aggregation/src /build/weight-aggregation/src
RUN pip install --no-cache-dir /build/weight-aggregation \
    && rm -rf /build

RUN mkdir -p /app/data/weight-aggregation && chown -R 65532:65532 /app
USER 65532:65532

EXPOSE 8080
VOLUME ["/app/data"]

CMD ["python", "-m", "weight_aggregation.main"]
