FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/KNoWS-Aggregator/k8s-fl-services"
LABEL org.opencontainers.image.description="Continuously prepares and aggregates Kvasir participant data"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/data \
    PORT=8080

WORKDIR /app

COPY images/data-preparation/pyproject.toml ./pyproject.toml
COPY images/data-preparation/src ./src

RUN pip install --no-cache-dir .

RUN mkdir -p /app/data && chown -R 65532:65532 /app
USER 65532:65532

EXPOSE 8080
VOLUME ["/app/data"]

CMD ["python", "-m", "data_preparation.main"]
