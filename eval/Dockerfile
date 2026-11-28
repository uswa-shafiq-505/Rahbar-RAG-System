# Image for the evaluation and gate Jobs (Assignment 03). Build context: the REPOSITORY ROOT:
#   docker build -f eval/Dockerfile -t rahbar-eval:dev .   (make eval-image does this and loads it)
# It carries the suite (eval/), the api code the suite may import (api/app), the gate and trace
# scripts (scripts/) and the default thresholds (config/). The thresholds the Jobs really use come
# from the gate-thresholds ConfigMap, so changing them never needs a new image.
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY eval/requirements.txt .
RUN python -m venv /opt/venv && /opt/venv/bin/pip install -r requirements.txt

FROM python:3.12-slim
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    PYTHONPATH=/app/eval:/app/api
RUN groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid 10001 --no-create-home app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY api/app ./api/app
COPY eval ./eval
COPY scripts ./scripts
COPY config ./config
USER 10001:10001
CMD ["python", "scripts/gate.py", "--help"]
