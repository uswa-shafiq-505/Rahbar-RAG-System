# RAHBAR api image. Build context: api/   (docker build -t rahbar-api:dev api)
# Two stages: dependencies are installed in a builder, the runtime image has no pip cache,
# no compiler and runs as a non-root user with a read-only root filesystem (section 4.7).
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv && /opt/venv/bin/pip install -r requirements.txt

FROM python:3.12-slim
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp
# A fixed numeric UID, so Kubernetes can verify runAsNonRoot without reading /etc/passwd.
RUN groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid 10001 --no-create-home app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY app ./app
USER 10001:10001
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
