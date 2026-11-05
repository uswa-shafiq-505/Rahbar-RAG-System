# RAHBAR indexer image: the Job's image, the only writer. Build context: indexer/
#   make stage-corpus SRC=corpus/raw     (copies the documents into indexer/corpus-stage/)
#   docker build -t rahbar-indexer:dev indexer
# The documents are baked into the image under /corpus (CORPUS_DIR). Why: the Job then needs
# no volume, no network fetch at start-up, and "what was indexed?" has the same one-word
# answer as everything else: the image tag. Explained in docs/PIPELINE.md and ADR 0001.
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv && /opt/venv/bin/pip install -r requirements.txt

FROM python:3.12-slim
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    CORPUS_DIR=/corpus
# OCR for the scanned PDF (corpus property 3): tesseract + poppler (pdf2image).
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng poppler-utils \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid 10001 --no-create-home app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY pipeline ./pipeline
COPY corpus-stage/ /corpus/
USER 10001:10001
CMD ["python", "-m", "pipeline.run"]
