#!/bin/sh
# Container startup script: build the index, then launch the API.
# Running ingestion here guarantees the vector store exists before the
# server starts serving requests.

set -e  # stop immediately if any command fails

echo "Running ingestion..."
python -m scripts.ingest

echo "Starting API..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000