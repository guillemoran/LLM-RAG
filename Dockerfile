# Start from a slim official Python image (small but complete).
FROM python:3.11-slim

# Set the working directory inside the container.
WORKDIR /app

# Install dependencies first, separately, so Docker can cache this layer and skip reinstalling when only the source code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project into the container.
COPY . .

# Build the vector index at image build time so the API is ready to serve.
# If you prefer to ingest at runtime instead, remove the next line.
# RUN python -m scripts.ingest

# Document the port the app listens on.
EXPOSE 8000

# Make the startup script executable.
RUN chmod +x entrypoint.sh

# On container start: run ingestion, then launch the API.
ENTRYPOINT ["./entrypoint.sh"]