# RAG Q&A API

A small Retrieval-Augmented Generation (RAG) system that answers questions
about a source document. It exposes a REST API and a simple web UI. Built with
FastAPI, ChromaDB and Cohere.

Each answer is: a single sentence, in the same language as the question
(Spanish / English / Portuguese), written in the third person, with summarizing
emojis, and based only on the document. The same question always returns the
same answer.

## How it works

The system has two phases:

- **Ingestion** (run once): the document is split into one chunk per thematic
  paragraph, each chunk is embedded with Cohere, and the vectors are stored in
  a persistent ChromaDB collection.
- **Query** (per request): the question is embedded, the most relevant chunk is
  retrieved by cosine similarity, and Cohere generates the final answer from
  that context.

## Requirements

- Python 3.10+
- A Cohere API key (free trial, no credit card): https://dashboard.cohere.com

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Cohere key
#    Create a file named .env in the project root with:
#    COHERE_API_KEY=your-trial-key
```

## Run

```bash
# 1. Build the index (run once, or whenever the document changes)
python -m scripts.ingest

# 2. Start the API
uvicorn app.main:app --reload
```

Then open:

- Web UI: http://127.0.0.1:8000/
- Interactive API docs (Swagger): http://127.0.0.1:8000/docs

## API

`POST /ask`

Request body:

```json
{
  "user_name": "John Doe",
  "question": "Who is Zara?"
}
```

Response:

```json
{
  "user_name": "John Doe",
  "question": "Who is Zara?",
  "answer": "Zara is an intrepid explorer who ... 🚀🌌"
}
```

`GET /health` returns the service status and the number of indexed chunks.

## Project structure
.

├── app/

│   ├── config.py              # central configuration (env vars)

│   ├── schemas.py             # request/response models

│   ├── main.py                # FastAPI app and endpoints

│   ├── core/                  # domain logic

│   │   ├── document.py        # load and chunk the document

│   │   ├── prompt.py          # answer rules and language detection

│   │   └── rag_service.py     # RAG orchestration + answer cache

│   └── infrastructure/        # external services

│       ├── llm.py             # Cohere client (chat + embeddings)

│       └── vector_store.py    # ChromaDB client

├── scripts/

│   └── ingest.py              # one-time ingestion pipeline

├── data/                      # source document

├── static/                    # web UI

└── requirements.txt

## Design decisions

- **Chunking by paragraph.** The document is a set of self-contained thematic
  paragraphs, so one chunk per paragraph keeps each answer fully inside a single
  chunk and makes top-1 retrieval sufficient.
- **Determinism via cache.** `temperature=0` reduces variation, but an in-memory
  answer cache guarantees that the same question always returns the exact same
  answer (text and emojis), independent of the provider.
- **Language handled in code.** The answer language is detected from the question
  and passed to the model as an explicit instruction, so it never echoes the
  document's language by mistake.
- **Clean layering.** The provider lives behind a single client class, so
  swapping Cohere for another provider only touches `infrastructure/`.