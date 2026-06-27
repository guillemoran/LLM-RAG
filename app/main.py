"""FastAPI application (API layer).

Exposes:
  GET  /health  -> liveness check + number of indexed chunks
  POST /ask     -> answer a question about the document

Clients are built once at import time and reused across requests. Interactive
docs are available at /docs (Swagger) when the server is running.
"""

from fastapi import FastAPI, HTTPException

from app.core.rag_service import RAGService
from app.infrastructure.llm import CohereClient
from app.infrastructure.vector_store import VectorStore
from app.schemas import AnswerResponse, QuestionRequest
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="RAG Challenge API", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Shared singletons: the Cohere client is lazy, the Chroma client opens the
# persisted index created by scripts/ingest.py.
llm = CohereClient()
store = VectorStore()
rag = RAGService(llm=llm, store=store)


@app.get("/")
def home():
    """Serve the colorful web UI at the root URL."""
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": store.count()}


@app.post("/ask", response_model=AnswerResponse)
def ask(payload: QuestionRequest):
    if store.count() == 0:
        raise HTTPException(
            status_code=503,
            detail="Index is empty. Run `python -m scripts.ingest` first.",
        )
    try:
        answer = rag.answer(payload.question)
    except Exception as error:  # surface provider/store failures as 502
        raise HTTPException(status_code=502, detail=str(error))

    return AnswerResponse(
        user_name=payload.user_name,
        question=payload.question,
        answer=answer,
    )
