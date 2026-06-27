"""One-time ingestion pipeline.

Run this once before starting the API (or whenever the document changes):

    python -m scripts.ingest

Steps: load the document -> split into thematic chunks -> embed each chunk ->
persist chunks + embeddings in ChromaDB. The API then only reads this index.
"""

from app.config import settings
from app.core.document import load_document, split_into_chunks
from app.infrastructure.llm import CohereClient
from app.infrastructure.vector_store import VectorStore


def main() -> None:
    print(f"Loading document from: {settings.document_path}")
    text = load_document(settings.document_path)

    chunks = split_into_chunks(text)
    print(f"Document split into {len(chunks)} chunks (one per thematic paragraph).")

    llm = CohereClient()
    store = VectorStore()

    print("Resetting collection and embedding chunks...")
    store.reset()
    embeddings = llm.embed_documents(chunks)
    ids = [f"chunk-{index}" for index in range(len(chunks))]
    store.add(ids=ids, embeddings=embeddings, documents=chunks)

    print(
        f"Done. Stored {store.count()} chunks in collection "
        f"'{settings.collection_name}' at '{settings.chroma_path}'."
    )


if __name__ == "__main__":
    main()