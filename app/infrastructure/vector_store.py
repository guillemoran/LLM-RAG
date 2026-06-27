"""ChromaDB vector store wrapper (infrastructure layer).

A PersistentClient is used so the index is built once by the ingestion script
and survives restarts; the API only reads from it. Embeddings are computed by
our own LLM client and passed in explicitly, keeping the retrieval logic
provider-agnostic. Similarity uses cosine distance.
"""

from typing import List

import chromadb

from app.config import settings


class VectorStore:
    def __init__(self, path: str = None, collection_name: str = None) -> None:
        self._path = path or settings.chroma_path
        self._name = collection_name or settings.collection_name
        self._client = chromadb.PersistentClient(path=self._path)
        self._collection = self._client.get_or_create_collection(
            name=self._name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """Drop and recreate the collection so ingestion is idempotent."""
        try:
            self._client.delete_collection(self._name)
        except Exception:
            pass  # collection may not exist yet on first run
        self._collection = self._client.get_or_create_collection(
            name=self._name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
    ) -> None:
        self._collection.add(ids=ids, embeddings=embeddings, documents=documents)

    def query(self, embedding: List[float], n_results: int = 1) -> List[str]:
        """Return the most relevant chunk texts for a query embedding."""
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
        )
        documents = result.get("documents") or []
        return documents[0] if documents else []

    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception:
            # Collection may not exist yet (e.g. ingestion not run). Treat as empty.
            return 0