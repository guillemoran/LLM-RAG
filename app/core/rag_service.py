"""RAG orchestration (domain layer).

Ties the pieces together without knowing which provider or vector store sits
underneath: embed the question, retrieve the most relevant chunk, build the
prompt, and generate the answer.

A small in-memory cache guarantees determinism: the first time a question is
asked, its answer is stored; identical questions then return the stored answer
without calling the model again. This makes "same question -> same answer"
hold 100% regardless of any residual non-determinism on the provider side, and
also saves API calls.
"""

from app.config import settings
from app.core.prompt import SYSTEM_PROMPT, build_user_prompt
from app.infrastructure.llm import CohereClient
from app.infrastructure.vector_store import VectorStore


class RAGService:
    def __init__(self, llm: CohereClient, store: VectorStore) -> None:
        self._llm = llm
        self._store = store
        # Cache: normalized question -> answer. In-memory, resets on restart.
        self._cache: dict[str, str] = {}

    def answer(self, question: str) -> str:
        # Normalize so trivial variations map to the same cache entry.
        cache_key = question.strip().lower()
        if cache_key in self._cache:
            return self._cache[cache_key]

        query_embedding = self._llm.embed_query(question)
        chunks = self._store.query(query_embedding, n_results=settings.top_k)
        context = "\n\n".join(chunks) if chunks else ""
        user_prompt = build_user_prompt(context, question)
        answer = self._llm.generate(SYSTEM_PROMPT, user_prompt)

        self._cache[cache_key] = answer
        return answer