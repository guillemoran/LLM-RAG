"""Cohere client wrapper (infrastructure layer).

This is the ONLY module that imports the Cohere SDK. The rest of the app depends
on these methods, not on Cohere directly, so swapping the provider (e.g. for
OpenAI) means rewriting just this file. The client is created lazily so the app
can boot (and /health can respond) even before a key is configured.
"""

from typing import List, Optional

import cohere

from app.config import settings


class CohereClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        chat_model: Optional[str] = None,
        embed_model: Optional[str] = None,
    ) -> None:
        self._api_key = api_key or settings.cohere_api_key
        self._chat_model = chat_model or settings.chat_model
        self._embed_model = embed_model or settings.embed_model
        self._client: Optional[cohere.ClientV2] = None

    def _ensure_client(self) -> cohere.ClientV2:
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "COHERE_API_KEY is not set. Add it to your .env file."
                )
            self._client = cohere.ClientV2(api_key=self._api_key)
        return self._client

    # Embeddings
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed corpus chunks (input_type tuned for indexing)."""
        return self._embed(texts, input_type="search_document")

    def embed_query(self, text: str) -> List[float]:
        """Embed a single user question (input_type tuned for querying)."""
        return self._embed([text], input_type="search_query")[0]

    def _embed(self, texts: List[str], input_type: str) -> List[List[float]]:
        response = self._ensure_client().embed(
            texts=texts,
            model=self._embed_model,
            input_type=input_type,
            embedding_types=["float"],
        )
        # The SDK exposes float embeddings as `.float_` (trailing underscore
        # because `float` is reserved); guard for both spellings across versions.
        embeddings = response.embeddings
        vectors = getattr(embeddings, "float_", None)
        if vectors is None:
            vectors = getattr(embeddings, "float", None)
        return [list(vector) for vector in vectors]

    # Generation
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate the final answer deterministically (temperature=0 + seed)."""
        response = self._ensure_client().chat(
            model=self._chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            seed=settings.seed,
        )
        return response.message.content[0].text.strip()