"""Application configuration.

All tunable values live here and can be overridden through environment
variables (or a local .env file). Keeping configuration in a single place
makes the rest of the code free of hard-coded literals.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Provider credentials -------------------------------------------------
    # Get a free trial key at https://dashboard.cohere.com (no credit card).
    cohere_api_key: str = ""

    # --- Models ---------------------------------------------------------------
    # Chat model used to generate the final answer. command-r is cheap and
    # available on the free trial tier. temperature is forced to 0 elsewhere.
    chat_model: str = "command-r-08-2024"
    # Embedding model. embed-v4.0 is multilingual by default (ES/EN/PT).
    # If a trial key rejects v4, fall back to "embed-multilingual-v3.0".
    embed_model: str = "embed-v4.0"
    # Fixed seed reinforces deterministic generation alongside temperature=0.
    seed: int = 42

    # --- Document & vector store ---------------------------------------------
    document_path: str = "data/documento.docx"
    chroma_path: str = "chroma_db"
    collection_name: str = "documents"

    # Number of chunks retrieved per question. The source document is organized
    # as self-contained thematic paragraphs, so the top-1 chunk already holds
    # the full answer; raising this only adds noise for this corpus.
    top_k: int = 1


settings = Settings()