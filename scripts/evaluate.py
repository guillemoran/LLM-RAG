"""Retrieval evaluation: measures hit rate.

Hit rate = fraction of test questions for which retrieval returns the correct
chunk. We identify the expected chunk by a unique keyword it must contain
(e.g. the Zara chunk must contain "Zara"). This evaluates ONLY retrieval,
not generation, so it is fast and cheap (no chat calls).

Run with:
    python -m scripts.evaluate
"""

from app.config import settings
from app.infrastructure.llm import CohereClient
from app.infrastructure.vector_store import VectorStore

# Each test case: the question, and a keyword the correct chunk must contain.
# Questions span Spanish, English and Portuguese to test multilingual retrieval.
TEST_CASES = [
    # Space theme (Zara)
    {"question": "¿Quién es Zara?", "expected_keyword": "Zara"},
    {"question": "Who is Zara?", "expected_keyword": "Zara"},
    {"question": "Quem é Zara?", "expected_keyword": "Zara"},
    {"question": "¿Qué pasa en la galaxia de Zenthoria?", "expected_keyword": "Zara"},
    # Tech theme (Alex)
    {"question": "¿Quién es Alex?", "expected_keyword": "Alex"},
    {"question": "Who is the young engineer?", "expected_keyword": "Alex"},
    {"question": "¿Qué es la singularidad en la historia?", "expected_keyword": "Alex"},
    # Nature theme (Luz de Luna)
    {"question": "¿Qué es la flor mágica?", "expected_keyword": "Luz de Luna"},
    {"question": "What is the magical flower?", "expected_keyword": "Luz de Luna"},
    {"question": "Qual é a flor mágica?", "expected_keyword": "Luz de Luna"},
    # Short story theme (Emma)
    {"question": "¿Quién es Emma?", "expected_keyword": "Emma"},
    {"question": "What did Emma receive?", "expected_keyword": "Emma"},
    # Hero theme (Sombra Silenciosa)
    {"question": "¿Quién es Sombra Silenciosa?", "expected_keyword": "Sombra Silenciosa"},
    {"question": "Who is the forgotten hero?", "expected_keyword": "Sombra Silenciosa"},
]


def main() -> None:
    llm = CohereClient()
    store = VectorStore()

    if store.count() == 0:
        print("Index is empty. Run `python -m scripts.ingest` first.")
        return

    hits = 0
    total = len(TEST_CASES)
    print(f"Evaluating retrieval on {total} test questions...\n")

    for case in TEST_CASES:
        question = case["question"]
        expected = case["expected_keyword"]

        # Use the real retrieval pipeline.
        embedding = llm.embed_query(question)
        chunks = store.query(embedding, n_results=settings.top_k)
        retrieved = chunks[0] if chunks else ""

        is_hit = expected.lower() in retrieved.lower()
        if is_hit:
            hits += 1

        mark = "OK " if is_hit else "MISS"
        print(f"[{mark}] {question}  ->  expected: '{expected}'")

    hit_rate = hits / total * 100
    print(f"\nHit rate: {hits}/{total} = {hit_rate:.1f}%")


if __name__ == "__main__":
    main()