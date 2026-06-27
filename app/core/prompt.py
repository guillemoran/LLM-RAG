"""Prompt construction.

The system prompt encodes the five answer requirements from the challenge.
The instructions are written in English (the model follows them regardless of
the answer language), but the answer language is resolved in code (see
detect_language) and injected as an explicit order, because the model tends to
echo the context's language (Spanish) instead of the question's. Determinism is
handled at the API level (temperature=0 + seed); a fixed prompt keeps the rest
of the input stable so identical questions yield identical answers.
"""


SYSTEM_PROMPT = (
    "You are a question-answering assistant that answers strictly based on the "
    "provided context.\n"
    "Follow these rules for EVERY answer:\n"
    "1. Answer in exactly ONE sentence.\n"
    "2. CRITICAL: Your answer MUST be written in the same language as the "
    "question. If the question is in English, answer in English. If in "
    "Spanish, answer in Spanish. If in Portuguese, answer in Portuguese. "
    "The context is written in Spanish, but you must IGNORE the context's "
    "language and use ONLY the question's language for your answer.\n"
    "3. Always answer in the THIRD PERSON.\n"
    "4. Include one or more emojis inside the sentence that summarize its "
    "content.\n"
    "5. Use ONLY the provided context. If the context does not contain the "
    "answer, say so in one sentence, in the question's language.\n"
    "Do not add greetings, prefaces, or any text beyond the single sentence."
)


def detect_language(text: str) -> str:
    """Lightweight language detector for ES / EN / PT based on common words.

    Good enough for short questions in three known languages; avoids pulling in
    a heavy dependency. Returns the explicit instruction to inject into the
    prompt so the model never has to infer the language itself.
    """
    lowered = f" {text.lower()} "

    english_markers = [" who ", " what ", " where ", " when ", " why ", " how ",
                       " is ", " are ", " was ", " were ", " did ", " does ",
                       " the ", " of ", " to ", " do "]
    portuguese_markers = [" quem ", " qual ", " onde ", " quando ", " porque ",
                          " como ", " é ", " são ", " foi ", " que ", " da ",
                          " do ", " decidiu "]

    english_score = sum(1 for word in english_markers if word in lowered)
    portuguese_score = sum(1 for word in portuguese_markers if word in lowered)

    if english_score > portuguese_score and english_score > 0:
        return "English"
    if portuguese_score > 0:
        return "Portuguese"
    return "Spanish"


def build_user_prompt(context: str, question: str) -> str:
    """Combine retrieved context and the user question into one message."""
    language = detect_language(question)
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer in ONE sentence. You MUST write your answer in {language}."
    )