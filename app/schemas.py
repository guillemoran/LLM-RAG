"""Request/response schemas.

Pydantic validates the incoming payload automatically, so the route handler
only ever sees well-formed data. The request shape matches the one defined in
the challenge: {"user_name": "...", "question": "..."}.
"""

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    user_name: str = Field(..., min_length=1, examples=["John Doe"])
    question: str = Field(..., min_length=1, examples=["Who is Zara?"])


class AnswerResponse(BaseModel):
    user_name: str
    question: str
    answer: str