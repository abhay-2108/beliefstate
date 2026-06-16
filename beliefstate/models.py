from typing import List
from pydantic import BaseModel, Field


class Belief(BaseModel):
    """Represents a single extracted factual belief."""

    subject: str
    predicate: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    turn: int
    source: str
    embedding: List[float] = Field(default_factory=list)
