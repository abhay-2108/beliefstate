from typing import List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class DeletionReceipt(BaseModel):
    """Auditable record of a GDPR data deletion request."""
    
    session_id: str = Field(description="Session ID that was deleted")
    beliefs_deleted: int = Field(description="Number of beliefs removed from store")
    deleted_at: datetime = Field(description="UTC timestamp when deletion completed")
    in_flight_tasks_drained: int = Field(default=0, description="Number of in-flight tasks that were drained before deletion")


class Belief(BaseModel):
    """Represents a single extracted factual belief."""

    subject: str
    predicate: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    turn: int
    source: str
    embedding: List[float] = Field(default_factory=list)
    embedding_model: str = Field(
        default="",
        description="Model used to generate the embedding (e.g., 'text-embedding-3-small', 'nomic-embed-text'). Empty string for backwards compatibility.",
    )
    embedding_dim: int = Field(
        default=0,
        description="Dimensionality of the embedding vector (e.g., 384 for MiniLM, 1536 for text-embedding-3-small). Prevents silent cosine corruption on model upgrade.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this belief was created.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID (user identity) this belief belongs to (optional, for schema flexibility).",
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Conversation ID (specific thread) this belief belongs to. Separates parallel conversations within same session.",
    )
    belief_type: str = Field(
        default="assertion",
        description="Type of belief: 'assertion' (normal), 'update' (temporal change), 'hypothetical' (if/imagine/example). Affects resolution strategy.",
    )
    is_hypothetical: bool = Field(
        default=False,
        description="True if belief is from a hypothetical scenario (if/imagine/example). Excluded from prompt injection.",
    )
    last_referenced_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this belief was last referenced/used in a session.",
    )
