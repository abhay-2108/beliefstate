from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable
from beliefstate.models import Belief


CATEGORY_LABELS = {
    "identity": "Identity",
    "technical": "Technical Decisions",
    "planning": "Tasks & Planning",
    "constraint": "Constraints & Requirements",
    "state": "Current State",
    "general": "Established Facts",
}


@runtime_checkable
class Store(Protocol):
    """Protocol for a Belief Storage backend."""

    async def add_belief(self, session_id: str, belief: Belief) -> None:
        """Add or overwrite a belief in the store."""
        ...

    async def get_beliefs(
        self, session_id: str, conversation_id: Optional[str] = None
    ) -> List[Belief]:
        """Retrieve all beliefs for a given session."""
        ...

    async def search_beliefs(
        self,
        session_id: str,
        embedding: List[float],
        threshold: float = 0.0,
        limit: int = 5,
        conversation_id: Optional[str] = None,
    ) -> List[Belief]:
        """Search the store for beliefs semantically similar to the target embedding."""
        ...

    async def remove_belief(
        self,
        session_id: str,
        subject: str,
        predicate: str,
        conversation_id: Optional[str] = None,
    ) -> None:
        """Remove a specific belief based on its subject and predicate."""
        ...

    async def update_belief(self, session_id: str, belief: Belief) -> None:
        """Update an existing belief."""
        ...

    async def clear(self, session_id: str) -> None:
        """Clear all beliefs for a given session."""
        ...

    async def belief_count(self, session_id: str) -> int:
        """Return the number of beliefs for a session without full deserialization."""
        ...

    async def health_check(self) -> bool:
        """Verify the store backend is reachable and functional."""
        ...

    async def upsert(self, belief: Belief) -> bool:
        """Insert or update a belief with turn-based optimistic concurrency.

        Returns True if the belief was written, False if discarded (stale write).
        """
        ...

    async def get_by_key(
        self,
        subject: str,
        predicate: str,
        session_id: str,
        conversation_id: Optional[str] = None,
    ) -> Optional[Belief]:
        """Retrieve a single belief by its composite key."""
        ...

    async def get_audit_history(
        self,
        session_id: str,
        subject: str,
        predicate: str,
    ) -> List[Dict[str, Any]]:
        """Return audit trail for a specific belief."""
        ...


def summary_for_prompt(
    beliefs: List[Belief],
    max_beliefs: int = 50,
    max_speculative_beliefs: int = 5,
) -> str:
    """Format beliefs into a structured, category-grouped summary for prompt injection.

    Deduplicates by (subject, predicate) keeping only the latest belief per key.
    Groups beliefs by category and excludes superseded beliefs.
    """
    # B5: Deduplicate by (subject, predicate) keeping highest turn (latest)
    deduped: Dict[Tuple[str, str], Belief] = {}
    for b in beliefs:
        key = (b.subject.lower(), b.predicate.lower())
        existing = deduped.get(key)
        if existing is None or b.turn > existing.turn:
            deduped[key] = b
    beliefs = list(deduped.values())

    real = [b for b in beliefs if not getattr(b, "is_hypothetical", False)]
    speculative = [b for b in beliefs if getattr(b, "is_hypothetical", False)]

    real.sort(key=lambda b: (b.confidence, b.turn), reverse=True)
    real = real[:max_beliefs]

    categories: Dict[str, List[Belief]] = {}
    for b in real:
        cat = b.category or "general"
        categories.setdefault(cat, []).append(b)

    lines = ["Established facts from this conversation:\n"]
    for cat, cat_beliefs in categories.items():
        lines.append(f"[{CATEGORY_LABELS.get(cat, cat)}]")
        for b in cat_beliefs:
            lines.append(f"- {b.subject} {b.predicate} {b.value}")
        lines.append("")

    if speculative:
        lines.append("[Speculative / Under Consideration]")
        for b in speculative[:max_speculative_beliefs]:
            lines.append(f"- {b.subject} {b.predicate} {b.value} (not committed)")

    return "\n".join(lines)
