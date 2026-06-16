from typing import List, Protocol, runtime_checkable
from beliefstate.models import Belief


@runtime_checkable
class Store(Protocol):
    """Protocol for a Belief Storage backend."""

    async def add_belief(self, session_id: str, belief: Belief) -> None:
        """Add or overwrite a belief in the store."""
        ...

    async def get_beliefs(self, session_id: str) -> List[Belief]:
        """Retrieve all beliefs for a given session."""
        ...

    async def search_beliefs(
        self,
        session_id: str,
        embedding: List[float],
        threshold: float = 0.0,
        limit: int = 5,
    ) -> List[Belief]:
        """Search the store for beliefs semantically similar to the target embedding."""
        ...

    async def remove_belief(
        self, session_id: str, subject: str, predicate: str
    ) -> None:
        """Remove a specific belief based on its subject and predicate."""
        ...

    async def update_belief(self, session_id: str, belief: Belief) -> None:
        """Update an existing belief."""
        ...

    async def clear(self, session_id: str) -> None:
        """Clear all beliefs for a given session."""
        ...
