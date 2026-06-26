from typing import Dict, List, Optional, Any
from collections import OrderedDict
import logging
from beliefstate.store.base import Store
from beliefstate.store.utils import cosine_similarity
from beliefstate.models import Belief

logger = logging.getLogger(__name__)


class InMemoryBeliefStore(Store):
    """In-memory storage for beliefs with LRU eviction policy.

    Features:
    - Simple in-memory dictionary-based storage
    - Per-session belief storage
    - Global size limit (max_bytes) with LRU eviction
    - Turn-based optimistic concurrency on upsert

    NOT suitable for: production, multi-process, persistent data needs.
    """

    def __init__(self, max_bytes: int = 100 * 1024 * 1024):
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self._beliefs: Dict[str, OrderedDict[str, Belief]] = {}

    def _estimate_belief_size(self, belief: Belief) -> int:
        return len(belief.model_dump_json().encode("utf-8"))

    def _evict_lru_belief(self) -> None:
        if not self._beliefs:
            return
        for session_id, beliefs_dict in self._beliefs.items():
            if beliefs_dict:
                field, belief = next(iter(beliefs_dict.items()))
                belief_size = self._estimate_belief_size(belief)
                del beliefs_dict[field]
                self.current_bytes -= belief_size
                logger.debug(f"LRU eviction: removed {field} from session {session_id}")
                if not beliefs_dict:
                    del self._beliefs[session_id]
                return

    async def add_belief(self, session_id: str, belief: Belief) -> None:
        if session_id not in self._beliefs:
            self._beliefs[session_id] = OrderedDict()

        cid = belief.conversation_id or ""
        field = f"{(belief.subject or '').lower()}::{(belief.predicate or '').lower()}::{cid}"
        belief_size = self._estimate_belief_size(belief)

        if field in self._beliefs[session_id]:
            old_size = self._estimate_belief_size(self._beliefs[session_id][field])
            self.current_bytes -= old_size

        self._beliefs[session_id][field] = belief
        self.current_bytes += belief_size

        while self.current_bytes > self.max_bytes:
            self._evict_lru_belief()

    async def get_beliefs(
        self, session_id: str, conversation_id: Optional[str] = None
    ) -> List[Belief]:
        if session_id not in self._beliefs:
            return []

        beliefs = list(self._beliefs[session_id].values())

        if conversation_id:
            beliefs = [b for b in beliefs if b.conversation_id == conversation_id]

        for field in list(self._beliefs[session_id].keys()):
            self._beliefs[session_id].move_to_end(field)

        return beliefs

    async def search_beliefs(
        self,
        session_id: str,
        embedding: List[float],
        threshold: float = 0.0,
        limit: int = 5,
        conversation_id: Optional[str] = None,
    ) -> List[Belief]:
        beliefs = await self.get_beliefs(session_id, conversation_id)
        scored_beliefs = []

        for b in beliefs:
            if not b.embedding or not embedding:
                continue
            if len(b.embedding) != len(embedding):
                continue
            sim = cosine_similarity(b.embedding, embedding)
            if sim >= threshold:
                scored_beliefs.append((b, sim))

        scored_beliefs.sort(key=lambda x: x[1], reverse=True)
        return [sb[0] for sb in scored_beliefs[:limit]]

    async def get_by_key(
        self,
        subject: str,
        predicate: str,
        session_id: str,
        conversation_id: Optional[str] = None,
    ) -> Optional[Belief]:
        """Retrieve a single belief by its composite key."""
        if session_id not in self._beliefs:
            return None
        cid = conversation_id or ""
        field = f"{subject.lower()}::{predicate.lower()}::{cid}"
        return self._beliefs[session_id].get(field)

    async def upsert(self, belief: Belief) -> bool:
        """Insert or update a belief with turn-based optimistic concurrency.

        Returns True if written, False if discarded (stale write).
        """
        existing = await self.get_by_key(
            belief.subject or "",
            belief.predicate or "",
            belief.session_id or "",
            belief.conversation_id or "",
        )
        if existing and existing.turn > belief.turn:
            return False
        await self.add_belief(belief.session_id or "", belief)
        return True

    async def remove_belief(
        self,
        session_id: str,
        subject: str,
        predicate: str,
        conversation_id: Optional[str] = None,
    ) -> None:
        if session_id not in self._beliefs:
            return

        cid = conversation_id or ""
        field = f"{subject.lower()}::{predicate.lower()}::{cid}"
        if field in self._beliefs[session_id]:
            belief = self._beliefs[session_id][field]
            belief_size = self._estimate_belief_size(belief)
            del self._beliefs[session_id][field]
            self.current_bytes -= belief_size

            if not self._beliefs[session_id]:
                del self._beliefs[session_id]

    async def update_belief(self, session_id: str, belief: Belief) -> None:
        await self.add_belief(session_id, belief)

    async def clear(self, session_id: str) -> None:
        if session_id in self._beliefs:
            total_size = sum(
                self._estimate_belief_size(b)
                for b in self._beliefs[session_id].values()
            )
            del self._beliefs[session_id]
            self.current_bytes -= total_size
            logger.debug(f"Cleared session {session_id} (freed {total_size} bytes)")

    async def belief_count(self, session_id: str) -> int:
        return len(self._beliefs.get(session_id, {}))

    async def health_check(self) -> bool:
        return True

    async def get_audit_history(
        self,
        session_id: str,
        subject: str,
        predicate: str,
    ) -> List[Dict[str, Any]]:
        """In-memory store does not persist audit history."""
        return []

    def get_stats(self) -> Dict[str, Any]:
        total_beliefs = sum(len(beliefs) for beliefs in self._beliefs.values())
        total_sessions = len(self._beliefs)

        return {
            "total_sessions": total_sessions,
            "total_beliefs": total_beliefs,
            "current_bytes": self.current_bytes,
            "max_bytes": self.max_bytes,
            "utilization_percent": (self.current_bytes / self.max_bytes) * 100
            if self.max_bytes > 0
            else 0,
        }

    async def close(self) -> None:
        """No-op for in-memory store."""
        pass

    async def __aenter__(self) -> "InMemoryBeliefStore":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
