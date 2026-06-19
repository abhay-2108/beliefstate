from typing import List, Dict, Any
from collections import OrderedDict
import logging
from beliefstate.store.base import Store
from beliefstate.models import Belief

logger = logging.getLogger(__name__)


class InMemoryBeliefStore(Store):
    """In-memory storage for beliefs with LRU eviction policy.
    
    Features:
    - Simple in-memory dictionary-based storage
    - Per-session belief storage
    - Global size limit (max_bytes) with LRU eviction
    - Useful for: testing, single-process deployments, development
    
    NOT suitable for: production, multi-process, persistent data needs.
    """

    def __init__(self, max_bytes: int = 100 * 1024 * 1024):  # 100MB default
        """Initialize in-memory store.
        
        Args:
            max_bytes: Maximum total size in bytes. When exceeded, evicts least-recently-used beliefs.
        """
        self.max_bytes = max_bytes
        self.current_bytes = 0
        # Store: session_id -> OrderedDict of (subject::predicate -> Belief)
        # OrderedDict maintains insertion/access order for LRU
        self._beliefs: Dict[str, OrderedDict[str, Belief]] = {}

    def _estimate_belief_size(self, belief: Belief) -> int:
        """Estimate memory size of a belief in bytes."""
        # Rough estimate: model_dump_json() length
        return len(belief.model_dump_json().encode('utf-8'))

    def _evict_lru_belief(self) -> None:
        """Evict the least-recently-used belief from the oldest session."""
        if not self._beliefs:
            return
        
        # Find the session with the oldest access pattern
        for session_id, beliefs_dict in self._beliefs.items():
            if beliefs_dict:
                # Get the first (oldest) belief in the OrderedDict
                field, belief = next(iter(beliefs_dict.items()))
                belief_size = self._estimate_belief_size(belief)
                
                # Remove it
                del beliefs_dict[field]
                self.current_bytes -= belief_size
                logger.debug(f"LRU eviction: removed {field} from session {session_id} (freed {belief_size} bytes)")
                
                # Clean up empty sessions
                if not beliefs_dict:
                    del self._beliefs[session_id]
                
                return

    async def add_belief(self, session_id: str, belief: Belief) -> None:
        """Add or update a belief."""
        if session_id not in self._beliefs:
            self._beliefs[session_id] = OrderedDict()
        
        field = f"{belief.subject}::{belief.predicate}"
        belief_size = self._estimate_belief_size(belief)
        
        # Check if belief already exists (for replacement)
        if field in self._beliefs[session_id]:
            old_size = self._estimate_belief_size(self._beliefs[session_id][field])
            self.current_bytes -= old_size
        
        # Add new belief (moves to end in OrderedDict = most recently used)
        self._beliefs[session_id][field] = belief
        self.current_bytes += belief_size
        
        # Evict if necessary
        while self.current_bytes > self.max_bytes:
            self._evict_lru_belief()
        
        logger.debug(f"Added belief: {field} (session {session_id}, size {belief_size}, total {self.current_bytes}/{self.max_bytes})")

    async def get_beliefs(self, session_id: str) -> List[Belief]:
        """Get all beliefs for a session."""
        if session_id not in self._beliefs:
            return []
        
        # Return as list, preserving order
        beliefs = list(self._beliefs[session_id].values())
        
        # Update access order for LRU (move all to end = most recently used)
        for field in list(self._beliefs[session_id].keys()):
            # Access and re-add to update order
            self._beliefs[session_id].move_to_end(field)
        
        return beliefs

    async def search_beliefs(
        self,
        session_id: str,
        embedding: List[float],
        threshold: float = 0.0,
        limit: int = 5,
    ) -> List[Belief]:
        """Search beliefs by embedding similarity."""
        import math

        beliefs = await self.get_beliefs(session_id)
        scored_beliefs = []

        for b in beliefs:
            if not b.embedding or not embedding:
                continue
            v1 = b.embedding
            v2 = embedding
            dot = sum(a * b for a, b in zip(v1, v2))
            mag1 = math.sqrt(sum(a * a for a in v1))
            mag2 = math.sqrt(sum(b * b for b in v2))
            sim = dot / (mag1 * mag2) if mag1 > 0 and mag2 > 0 else 0.0
            if sim >= threshold:
                scored_beliefs.append((b, sim))

        scored_beliefs.sort(key=lambda x: x[1], reverse=True)
        return [sb[0] for sb in scored_beliefs[:limit]]

    async def remove_belief(
        self, session_id: str, subject: str, predicate: str
    ) -> None:
        """Remove a belief."""
        if session_id not in self._beliefs:
            return
        
        field = f"{subject}::{predicate}"
        if field in self._beliefs[session_id]:
            belief = self._beliefs[session_id][field]
            belief_size = self._estimate_belief_size(belief)
            del self._beliefs[session_id][field]
            self.current_bytes -= belief_size
            
            # Clean up empty sessions
            if not self._beliefs[session_id]:
                del self._beliefs[session_id]

    async def update_belief(self, session_id: str, belief: Belief) -> None:
        """Update a belief (same as add)."""
        await self.add_belief(session_id, belief)

    async def clear(self, session_id: str) -> None:
        """Clear all beliefs for a session."""
        if session_id in self._beliefs:
            # Calculate total size freed
            total_size = sum(self._estimate_belief_size(b) for b in self._beliefs[session_id].values())
            del self._beliefs[session_id]
            self.current_bytes -= total_size
            logger.debug(f"Cleared session {session_id} (freed {total_size} bytes)")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the in-memory store."""
        total_beliefs = sum(len(beliefs) for beliefs in self._beliefs.values())
        total_sessions = len(self._beliefs)
        
        return {
            "total_sessions": total_sessions,
            "total_beliefs": total_beliefs,
            "current_bytes": self.current_bytes,
            "max_bytes": self.max_bytes,
            "utilization_percent": (self.current_bytes / self.max_bytes) * 100 if self.max_bytes > 0 else 0,
        }
