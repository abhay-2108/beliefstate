from typing import List, Tuple, Dict
from beliefstate.models import Belief
from beliefstate.store.base import Store


class BeliefResolver:
    """Handles what to do when a contradiction is detected."""

    def __init__(self, store: Store, strategy: str = "overwrite"):
        self.store = store
        # strategies: 'overwrite' (prefer new), 'keep_old' (prefer existing), 'raise' (throw error)
        self.strategy = strategy
        self.pending_conflicts: Dict[str, List[str]] = {}

    async def resolve(
        self, session_id: str, contradictions: List[Tuple[Belief, Belief, float, str]]
    ) -> None:
        if not contradictions:
            return

        for old_b, new_b, score, reason in contradictions:
            if self.strategy == "overwrite":
                # Remove the old belief to replace with new one
                await self.store.remove_belief(
                    session_id, old_b.subject, old_b.predicate
                )
                await self.store.add_belief(session_id, new_b)

            elif self.strategy == "keep_old":
                # Ignore the new belief
                pass

            elif self.strategy == "raise":
                raise ValueError(
                    f"Contradiction detected: {old_b.value} vs {new_b.value} - {reason}"
                )

            # Queue a conflict note for the tracker to optionally inject into next prompt
            if session_id not in self.pending_conflicts:
                self.pending_conflicts[session_id] = []

            note = f"[BELIEF CONFLICT] Previously stated: '{old_b.value}'. Now asserting: '{new_b.value}'. Reason: {reason}."
            self.pending_conflicts[session_id].append(note)

    def pop_pending_conflicts(self, session_id: str) -> List[str]:
        """Retrieve and clear pending conflict notes for a session."""
        return self.pending_conflicts.pop(session_id, [])
