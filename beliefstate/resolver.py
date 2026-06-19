from typing import List, Tuple, Dict
import logging
from beliefstate.models import Belief
from beliefstate.store.base import Store

logger = logging.getLogger(__name__)


class BeliefResolver:
    """Handles what to do when a contradiction is detected.
    
    Implements escalation logic: ASK → BLOCK to prevent conflict note stacking.
    - First occurrence: ASK (inject conflict note into prompt)
    - If same conflict fires again after ASK: BLOCK (stop injecting, log warning)
    """

    def __init__(self, store: Store, strategy: str = "overwrite"):
        self.store = store
        # strategies: 'overwrite' (prefer new), 'keep_old' (prefer existing), 'raise' (throw error)
        self.strategy = strategy
        self.pending_conflicts: Dict[str, List[str]] = {}
        # Track conflicts: {session_id -> {(old_subject, old_pred, new_subject, new_pred) -> count}}
        self.conflict_history: Dict[str, Dict[Tuple[str, str, str, str], int]] = {}

    def _get_conflict_key(self, old_b: Belief, new_b: Belief) -> Tuple[str, str, str, str]:
        """Create a unique key for a contradiction pair."""
        return (old_b.subject, old_b.predicate, new_b.subject, new_b.predicate)

    async def resolve(
        self, session_id: str, contradictions: List[Tuple[Belief, Belief, float, str]]
    ) -> None:
        if not contradictions:
            return

        # Initialize conflict history for this session if needed
        if session_id not in self.conflict_history:
            self.conflict_history[session_id] = {}
        if session_id not in self.pending_conflicts:
            self.pending_conflicts[session_id] = []

        for old_b, new_b, score, reason in contradictions:
            # Skip resolution for temporal updates (belief_type='update')
            # These are intentional corrections/changes, not real contradictions
            if new_b.belief_type == "update":
                logger.info(
                    f"Session {session_id}: Skipping contradiction resolution for temporal update. "
                    f"Replacing '{old_b.value}' with '{new_b.value}'"
                )
                # Always replace with the update (overwrite strategy implicit for updates)
                await self.store.remove_belief(
                    session_id, old_b.subject, old_b.predicate
                )
                await self.store.add_belief(session_id, new_b)
                continue
            
            conflict_key = self._get_conflict_key(old_b, new_b)
            current_count = self.conflict_history[session_id].get(conflict_key, 0)

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

            # Escalation logic: first time = ASK, repeat = BLOCK
            if current_count == 0:
                # First occurrence: ASK - inject conflict note
                note = f"[BELIEF CONFLICT] Previously stated: '{old_b.value}'. Now asserting: '{new_b.value}'. Reason: {reason}."
                self.pending_conflicts[session_id].append(note)
                logger.info(
                    f"Session {session_id}: Conflict ASK - injecting resolution prompt"
                )

            elif current_count == 1:
                # Second occurrence: BLOCK - don't inject (user ignored the ASK)
                logger.warning(
                    f"Session {session_id}: Conflict BLOCK - user ignored previous ASK. "
                    f"Conflict between '{old_b.value}' vs '{new_b.value}' still unresolved."
                )

            else:
                # Subsequent occurrences: silently drop
                logger.debug(
                    f"Session {session_id}: Conflict suppressed (escalated to BLOCK)"
                )

            # Increment conflict count
            self.conflict_history[session_id][conflict_key] = current_count + 1

    def pop_pending_conflicts(self, session_id: str) -> List[str]:
        """Retrieve and clear pending conflict notes for a session."""
        return self.pending_conflicts.pop(session_id, [])

    def clear_session(self, session_id: str) -> None:
        """Clear all conflict tracking for a session."""
        self.pending_conflicts.pop(session_id, None)
        self.conflict_history.pop(session_id, None)
