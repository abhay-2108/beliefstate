from typing import List, Tuple, Dict, Optional
import logging
from beliefstate.models import Belief
from beliefstate.store.base import Store

logger = logging.getLogger(__name__)


class BeliefResolver:
    """Handles what to do when a contradiction is detected.

    Implements escalation logic: ASK -> BLOCK to prevent conflict note stacking.
    - First occurrence: ASK (inject conflict note into prompt)
    - If same conflict fires again after ASK: BLOCK (stop injecting, log warning)

    Strategies:
    - overwrite: Remove old belief, store new one
    - keep_old: Ignore new belief
    - raise: Throw ValueError
    """

    def __init__(self, store: Store, strategy: str = "overwrite"):
        self.store = store
        self.strategy = strategy
        self.pending_conflicts: Dict[str, List[str]] = {}
        self.conflict_history: Dict[str, Dict[Tuple[str, str, str, str], int]] = {}
        self._MAX_SESSIONS = 1000

    def _get_conflict_key(
        self, old_b: Belief, new_b: Belief
    ) -> Tuple[str, str, str, str]:
        return (old_b.subject, old_b.predicate, new_b.subject, new_b.predicate)

    async def resolve(
        self,
        session_id: str,
        contradictions: List[Tuple[Belief, Belief, float, str]],
        store: Optional[Store] = None,
    ) -> None:
        """Resolve contradictions using the configured strategy.

        Args:
            session_id: Session ID
            contradictions: List of (old_belief, new_belief, score, reason) tuples
            store: Optional store override (for backwards compatibility)
        """
        if not contradictions:
            return

        # Evict oldest sessions if at capacity
        while len(self.pending_conflicts) > self._MAX_SESSIONS:
            oldest_sid = next(iter(self.pending_conflicts))
            del self.pending_conflicts[oldest_sid]
            self.conflict_history.pop(oldest_sid, None)
        while len(self.conflict_history) > self._MAX_SESSIONS:
            oldest_sid = next(iter(self.conflict_history))
            del self.conflict_history[oldest_sid]
            self.pending_conflicts.pop(oldest_sid, None)

        target_store = store or self.store

        if session_id not in self.conflict_history:
            self.conflict_history[session_id] = {}
        if session_id not in self.pending_conflicts:
            self.pending_conflicts[session_id] = []

        for old_b, new_b, score, reason in contradictions:
            # Skip resolution for temporal updates (belief_type='update')
            if new_b.belief_type == "update":
                logger.info(
                    f"Session {session_id}: Skipping contradiction resolution for temporal update. "
                    f"Replacing '{old_b.value}' with '{new_b.value}'"
                )
                await target_store.remove_belief(
                    session_id, old_b.subject, old_b.predicate
                )
                await target_store.add_belief(session_id, new_b)
                continue

            conflict_key = self._get_conflict_key(old_b, new_b)
            current_count = self.conflict_history[session_id].get(conflict_key, 0)

            if self.strategy == "overwrite":
                await target_store.remove_belief(
                    session_id, old_b.subject, old_b.predicate
                )
                await target_store.add_belief(session_id, new_b)

            elif self.strategy == "keep_old":
                note = f"[BELIEF CONFLICT — kept old] Previously stated: '{old_b.value}'. New assertion: '{new_b.value}' was discarded. Reason: {reason}."
                if current_count == 0:
                    self.pending_conflicts[session_id].append(note)
                self.conflict_history[session_id][conflict_key] = current_count + 1
                logger.info(
                    f"Session {session_id}: keep_old — discarded '{new_b.value}', keeping '{old_b.value}'"
                )
                continue

            elif self.strategy == "raise":
                error_msg = (
                    f"Contradiction detected (strategy=raise): "
                    f"'{old_b.value}' vs '{new_b.value}' — {reason}"
                )
                self.pending_conflicts[session_id].append(f"[RAISE] {error_msg}")
                raise ValueError(error_msg)

            # Escalation logic: first time = ASK, repeat = BLOCK
            if current_count == 0:
                note = f"[BELIEF CONFLICT] Previously stated: '{old_b.value}'. Now asserting: '{new_b.value}'. Reason: {reason}."
                self.pending_conflicts[session_id].append(note)
                logger.info(
                    f"Session {session_id}: Conflict ASK - injecting resolution prompt"
                )
            elif current_count == 1:
                logger.warning(
                    f"Session {session_id}: Conflict BLOCK - user ignored previous ASK. "
                    f"Conflict between '{old_b.value}' vs '{new_b.value}' still unresolved."
                )
            else:
                logger.debug(
                    f"Session {session_id}: Conflict suppressed (escalated to BLOCK)"
                )

            self.conflict_history[session_id][conflict_key] = current_count + 1

    def pop_pending_conflicts(self, session_id: str) -> List[str]:
        return self.pending_conflicts.pop(session_id, [])

    def clear_session(self, session_id: str) -> None:
        self.pending_conflicts.pop(session_id, None)
        self.conflict_history.pop(session_id, None)
