import logging
import math
from typing import Any, List, Optional, Tuple
from beliefstate.config import TrackerConfig
from beliefstate.models import Belief
from beliefstate.adapters.base import ProviderAdapter
from beliefstate.store.base import Store

logger = logging.getLogger(__name__)


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not v1 or not v2:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0
    return dot / (mag1 * mag2)


class ContradictionDetector:
    def __init__(
        self,
        adapter: ProviderAdapter,
        store: Store,
        config: TrackerConfig,
        judge: Optional[Any] = None,
    ):
        self.adapter = adapter
        self.store = store
        self.config = config
        if judge:
            self.judge = judge
        else:
            from beliefstate.judge import LLMJudge

            self.judge = LLMJudge(adapter, config)

    async def detect(
        self, session_id: str, new_beliefs: List[Belief]
    ) -> List[Tuple[Belief, Belief, float, str]]:
        """Detect contradictions between new beliefs and existing store."""
        contradictions = []

        for new_b in new_beliefs:
            if not new_b.embedding:
                continue

            # Database-side vector search for top candidate beliefs
            matched_beliefs = await self.store.search_beliefs(
                session_id=session_id,
                embedding=new_b.embedding,
                threshold=self.config.similarity_threshold,
                limit=5,
            )

            for old_b in matched_beliefs:
                is_contradiction, score, reason = await self.judge.check(old_b, new_b)
                if is_contradiction:
                    contradictions.append((old_b, new_b, score, reason))

        return contradictions
