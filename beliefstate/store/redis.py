from typing import List, Any
from beliefstate.store.base import Store
from beliefstate.models import Belief

try:
    import redis.asyncio as redis
except ImportError:
    redis = None  # type: ignore[assignment]


class RedisStore(Store):
    """Redis-based asynchronous storage for beliefs."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self._client: Any
        if not redis:
            self._client = None
        else:
            self._client = redis.Redis.from_url(redis_url, decode_responses=True)

    def _get_key(self, session_id: str) -> str:
        return f"beliefstate:session:{session_id}"

    async def add_belief(self, session_id: str, belief: Belief) -> None:
        if not self._client:
            raise RuntimeError(
                "redis package is not installed. Run `pip install redis`"
            )

        # We store beliefs as a hash map where field="subject::predicate" and value=JSON
        field = f"{belief.subject}::{belief.predicate}"
        await self._client.hset(
            self._get_key(session_id), field, belief.model_dump_json()
        )

    async def get_beliefs(self, session_id: str) -> List[Belief]:
        if not self._client:
            raise RuntimeError(
                "redis package is not installed. Run `pip install redis`"
            )

        data = await self._client.hgetall(self._get_key(session_id))
        beliefs = []
        for value_str in data.values():
            beliefs.append(Belief.model_validate_json(value_str))
        return beliefs

    async def search_beliefs(
        self,
        session_id: str,
        embedding: List[float],
        threshold: float = 0.0,
        limit: int = 5,
    ) -> List[Belief]:
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
        if not self._client:
            raise RuntimeError(
                "redis package is not installed. Run `pip install redis`"
            )

        field = f"{subject}::{predicate}"
        await self._client.hdel(self._get_key(session_id), field)

    async def update_belief(self, session_id: str, belief: Belief) -> None:
        await self.add_belief(session_id, belief)

    async def clear(self, session_id: str) -> None:
        if not self._client:
            raise RuntimeError(
                "redis package is not installed. Run `pip install redis`"
            )

        await self._client.delete(self._get_key(session_id))
