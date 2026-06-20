from typing import List, Optional, Any
from beliefstate.store.base import Store
from beliefstate.models import Belief

try:
    import redis.asyncio as redis
except ImportError:
    redis = None  # type: ignore[assignment]


class RedisStore(Store):
    """Redis-based asynchronous storage for beliefs.

    FUTURE OPTIMIZATION: Use binary format (struct.pack float32) for embeddings to reduce storage by ~75%.
    Current implementation stores embeddings as JSON for easier testing and debugging.
    """

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

        # Optionally set TTL on the hash (from config or override)
        # Note: Redis TTL applies to the entire key, not individual fields
        # If you want per-belief TTL, store each belief as a separate key
        # For now, we store as a hash but don't auto-expire by default

    async def get_beliefs(
        self, session_id: str, conversation_id: Optional[str] = None
    ) -> List[Belief]:
        if not self._client:
            raise RuntimeError(
                "redis package is not installed. Run `pip install redis`"
            )

        data = await self._client.hgetall(self._get_key(session_id))
        beliefs = []
        for value_str in data.values():
            belief = Belief.model_validate_json(value_str)
            if conversation_id and belief.conversation_id != conversation_id:
                continue
            beliefs.append(belief)
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

    async def belief_count(self, session_id: str) -> int:
        """Return belief count using efficient HLEN — no deserialization."""
        if not self._client:
            raise RuntimeError(
                "redis package is not installed. Run `pip install redis`"
            )
        return int(await self._client.hlen(self._get_key(session_id)))

    async def health_check(self) -> bool:
        """Verify Redis connection is functional."""
        try:
            if not self._client:
                return False
            return bool(await self._client.ping())
        except Exception:
            return False

    async def set_session_ttl(self, session_id: str, ttl_seconds: int) -> None:
        """Set time-to-live (expiration) for all beliefs in a session.

        When TTL expires, Redis automatically deletes the entire session hash.

        Args:
            session_id: Session ID
            ttl_seconds: Time in seconds before beliefs expire

        Example:
            await store.set_session_ttl("user-123", 86400)  # 24 hours
        """
        if not self._client:
            raise RuntimeError(
                "redis package is not installed. Run `pip install redis`"
            )

        key = self._get_key(session_id)
        await self._client.expire(key, ttl_seconds)

    async def get_session_ttl(self, session_id: str) -> Optional[int]:
        """Get remaining TTL for a session's beliefs in seconds.

        Returns:
            - Positive int: seconds until expiration
            - -1: key exists but no TTL is set
            - -2 or None: key doesn't exist
        """
        if not self._client:
            raise RuntimeError(
                "redis package is not installed. Run `pip install redis`"
            )

        key = self._get_key(session_id)
        ttl = await self._client.ttl(key)
        return ttl if ttl >= -1 else None
