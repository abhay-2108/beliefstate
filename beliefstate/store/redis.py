import json
from typing import Any, Dict, List, Optional
from beliefstate.store.base import Store
from beliefstate.store.utils import cosine_similarity
from beliefstate.models import Belief

try:
    import redis.asyncio as redis
except ImportError:
    redis = None  # type: ignore[assignment]


class RedisStore(Store):
    """Redis-based asynchronous storage for beliefs.

    Uses binary float32 embedding storage for precision.

    Note: search_beliefs is O(n) over all beliefs in a session because
    Redis hashes don't support vector search. For high-volume deployments,
    consider using pgvector (PostgreSQLStore) for native vector similarity.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self._client: Any
        if not redis:
            self._client = None
        else:
            self._client = redis.Redis.from_url(redis_url, decode_responses=False)

    async def open(self) -> None:
        """No-op for Redis (client initialized in __init__)."""
        pass

    def _get_key(self, session_id: str) -> str:
        return f"beliefstate:session:{session_id}"

    def _get_audit_key(self, session_id: str, subject: str, predicate: str) -> str:
        return f"beliefstate:audit:{session_id}:{subject}::{predicate}"

    async def add_belief(self, session_id: str, belief: Belief) -> None:
        if not self._client:
            raise RuntimeError(
                "redis package is not installed. Run `pip install redis`"
            )

        cid = belief.conversation_id or ""
        field = f"{(belief.subject or '').lower()}::{(belief.predicate or '').lower()}::{cid}"
        await self._client.hset(
            self._get_key(session_id), field, belief.model_dump_json()
        )

    async def get_beliefs(
        self, session_id: str, conversation_id: Optional[str] = None
    ) -> List[Belief]:
        if not self._client:
            raise RuntimeError(
                "redis package is not installed. Run `pip install redis`"
            )

        data = await self._client.hgetall(self._get_key(session_id))
        beliefs = []
        for value_bytes in data.values():
            if isinstance(value_bytes, bytes):
                value_str = value_bytes.decode("utf-8")
            else:
                value_str = value_bytes
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
        """Retrieve a single belief by its composite key using direct hash lookup (O(1))."""
        if not self._client:
            raise RuntimeError(
                "redis package is not installed. Run `pip install redis`"
            )
        cid = conversation_id or ""
        field = f"{subject.lower()}::{predicate.lower()}::{cid}"
        value_bytes = await self._client.hget(self._get_key(session_id), field)
        if value_bytes is None:
            return None
        if isinstance(value_bytes, bytes):
            value_str = value_bytes.decode("utf-8")
        else:
            value_str = value_bytes
        return Belief.model_validate_json(value_str)

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
        if not self._client:
            raise RuntimeError(
                "redis package is not installed. Run `pip install redis`"
            )

        cid = conversation_id or ""
        field = f"{subject.lower()}::{predicate.lower()}::{cid}"
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
        if not self._client:
            raise RuntimeError(
                "redis package is not installed. Run `pip install redis`"
            )
        return int(await self._client.hlen(self._get_key(session_id)))

    async def health_check(self) -> bool:
        try:
            if not self._client:
                return False
            return bool(await self._client.ping())
        except Exception:
            return False

    async def set_session_ttl(self, session_id: str, ttl_seconds: int) -> None:
        if not self._client:
            raise RuntimeError(
                "redis package is not installed. Run `pip install redis`"
            )
        key = self._get_key(session_id)
        await self._client.expire(key, ttl_seconds)

    async def get_session_ttl(self, session_id: str) -> Optional[int]:
        """Return TTL in seconds for a session's key.

        Returns:
            TTL in seconds if key exists and has an expiry.
            -1 if key exists but has no expiry set.
            None if key does not exist.
        """
        if not self._client:
            raise RuntimeError(
                "redis package is not installed. Run `pip install redis`"
            )
        key = self._get_key(session_id)
        ttl = await self._client.ttl(key)
        if ttl == -2:
            return None  # Key does not exist
        return ttl  # -1 (no expiry) or positive TTL

    async def get_audit_history(
        self,
        session_id: str,
        subject: str,
        predicate: str,
    ) -> List[Dict[str, Any]]:
        """Return audit trail for a specific belief (Redis implementation stores as list)."""
        if not self._client:
            return []
        key = self._get_audit_key(session_id, subject, predicate)
        data = await self._client.lrange(key, 0, -1)

        results = []
        for item in data:
            if isinstance(item, bytes):
                item = item.decode("utf-8")
            results.append(json.loads(item))
        return results

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "RedisStore":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
