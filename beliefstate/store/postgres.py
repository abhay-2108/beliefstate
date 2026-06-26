import logging
from typing import List, Optional, Any, Dict
from datetime import datetime, timezone, timedelta
from beliefstate.store.base import Store
from beliefstate.models import Belief

logger = logging.getLogger(__name__)

try:
    import asyncpg
except ImportError:
    asyncpg = None


class PostgreSQLStore(Store):
    """PostgreSQL-based asynchronous storage for beliefs using asyncpg."""

    def __init__(self, dsn: Optional[str] = None, **kwargs: Any):
        """Initialize PostgreSQLStore.

        Args:
            dsn: Database connection string (e.g., "postgresql://user:pass@host/db")
            **kwargs: Connection parameters passed directly to asyncpg (host, port, user, etc.)
        """
        self.dsn = dsn
        self.connection_kwargs = kwargs
        self._pool: Optional[Any] = None

    async def open(self) -> None:
        """Initialize connection pool and tables/functions."""
        if self._pool is None:
            if not asyncpg:
                raise RuntimeError(
                    "asyncpg is not installed. Run `pip install asyncpg` or `pip install beliefstate[postgres]`"
                )
            if self.dsn:
                self._pool = await asyncpg.create_pool(
                    self.dsn, **self.connection_kwargs
                )
            else:
                self._pool = await asyncpg.create_pool(**self.connection_kwargs)

            await self._init_db()

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def __aenter__(self) -> "PostgreSQLStore":
        await self.open()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def _get_pool(self) -> Any:
        if self._pool is None:
            await self.open()
        return self._pool

    async def _init_db(self) -> None:
        pool = self._pool
        if pool is None:
            return

        async with pool.acquire() as conn:
            # 1. Create cosine_similarity helper function
            await conn.execute("""
                CREATE OR REPLACE FUNCTION cosine_similarity(a double precision[], b double precision[])
                RETURNS double precision AS $$
                DECLARE
                    dot double precision := 0;
                    mag_a double precision := 0;
                    mag_b double precision := 0;
                    i integer;
                BEGIN
                    IF a IS NULL OR b IS NULL OR cardinality(a) = 0 OR cardinality(b) = 0 OR cardinality(a) <> cardinality(b) THEN
                        RETURN 0.0;
                    END IF;
                    FOR i IN 1..cardinality(a) LOOP
                        dot := dot + a[i] * b[i];
                        mag_a := mag_a + a[i] * a[i];
                        mag_b := mag_b + b[i] * b[i];
                    END LOOP;
                    IF mag_a = 0 OR mag_b = 0 THEN
                        RETURN 0.0;
                    END IF;
                    RETURN dot / (sqrt(mag_a) * sqrt(mag_b));
                END;
                $$ LANGUAGE plpgsql IMMUTABLE;
            """)

            # 2. Create beliefs table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS beliefs (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    conversation_id VARCHAR(255) NOT NULL DEFAULT '',
                    subject VARCHAR(255) NOT NULL,
                    predicate VARCHAR(255) NOT NULL,
                    value TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    turn INTEGER NOT NULL,
                    source VARCHAR(50) NOT NULL,
                    source_quote TEXT DEFAULT '',
                    category VARCHAR(50) DEFAULT '',
                    embedding DOUBLE PRECISION[],
                    embedding_model VARCHAR(255) DEFAULT '',
                    embedding_dim INTEGER DEFAULT 0,
                    belief_type VARCHAR(50) DEFAULT 'assertion',
                    is_hypothetical BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    last_referenced_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(session_id, conversation_id, subject, predicate)
                );
            """)

            # 3. Audit table for belief mutation history
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS beliefs_audit (
                    id          SERIAL PRIMARY KEY,
                    session_id  VARCHAR(255) NOT NULL,
                    conversation_id VARCHAR(255) NOT NULL DEFAULT '',
                    subject     VARCHAR(255) NOT NULL,
                    predicate   VARCHAR(255) NOT NULL,
                    old_value   TEXT,
                    new_value   TEXT NOT NULL,
                    operation   VARCHAR(50) NOT NULL,
                    source_quote TEXT DEFAULT '',
                    confidence  REAL,
                    turn        INTEGER,
                    created_at  TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 3. Create indices
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pg_session ON beliefs(session_id);"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pg_conversation ON beliefs(conversation_id);"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pg_session_conv ON beliefs(session_id, conversation_id);"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pg_created_at ON beliefs(created_at);"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pg_last_ref ON beliefs(last_referenced_at);"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pg_session_subject ON beliefs(session_id, subject, predicate);"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pg_audit_session ON beliefs_audit(session_id, subject, predicate);"
            )

            # Migrate beliefs_audit: add conversation_id if missing
            try:
                await conn.execute(
                    "ALTER TABLE beliefs_audit ADD COLUMN IF NOT EXISTS conversation_id VARCHAR(255) NOT NULL DEFAULT ''"
                )
            except Exception as e:
                logger.debug(f"Audit table migration check failed (non-critical): {e}")

    async def add_belief(self, session_id: str, belief: Belief) -> None:
        pool = await self._get_pool()
        conversation_id = belief.conversation_id or ""
        subject = (belief.subject or "").lower()
        predicate = (belief.predicate or "").lower()
        created_at = belief.created_at or datetime.now(timezone.utc)
        last_referenced_at = belief.last_referenced_at or datetime.now(timezone.utc)

        # Handle naive datetime mapping
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if last_referenced_at.tzinfo is None:
            last_referenced_at = last_referenced_at.replace(tzinfo=timezone.utc)

        # Check for existing belief for audit trail
        old_value = None
        try:
            existing = await self.get_by_key(
                subject, predicate, session_id, conversation_id
            )
            if existing:
                old_value = existing.value
        except Exception as e:
            logger.debug(f"Audit lookup failed (non-critical): {e}")

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO beliefs (
                    session_id, conversation_id, subject, predicate, value, confidence, turn, source,
                    source_quote, category, embedding, embedding_model, embedding_dim, belief_type,
                    is_hypothetical, created_at, last_referenced_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                ON CONFLICT(session_id, conversation_id, subject, predicate) DO UPDATE SET
                    value = EXCLUDED.value,
                    confidence = EXCLUDED.confidence,
                    turn = EXCLUDED.turn,
                    source = EXCLUDED.source,
                    source_quote = EXCLUDED.source_quote,
                    category = EXCLUDED.category,
                    embedding = EXCLUDED.embedding,
                    embedding_model = EXCLUDED.embedding_model,
                    embedding_dim = EXCLUDED.embedding_dim,
                    belief_type = EXCLUDED.belief_type,
                    is_hypothetical = EXCLUDED.is_hypothetical,
                    created_at = EXCLUDED.created_at,
                    last_referenced_at = EXCLUDED.last_referenced_at
            """,
                session_id,
                conversation_id,
                subject,
                predicate,
                belief.value,
                belief.confidence,
                belief.turn,
                belief.source,
                getattr(belief, "source_quote", ""),
                getattr(belief, "category", ""),
                belief.embedding,
                belief.embedding_model,
                belief.embedding_dim,
                belief.belief_type,
                belief.is_hypothetical,
                created_at,
                last_referenced_at,
            )

            # Audit: create or update
            if old_value is not None and old_value != belief.value:
                await self._audit(conn, belief, "contradiction_update", old_value)
            elif old_value is None:
                await self._audit(conn, belief, "create")

    def _row_to_belief(self, r: Any) -> Belief:
        """Convert a database row to a Belief object."""
        c_at = r["created_at"]
        if c_at and c_at.tzinfo is None:
            c_at = c_at.replace(tzinfo=timezone.utc)
        l_ref = r["last_referenced_at"]
        if l_ref and l_ref.tzinfo is None:
            l_ref = l_ref.replace(tzinfo=timezone.utc)

        return Belief(
            subject=r["subject"],
            predicate=r["predicate"],
            value=r["value"],
            confidence=r["confidence"],
            turn=r["turn"],
            source=r["source"],
            source_quote=r.get("source_quote") or "",
            category=r.get("category") or "",
            embedding=r["embedding"] or [],
            embedding_model=r["embedding_model"] or "",
            embedding_dim=r["embedding_dim"] or 0,
            belief_type=r["belief_type"] or "assertion",
            is_hypothetical=bool(r["is_hypothetical"]),
            created_at=c_at or datetime.now(timezone.utc),
            last_referenced_at=l_ref or datetime.now(timezone.utc),
            session_id=r["session_id"],
            conversation_id=r["conversation_id"],
        )

    async def get_beliefs(
        self, session_id: str, conversation_id: Optional[str] = None
    ) -> List[Belief]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            if conversation_id:
                rows = await conn.fetch(
                    """
                    SELECT subject, predicate, value, confidence, turn, source, source_quote, category, embedding, embedding_model, embedding_dim, belief_type, is_hypothetical, created_at, last_referenced_at, session_id, conversation_id
                    FROM beliefs WHERE session_id = $1 AND conversation_id = $2
                """,
                    session_id,
                    conversation_id,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT subject, predicate, value, confidence, turn, source, source_quote, category, embedding, embedding_model, embedding_dim, belief_type, is_hypothetical, created_at, last_referenced_at, session_id, conversation_id
                    FROM beliefs WHERE session_id = $1
                """,
                    session_id,
                )

        return [self._row_to_belief(r) for r in rows]

    async def search_beliefs(
        self,
        session_id: str,
        embedding: List[float],
        threshold: float = 0.0,
        limit: int = 5,
        conversation_id: Optional[str] = None,
    ) -> List[Belief]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            if conversation_id:
                rows = await conn.fetch(
                    """
                    SELECT subject, predicate, value, confidence, turn, source, source_quote, category, embedding, embedding_model, embedding_dim, belief_type, is_hypothetical, created_at, last_referenced_at, session_id, conversation_id,
                           cosine_similarity($1, embedding) as similarity
                    FROM beliefs 
                    WHERE session_id = $2 AND conversation_id = $3 AND cosine_similarity($1, embedding) >= $4
                    ORDER BY similarity DESC
                    LIMIT $5
                """,
                    embedding,
                    session_id,
                    conversation_id,
                    threshold,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT subject, predicate, value, confidence, turn, source, source_quote, category, embedding, embedding_model, embedding_dim, belief_type, is_hypothetical, created_at, last_referenced_at, session_id, conversation_id,
                           cosine_similarity($1, embedding) as similarity
                    FROM beliefs 
                    WHERE session_id = $2 AND cosine_similarity($1, embedding) >= $3
                    ORDER BY similarity DESC
                    LIMIT $4
                """,
                    embedding,
                    session_id,
                    threshold,
                    limit,
                )

        return [self._row_to_belief(r) for r in rows]

    async def _audit(
        self,
        conn: Any,
        belief: Belief,
        operation: str,
        old_value: Optional[str] = None,
    ) -> None:
        """Write an immutable audit record for a belief mutation."""
        await conn.execute(
            """INSERT INTO beliefs_audit
               (session_id, conversation_id, subject, predicate, old_value, new_value,
                operation, source_quote, confidence, turn)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)""",
            belief.session_id or "",
            belief.conversation_id or "",
            (belief.subject or "").lower(),
            (belief.predicate or "").lower(),
            old_value,
            belief.value,
            operation,
            getattr(belief, "source_quote", ""),
            belief.confidence,
            belief.turn,
        )

    async def upsert(self, belief: Belief) -> bool:
        """Insert or update a belief with turn-based optimistic concurrency.

        Returns True if the belief was written, False if discarded (stale write).
        """
        existing = await self.get_by_key(
            belief.subject.lower(),
            belief.predicate.lower(),
            belief.session_id or "",
            belief.conversation_id or "",
        )
        if existing and existing.turn > belief.turn:
            return False
        await self.add_belief(belief.session_id or "", belief)
        return True

    async def get_by_key(
        self,
        subject: str,
        predicate: str,
        session_id: str,
        conversation_id: Optional[str] = None,
    ) -> Optional[Belief]:
        """Retrieve a single belief by its composite key."""
        pool = await self._get_pool()
        cid = conversation_id or ""
        sub = subject.lower()
        pred = predicate.lower()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT subject, predicate, value, confidence, turn, source, source_quote, category, embedding, embedding_model, embedding_dim, belief_type, is_hypothetical, created_at, last_referenced_at, session_id, conversation_id
                FROM beliefs
                WHERE session_id = $1 AND conversation_id = $2 AND subject = $3 AND predicate = $4
                LIMIT 1
            """,
                session_id,
                cid,
                sub,
                pred,
            )

        if row is None:
            return None

        return self._row_to_belief(row)

    async def get_audit_history(
        self,
        session_id: str,
        subject: str,
        predicate: str,
    ) -> List[Dict[str, Any]]:
        """Return audit trail for a specific belief."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT session_id, conversation_id, subject, predicate, old_value, new_value,
                       operation, source_quote, confidence, turn, created_at
                FROM beliefs_audit
                WHERE session_id = $1 AND subject = $2 AND predicate = $3
                ORDER BY created_at ASC
            """,
                session_id,
                subject,
                predicate,
            )

        return [dict(r) for r in rows]

    async def remove_belief(
        self,
        session_id: str,
        subject: str,
        predicate: str,
        conversation_id: Optional[str] = None,
    ) -> None:
        pool = await self._get_pool()
        cid = conversation_id or ""
        async with pool.acquire() as conn:
            # Audit before delete
            try:
                existing = await self.get_by_key(
                    subject, predicate, session_id, conversation_id
                )
                if existing:
                    await self._audit(conn, existing, "delete")
            except Exception as e:
                logger.debug(f"Audit lookup failed (non-critical): {e}")

            await conn.execute(
                "DELETE FROM beliefs WHERE session_id = $1 AND conversation_id = $2 AND subject = $3 AND predicate = $4",
                session_id,
                cid,
                subject.lower(),
                predicate.lower(),
            )

    async def update_belief(self, session_id: str, belief: Belief) -> None:
        await self.add_belief(session_id, belief)

    async def clear(self, session_id: str) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM beliefs WHERE session_id = $1", session_id)

    async def belief_count(self, session_id: str) -> int:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            val = await conn.fetchval(
                "SELECT COUNT(*) FROM beliefs WHERE session_id = $1", session_id
            )
            return int(val) if val else 0

    async def health_check(self) -> bool:
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                val = await conn.fetchval("SELECT 1")
                return bool(val == 1)
        except Exception:
            return False

    async def prune_expired_beliefs(
        self, max_age_seconds: int, session_id: Optional[str] = None
    ) -> int:
        pool = await self._get_pool()
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)

        async with pool.acquire() as conn:
            if session_id:
                res = await conn.execute(
                    "DELETE FROM beliefs WHERE session_id = $1 AND created_at < $2",
                    session_id,
                    cutoff_time,
                )
            else:
                res = await conn.execute(
                    "DELETE FROM beliefs WHERE created_at < $1",
                    cutoff_time,
                )

        # res is a string like "DELETE 5"
        try:
            parts = res.split()
            if len(parts) >= 2:
                return int(parts[-1])
        except Exception:
            pass
        return 0

    async def get_session_belief_age_stats(self, session_id: str) -> Dict[str, Any]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT 
                    MIN(EXTRACT(EPOCH FROM (NOW() - created_at))) as oldest_age,
                    MAX(EXTRACT(EPOCH FROM (NOW() - created_at))) as newest_age,
                    AVG(EXTRACT(EPOCH FROM (NOW() - created_at))) as avg_age
                FROM beliefs WHERE session_id = $1
            """,
                session_id,
            )

        if row is None or row["oldest_age"] is None:
            return {
                "oldest_belief_age_seconds": 0,
                "newest_belief_age_seconds": 0,
                "avg_age_seconds": 0,
            }

        return {
            "oldest_belief_age_seconds": int(row["oldest_age"]),
            "newest_belief_age_seconds": int(row["newest_age"]),
            "avg_age_seconds": int(row["avg_age"]),
        }
