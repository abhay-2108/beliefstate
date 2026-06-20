import json
import logging
from typing import List, Optional, Any
from datetime import datetime, timezone
from beliefstate.store.base import Store
from beliefstate.models import Belief

logger = logging.getLogger(__name__)

try:
    import aiosqlite
except ImportError:
    aiosqlite = None  # type: ignore[assignment]


def cosine_similarity_py(emb1_str: str, emb2_json_str: str) -> float:
    try:
        import json
        import math

        v1 = json.loads(emb1_str)
        v2 = json.loads(emb2_json_str)
        if not v1 or not v2:
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(a * a for a in v1))
        mag2 = math.sqrt(sum(b * b for b in v2))
        if mag1 == 0.0 or mag2 == 0.0:
            return 0.0
        return float(dot / (mag1 * mag2))
    except Exception:
        return 0.0


class SQLiteStore(Store):
    """SQLite-based asynchronous storage for beliefs."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._conn: Optional[Any] = None

    async def open(self) -> None:
        """Explicitly open and initialize the database connection."""
        if self._conn is None:
            if not aiosqlite:
                raise RuntimeError(
                    "aiosqlite is not installed. Run `pip install aiosqlite`"
                )
            import os

            if self.db_path != ":memory:":
                parent = os.path.dirname(self.db_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
            self._conn = await aiosqlite.connect(self.db_path)
            self._conn.row_factory = aiosqlite.Row

            # Enable WAL mode for robustness against abrupt shutdowns
            # WAL (Write-Ahead Log) survives crashes and allows concurrent reads during writes
            if self.db_path != ":memory:":
                await self._conn.execute("PRAGMA journal_mode=WAL")
                await self._conn.execute("PRAGMA synchronous=NORMAL")
                await self._conn.execute("PRAGMA foreign_keys=ON")
                await self._conn.commit()

            await self._conn.create_function(
                "cosine_similarity", 2, cosine_similarity_py
            )
            await self._init_db()

    async def close(self) -> None:
        """Explicitly close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self) -> "SQLiteStore":
        """Async context manager entry."""
        await self.open()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def _get_connection(self) -> Any:
        """Get or lazily initialize the connection."""
        if not aiosqlite:
            raise RuntimeError(
                "aiosqlite is not installed. Run `pip install aiosqlite`"
            )
        if self._conn is None:
            await self.open()
        return self._conn

    async def _init_db(self) -> None:
        conn = self._conn
        if conn is None:
            return

        # Check if table exists and what columns it has
        async with conn.execute("PRAGMA table_info(beliefs)") as cursor:
            existing_columns = await cursor.fetchall()

        if not existing_columns:
            # Table doesn't exist, create fresh
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS beliefs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL DEFAULT '',
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    value TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    turn INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    embedding TEXT,
                    embedding_model TEXT DEFAULT '',
                    embedding_dim INTEGER DEFAULT 0,
                    belief_type TEXT DEFAULT 'assertion',
                    is_hypothetical INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_referenced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(session_id, conversation_id, subject, predicate)
                )
            """)
        else:
            # Table exists, migrate schema if needed
            existing_column_names = {row[1] for row in existing_columns}

            # ALTER TABLE can't add DEFAULT CURRENT_TIMESTAMP, so use NULL instead
            columns_to_add = [
                ("conversation_id", "TEXT NOT NULL DEFAULT ''"),
                ("embedding_dim", "INTEGER DEFAULT 0"),
                ("belief_type", "TEXT DEFAULT 'assertion'"),
                ("is_hypothetical", "INTEGER DEFAULT 0"),
                ("last_referenced_at", "TIMESTAMP"),
            ]

            for col_name, col_def in columns_to_add:
                if col_name not in existing_column_names:
                    try:
                        await conn.execute(
                            f"ALTER TABLE beliefs ADD COLUMN {col_name} {col_def}"
                        )
                        logger.info(f"Added column '{col_name}' to beliefs table")
                    except Exception as e:
                        logger.warning(f"Could not add column '{col_name}': {e}")

        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_session ON beliefs(session_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversation ON beliefs(conversation_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_conversation ON beliefs(session_id, conversation_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_created_at ON beliefs(created_at)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_last_referenced ON beliefs(last_referenced_at)"
        )
        await conn.commit()

    async def add_belief(self, session_id: str, belief: Belief) -> None:
        conn = await self._get_connection()
        embedding_json = json.dumps(belief.embedding) if belief.embedding else "[]"

        # Use empty string as default conversation_id for backwards compatibility
        conversation_id = belief.conversation_id or ""

        await conn.execute(
            """
            INSERT INTO beliefs (session_id, conversation_id, subject, predicate, value, confidence, turn, source, embedding, embedding_model, embedding_dim, belief_type, is_hypothetical, created_at, last_referenced_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, conversation_id, subject, predicate) DO UPDATE SET
                value=excluded.value,
                confidence=excluded.confidence,
                turn=excluded.turn,
                source=excluded.source,
                embedding=excluded.embedding,
                embedding_model=excluded.embedding_model,
                embedding_dim=excluded.embedding_dim,
                belief_type=excluded.belief_type,
                is_hypothetical=excluded.is_hypothetical,
                created_at=excluded.created_at,
                last_referenced_at=excluded.last_referenced_at
        """,
            (
                session_id,
                conversation_id,
                belief.subject,
                belief.predicate,
                belief.value,
                belief.confidence,
                belief.turn,
                belief.source,
                embedding_json,
                belief.embedding_model,
                belief.embedding_dim,
                belief.belief_type,
                1 if belief.is_hypothetical else 0,
                belief.created_at.isoformat() if belief.created_at else None,
                belief.last_referenced_at.isoformat()
                if belief.last_referenced_at
                else None,
            ),
        )
        await conn.commit()

    async def get_beliefs(
        self, session_id: str, conversation_id: Optional[str] = None
    ) -> List[Belief]:
        conn = await self._get_connection()

        if conversation_id:
            # Get beliefs for specific conversation
            async with conn.execute(
                """
                SELECT subject, predicate, value, confidence, turn, source, embedding, embedding_model, embedding_dim, belief_type, is_hypothetical, created_at, last_referenced_at, session_id, conversation_id
                FROM beliefs WHERE session_id = ? AND conversation_id = ?
            """,
                (session_id, conversation_id),
            ) as cursor:
                rows = await cursor.fetchall()
        else:
            # Get beliefs for session (all conversations)
            async with conn.execute(
                """
                SELECT subject, predicate, value, confidence, turn, source, embedding, embedding_model, embedding_dim, belief_type, is_hypothetical, created_at, last_referenced_at, session_id, conversation_id
                FROM beliefs WHERE session_id = ?
            """,
                (session_id,),
            ) as cursor:
                rows = await cursor.fetchall()

        beliefs = []
        for r in rows:
            beliefs.append(
                Belief(
                    subject=r["subject"],
                    predicate=r["predicate"],
                    value=r["value"],
                    confidence=r["confidence"],
                    turn=r["turn"],
                    source=r["source"],
                    embedding=json.loads(r["embedding"]) if r["embedding"] else [],
                    embedding_model=r["embedding_model"] or "",
                    embedding_dim=r["embedding_dim"] or 0,
                    belief_type=r["belief_type"] or "assertion",
                    is_hypothetical=bool(r["is_hypothetical"]),
                    created_at=datetime.fromisoformat(r["created_at"])
                    if r["created_at"]
                    else datetime.now(timezone.utc),
                    last_referenced_at=datetime.fromisoformat(r["last_referenced_at"])
                    if r["last_referenced_at"]
                    else datetime.now(timezone.utc),
                    session_id=r["session_id"],
                    conversation_id=r["conversation_id"],
                )
            )
        return beliefs

    async def search_beliefs(
        self,
        session_id: str,
        embedding: List[float],
        threshold: float = 0.0,
        limit: int = 5,
        conversation_id: Optional[str] = None,
    ) -> List[Belief]:
        conn = await self._get_connection()
        embedding_json = json.dumps(embedding)

        if conversation_id:
            # Search within specific conversation
            async with conn.execute(
                """
                SELECT subject, predicate, value, confidence, turn, source, embedding, embedding_model, embedding_dim, belief_type, is_hypothetical, created_at, last_referenced_at, session_id, conversation_id,
                       cosine_similarity(?, embedding) as similarity
                FROM beliefs 
                WHERE session_id = ? AND conversation_id = ? AND similarity >= ?
                ORDER BY similarity DESC
                LIMIT ?
             """,
                (embedding_json, session_id, conversation_id, threshold, limit),
            ) as cursor:
                rows = await cursor.fetchall()
        else:
            # Search across all conversations in session
            async with conn.execute(
                """
                SELECT subject, predicate, value, confidence, turn, source, embedding, embedding_model, embedding_dim, belief_type, is_hypothetical, created_at, last_referenced_at, session_id, conversation_id,
                       cosine_similarity(?, embedding) as similarity
                FROM beliefs 
                WHERE session_id = ? AND similarity >= ?
                ORDER BY similarity DESC
                LIMIT ?
             """,
                (embedding_json, session_id, threshold, limit),
            ) as cursor:
                rows = await cursor.fetchall()

        beliefs = []
        for r in rows:
            beliefs.append(
                Belief(
                    subject=r["subject"],
                    predicate=r["predicate"],
                    value=r["value"],
                    confidence=r["confidence"],
                    turn=r["turn"],
                    source=r["source"],
                    embedding=json.loads(r["embedding"]) if r["embedding"] else [],
                    embedding_model=r["embedding_model"] or "",
                    embedding_dim=r["embedding_dim"] or 0,
                    belief_type=r["belief_type"] or "assertion",
                    is_hypothetical=bool(r["is_hypothetical"]),
                    created_at=datetime.fromisoformat(r["created_at"])
                    if r["created_at"]
                    else datetime.now(timezone.utc),
                    last_referenced_at=datetime.fromisoformat(r["last_referenced_at"])
                    if r["last_referenced_at"]
                    else datetime.now(timezone.utc),
                    session_id=r["session_id"],
                    conversation_id=r["conversation_id"],
                )
            )
        return beliefs

    async def remove_belief(
        self, session_id: str, subject: str, predicate: str
    ) -> None:
        conn = await self._get_connection()
        await conn.execute(
            """
            DELETE FROM beliefs 
            WHERE session_id = ? AND subject = ? AND predicate = ?
        """,
            (session_id, subject, predicate),
        )
        await conn.commit()

    async def update_belief(self, session_id: str, belief: Belief) -> None:
        await self.add_belief(session_id, belief)

    async def clear(self, session_id: str) -> None:
        conn = await self._get_connection()
        await conn.execute("DELETE FROM beliefs WHERE session_id = ?", (session_id,))
        await conn.commit()

    async def belief_count(self, session_id: str) -> int:
        """Return belief count using efficient SELECT COUNT(*) — no deserialization."""
        conn = await self._get_connection()
        async with conn.execute(
            "SELECT COUNT(*) FROM beliefs WHERE session_id = ?", (session_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def health_check(self) -> bool:
        """Verify SQLite connection is functional."""
        try:
            conn = await self._get_connection()
            async with conn.execute("SELECT 1") as cursor:
                row = await cursor.fetchone()
            return row is not None
        except Exception:
            return False

    async def prune_expired_beliefs(
        self, max_age_seconds: int, session_id: Optional[str] = None
    ) -> int:
        """Remove beliefs older than max_age_seconds.

        Args:
            max_age_seconds: Age threshold in seconds
            session_id: Optional - prune only for specific session, None = all sessions

        Returns:
            Number of beliefs deleted
        """
        import logging
        from datetime import timedelta, timezone

        logger = logging.getLogger(__name__)
        conn = await self._get_connection()
        cutoff_time = (
            datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)
        ).isoformat()

        if session_id:
            cursor = await conn.execute(
                "DELETE FROM beliefs WHERE session_id = ? AND created_at < ?",
                (session_id, cutoff_time),
            )
        else:
            cursor = await conn.execute(
                "DELETE FROM beliefs WHERE created_at < ?", (cutoff_time,)
            )

        await conn.commit()
        deleted_count = cursor.rowcount
        if deleted_count > 0:
            logger.debug(
                f"Pruned {deleted_count} expired beliefs (older than {max_age_seconds}s)"
            )
        return int(deleted_count)

    async def get_session_belief_age_stats(self, session_id: str) -> dict[str, Any]:
        """Get age statistics for beliefs in a session.

        Returns:
            Dict with: oldest_belief_age_seconds, newest_belief_age_seconds, avg_age_seconds
        """
        conn = await self._get_connection()
        async with conn.execute(
            """
            SELECT 
                MIN(CAST((julianday('now') - julianday(created_at)) * 86400 AS INTEGER)) as oldest_age,
                MAX(CAST((julianday('now') - julianday(created_at)) * 86400 AS INTEGER)) as newest_age,
                AVG(CAST((julianday('now') - julianday(created_at)) * 86400 AS INTEGER)) as avg_age
            FROM beliefs WHERE session_id = ?
        """,
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()

        if row is None or row["oldest_age"] is None:
            return {
                "oldest_belief_age_seconds": 0,
                "newest_belief_age_seconds": 0,
                "avg_age_seconds": 0,
            }

        return {
            "oldest_belief_age_seconds": row["oldest_age"] or 0,
            "newest_belief_age_seconds": row["newest_age"] or 0,
            "avg_age_seconds": row["avg_age"] or 0,
        }
