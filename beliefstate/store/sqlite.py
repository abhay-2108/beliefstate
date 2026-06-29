import json
import logging
import os
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from beliefstate.store.base import Store
from beliefstate.store.utils import cosine_similarity, pack_embedding, unpack_embedding
from beliefstate.models import Belief

logger = logging.getLogger(__name__)

try:
    import aiosqlite
except ImportError:
    aiosqlite = None  # type: ignore[assignment]


def cosine_similarity_py(emb1_str: str, emb2_json_str: str) -> float:
    """Compute cosine similarity between two JSON-encoded embedding strings.

    Kept for backwards compatibility with existing in-progress queries.
    New code should use binary embeddings.
    """
    try:
        v1 = json.loads(emb1_str)
        v2 = json.loads(emb2_json_str)
        if not v1 or not v2:
            return 0.0
        if len(v1) != len(v2):
            return 0.0
        return float(cosine_similarity(v1, v2))
    except Exception:
        return 0.0


def cosine_similarity_binary(emb1_bytes: bytes, emb2_bytes: bytes) -> float:
    """Compute cosine similarity between two binary-packed float32 embeddings."""
    try:
        if not emb1_bytes or not emb2_bytes:
            return 0.0
        n1 = len(emb1_bytes) // 4
        n2 = len(emb2_bytes) // 4
        if n1 == 0 or n2 == 0:
            return 0.0
        v1 = unpack_embedding(emb1_bytes)
        v2 = unpack_embedding(emb2_bytes)
        if n1 != n2:
            return 0.0
        return float(cosine_similarity(v1, v2))
    except Exception:
        return 0.0


class SQLiteStore(Store):
    """SQLite-based asynchronous storage for beliefs.

    Uses a single persistent connection, binary float32 embedding storage,
    WAL mode for crash resilience, and turn-based optimistic concurrency.
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._conn: Optional[Any] = None

    async def open(self) -> None:
        """Open and initialize the database connection (called once)."""
        if self._conn is not None:
            return
        if not aiosqlite:
            raise RuntimeError(
                "aiosqlite is not installed. Run `pip install aiosqlite`"
            )

        if self.db_path != ":memory:":
            parent = os.path.dirname(self.db_path)
            if parent:
                os.makedirs(parent, exist_ok=True)

        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row

        if self.db_path != ":memory:":
            await self._conn.execute("PRAGMA journal_mode=WAL")
            await self._conn.execute("PRAGMA synchronous=NORMAL")
            await self._conn.execute("PRAGMA busy_timeout=5000")
            await self._conn.execute("PRAGMA foreign_keys=ON")
            await self._conn.commit()

        # Register cosine similarity for JSON fallback
        await self._conn.create_function("cosine_similarity", 2, cosine_similarity_py)

        # Register binary cosine similarity
        await self._conn.create_function(
            "cosine_similarity_bin", 2, cosine_similarity_binary
        )

        await self._init_db()

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self) -> "SQLiteStore":
        await self.open()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
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

        async with conn.execute("PRAGMA table_info(beliefs)") as cursor:
            existing_columns = await cursor.fetchall()

        if not existing_columns:
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
                    source_quote TEXT NOT NULL DEFAULT '',
                    category TEXT NOT NULL DEFAULT '',
                    belief_type TEXT DEFAULT 'assertion',
                    is_hypothetical INTEGER DEFAULT 0,
                    embedding BLOB,
                    embedding_model TEXT DEFAULT '',
                    embedding_dim INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_referenced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(session_id, conversation_id, subject, predicate)
                )
            """)
        else:
            existing_column_names = {row[1] for row in existing_columns}

            _VALID_COLUMNS = {
                "conversation_id": "TEXT NOT NULL DEFAULT ''",
                "embedding_dim": "INTEGER DEFAULT 0",
                "belief_type": "TEXT DEFAULT 'assertion'",
                "is_hypothetical": "INTEGER DEFAULT 0",
                "last_referenced_at": "TIMESTAMP",
                "source_quote": "TEXT NOT NULL DEFAULT ''",
                "category": "TEXT NOT NULL DEFAULT ''",
            }

            for col_name, col_def in _VALID_COLUMNS.items():
                if col_name not in existing_column_names:
                    if not col_name.isidentifier():
                        logger.warning(f"Skipping invalid column name: {col_name}")
                        continue
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
            "CREATE INDEX IF NOT EXISTS idx_session_subject ON beliefs(session_id, subject, predicate)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_created_at ON beliefs(created_at)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_last_referenced ON beliefs(last_referenced_at)"
        )

        # Audit table for belief mutation history
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS beliefs_audit (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                conversation_id TEXT NOT NULL DEFAULT '',
                subject     TEXT NOT NULL,
                predicate   TEXT NOT NULL,
                old_value   TEXT,
                new_value   TEXT NOT NULL,
                operation   TEXT NOT NULL,
                source_quote TEXT,
                confidence  REAL,
                turn        INTEGER,
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        # Migrate beliefs_audit: add conversation_id if missing
        try:
            async with conn.execute("PRAGMA table_info(beliefs_audit)") as cursor:
                audit_cols = {row[1] for row in await cursor.fetchall()}
            if "conversation_id" not in audit_cols:
                await conn.execute(
                    "ALTER TABLE beliefs_audit ADD COLUMN conversation_id TEXT NOT NULL DEFAULT ''"
                )
                logger.info("Added 'conversation_id' to beliefs_audit table")
        except Exception as e:
            logger.debug(f"Audit table migration check failed (non-critical): {e}")

        await conn.commit()

    async def _audit(
        self,
        belief: Belief,
        operation: str,
        old_value: Optional[str] = None,
    ) -> None:
        """Write an immutable audit record for a belief mutation."""
        conn = await self._get_connection()
        await conn.execute(
            """INSERT INTO beliefs_audit
               (session_id, conversation_id, subject, predicate, old_value, new_value,
                operation, source_quote, confidence, turn)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
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
            ),
        )

    async def add_belief(self, session_id: str, belief: Belief) -> None:
        conn = await self._get_connection()
        embedding_blob = pack_embedding(belief.embedding) if belief.embedding else b""
        conversation_id = belief.conversation_id or ""
        # Normalize to lowercase for case-insensitive matching
        subject = (belief.subject or "").lower()
        predicate = (belief.predicate or "").lower()

        # Check for existing belief for audit trail
        old_value = None
        try:
            existing = await self.get_by_key(
                subject,
                predicate,
                session_id,
                conversation_id,
            )
            if existing:
                old_value = existing.value
        except Exception as e:
            logger.debug(f"Audit lookup failed (non-critical): {e}")

        await conn.execute(
            """
            INSERT INTO beliefs (session_id, conversation_id, subject, predicate, value, confidence, turn, source, source_quote, category, embedding, embedding_model, embedding_dim, belief_type, is_hypothetical, created_at, last_referenced_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, conversation_id, subject, predicate) DO UPDATE SET
                value=excluded.value,
                confidence=excluded.confidence,
                turn=excluded.turn,
                source=excluded.source,
                source_quote=excluded.source_quote,
                category=excluded.category,
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
                subject,
                predicate,
                belief.value,
                belief.confidence,
                belief.turn,
                belief.source,
                getattr(belief, "source_quote", ""),
                getattr(belief, "category", ""),
                embedding_blob,
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

        # Audit: create or update (single commit for both operations)
        if old_value is not None and old_value != belief.value:
            await self._audit(belief, "contradiction_update", old_value)
        elif old_value is None:
            await self._audit(belief, "create")
        await conn.commit()

    async def get_beliefs(
        self, session_id: str, conversation_id: Optional[str] = None
    ) -> List[Belief]:
        conn = await self._get_connection()

        if conversation_id:
            async with conn.execute(
                """
                SELECT subject, predicate, value, confidence, turn, source, source_quote, category, embedding, embedding_model, embedding_dim, belief_type, is_hypothetical, created_at, last_referenced_at, session_id, conversation_id
                FROM beliefs WHERE session_id = ? AND conversation_id = ?
            """,
                (session_id, conversation_id),
            ) as cursor:
                rows = await cursor.fetchall()
        else:
            async with conn.execute(
                """
                SELECT subject, predicate, value, confidence, turn, source, source_quote, category, embedding, embedding_model, embedding_dim, belief_type, is_hypothetical, created_at, last_referenced_at, session_id, conversation_id
                FROM beliefs WHERE session_id = ?
            """,
                (session_id,),
            ) as cursor:
                rows = await cursor.fetchall()

        return [self._row_to_belief(r) for r in rows]

    def _row_to_belief(self, r: Any) -> Belief:
        """Convert a database row to a Belief object."""
        embedding_data = r["embedding"]
        if embedding_data:
            if isinstance(embedding_data, (bytes, bytearray)):
                emb = unpack_embedding(bytes(embedding_data))
            else:
                try:
                    emb = json.loads(embedding_data)
                except (json.JSONDecodeError, TypeError):
                    emb = []
        else:
            emb = []

        def _ensure_aware(dt: datetime) -> datetime:
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt

        return Belief(
            subject=r["subject"],
            predicate=r["predicate"],
            value=r["value"],
            confidence=r["confidence"],
            turn=r["turn"],
            source=r["source"],
            source_quote=r["source_quote"] or "",
            category=r["category"] or "",
            embedding=emb,
            embedding_model=r["embedding_model"] or "",
            embedding_dim=r["embedding_dim"] or 0,
            belief_type=r["belief_type"] or "assertion",
            is_hypothetical=bool(r["is_hypothetical"]),
            created_at=_ensure_aware(datetime.fromisoformat(r["created_at"]))
            if r["created_at"]
            else datetime.now(timezone.utc),
            last_referenced_at=_ensure_aware(
                datetime.fromisoformat(r["last_referenced_at"])
            )
            if r["last_referenced_at"]
            else datetime.now(timezone.utc),
            session_id=r["session_id"],
            conversation_id=r["conversation_id"],
        )

    async def search_beliefs(
        self,
        session_id: str,
        embedding: List[float],
        threshold: float = 0.0,
        limit: int = 5,
        conversation_id: Optional[str] = None,
    ) -> List[Belief]:
        conn = await self._get_connection()
        embedding_blob = pack_embedding(embedding) if embedding else b""

        if conversation_id:
            async with conn.execute(
                """
                SELECT subject, predicate, value, confidence, turn, source, source_quote, category, embedding, embedding_model, embedding_dim, belief_type, is_hypothetical, created_at, last_referenced_at, session_id, conversation_id
                FROM beliefs
                WHERE session_id = ? AND conversation_id = ? AND cosine_similarity_bin(?, embedding) >= ?
                ORDER BY cosine_similarity_bin(?, embedding) DESC
                LIMIT ?
             """,
                (
                    session_id,
                    conversation_id,
                    embedding_blob,
                    threshold,
                    embedding_blob,
                    limit,
                ),
            ) as cursor:
                rows = await cursor.fetchall()
        else:
            async with conn.execute(
                """
                SELECT subject, predicate, value, confidence, turn, source, source_quote, category, embedding, embedding_model, embedding_dim, belief_type, is_hypothetical, created_at, last_referenced_at, session_id, conversation_id
                FROM beliefs
                WHERE session_id = ? AND cosine_similarity_bin(?, embedding) >= ?
                ORDER BY cosine_similarity_bin(?, embedding) DESC
                LIMIT ?
             """,
                (session_id, embedding_blob, threshold, embedding_blob, limit),
            ) as cursor:
                rows = await cursor.fetchall()

        return [self._row_to_belief(r) for r in rows]

    async def get_by_key(
        self,
        subject: str,
        predicate: str,
        session_id: str,
        conversation_id: Optional[str] = None,
    ) -> Optional[Belief]:
        """Retrieve a single belief by its composite key."""
        conn = await self._get_connection()
        cid = conversation_id or ""
        subject = subject.lower()
        predicate = predicate.lower()

        async with conn.execute(
            """
            SELECT subject, predicate, value, confidence, turn, source, source_quote, category, embedding, embedding_model, embedding_dim, belief_type, is_hypothetical, created_at, last_referenced_at, session_id, conversation_id
            FROM beliefs
            WHERE session_id = ? AND conversation_id = ? AND subject = ? AND predicate = ?
            LIMIT 1
        """,
            (session_id, cid, subject, predicate),
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None
        return self._row_to_belief(row)

    async def remove_belief(
        self,
        session_id: str,
        subject: str,
        predicate: str,
        conversation_id: Optional[str] = None,
    ) -> None:
        conn = await self._get_connection()
        subject = subject.lower()
        predicate = predicate.lower()
        cid = conversation_id or ""

        # Audit before delete
        existing = await self.get_by_key(
            subject, predicate, session_id, conversation_id
        )
        if existing:
            await self._audit(existing, "delete")

        await conn.execute(
            """
            DELETE FROM beliefs
            WHERE session_id = ? AND conversation_id = ? AND subject = ? AND predicate = ?
        """,
            (session_id, cid, subject, predicate),
        )
        await conn.commit()

    async def update_belief(self, session_id: str, belief: Belief) -> None:
        await self.add_belief(session_id, belief)

    async def clear(self, session_id: str) -> None:
        conn = await self._get_connection()
        await conn.execute("DELETE FROM beliefs WHERE session_id = ?", (session_id,))
        await conn.commit()

    async def belief_count(self, session_id: str) -> int:
        conn = await self._get_connection()
        async with conn.execute(
            "SELECT COUNT(*) FROM beliefs WHERE session_id = ?", (session_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def upsert(self, belief: Belief) -> bool:
        """Insert or update a belief with turn-based optimistic concurrency.

        Returns True if the belief was written, False if discarded (stale write).
        """
        existing = await self.get_by_key(
            (belief.subject or "").lower(),
            (belief.predicate or "").lower(),
            belief.session_id or "",
            belief.conversation_id or "",
        )
        if existing and existing.turn > belief.turn:
            return False
        await self.add_belief(belief.session_id or "", belief)
        return True

    async def health_check(self) -> bool:
        try:
            conn = await self._get_connection()
            async with conn.execute("SELECT 1") as cursor:
                row = await cursor.fetchone()
            return row is not None
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    async def prune_expired_beliefs(
        self, max_age_seconds: int, session_id: Optional[str] = None
    ) -> int:
        from datetime import timedelta

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

    async def get_session_belief_age_stats(self, session_id: str) -> Dict[str, Any]:
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

    async def get_audit_history(
        self,
        session_id: str,
        subject: str,
        predicate: str,
    ) -> List[Dict[str, Any]]:
        """Return audit trail for a specific belief."""
        conn = await self._get_connection()
        async with conn.execute(
            """
            SELECT turn, old_value, new_value, operation, confidence, created_at, conversation_id
            FROM beliefs_audit
            WHERE session_id = ? AND subject = ? AND predicate = ?
            ORDER BY id ASC
        """,
            (session_id, subject.lower(), predicate.lower()),
        ) as cursor:
            rows = await cursor.fetchall()

        return [
            {
                "turn": r["turn"],
                "old_value": r["old_value"],
                "new_value": r["new_value"],
                "operation": r["operation"],
                "confidence": r["confidence"],
                "created_at": r["created_at"],
                "conversation_id": r["conversation_id"],
            }
            for r in rows
        ]
