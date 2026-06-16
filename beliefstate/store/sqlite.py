import json
from typing import List, Optional, Any
from beliefstate.store.base import Store
from beliefstate.models import Belief

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

    async def _get_connection(self) -> Any:
        if not aiosqlite:
            raise RuntimeError(
                "aiosqlite is not installed. Run `pip install aiosqlite`"
            )
        if self._conn is None:
            import os

            if self.db_path != ":memory:":
                parent = os.path.dirname(self.db_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
            self._conn = await aiosqlite.connect(self.db_path)
            self._conn.row_factory = aiosqlite.Row
            await self._conn.create_function(
                "cosine_similarity", 2, cosine_similarity_py
            )
            await self._init_db()
        return self._conn

    async def _init_db(self) -> None:
        conn = self._conn
        if conn is None:
            return
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS beliefs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                value TEXT NOT NULL,
                confidence REAL NOT NULL,
                turn INTEGER NOT NULL,
                source TEXT NOT NULL,
                embedding TEXT,
                UNIQUE(session_id, subject, predicate)
            )
        """)
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_session ON beliefs(session_id)"
        )
        await conn.commit()

    async def add_belief(self, session_id: str, belief: Belief) -> None:
        conn = await self._get_connection()
        embedding_json = json.dumps(belief.embedding) if belief.embedding else "[]"
        await conn.execute(
            """
            INSERT INTO beliefs (session_id, subject, predicate, value, confidence, turn, source, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, subject, predicate) DO UPDATE SET
                value=excluded.value,
                confidence=excluded.confidence,
                turn=excluded.turn,
                source=excluded.source,
                embedding=excluded.embedding
        """,
            (
                session_id,
                belief.subject,
                belief.predicate,
                belief.value,
                belief.confidence,
                belief.turn,
                belief.source,
                embedding_json,
            ),
        )
        await conn.commit()

    async def get_beliefs(self, session_id: str) -> List[Belief]:
        conn = await self._get_connection()
        async with conn.execute(
            """
            SELECT subject, predicate, value, confidence, turn, source, embedding 
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
                )
            )
        return beliefs

    async def search_beliefs(
        self,
        session_id: str,
        embedding: List[float],
        threshold: float = 0.0,
        limit: int = 5,
    ) -> List[Belief]:
        conn = await self._get_connection()
        embedding_json = json.dumps(embedding)
        async with conn.execute(
            """
            SELECT subject, predicate, value, confidence, turn, source, embedding,
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
