"""Advanced tests for SQLiteStore — context manager, conversations, pruning, stats."""
import pytest
from datetime import datetime, timezone, timedelta

from beliefstate.models import Belief
from beliefstate.store.sqlite import SQLiteStore


def _make_belief(subject="USER", predicate="likes", value="Python", **kwargs) -> Belief:
    defaults = dict(confidence=1.0, turn=1, source="user")
    defaults.update(kwargs)
    return Belief(subject=subject, predicate=predicate, value=value, **defaults)


class TestSQLiteContextManager:
    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        async with SQLiteStore(db_path=":memory:") as store:
            await store.add_belief("s1", _make_belief())
            beliefs = await store.get_beliefs("s1")
            assert len(beliefs) == 1
        # After exit, connection should be closed
        assert store._conn is None

    @pytest.mark.asyncio
    async def test_explicit_open_close(self):
        store = SQLiteStore(db_path=":memory:")
        await store.open()
        assert store._conn is not None
        await store.close()
        assert store._conn is None


class TestSQLiteConversationFiltering:
    @pytest.mark.asyncio
    async def test_get_beliefs_by_conversation_id(self):
        store = SQLiteStore(db_path=":memory:")
        b1 = _make_belief(value="Python", conversation_id="conv_1")
        b2 = _make_belief(predicate="hates", value="Java", conversation_id="conv_2")
        await store.add_belief("s1", b1)
        await store.add_belief("s1", b2)

        conv1 = await store.get_beliefs("s1", conversation_id="conv_1")
        assert len(conv1) == 1
        assert conv1[0].value == "Python"

        conv2 = await store.get_beliefs("s1", conversation_id="conv_2")
        assert len(conv2) == 1
        assert conv2[0].value == "Java"

    @pytest.mark.asyncio
    async def test_get_all_beliefs_across_conversations(self):
        store = SQLiteStore(db_path=":memory:")
        b1 = _make_belief(value="Python", conversation_id="c1")
        b2 = _make_belief(predicate="hates", value="Java", conversation_id="c2")
        await store.add_belief("s1", b1)
        await store.add_belief("s1", b2)

        all_beliefs = await store.get_beliefs("s1")
        assert len(all_beliefs) == 2


class TestSQLiteUpsert:
    @pytest.mark.asyncio
    async def test_upsert_overwrites_on_conflict(self):
        store = SQLiteStore(db_path=":memory:")
        b1 = _make_belief(value="Python", confidence=0.5)
        await store.add_belief("s1", b1)

        b2 = _make_belief(value="Rust", confidence=0.9)
        await store.add_belief("s1", b2)

        beliefs = await store.get_beliefs("s1")
        assert len(beliefs) == 1
        assert beliefs[0].value == "Rust"
        assert beliefs[0].confidence == 0.9

    @pytest.mark.asyncio
    async def test_different_predicates_not_conflict(self):
        store = SQLiteStore(db_path=":memory:")
        b1 = _make_belief(predicate="likes", value="Python")
        b2 = _make_belief(predicate="hates", value="Java")
        await store.add_belief("s1", b1)
        await store.add_belief("s1", b2)

        beliefs = await store.get_beliefs("s1")
        assert len(beliefs) == 2


class TestSQLiteSessionIsolation:
    @pytest.mark.asyncio
    async def test_sessions_isolated(self):
        store = SQLiteStore(db_path=":memory:")
        await store.add_belief("s1", _make_belief(value="Python"))
        await store.add_belief("s2", _make_belief(value="Rust"))

        s1 = await store.get_beliefs("s1")
        s2 = await store.get_beliefs("s2")
        assert len(s1) == 1 and s1[0].value == "Python"
        assert len(s2) == 1 and s2[0].value == "Rust"

    @pytest.mark.asyncio
    async def test_clear_one_session_preserves_other(self):
        store = SQLiteStore(db_path=":memory:")
        await store.add_belief("s1", _make_belief(value="Python"))
        await store.add_belief("s2", _make_belief(value="Rust"))
        await store.clear("s1")
        assert await store.get_beliefs("s1") == []
        assert len(await store.get_beliefs("s2")) == 1


class TestSQLitePruning:
    @pytest.mark.asyncio
    async def test_prune_expired_beliefs(self):
        store = SQLiteStore(db_path=":memory:")
        # Add a belief with an old created_at
        old_belief = _make_belief(
            value="old_fact",
            created_at=datetime.now(timezone.utc) - timedelta(days=30),
        )
        new_belief = _make_belief(
            predicate="knows",
            value="new_fact",
            created_at=datetime.now(timezone.utc),
        )
        await store.add_belief("s1", old_belief)
        await store.add_belief("s1", new_belief)

        # Prune beliefs older than 1 day
        deleted = await store.prune_expired_beliefs(86400, session_id="s1")
        assert deleted >= 1

        remaining = await store.get_beliefs("s1")
        assert len(remaining) == 1
        assert remaining[0].value == "new_fact"

    @pytest.mark.asyncio
    async def test_prune_nothing_when_fresh(self):
        store = SQLiteStore(db_path=":memory:")
        await store.add_belief("s1", _make_belief())
        deleted = await store.prune_expired_beliefs(86400, session_id="s1")
        assert deleted == 0


class TestSQLiteAgeStats:
    @pytest.mark.asyncio
    async def test_age_stats_empty_session(self):
        store = SQLiteStore(db_path=":memory:")
        stats = await store.get_session_belief_age_stats("nonexistent")
        assert stats["oldest_belief_age_seconds"] == 0

    @pytest.mark.asyncio
    async def test_age_stats_with_beliefs(self):
        store = SQLiteStore(db_path=":memory:")
        await store.add_belief("s1", _make_belief())
        stats = await store.get_session_belief_age_stats("s1")
        # Just created, so age should be very small
        assert stats["oldest_belief_age_seconds"] >= 0
        assert stats["avg_age_seconds"] >= 0


class TestSQLiteSearchConversation:
    @pytest.mark.asyncio
    async def test_search_within_conversation(self):
        store = SQLiteStore(db_path=":memory:")
        b1 = _make_belief(value="apples", embedding=[1.0, 0.0, 0.0], conversation_id="c1")
        b2 = _make_belief(predicate="hates", value="bananas", embedding=[1.0, 0.1, 0.0], conversation_id="c2")
        await store.add_belief("s1", b1)
        await store.add_belief("s1", b2)

        # Search only within conversation c1
        results = await store.search_beliefs(
            "s1", [1.0, 0.0, 0.0], threshold=0.5, conversation_id="c1"
        )
        assert len(results) == 1
        assert results[0].value == "apples"
