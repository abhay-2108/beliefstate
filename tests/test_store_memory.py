"""Tests for InMemoryBeliefStore — CRUD, LRU eviction, search, stats."""
import pytest

from beliefstate.models import Belief
from beliefstate.store.memory import InMemoryBeliefStore


def _make_belief(subject="USER", predicate="likes", value="Python", **kwargs) -> Belief:
    """Helper to create a Belief with sensible defaults."""
    defaults = dict(confidence=1.0, turn=1, source="user")
    defaults.update(kwargs)
    return Belief(subject=subject, predicate=predicate, value=value, **defaults)


class TestInMemoryBasicCRUD:
    @pytest.mark.asyncio
    async def test_add_and_get(self):
        store = InMemoryBeliefStore()
        b = _make_belief()
        await store.add_belief("s1", b)
        beliefs = await store.get_beliefs("s1")
        assert len(beliefs) == 1
        assert beliefs[0].value == "Python"

    @pytest.mark.asyncio
    async def test_get_empty_session(self):
        store = InMemoryBeliefStore()
        assert await store.get_beliefs("nonexistent") == []

    @pytest.mark.asyncio
    async def test_upsert_overwrites(self):
        store = InMemoryBeliefStore()
        b1 = _make_belief(value="Python")
        b2 = _make_belief(value="Rust")  # same subject+predicate
        await store.add_belief("s1", b1)
        await store.add_belief("s1", b2)
        beliefs = await store.get_beliefs("s1")
        assert len(beliefs) == 1
        assert beliefs[0].value == "Rust"

    @pytest.mark.asyncio
    async def test_remove_belief(self):
        store = InMemoryBeliefStore()
        await store.add_belief("s1", _make_belief())
        await store.remove_belief("s1", "USER", "likes")
        assert await store.get_beliefs("s1") == []

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self):
        store = InMemoryBeliefStore()
        # Should not raise
        await store.remove_belief("s1", "USER", "likes")

    @pytest.mark.asyncio
    async def test_clear_session(self):
        store = InMemoryBeliefStore()
        await store.add_belief("s1", _make_belief())
        await store.add_belief("s1", _make_belief(predicate="hates", value="Java"))
        await store.clear("s1")
        assert await store.get_beliefs("s1") == []

    @pytest.mark.asyncio
    async def test_update_belief(self):
        store = InMemoryBeliefStore()
        b = _make_belief(confidence=0.5)
        await store.add_belief("s1", b)
        b_updated = _make_belief(confidence=0.99)
        await store.update_belief("s1", b_updated)
        beliefs = await store.get_beliefs("s1")
        assert beliefs[0].confidence == 0.99


class TestInMemorySessionIsolation:
    @pytest.mark.asyncio
    async def test_sessions_isolated(self):
        store = InMemoryBeliefStore()
        await store.add_belief("s1", _make_belief(value="Python"))
        await store.add_belief("s2", _make_belief(value="Rust"))
        s1 = await store.get_beliefs("s1")
        s2 = await store.get_beliefs("s2")
        assert len(s1) == 1 and s1[0].value == "Python"
        assert len(s2) == 1 and s2[0].value == "Rust"

    @pytest.mark.asyncio
    async def test_clear_one_session_preserves_other(self):
        store = InMemoryBeliefStore()
        await store.add_belief("s1", _make_belief(value="Python"))
        await store.add_belief("s2", _make_belief(value="Rust"))
        await store.clear("s1")
        assert await store.get_beliefs("s1") == []
        assert len(await store.get_beliefs("s2")) == 1


class TestInMemoryLRUEviction:
    @pytest.mark.asyncio
    async def test_eviction_when_over_limit(self):
        # Use a very small max_bytes so beliefs get evicted
        store = InMemoryBeliefStore(max_bytes=500)
        # Add beliefs until eviction kicks in
        for i in range(20):
            await store.add_belief(
                "s1",
                _make_belief(predicate=f"fact_{i}", value=f"value_{i}"),
            )
        beliefs = await store.get_beliefs("s1")
        # Some should have been evicted
        assert len(beliefs) < 20
        # Total bytes should be within limit
        assert store.current_bytes <= store.max_bytes


class TestInMemorySearch:
    @pytest.mark.asyncio
    async def test_search_by_similarity(self):
        store = InMemoryBeliefStore()
        b1 = _make_belief(value="apples", embedding=[1.0, 0.0, 0.0])
        b2 = _make_belief(predicate="hates", value="bananas", embedding=[0.0, 1.0, 0.0])
        await store.add_belief("s1", b1)
        await store.add_belief("s1", b2)

        results = await store.search_beliefs("s1", [1.0, 0.1, 0.0], threshold=0.7)
        assert len(results) == 1
        assert results[0].value == "apples"

    @pytest.mark.asyncio
    async def test_search_empty_store(self):
        store = InMemoryBeliefStore()
        results = await store.search_beliefs("s1", [1.0, 0.0], threshold=0.5)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_respects_limit(self):
        store = InMemoryBeliefStore()
        for i in range(10):
            await store.add_belief(
                "s1",
                _make_belief(predicate=f"fact_{i}", embedding=[1.0, 0.0, 0.0]),
            )
        results = await store.search_beliefs("s1", [1.0, 0.0, 0.0], threshold=0.0, limit=3)
        assert len(results) <= 3


class TestInMemoryStats:
    @pytest.mark.asyncio
    async def test_stats_empty_store(self):
        store = InMemoryBeliefStore()
        stats = store.get_stats()
        assert stats["total_sessions"] == 0
        assert stats["total_beliefs"] == 0

    @pytest.mark.asyncio
    async def test_stats_with_data(self):
        store = InMemoryBeliefStore()
        await store.add_belief("s1", _make_belief())
        await store.add_belief("s2", _make_belief(value="Rust"))
        stats = store.get_stats()
        assert stats["total_sessions"] == 2
        assert stats["total_beliefs"] == 2
        assert stats["current_bytes"] > 0
