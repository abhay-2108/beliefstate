"""Tests for production bug fixes: search, upsert, shutdown, resolver, audit."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from beliefstate.models import Belief
from beliefstate.config import TrackerConfig


# ─── SQLite search_beliefs() SQL alias bug fix ───
@pytest.mark.asyncio
async def test_sqlite_search_beliefs_returns_results():
    """search_beliefs must not crash with 'no such column: similarity'."""
    from beliefstate.store.sqlite import SQLiteStore

    store = SQLiteStore(":memory:")
    await store.open()

    belief = Belief(
        subject="USER",
        predicate="likes",
        value="Python",
        confidence=0.9,
        turn=1,
        source="user",
        embedding=[1.0, 0.0, 0.0],
        embedding_model="test",
        embedding_dim=3,
        session_id="s1",
    )
    await store.add_belief("s1", belief)

    results = await store.search_beliefs(
        "s1", embedding=[1.0, 0.0, 0.0], threshold=0.5, limit=5
    )
    assert len(results) >= 1
    assert results[0].value == "Python"
    await store.close()


# ─── SQLite search_beliefs with conversation_id ───
@pytest.mark.asyncio
async def test_sqlite_search_beliefs_conversation_filter():
    """search_beliefs with conversation_id must filter correctly."""
    from beliefstate.store.sqlite import SQLiteStore

    store = SQLiteStore(":memory:")
    await store.open()

    b1 = Belief(
        subject="USER",
        predicate="likes",
        value="A",
        confidence=0.9,
        turn=1,
        source="user",
        embedding=[1.0, 0.0],
        embedding_model="test",
        embedding_dim=2,
        session_id="s1",
        conversation_id="c1",
    )
    b2 = Belief(
        subject="USER",
        predicate="likes",
        value="B",
        confidence=0.9,
        turn=1,
        source="user",
        embedding=[1.0, 0.0],
        embedding_model="test",
        embedding_dim=2,
        session_id="s1",
        conversation_id="c2",
    )
    await store.add_belief("s1", b1)
    await store.add_belief("s1", b2)

    results_c1 = await store.search_beliefs(
        "s1", [1.0, 0.0], threshold=0.5, limit=5, conversation_id="c1"
    )
    assert all(r.conversation_id == "c1" for r in results_c1)
    await store.close()


# ─── Upsert case-sensitivity fix ───
@pytest.mark.asyncio
async def test_sqlite_upsert_case_insensitive_no_duplicates():
    """Upsert with different casing must not create duplicate rows."""
    from beliefstate.store.sqlite import SQLiteStore

    store = SQLiteStore(":memory:")
    await store.open()

    b1 = Belief(
        subject="Alice",
        predicate="likes",
        value="tea",
        confidence=0.9,
        turn=1,
        source="user",
        session_id="s1",
    )
    await store.add_belief("s1", b1)

    b2 = Belief(
        subject="alice",
        predicate="likes",
        value="coffee",
        confidence=0.9,
        turn=2,
        source="user",
        session_id="s1",
    )
    result = await store.upsert(b2)
    assert result is True

    beliefs = await store.get_beliefs("s1")
    assert len(beliefs) == 1
    assert beliefs[0].value == "coffee"
    await store.close()


# ─── Memory upsert case-sensitivity fix ───
@pytest.mark.asyncio
async def test_memory_upsert_case_insensitive_no_duplicates():
    """Upsert with different casing must not create duplicate rows."""
    from beliefstate.store.memory import InMemoryBeliefStore

    store = InMemoryBeliefStore()

    b1 = Belief(
        subject="Alice",
        predicate="likes",
        value="tea",
        confidence=0.9,
        turn=1,
        source="user",
        session_id="s1",
    )
    await store.add_belief("s1", b1)

    b2 = Belief(
        subject="alice",
        predicate="likes",
        value="coffee",
        confidence=0.9,
        turn=2,
        source="user",
        session_id="s1",
    )
    result = await store.upsert(b2)
    assert result is True

    beliefs = await store.get_beliefs("s1")
    assert len(beliefs) == 1
    assert beliefs[0].value == "coffee"


# ─── Memory search_beliefs with conversation_id ───
@pytest.mark.asyncio
async def test_memory_search_beliefs_conversation_filter():
    """search_beliefs with conversation_id must filter correctly."""
    from beliefstate.store.memory import InMemoryBeliefStore

    store = InMemoryBeliefStore()

    b1 = Belief(
        subject="USER",
        predicate="likes",
        value="A",
        confidence=0.9,
        turn=1,
        source="user",
        embedding=[1.0, 0.0],
        embedding_model="test",
        embedding_dim=2,
        session_id="s1",
        conversation_id="c1",
    )
    b2 = Belief(
        subject="USER",
        predicate="likes",
        value="B",
        confidence=0.9,
        turn=1,
        source="user",
        embedding=[1.0, 0.0],
        embedding_model="test",
        embedding_dim=2,
        session_id="s1",
        conversation_id="c2",
    )
    await store.add_belief("s1", b1)
    await store.add_belief("s1", b2)

    results_c1 = await store.search_beliefs(
        "s1", [1.0, 0.0], threshold=0.5, limit=5, conversation_id="c1"
    )
    assert all(r.conversation_id == "c1" for r in results_c1)


# ─── Judge broad except fix ───
@pytest.mark.asyncio
async def test_judge_json_decode_error_handled():
    """Judge should handle malformed JSON gracefully without catching all exceptions."""
    from beliefstate.judge import LLMJudge

    config = TrackerConfig()
    mock_adapter = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.text = "this is not json at all"
    mock_adapter.generate = AsyncMock(return_value=mock_resp)

    judge = LLMJudge(mock_adapter, config)
    old = Belief(
        subject="USER",
        predicate="likes",
        value="A",
        confidence=0.9,
        turn=1,
        source="user",
        session_id="s1",
    )
    new = Belief(
        subject="USER",
        predicate="likes",
        value="B",
        confidence=0.9,
        turn=2,
        source="user",
        session_id="s1",
    )
    is_contradiction, score, reason = await judge.check(old, new)
    assert isinstance(is_contradiction, bool)
    assert isinstance(score, float)


# ─── Resolver keep_old no phantom escalation ───
@pytest.mark.asyncio
async def test_resolver_keep_old_generates_feedback():
    """keep_old strategy must generate a conflict note so the user gets feedback."""
    from beliefstate.resolver import BeliefResolver
    from beliefstate.store.memory import InMemoryBeliefStore

    store = InMemoryBeliefStore()
    resolver = BeliefResolver(store, strategy="keep_old")

    old_b = Belief(
        subject="USER",
        predicate="likes",
        value="A",
        confidence=0.9,
        turn=1,
        source="user",
        session_id="s1",
    )
    new_b = Belief(
        subject="USER",
        predicate="likes",
        value="B",
        confidence=0.9,
        turn=2,
        source="user",
        session_id="s1",
    )

    await resolver.resolve("s1", [(old_b, new_b, 0.9, "test reason")])
    conflicts = resolver.pop_pending_conflicts("s1")
    assert len(conflicts) == 1
    assert "kept old" in conflicts[0]
    assert "A" in conflicts[0]
    assert "B" in conflicts[0]


# ─── Shutdown closes store ───
@pytest.mark.asyncio
async def test_shutdown_closes_store():
    """shutdown() should close the store connection."""
    from beliefstate.tracker import BeliefTracker
    from beliefstate.store.sqlite import SQLiteStore

    store = SQLiteStore(":memory:")
    config = TrackerConfig()
    tracker = BeliefTracker(config=config, store=store)
    await store.open()

    # Verify connection is open
    conn = await store._get_connection()
    assert conn is not None

    await tracker.shutdown(grace_seconds=0.1)
    # After shutdown, store should be closed
    assert store._conn is None


# ─── Audit trail includes conversation_id ───
@pytest.mark.asyncio
async def test_sqlite_audit_includes_conversation_id():
    """Audit records must include conversation_id."""
    from beliefstate.store.sqlite import SQLiteStore

    store = SQLiteStore(":memory:")
    await store.open()

    b = Belief(
        subject="USER",
        predicate="likes",
        value="A",
        confidence=0.9,
        turn=1,
        source="user",
        session_id="s1",
        conversation_id="c1",
    )
    await store.add_belief("s1", b)

    # Update the belief
    b2 = Belief(
        subject="USER",
        predicate="likes",
        value="B",
        confidence=0.9,
        turn=2,
        source="user",
        session_id="s1",
        conversation_id="c1",
    )
    await store.add_belief("s1", b2)

    audit = await store.get_audit_history("s1", "user", "likes")
    assert len(audit) >= 2
    assert audit[0].get("conversation_id") == "c1"
    await store.close()


# ─── Store context managers work ───
@pytest.mark.asyncio
async def test_memory_store_context_manager():
    """Memory store should support async context manager."""
    from beliefstate.store.memory import InMemoryBeliefStore

    async with InMemoryBeliefStore() as store:
        b = Belief(
            subject="USER",
            predicate="likes",
            value="X",
            confidence=0.9,
            turn=1,
            source="user",
            session_id="s1",
        )
        await store.add_belief("s1", b)
        assert await store.belief_count("s1") == 1
