import pytest
from beliefstate.models import Belief
from beliefstate.resolver import BeliefResolver
from beliefstate.store.sqlite import SQLiteStore


@pytest.mark.asyncio
async def test_resolver_overwrite():
    store = SQLiteStore(db_path=":memory:")
    resolver = BeliefResolver(store=store, strategy="overwrite")

    b_old = Belief(
        subject="USER",
        predicate="likes",
        value="Python",
        confidence=1.0,
        turn=1,
        source="user",
    )
    b_new = Belief(
        subject="USER",
        predicate="likes",
        value="Rust",
        confidence=1.0,
        turn=2,
        source="user",
    )

    # Store old belief first
    await store.add_belief("session_1", b_old)

    # Resolve contradiction
    await resolver.resolve("session_1", [(b_old, b_new, 0.9, "User changed mind")])

    # Check that new belief replaced the old one
    beliefs = await store.get_beliefs("session_1")
    assert len(beliefs) == 1
    assert beliefs[0].value == "Rust"

    # Check pending conflict notes
    conflicts = resolver.pop_pending_conflicts("session_1")
    assert len(conflicts) == 1
    assert "Previously stated: 'Python'. Now asserting: 'Rust'" in conflicts[0]


@pytest.mark.asyncio
async def test_resolver_keep_old():
    store = SQLiteStore(db_path=":memory:")
    resolver = BeliefResolver(store=store, strategy="keep_old")

    b_old = Belief(
        subject="USER",
        predicate="likes",
        value="Python",
        confidence=1.0,
        turn=1,
        source="user",
    )
    b_new = Belief(
        subject="USER",
        predicate="likes",
        value="Rust",
        confidence=1.0,
        turn=2,
        source="user",
    )

    # Store old belief first
    await store.add_belief("session_1", b_old)

    # Resolve contradiction
    await resolver.resolve("session_1", [(b_old, b_new, 0.9, "User changed mind")])

    # Check that old belief was NOT replaced (remains Python)
    beliefs = await store.get_beliefs("session_1")
    assert len(beliefs) == 1
    assert beliefs[0].value == "Python"


@pytest.mark.asyncio
async def test_resolver_raise():
    store = SQLiteStore(db_path=":memory:")
    resolver = BeliefResolver(store=store, strategy="raise")

    b_old = Belief(
        subject="USER",
        predicate="likes",
        value="Python",
        confidence=1.0,
        turn=1,
        source="user",
    )
    b_new = Belief(
        subject="USER",
        predicate="likes",
        value="Rust",
        confidence=1.0,
        turn=2,
        source="user",
    )

    # Store old belief first
    await store.add_belief("session_1", b_old)

    # Resolve contradiction should raise ValueError
    with pytest.raises(ValueError, match="Contradiction detected"):
        await resolver.resolve("session_1", [(b_old, b_new, 0.9, "User changed mind")])
