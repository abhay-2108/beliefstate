import pytest
from unittest.mock import AsyncMock

from beliefstate.models import Belief
from beliefstate.store.sqlite import SQLiteStore
from beliefstate.store.redis import RedisStore


@pytest.mark.asyncio
async def test_sqlite_store():
    # SQLite memory DB
    store = SQLiteStore(db_path=":memory:")

    b1 = Belief(
        subject="user_name",
        predicate="is",
        value="Alice",
        confidence=0.95,
        turn=10,
        source="user",
        embedding=[0.5, -0.5, 0.1],
    )
    await store.add_belief("session_123", b1)

    # Retrieves should be retrieved exactly
    retrieved = await store.get_beliefs("session_123")
    assert len(retrieved) == 1
    assert retrieved[0].subject == "user_name"
    assert retrieved[0].value == "Alice"
    assert retrieved[0].embedding == [0.5, -0.5, 0.1]

    # Overwrite on conflict
    b2 = Belief(
        subject="user_name",
        predicate="is",
        value="Bob",
        confidence=0.80,
        turn=11,
        source="assistant",
        embedding=[0.9, 0.9, 0.9],
    )
    await store.add_belief("session_123", b2)

    all_b = await store.get_beliefs("session_123")
    assert len(all_b) == 1
    assert all_b[0].value == "Bob"

    # Remove
    await store.remove_belief("session_123", "user_name", "is")
    all_b = await store.get_beliefs("session_123")
    assert len(all_b) == 0


@pytest.mark.asyncio
async def test_redis_store_mock():
    # Instantiate with dummy namespace config (Note: RedisStore gets key as beliefstate:session:{session_id})
    store = RedisStore(redis_url="redis://localhost:6379")

    # Mock redis client
    mock_client = AsyncMock()
    store._client = mock_client

    b = Belief(
        subject="redis_key",
        predicate="has",
        value="value",
        confidence=1.0,
        turn=1,
        source="user",
        embedding=[0.1, 0.2],
    )

    await store.add_belief("session_123", b)

    # Check hset arguments
    mock_client.hset.assert_called_once()
    args, _ = mock_client.hset.call_args
    assert args[0] == "beliefstate:session:session_123"
    assert args[1] == "redis_key::has"

    # Mock hgetall
    payload = b.model_dump_json()
    mock_client.hgetall.return_value = {"redis_key::has": payload}

    retrieved = await store.get_beliefs("session_123")
    assert len(retrieved) == 1
    assert retrieved[0].value == "value"
    assert retrieved[0].embedding == [0.1, 0.2]


@pytest.mark.asyncio
async def test_sqlite_search_beliefs():
    store = SQLiteStore(db_path=":memory:")

    b1 = Belief(
        subject="user",
        predicate="likes",
        value="apples",
        confidence=1.0,
        turn=1,
        source="user",
        embedding=[1.0, 0.0, 0.0],
    )
    b2 = Belief(
        subject="user",
        predicate="hates",
        value="bananas",
        confidence=1.0,
        turn=2,
        source="user",
        embedding=[0.0, 1.0, 0.0],
    )
    await store.add_belief("session_123", b1)
    await store.add_belief("session_123", b2)

    # Search with query embedding close to b1
    results = await store.search_beliefs(
        "session_123", [1.0, 0.1, 0.0], threshold=0.7, limit=5
    )
    assert len(results) == 1
    assert results[0].value == "apples"

    # Search with limit=1
    results_limited = await store.search_beliefs(
        "session_123", [0.1, 1.0, 0.0], threshold=0.1, limit=1
    )
    assert len(results_limited) == 1
    assert results_limited[0].value == "bananas"


@pytest.mark.asyncio
async def test_redis_search_beliefs():
    store = RedisStore(redis_url="redis://localhost:6379")
    mock_client = AsyncMock()
    store._client = mock_client

    b1 = Belief(
        subject="user",
        predicate="likes",
        value="apples",
        confidence=1.0,
        turn=1,
        source="user",
        embedding=[1.0, 0.0, 0.0],
    )
    b2 = Belief(
        subject="user",
        predicate="hates",
        value="bananas",
        confidence=1.0,
        turn=2,
        source="user",
        embedding=[0.0, 1.0, 0.0],
    )

    mock_client.hgetall.return_value = {
        "user::likes": b1.model_dump_json(),
        "user::hates": b2.model_dump_json(),
    }

    results = await store.search_beliefs(
        "session_123", [1.0, 0.1, 0.0], threshold=0.7, limit=5
    )
    assert len(results) == 1
    assert results[0].value == "apples"
