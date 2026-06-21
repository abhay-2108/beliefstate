import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from beliefstate import TrackerConfig, BeliefTracker, Belief
from beliefstate.store.postgres import PostgreSQLStore
from beliefstate.tracker import AsyncStreamWrapper
from beliefstate.extractor import BeliefExtractor


@pytest.mark.asyncio
async def test_postgres_store_mocked():
    mock_pool = MagicMock()
    mock_pool.close = AsyncMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

    with patch("beliefstate.store.postgres.asyncpg") as mock_asyncpg:
        mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)

        store = PostgreSQLStore(dsn="postgresql://localhost/db")
        await store.open()

        mock_asyncpg.create_pool.assert_called_once_with("postgresql://localhost/db")
        assert store._pool == mock_pool

        # Verify schema initialization executes SQL
        assert mock_conn.execute.call_count >= 2

        # Test add_belief call
        belief = Belief(
            subject="USER",
            predicate="likes",
            value="coffee",
            turn=1,
            source="user",
            confidence=1.0,
        )
        await store.add_belief("session1", belief)

        mock_conn.execute.assert_called()

        # Test get_beliefs call
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "subject": "USER",
                    "predicate": "likes",
                    "value": "coffee",
                    "confidence": 1.0,
                    "turn": 1,
                    "source": "user",
                    "embedding": [0.1, 0.2],
                    "embedding_model": "test",
                    "embedding_dim": 2,
                    "belief_type": "assertion",
                    "is_hypothetical": False,
                    "created_at": None,
                    "last_referenced_at": None,
                    "session_id": "session1",
                    "conversation_id": "",
                }
            ]
        )
        beliefs = await store.get_beliefs("session1")
        assert len(beliefs) == 1
        assert beliefs[0].subject == "USER"
        assert beliefs[0].value == "coffee"

        # clean up
        await store.close()


@pytest.mark.asyncio
async def test_decoupled_embeddings():
    mock_chat_adapter = MagicMock()
    mock_embed_adapter = MagicMock()
    mock_embed_adapter.get_embeddings = AsyncMock(return_value=[[0.5, 0.6]])
    mock_embed_adapter.embed_model = "custom-embed-model"
    mock_embed_adapter.embedding_dim = 256

    config = TrackerConfig(
        embed_provider=mock_embed_adapter,
        embed_model="custom-embed-model",
    )

    extractor = BeliefExtractor(adapter=mock_chat_adapter, config=config)
    assert extractor.adapter == mock_chat_adapter
    assert extractor.embedding_adapter == mock_embed_adapter
    assert extractor.embedding_model == "custom-embed-model"
    assert extractor.embedding_dim == 256


@pytest.mark.asyncio
async def test_async_stream_wrapper():
    # An async generator yielding chunks
    async def sample_stream():
        chunks = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content="I "))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="like "))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Python."))]),
        ]
        for c in chunks:
            yield c

    mock_tracker = MagicMock()
    mock_tracker.config.enable_background_tasks = False
    mock_tracker._auto_detect_adapter = False
    mock_tracker.app_adapter = MagicMock()

    # Mock to_llm_call and to_llm_response
    mock_tracker.app_adapter.to_llm_call.return_value = "mock_call"
    mock_tracker.app_adapter.to_llm_response.return_value = "mock_response"

    # We want to mock _track_background
    mock_tracker._track_background = AsyncMock()

    stream = sample_stream()
    wrapper = AsyncStreamWrapper(
        stream_gen=stream,
        tracker=mock_tracker,
        args=(),
        kwargs={},
        session_id="session123",
        turn=2,
    )

    # Consume the wrapper stream
    yielded_chunks = []
    async for chunk in wrapper:
        yielded_chunks.append(chunk)

    assert len(yielded_chunks) == 3
    assert wrapper.accumulated_text == "I like Python."

    # Verify tracking was dispatched when the stream completed
    mock_tracker._track_background.assert_called_once_with(
        "mock_call", "mock_response", "session123", 2
    )


@pytest.mark.asyncio
async def test_fail_fast_generic_adapter():
    config = TrackerConfig()
    tracker = BeliefTracker(config=config)

    # Manually configure internal_adapter to be GenericAdapter or mock it
    class GenericAdapter:
        pass

    mock_generic = GenericAdapter()

    tracker.internal_adapter = mock_generic
    tracker.app_adapter = mock_generic

    with pytest.raises(
        ValueError,
        match="Auto-detected LLM provider adapter does not support belief extraction",
    ):
        tracker._ensure_initialized()
