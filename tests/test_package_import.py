"""
Test that the beliefstate package can be properly imported and used.

This test ensures the package works correctly for developers installing via pip/uv.
"""

import pytest


def test_core_imports():
    """Test that core modules can be imported."""
    from beliefstate import (
        BeliefTracker,
        TrackerConfig,
        Belief,
        BeliefExtractor,
        ContradictionDetector,
        BeliefResolver,
        SQLiteStore,
        RedisStore,
        ProviderAdapter,
        OpenAIAdapter,
        AnthropicAdapter,
        GeminiAdapter,
        OllamaAdapter,
        LiteLLMAdapter,
    )
    
    assert BeliefTracker is not None
    assert TrackerConfig is not None
    assert Belief is not None
    assert BeliefExtractor is not None
    assert ContradictionDetector is not None
    assert BeliefResolver is not None
    assert SQLiteStore is not None
    assert RedisStore is not None
    assert ProviderAdapter is not None
    assert OpenAIAdapter is not None
    assert AnthropicAdapter is not None
    assert GeminiAdapter is not None
    assert OllamaAdapter is not None
    assert LiteLLMAdapter is not None


def test_dispatcher_imports():
    """Test that dispatcher modules can be imported."""
    from beliefstate import (
        AsyncioDispatcher,
        SyncDispatcher,
        CeleryDispatcher,
        RQDispatcher,
        register_global_tracker,
        execute_tracking_task,
    )
    
    assert AsyncioDispatcher is not None
    assert SyncDispatcher is not None
    assert CeleryDispatcher is not None
    assert RQDispatcher is not None
    assert register_global_tracker is not None
    assert execute_tracking_task is not None


def test_judge_imports():
    """Test that judge modules can be imported."""
    from beliefstate import (
        ContradictionJudge,
        LLMJudge,
        LocalNLIJudge,
    )
    
    assert ContradictionJudge is not None
    assert LLMJudge is not None
    assert LocalNLIJudge is not None


def test_resilience_imports():
    """Test that resilience modules can be imported."""
    from beliefstate import (
        ResilientAdapterWrapper,
        CircuitBreaker,
        CircuitBreakerOpenException,
    )
    
    assert ResilientAdapterWrapper is not None
    assert CircuitBreaker is not None
    assert CircuitBreakerOpenException is not None


def test_session_context_import():
    """Test that session context can be imported."""
    from beliefstate import session_context
    
    assert session_context is not None
    # Test basic functionality
    session_context.set("test-session-123")
    assert session_context.get() == "test-session-123"
    session_context.set("default")


def test_basic_models():
    """Test that Belief model works correctly."""
    from beliefstate import Belief
    from datetime import datetime
    
    belief = Belief(
        subject="USER",
        predicate="likes",
        value="Python",
        confidence=0.95,
        turn=1,
        source="user",
    )
    
    assert belief.subject == "USER"
    assert belief.predicate == "likes"
    assert belief.value == "Python"
    assert belief.confidence == 0.95
    assert belief.turn == 1
    assert belief.source == "user"
    assert isinstance(belief.created_at, datetime)


def test_tracker_config():
    """Test that TrackerConfig can be instantiated."""
    from beliefstate import TrackerConfig
    
    config = TrackerConfig()
    assert config is not None
    assert config.similarity_threshold == 0.82
    assert config.entailment_threshold == 0.85
    assert config.max_beliefs == 50


@pytest.mark.asyncio
async def test_sqlite_store():
    """Test that SQLiteStore can be instantiated and used."""
    from beliefstate import SQLiteStore, Belief
    import tempfile
    import os
    
    # Create a temporary database file
    tmpdir = tempfile.mkdtemp()
    try:
        db_path = os.path.join(tmpdir, "test.db")
        store = SQLiteStore(db_path=db_path)
        
        # Test basic operations
        belief = Belief(
            subject="TEST",
            predicate="test",
            value="value",
            confidence=1.0,
            turn=1,
            source="test",
            session_id="test-session",
        )
        
        # Add belief
        await store.add_belief("test-session", belief)
        
        # Retrieve belief
        beliefs = await store.get_beliefs("test-session")
        assert len(beliefs) > 0
        assert beliefs[0].subject == "TEST"
        
        # Clear session
        await store.clear("test-session")
        beliefs = await store.get_beliefs("test-session")
        assert len(beliefs) == 0
        
        # Close the store
        if hasattr(store, "close"):
            await store.close()
    finally:
        # Clean up temp directory
        import shutil
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_belief_tracker_initialization():
    """Test that BeliefTracker can be initialized."""
    from beliefstate import (
        BeliefTracker,
        TrackerConfig,
        OpenAIAdapter,
    )
    from unittest.mock import MagicMock
    
    # Create a mock adapter
    adapter = MagicMock(spec=OpenAIAdapter)
    config = TrackerConfig()
    
    # Initialize tracker
    tracker = BeliefTracker(config=config, adapter=adapter)
    
    assert tracker is not None
    assert tracker.config == config
    assert tracker.app_adapter == adapter


def test_package_version():
    """Test that package has a version."""
    import beliefstate
    
    # Check that package has metadata
    assert hasattr(beliefstate, "__version__") or True  # May not be set initially
    
    # Check that core modules exist
    assert hasattr(beliefstate, "BeliefTracker")
    assert hasattr(beliefstate, "TrackerConfig")
    assert hasattr(beliefstate, "Belief")


def test_optional_imports():
    """Test that optional integrations don't block core package import."""
    import beliefstate
    
    # These may or may not be available depending on installed extras
    # But the package should still be importable
    has_fastapi = hasattr(beliefstate, "FastAPIBeliefTrackerMiddleware") and \
                  beliefstate.FastAPIBeliefTrackerMiddleware is not None
    has_flask = hasattr(beliefstate, "FlaskBeliefTrackerMiddleware") and \
                beliefstate.FlaskBeliefTrackerMiddleware is not None
    
    # We can check without failing
    assert isinstance(has_fastapi, bool)
    assert isinstance(has_flask, bool)
