"""Tests for production hardening features.

Covers:
1. Per-session turn counters (no cross-session collision)
2. Async context manager on tracker
3. belief_count() on SQLite and InMemory stores
4. export_beliefs() / import_beliefs() round-trip
5. inject_context() correctly passes format_template as keyword
6. Session lock cleanup in clear_session()
7. health_check() on tracker and stores
8. Structured logging event format
"""

import asyncio
import pytest
from unittest.mock import MagicMock

from beliefstate.config import TrackerConfig
from beliefstate.models import Belief
from beliefstate.tracker import (
    BeliefTracker,
    session_context,
    _session_locks,
)
from beliefstate.store.sqlite import SQLiteStore
from beliefstate.store.memory import InMemoryBeliefStore
from beliefstate.logging_utils import TrackerEvent, log_event


def make_config(**kwargs):
    store_kwargs = kwargs.pop("store_kwargs", {"db_path": ":memory:"})
    return TrackerConfig(
        enable_background_tasks=False, store_kwargs=store_kwargs, **kwargs
    )


def make_belief(subject="USER", predicate="likes", value="Python", turn=1, **kwargs):
    return Belief(
        subject=subject,
        predicate=predicate,
        value=value,
        confidence=kwargs.get("confidence", 1.0),
        turn=turn,
        source=kwargs.get("source", "user"),
        **{k: v for k, v in kwargs.items() if k not in ("confidence", "source")},
    )


# ── 1. Per-session turn counters ──────────────────────────────────────────


class TestPerSessionTurnCounters:
    def test_initial_turn_counter_is_zero(self):
        config = make_config()
        mock_adapter = MagicMock()
        tracker = BeliefTracker(config=config, adapter=mock_adapter)
        assert tracker.turn_counter == 0

    def test_turn_counter_property_backward_compat(self):
        config = make_config()
        mock_adapter = MagicMock()
        tracker = BeliefTracker(config=config, adapter=mock_adapter)
        # Setter should work
        tracker.turn_counter = 5
        assert tracker.turn_counter == 5

    def test_per_session_turn_isolation(self):
        config = make_config()
        mock_adapter = MagicMock()
        tracker = BeliefTracker(config=config, adapter=mock_adapter)
        # Simulate turns for different sessions
        tracker._session_turn_counters["session_a"] = 3
        tracker._session_turn_counters["session_b"] = 7
        # turn_counter now returns current session's value (not max)
        assert tracker.turn_counter == 0  # default session has 0 turns
        # get_session_turn should return per-session value
        assert tracker.get_session_turn("session_a") == 3
        assert tracker.get_session_turn("session_b") == 7
        assert tracker.get_session_turn("session_c") == 0  # non-existent

    def test_get_session_turn_default_context(self):
        config = make_config()
        mock_adapter = MagicMock()
        tracker = BeliefTracker(config=config, adapter=mock_adapter)
        token = session_context.set("test_session")
        tracker._session_turn_counters["test_session"] = 42
        try:
            assert tracker.get_session_turn() == 42
        finally:
            session_context.reset(token)


# ── 2. Async context manager ─────────────────────────────────────────────


class TestAsyncContextManager:
    @pytest.mark.asyncio
    async def test_context_manager_opens_and_closes_sqlite(self):
        config = make_config()
        mock_adapter = MagicMock()
        tracker = BeliefTracker(config=config, adapter=mock_adapter)

        async with tracker:
            # Store should be usable
            await tracker.store.add_belief("s1", make_belief())
            beliefs = await tracker.get_beliefs("s1")
            assert len(beliefs) == 1

    @pytest.mark.asyncio
    async def test_context_manager_with_memory_store(self):
        config = make_config()
        mock_adapter = MagicMock()
        store = InMemoryBeliefStore()
        tracker = BeliefTracker(config=config, adapter=mock_adapter, store=store)

        async with tracker:
            await store.add_belief("s1", make_belief())
            beliefs = await tracker.get_beliefs("s1")
            assert len(beliefs) == 1


# ── 3. belief_count() ────────────────────────────────────────────────────


class TestBeliefCount:
    @pytest.mark.asyncio
    async def test_sqlite_belief_count_empty(self):
        store = SQLiteStore(db_path=":memory:")
        await store.open()
        try:
            count = await store.belief_count("empty_session")
            assert count == 0
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_sqlite_belief_count_with_beliefs(self):
        store = SQLiteStore(db_path=":memory:")
        await store.open()
        try:
            for i in range(5):
                await store.add_belief(
                    "s1", make_belief(predicate=f"fact_{i}", value=f"v{i}")
                )
            count = await store.belief_count("s1")
            assert count == 5
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_sqlite_belief_count_different_sessions(self):
        store = SQLiteStore(db_path=":memory:")
        await store.open()
        try:
            for i in range(3):
                await store.add_belief(
                    "s1", make_belief(predicate=f"f{i}", value=f"v{i}")
                )
            for i in range(7):
                await store.add_belief(
                    "s2", make_belief(predicate=f"g{i}", value=f"v{i}")
                )
            assert await store.belief_count("s1") == 3
            assert await store.belief_count("s2") == 7
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_memory_belief_count(self):
        store = InMemoryBeliefStore()
        assert await store.belief_count("s1") == 0
        await store.add_belief("s1", make_belief())
        assert await store.belief_count("s1") == 1
        await store.add_belief("s1", make_belief(predicate="hates", value="Java"))
        assert await store.belief_count("s1") == 2


# ── 4. export/import beliefs ─────────────────────────────────────────────


class TestExportImportBeliefs:
    @pytest.mark.asyncio
    async def test_export_beliefs_returns_dicts(self):
        config = make_config()
        mock_adapter = MagicMock()
        tracker = BeliefTracker(config=config, adapter=mock_adapter)

        await tracker.store.add_belief("s1", make_belief())
        await tracker.store.add_belief(
            "s1", make_belief(predicate="lives in", value="Tokyo")
        )

        exported = await tracker.export_beliefs("s1")
        assert len(exported) == 2
        assert all(isinstance(d, dict) for d in exported)
        # Should have all expected keys
        assert "subject" in exported[0]
        assert "predicate" in exported[0]
        assert "value" in exported[0]

    @pytest.mark.asyncio
    async def test_export_empty_session(self):
        config = make_config()
        mock_adapter = MagicMock()
        tracker = BeliefTracker(config=config, adapter=mock_adapter)
        exported = await tracker.export_beliefs("nonexistent")
        assert exported == []

    @pytest.mark.asyncio
    async def test_import_beliefs_round_trip(self):
        config = make_config()
        mock_adapter = MagicMock()
        tracker = BeliefTracker(config=config, adapter=mock_adapter)

        # Add beliefs to session 1
        await tracker.store.add_belief("s1", make_belief())
        await tracker.store.add_belief(
            "s1", make_belief(predicate="lives in", value="Tokyo")
        )

        # Export from session 1
        exported = await tracker.export_beliefs("s1")
        assert len(exported) == 2

        # Import into session 2
        count = await tracker.import_beliefs("s2", exported)
        assert count == 2

        # Verify session 2 has the same beliefs
        beliefs = await tracker.get_beliefs("s2")
        assert len(beliefs) == 2
        values = {b.value for b in beliefs}
        assert "Python" in values
        assert "Tokyo" in values

    @pytest.mark.asyncio
    async def test_import_skips_invalid_entries(self):
        config = make_config()
        mock_adapter = MagicMock()
        tracker = BeliefTracker(config=config, adapter=mock_adapter)

        data = [
            {
                "subject": "USER",
                "predicate": "likes",
                "value": "Python",
                "confidence": 1.0,
                "turn": 1,
                "source": "user",
            },
            {"invalid": "missing required fields"},
        ]
        count = await tracker.import_beliefs("s1", data)
        # Only valid entry should be imported
        assert count == 1

    @pytest.mark.asyncio
    async def test_import_empty_data(self):
        config = make_config()
        mock_adapter = MagicMock()
        tracker = BeliefTracker(config=config, adapter=mock_adapter)
        count = await tracker.import_beliefs("s1", [])
        assert count == 0
        count = await tracker.import_beliefs("s1", None)
        assert count == 0


# ── 5. inject_context fix ────────────────────────────────────────────────


class TestInjectContextFix:
    @pytest.mark.asyncio
    async def test_inject_context_with_format_template(self):
        """Verify format_template is NOT passed as conversation_id."""
        config = make_config(enable_staleness_scoring=False)
        mock_adapter = MagicMock()
        tracker = BeliefTracker(config=config, adapter=mock_adapter)

        await tracker.store.add_belief("s1", make_belief())

        messages = [{"role": "user", "content": "hello"}]
        result = await tracker.inject_context(
            messages,
            session_id="s1",
            format_template="User context:\n{facts}",
        )

        # Should have injected a system message
        assert len(result) == 2
        system_msg = result[0]
        assert system_msg["role"] == "system"
        assert "User context:" in system_msg["content"]
        assert "user likes Python" in system_msg["content"]

    @pytest.mark.asyncio
    async def test_inject_context_appends_to_existing_system(self):
        config = make_config(enable_staleness_scoring=False)
        mock_adapter = MagicMock()
        tracker = BeliefTracker(config=config, adapter=mock_adapter)

        await tracker.store.add_belief("s1", make_belief())

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello"},
        ]
        result = await tracker.inject_context(messages, session_id="s1")

        assert len(result) == 2
        assert "You are a helpful assistant." in result[0]["content"]
        assert "user likes Python" in result[0]["content"]

    @pytest.mark.asyncio
    async def test_inject_context_empty_session(self):
        config = make_config()
        mock_adapter = MagicMock()
        tracker = BeliefTracker(config=config, adapter=mock_adapter)

        messages = [{"role": "user", "content": "hello"}]
        result = await tracker.inject_context(messages, session_id="empty")
        # No beliefs = no injection
        assert result == messages


# ── 6. Session lock cleanup ──────────────────────────────────────────────


class TestSessionCleanup:
    @pytest.mark.asyncio
    async def test_clear_session_cleans_up_state(self):
        config = make_config()
        mock_adapter = MagicMock()
        tracker = BeliefTracker(config=config, adapter=mock_adapter)

        sid = "cleanup_test"

        # Populate session state
        await tracker.store.add_belief(sid, make_belief())
        tracker._session_turn_counters[sid] = 5
        tracker._session_turn_states[sid] = 4
        tracker._session_providers[sid] = "TestAdapter"
        _session_locks[sid] = asyncio.Lock()

        # Clear session
        receipt = await tracker.clear_session(sid)

        # Verify all state is cleaned up
        assert receipt.beliefs_deleted == 1
        assert sid not in tracker._session_turn_counters
        assert sid not in tracker._session_turn_states
        assert sid not in tracker._session_providers
        assert sid not in _session_locks

    @pytest.mark.asyncio
    async def test_clear_session_no_state_doesnt_crash(self):
        config = make_config()
        mock_adapter = MagicMock()
        tracker = BeliefTracker(config=config, adapter=mock_adapter)

        # Clear a session that has no state at all
        receipt = await tracker.clear_session("nonexistent_session")
        assert receipt.beliefs_deleted == 0


# ── 7. health_check ──────────────────────────────────────────────────────


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_sqlite_store_health_check(self):
        store = SQLiteStore(db_path=":memory:")
        await store.open()
        try:
            assert await store.health_check() is True
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_memory_store_health_check(self):
        store = InMemoryBeliefStore()
        assert await store.health_check() is True

    @pytest.mark.asyncio
    async def test_tracker_health_check_store_only(self):
        config = make_config()
        mock_adapter = MagicMock()
        tracker = BeliefTracker(config=config, adapter=mock_adapter)

        health = await tracker.health_check()
        # Store (SQLite in-memory) should be healthy
        assert health["store"] is True
        # Adapter is a Mock — health_check will return Mock, which is truthy
        # but the actual bool depends on the mock

    @pytest.mark.asyncio
    async def test_tracker_health_check_no_adapter(self):
        config = make_config()
        tracker = BeliefTracker(config=config)

        health = await tracker.health_check()
        assert health["store"] is True
        # No adapter set, should be False
        assert health["adapter"] is False


# ── 8. Structured logging ────────────────────────────────────────────────


class TestStructuredLogging:
    def test_tracker_event_to_dict(self):
        event = TrackerEvent(
            session_id="user-123",
            operation="extract_beliefs",
            turn=5,
            detail="Extracted 3 beliefs",
            latency_ms=142.5,
        )
        d = event.to_dict()
        assert d["session_id"] == "user-123"
        assert d["operation"] == "extract_beliefs"
        assert d["turn"] == 5
        assert d["latency_ms"] == 142.5

    def test_tracker_event_to_dict_omits_empty(self):
        event = TrackerEvent(operation="test")
        d = event.to_dict()
        # session_id is empty string, should be omitted
        assert "session_id" not in d
        # latency_ms is None, should be omitted
        assert "latency_ms" not in d
        # extra is empty dict, should be omitted
        assert "extra" not in d

    def test_tracker_event_to_json(self):
        event = TrackerEvent(
            session_id="s1",
            operation="detect",
            turn=1,
            detail="Found contradiction",
        )
        json_str = event.to_json()
        assert '"session_id": "s1"' in json_str or '"session_id":"s1"' in json_str

    def test_log_event_does_not_raise(self):
        """log_event should never raise, even without handlers."""
        event = TrackerEvent(
            session_id="s1",
            operation="test",
            turn=1,
            detail="Test event",
            latency_ms=10.0,
        )
        # Should not raise
        log_event(event)

    def test_tracker_event_with_extra(self):
        event = TrackerEvent(
            session_id="s1",
            operation="test",
            turn=1,
            detail="test",
            extra={"beliefs_count": 5, "model": "gpt-4"},
        )
        d = event.to_dict()
        assert d["extra"]["beliefs_count"] == 5
        assert d["extra"]["model"] == "gpt-4"
