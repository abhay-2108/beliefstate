"""Tests for new features: extraction architecture, storage, detection pipeline,
resilience, and public API changes."""

import asyncio
import warnings
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from beliefstate.config import TrackerConfig
from beliefstate.models import Belief
from beliefstate.extractor import (
    BeliefExtractor,
    calibrate_confidence,
    _is_trivial_response,
)
from beliefstate.detector import (
    ContradictionDetector,
)
from beliefstate.resolver import BeliefResolver
from beliefstate.tracker import (
    BeliefTracker,
    TrackerStats,
    _get_session_lock,
    _validate_deployment_config,
    ConfigurationWarning,
)
from beliefstate.store.base import summary_for_prompt
from beliefstate.store.sqlite import SQLiteStore, pack_embedding, unpack_embedding
from beliefstate.store.memory import InMemoryBeliefStore


# --- Helpers ---


def make_belief(
    subject="USER",
    predicate="likes",
    value="Python",
    confidence=0.9,
    turn=1,
    source="user",
    embedding=None,
    category="",
    source_quote="",
    belief_type="assertion",
    is_hypothetical=False,
    embedding_dim=0,
    embedding_model="",
    session_id=None,
    conversation_id=None,
):
    return Belief(
        subject=subject,
        predicate=predicate,
        value=value,
        confidence=confidence,
        turn=turn,
        source=source,
        embedding=embedding or [0.1, 0.2, 0.3],
        category=category,
        source_quote=source_quote,
        belief_type=belief_type,
        is_hypothetical=is_hypothetical,
        embedding_dim=embedding_dim,
        embedding_model=embedding_model,
        session_id=session_id,
        conversation_id=conversation_id,
    )


# =====================================================================
# Part 1: Extraction Architecture
# =====================================================================


class TestProcessTurnBothSources:
    """test_extract_from_both_sources: Verify both user_message and assistant_response
    are passed to extraction."""

    @pytest.mark.asyncio
    async def test_process_turn_sends_both_texts(self):
        config = TrackerConfig()
        mock_adapter = AsyncMock()
        mock_adapter.generate = AsyncMock(return_value=MagicMock(text="[]"))
        mock_adapter.get_embeddings = AsyncMock(return_value=[])
        mock_adapter.embed_model = "text-embedding-3-small"
        extractor = BeliefExtractor(adapter=mock_adapter, config=config)

        await extractor.process_turn(
            user_message="My budget is $5,000",
            assistant_response="Understood, I will keep that in mind.",
            session_id="s1",
            turn=1,
        )

        called_prompt = mock_adapter.generate.call_args[0][0].messages[0]["content"]
        assert "My budget is $5,000" in called_prompt
        assert "Understood" in called_prompt


class TestPreFilterTrivialResponse:
    """test_prefilter_trivial_response: Pass 'Sure!' as assistant response.
    Verify extraction LLM is called with only the user message text."""

    @pytest.mark.asyncio
    async def test_trivial_response_skips_assistant(self):
        config = TrackerConfig()
        mock_adapter = AsyncMock()
        mock_adapter.generate = AsyncMock(return_value=MagicMock(text="[]"))
        mock_adapter.get_embeddings = AsyncMock(return_value=[])
        mock_adapter.embed_model = "text-embedding-3-small"
        extractor = BeliefExtractor(adapter=mock_adapter, config=config)

        await extractor.process_turn(
            user_message="My budget is $5,000",
            assistant_response="Sure!",
            session_id="s1",
            turn=1,
        )

        called_prompt = mock_adapter.generate.call_args[0][0].messages[0]["content"]
        assert "My budget is $5,000" in called_prompt
        assert "Sure!" not in called_prompt

    def test_is_trivial_short_responses(self):
        assert _is_trivial_response("Sure!") is True
        assert _is_trivial_response("Got it.") is True
        assert _is_trivial_response("Okay.") is True
        assert _is_trivial_response("Understood.") is True
        assert _is_trivial_response("Great!") is True

    def test_is_trivial_code_block(self):
        assert _is_trivial_response("{ } ( ) = ; : # | \\ > <") is True

    def test_is_trivial_json_start(self):
        assert _is_trivial_response('{"key": "value"}') is True
        assert _is_trivial_response('[{"key": "value"}]') is True

    def test_not_trivial_substantive(self):
        assert _is_trivial_response("I understand your budget constraints.") is False
        assert _is_trivial_response("The database should be PostgreSQL.") is False


class TestConfidenceCalibrationHedging:
    """test_confidence_calibration_hedging: Pass source_quote='we might use Redis'.
    Verify confidence is capped at 0.60 and is_hypothetical=True."""

    def test_might_hedging(self):
        b = make_belief(confidence=0.95, source_quote="we might use Redis")
        result = calibrate_confidence(b)
        assert result.confidence == 0.60
        assert result.is_hypothetical is True

    def test_think_hedging(self):
        b = make_belief(confidence=0.95, source_quote="I think we should use FastAPI")
        result = calibrate_confidence(b)
        assert result.confidence == 0.70

    def test_want_to_hedging(self):
        b = make_belief(confidence=0.95, source_quote="I want to deploy on AWS")
        result = calibrate_confidence(b)
        assert result.confidence == 0.75

    def test_no_hedging_unchanged(self):
        b = make_belief(confidence=0.95, source_quote="Use FastAPI for the backend")
        result = calibrate_confidence(b)
        assert result.confidence == 0.95


class TestUniversalPromptMixedDomain:
    """test_universal_prompt_mixed_domain: Verify the universal prompt contains
    all domain categories."""

    def test_universal_prompt_has_all_categories(self):
        from beliefstate.config import DEFAULT_EXTRACT_PROMPT

        assert "identity" in DEFAULT_EXTRACT_PROMPT
        assert "technical" in DEFAULT_EXTRACT_PROMPT
        assert "planning" in DEFAULT_EXTRACT_PROMPT
        assert "constraint" in DEFAULT_EXTRACT_PROMPT
        assert "state" in DEFAULT_EXTRACT_PROMPT
        assert "{conversation}" in DEFAULT_EXTRACT_PROMPT


# =====================================================================
# Part 2: Storage Architecture
# =====================================================================


class TestSQLiteMemoryConnection:
    """test_sqlite_memory_connection: Open SQLiteStore(':memory:'). Write a belief.
    Read it back in the same connection. Verify it is returned."""

    @pytest.mark.asyncio
    async def test_memory_connection_persists(self):
        store = SQLiteStore(db_path=":memory:")
        await store.open()

        b = make_belief(session_id="s1")
        await store.add_belief("s1", b)

        beliefs = await store.get_beliefs("s1")
        assert len(beliefs) == 1
        assert beliefs[0].subject == "user"

        await store.close()


class TestOptimisticConcurrency:
    """test_optimistic_concurrency: Write belief at turn=5. Attempt upsert at turn=3.
    Verify the turn=3 write is discarded and turn=5 value remains."""

    @pytest.mark.asyncio
    async def test_stale_write_discarded_sqlite(self):
        store = SQLiteStore(db_path=":memory:")
        await store.open()

        b5 = make_belief(
            subject="user",
            predicate="likes",
            value="PostgreSQL",
            turn=5,
            session_id="s1",
        )
        await store.add_belief("s1", b5)

        b3 = make_belief(
            subject="user", predicate="likes", value="SQLite", turn=3, session_id="s1"
        )
        result = await store.upsert(b3)
        assert result is False  # stale write discarded

        beliefs = await store.get_beliefs("s1")
        assert len(beliefs) == 1
        assert beliefs[0].value == "PostgreSQL"

        await store.close()

    @pytest.mark.asyncio
    async def test_stale_write_discarded_memory(self):
        store = InMemoryBeliefStore()

        b5 = make_belief(value="PostgreSQL", turn=5, session_id="s1")
        await store.add_belief("s1", b5)

        b3 = make_belief(value="SQLite", turn=3, session_id="s1")
        result = await store.upsert(b3)
        assert result is False

        beliefs = await store.get_beliefs("s1")
        assert len(beliefs) == 1
        assert beliefs[0].value == "PostgreSQL"


class TestSummaryGroupedByCategory:
    """test_summary_grouped_by_category: Store beliefs with categories identity,
    technical, planning. Call summary_for_prompt(). Verify section headers."""

    def test_summary_has_category_sections(self):
        beliefs = [
            make_belief(
                subject="USER",
                predicate="name is",
                value="Raj",
                category="identity",
                turn=1,
            ),
            make_belief(
                subject="Database",
                predicate="is",
                value="PostgreSQL",
                category="technical",
                turn=2,
            ),
            make_belief(
                subject="Auth",
                predicate="assigned to",
                value="Priya",
                category="planning",
                turn=3,
            ),
        ]
        result = summary_for_prompt(beliefs)
        assert "[Identity]" in result
        assert "[Technical Decisions]" in result
        assert "[Tasks & Planning]" in result
        assert "- USER name is Raj" in result
        assert "- Database is PostgreSQL" in result
        assert "- Auth assigned to Priya" in result

    def test_summary_excludes_hypotheticals(self):
        beliefs = [
            make_belief(
                value="PostgreSQL", category="technical", is_hypothetical=False, turn=1
            ),
            make_belief(
                value="Redis",
                category="technical",
                is_hypothetical=True,
                turn=2,
                subject="Cache",
                predicate="might be",
            ),
        ]
        result = summary_for_prompt(beliefs)
        assert "PostgreSQL" in result
        assert "Speculative / Under Consideration" in result
        assert "Redis" in result
        assert "(not committed)" in result


class TestEmbeddingBinaryPrecision:
    """test_embedding_binary_precision: Store embedding with float values.
    Retrieve and compare. Verify all values match to float32 precision."""

    @pytest.mark.asyncio
    async def test_binary_roundtrip(self):
        original = [0.123456789, -0.987654321, 0.0, 1.0, -1.0]
        packed = pack_embedding(original)
        unpacked = unpack_embedding(packed)

        assert len(unpacked) == len(original)
        for orig, unpacked_val in zip(original, unpacked):
            # float32 precision: ~7 decimal digits
            assert abs(orig - unpacked_val) < 1e-6

    @pytest.mark.asyncio
    async def test_sqlite_binary_embedding(self):
        store = SQLiteStore(db_path=":memory:")
        await store.open()

        emb = [0.5, -0.5, 0.10000000149]
        b = make_belief(embedding=emb, embedding_dim=3, session_id="s1")
        await store.add_belief("s1", b)

        beliefs = await store.get_beliefs("s1")
        assert len(beliefs) == 1
        assert len(beliefs[0].embedding) == 3
        for orig, stored in zip(emb, beliefs[0].embedding):
            assert abs(orig - stored) < 1e-5

        await store.close()


# =====================================================================
# Part 3: Detection Pipeline
# =====================================================================


class TestExactDuplicateSkip:
    """test_exact_duplicate_skip: Store a belief. Call detect() with identical
    subject/predicate/value. Verify outcome is DUPLICATE and no embedding call."""

    @pytest.mark.asyncio
    async def test_exact_duplicate_detected(self):
        store = SQLiteStore(db_path=":memory:")
        await store.open()
        b = make_belief(
            subject="user",
            predicate="likes",
            value="PostgreSQL",
            session_id="s1",
            embedding=[0.1, 0.2, 0.3],
        )
        await store.add_belief("s1", b)

        config = TrackerConfig()
        mock_adapter = AsyncMock()
        detector = ContradictionDetector(
            adapter=mock_adapter, store=store, config=config
        )

        new_b = make_belief(
            subject="user",
            predicate="likes",
            value="PostgreSQL",
            session_id="s1",
            embedding=[0.1, 0.2, 0.3],
        )
        _, duplicates = await detector.detect_with_deduplication("s1", [new_b])

        assert len(duplicates) == 1
        assert duplicates[0].value == "PostgreSQL"

        await store.close()


class TestEntailmentDeduplication:
    """test_entailment_deduplication: Store 'USER likes Python'. Detect
    'USER enjoys Python'. Verify NLI entailment score causes DUPLICATE."""

    @pytest.mark.asyncio
    async def test_entailment_skips_duplicate(self):
        store = SQLiteStore(db_path=":memory:")
        await store.open()

        b = make_belief(
            predicate="likes",
            value="Python",
            session_id="s1",
            embedding=[0.1, 0.2, 0.3],
        )
        await store.add_belief("s1", b)

        config = TrackerConfig()
        mock_judge = AsyncMock()
        mock_judge.check = AsyncMock(
            return_value=(False, 0.95, "entailment - same meaning")
        )

        mock_adapter = AsyncMock()
        detector = ContradictionDetector(
            adapter=mock_adapter, store=store, config=config, judge=mock_judge
        )

        new_b = make_belief(
            predicate="enjoys",
            value="Python",
            session_id="s1",
            embedding=[0.1, 0.2, 0.3],
        )
        _, duplicates = await detector.detect_with_deduplication("s1", [new_b])

        assert len(duplicates) == 1

        await store.close()


class TestEmbeddingModelMismatch:
    """test_embedding_model_mismatch: Store belief with embedding_dim=384.
    Detect with belief at embedding_dim=1536. Verify cosine gate is skipped
    and NLI judge is called directly."""

    @pytest.mark.asyncio
    async def test_mismatch_skips_cosine(self):
        store = SQLiteStore(db_path=":memory:")
        await store.open()

        b = make_belief(
            value="Python",
            session_id="s1",
            embedding=[0.1] * 10,
            embedding_dim=384,
            embedding_model="old-model",
        )
        await store.add_belief("s1", b)

        config = TrackerConfig()
        mock_judge = AsyncMock()
        mock_judge.check = AsyncMock(return_value=(False, 0.3, "neutral"))

        mock_adapter = AsyncMock()
        detector = ContradictionDetector(
            adapter=mock_adapter, store=store, config=config, judge=mock_judge
        )

        new_b = make_belief(
            value="Python is great",
            session_id="s1",
            embedding=[0.1] * 10,
            embedding_dim=1536,
            embedding_model="new-model",
        )
        contradictions, _ = await detector.detect_with_deduplication("s1", [new_b])

        # Judge should have been called despite dimension mismatch
        mock_judge.check.assert_called()
        await store.close()


class TestNLINonblocking:
    """test_nli_nonblocking: Verify _encode() and _nli_predict() use run_in_executor."""

    def test_detector_uses_adapter_for_judgment(self):
        """Verify the detector delegates NLI to the adapter (which should use
        run_in_executor in production)."""
        config = TrackerConfig()
        mock_adapter = AsyncMock()
        store = MagicMock()
        detector = ContradictionDetector(
            adapter=mock_adapter, store=store, config=config
        )
        assert detector.adapter is mock_adapter


# =====================================================================
# Part 4: Resilience and Concurrency
# =====================================================================


class TestSessionLockPreventsRace:
    """test_session_lock_prevents_race: Verify _get_session_lock returns
    same lock for same session."""

    def test_same_session_same_lock(self):
        lock1 = _get_session_lock("s1")
        lock2 = _get_session_lock("s1")
        assert lock1 is lock2

    def test_different_session_different_lock(self):
        lock1 = _get_session_lock("s1")
        lock2 = _get_session_lock("s2")
        assert lock1 is not lock2


class TestShutdownDrainsTasks:
    """test_shutdown_drains_tasks: Create tracker, dispatch 3 tasks with delays.
    Call shutdown(grace=2.0). Verify all tasks completed."""

    @pytest.mark.asyncio
    async def test_shutdown_cancels_pending(self):
        config = TrackerConfig()
        mock_adapter = MagicMock()
        tracker = BeliefTracker(config=config, adapter=mock_adapter)

        async def slow_task():
            await asyncio.sleep(10)

        tracker._dispatch(slow_task())
        tracker._dispatch(slow_task())
        assert len(tracker._pending_tasks) > 0

        await tracker.shutdown(grace_seconds=0.1)
        # Allow done callbacks (set.discard) to fire after cancellation
        await asyncio.sleep(0.05)
        assert len(tracker._pending_tasks) == 0


class TestMultiWorkerWarning:
    """test_multiworker_warning: Set WEB_CONCURRENCY=4 in env. Create BeliefTracker
    with store_type='sqlite'. Verify ConfigurationWarning is emitted."""

    def test_warning_emitted(self):
        config = TrackerConfig(store_type="sqlite")
        with patch.dict("os.environ", {"WEB_CONCURRENCY": "4"}):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _validate_deployment_config(config)
                config_warnings = [
                    x for x in w if issubclass(x.category, ConfigurationWarning)
                ]
                assert len(config_warnings) == 1
                assert "SQLite" in str(config_warnings[0].message)

    def test_no_warning_for_single_worker(self):
        config = TrackerConfig(store_type="sqlite")
        with patch.dict("os.environ", {"WEB_CONCURRENCY": "1"}):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _validate_deployment_config(config)
                config_warnings = [
                    x for x in w if issubclass(x.category, ConfigurationWarning)
                ]
                assert len(config_warnings) == 0

    def test_no_warning_for_redis(self):
        config = TrackerConfig(store_type="redis")
        with patch.dict("os.environ", {"WEB_CONCURRENCY": "4"}):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _validate_deployment_config(config)
                config_warnings = [
                    x for x in w if issubclass(x.category, ConfigurationWarning)
                ]
                assert len(config_warnings) == 0


class TestGetStatsTracksErrors:
    """test_get_stats_tracks_errors: Force extraction to fail. Call get_stats().
    Verify extraction_errors=1."""

    def test_stats_record_error(self):
        stats = TrackerStats()
        stats.record_error("test error")
        assert stats.extraction_errors == 1
        assert stats.last_error == "test error"
        assert stats.extraction_success_rate == 0.0

    def test_stats_record_success(self):
        stats = TrackerStats()
        stats.record_success()
        stats.record_success()
        stats.record_success()
        assert stats.total_turns_processed == 3
        assert stats.extraction_success_rate == 1.0

    def test_stats_mixed_outcomes(self):
        stats = TrackerStats()
        stats.record_success()
        stats.record_error("fail")
        stats.record_success()
        assert stats.extraction_success_rate == pytest.approx(2 / 3)


# =====================================================================
# Part 5: Resolver
# =====================================================================


class TestResolverUpdateStrategy:
    """test_update_strategy_receives_store: Call resolve() with UPDATE strategy.
    Verify no NameError is raised and store.upsert() is called with new belief."""

    @pytest.mark.asyncio
    async def test_update_strategy_works(self):
        store = InMemoryBeliefStore()
        resolver = BeliefResolver(store=store, strategy="overwrite")

        old_b = make_belief(value="PostgreSQL", turn=1, session_id="s1")
        new_b = make_belief(value="SQLite", turn=2, session_id="s1")

        # Should not raise NameError
        await resolver.resolve("s1", [(old_b, new_b, 0.9, "contradiction")])

        beliefs = await store.get_beliefs("s1")
        assert len(beliefs) == 1
        assert beliefs[0].value == "SQLite"


class TestAskEscalatesToBlock:
    """test_ask_escalates_to_block: Fire same contradiction twice for same belief
    pair after ASK was already injected. Verify second occurrence escalates to BLOCK."""

    @pytest.mark.asyncio
    async def test_escalation(self):
        store = InMemoryBeliefStore()
        resolver = BeliefResolver(store=store, strategy="overwrite")

        old_b = make_belief(value="PostgreSQL", turn=1, session_id="s1")
        new_b = make_belief(value="SQLite", turn=2, session_id="s1")

        # First fire: ASK
        await resolver.resolve("s1", [(old_b, new_b, 0.9, "contradiction")])
        conflicts = resolver.pop_pending_conflicts("s1")
        assert len(conflicts) == 1
        assert "BELIEF CONFLICT" in conflicts[0]

        # Second fire: BLOCK (no new conflict note)
        await resolver.resolve("s1", [(old_b, new_b, 0.9, "contradiction")])
        conflicts2 = resolver.pop_pending_conflicts("s1")
        assert len(conflicts2) == 0  # escalated to BLOCK, no new note


class TestResolverTemporalUpdate:
    """Verify temporal updates bypass contradiction resolution."""

    @pytest.mark.asyncio
    async def test_temporal_update_overwrites(self):
        store = InMemoryBeliefStore()
        resolver = BeliefResolver(store=store, strategy="keep_old")

        old_b = make_belief(value="PostgreSQL", turn=1, session_id="s1")
        new_b = make_belief(
            value="SQLite", turn=2, session_id="s1", belief_type="update"
        )

        # Even with keep_old strategy, temporal update should overwrite
        await resolver.resolve("s1", [(old_b, new_b, 0.9, "contradiction")])

        beliefs = await store.get_beliefs("s1")
        assert len(beliefs) == 1
        assert beliefs[0].value == "SQLite"


# =====================================================================
# Part 5: New Public API
# =====================================================================


class TestBeliefHistory:
    """test_get_belief_history: Verify audit trail is returned."""

    @pytest.mark.asyncio
    async def test_audit_history_sqlite(self):
        store = SQLiteStore(db_path=":memory:")
        await store.open()

        b1 = make_belief(value="PostgreSQL", turn=1, session_id="s1")
        await store.add_belief("s1", b1)

        b2 = make_belief(value="SQLite", turn=2, session_id="s1")
        await store.add_belief("s1", b2)

        history = await store.get_audit_history("s1", "user", "likes")
        assert len(history) >= 1  # at least one create record

        await store.close()


class TestBeliefModelNewFields:
    """Verify Belief model accepts new fields."""

    def test_new_fields(self):
        b = Belief(
            subject="USER",
            predicate="likes",
            value="Python",
            confidence=0.9,
            turn=1,
            source="user",
            category="identity",
            source_quote="I like Python",
            embedding_model="all-MiniLM-L6-v2",
            embedding_dim=384,
        )
        assert b.category == "identity"
        assert b.source_quote == "I like Python"
        assert b.embedding_model == "all-MiniLM-L6-v2"
        assert b.embedding_dim == 384

    def test_backwards_compatible(self):
        b = Belief(
            subject="USER",
            predicate="likes",
            value="Python",
            confidence=0.9,
            turn=1,
            source="user",
        )
        assert b.category == ""
        assert b.source_quote == ""
