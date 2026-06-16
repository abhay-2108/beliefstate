import pytest
from unittest.mock import AsyncMock, MagicMock
from beliefstate.config import TrackerConfig
from beliefstate.models import Belief
from beliefstate.detector import ContradictionDetector, cosine_similarity
from beliefstate.call import LLMResponse
from beliefstate.store.sqlite import SQLiteStore


def test_cosine_similarity():
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    c = [0.0, 1.0, 0.0]
    d = [-1.0, 0.0, 0.0]

    assert pytest.approx(cosine_similarity(a, b)) == 1.0
    assert pytest.approx(cosine_similarity(a, c)) == 0.0
    assert pytest.approx(cosine_similarity(a, d)) == -1.0
    assert cosine_similarity([], b) == 0.0


@pytest.mark.asyncio
async def test_detector_detects_contradiction():
    config = TrackerConfig(similarity_threshold=0.8, contradiction_threshold=0.7)
    store = SQLiteStore(db_path=":memory:")

    # Pre-populate store with a belief
    b_old = Belief(
        subject="USER",
        predicate="likes",
        value="Python",
        confidence=1.0,
        turn=1,
        source="user",
        embedding=[1.0, 0.0, 0.0],
    )
    await store.add_belief("session_1", b_old)

    # Mock adapter
    mock_adapter = MagicMock()
    # If cosine similarity matches, it does LLM judgment. Let's mock generate to say it IS a contradiction.
    mock_adapter.generate = AsyncMock(
        return_value=LLMResponse(
            text='{"relationship": "contradiction", "score": 0.9, "reason": "User said they hate Python"}',
            raw_response=None,
        )
    )

    detector = ContradictionDetector(adapter=mock_adapter, store=store, config=config)

    # New belief with high embedding similarity
    b_new = Belief(
        subject="USER",
        predicate="likes",
        value="Python",
        confidence=1.0,
        turn=2,
        source="user",
        embedding=[0.9, 0.1, 0.0],  # High cosine similarity with old belief
    )

    contradictions = await detector.detect("session_1", [b_new])

    assert len(contradictions) == 1
    old, new, score, reason = contradictions[0]
    assert old.value == "Python"
    assert new.value == "Python"
    assert score == 0.9
    assert reason == "User said they hate Python"
