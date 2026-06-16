import sys
import pytest
from unittest.mock import AsyncMock, MagicMock
from beliefstate.models import Belief
from beliefstate.config import TrackerConfig
from beliefstate.call import LLMResponse
from beliefstate.judge import LLMJudge, LocalNLIJudge


@pytest.mark.asyncio
async def test_llm_judge_contradiction():
    config = TrackerConfig()
    mock_adapter = MagicMock()

    # Mock contradiction result
    mock_adapter.generate = AsyncMock(
        return_value=LLMResponse(
            text='{"relationship": "contradiction", "score": 0.92, "reason": "different locations"}',
            raw_response=None,
        )
    )

    judge = LLMJudge(adapter=mock_adapter, config=config)

    b1 = Belief(
        subject="USER",
        predicate="lives in",
        value="Paris",
        confidence=1.0,
        turn=1,
        source="user",
    )
    b2 = Belief(
        subject="USER",
        predicate="lives in",
        value="Tokyo",
        confidence=1.0,
        turn=2,
        source="user",
    )

    is_contra, score, reason = await judge.check(b1, b2)
    assert is_contra is True
    assert score == 0.92
    assert "different locations" in reason

    mock_adapter.generate.assert_called_once()


@pytest.mark.asyncio
async def test_llm_judge_neutral():
    config = TrackerConfig()
    mock_adapter = MagicMock()

    # Mock neutral result
    mock_adapter.generate = AsyncMock(
        return_value=LLMResponse(
            text='{"relationship": "neutral", "score": 0.1, "reason": "unrelated"}',
            raw_response=None,
        )
    )

    judge = LLMJudge(adapter=mock_adapter, config=config)

    b1 = Belief(
        subject="USER",
        predicate="lives in",
        value="Paris",
        confidence=1.0,
        turn=1,
        source="user",
    )
    b2 = Belief(
        subject="USER",
        predicate="likes",
        value="coffee",
        confidence=1.0,
        turn=2,
        source="user",
    )

    is_contra, score, reason = await judge.check(b1, b2)
    assert is_contra is False
    assert score == 0.1


@pytest.mark.asyncio
async def test_local_nli_judge_mock():
    # Mock the pipeline return values
    mock_pipeline_fn = MagicMock()
    mock_pipeline_fn.return_value = [{"label": "contradiction", "score": 0.88}]

    # Mock transformers module using sys.modules
    mock_transformers = MagicMock()
    mock_transformers.pipeline.return_value = mock_pipeline_fn
    sys.modules["transformers"] = mock_transformers

    try:
        judge = LocalNLIJudge(threshold=0.8)

        # Test initialization triggers pipeline creation
        judge._init_pipeline()
        mock_transformers.pipeline.assert_called_once_with(
            "text-classification", model="cross-encoder/nli-deberta-v3-xsmall"
        )

        b1 = Belief(
            subject="USER",
            predicate="lives in",
            value="Paris",
            confidence=1.0,
            turn=1,
            source="user",
        )
        b2 = Belief(
            subject="USER",
            predicate="lives in",
            value="Tokyo",
            confidence=1.0,
            turn=2,
            source="user",
        )

        # Run check
        is_contra, score, reason = await judge.check(b1, b2)
        assert is_contra is True
        assert score == 0.88
        assert "contradiction" in reason

        # Test below threshold
        mock_pipeline_fn.return_value = [{"label": "contradiction", "score": 0.65}]
        is_contra_low, score_low, _ = await judge.check(b1, b2)
        assert is_contra_low is False
        assert score_low == 0.65

        # Test neutral label
        mock_pipeline_fn.return_value = [{"label": "neutral", "score": 0.95}]
        is_contra_neut, score_neut, _ = await judge.check(b1, b2)
        assert is_contra_neut is False
        assert score_neut == 0.95
    finally:
        # Clean up sys.modules
        if "transformers" in sys.modules:
            del sys.modules["transformers"]
