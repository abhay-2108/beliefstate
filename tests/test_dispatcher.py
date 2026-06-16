import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from beliefstate import BeliefTracker, TrackerConfig
from beliefstate.call import LLMCall, LLMResponse
from beliefstate.dispatcher import (
    AsyncioDispatcher,
    SyncDispatcher,
    CeleryDispatcher,
    RQDispatcher,
    register_global_tracker,
    get_global_tracker,
    execute_tracking_task,
)


@pytest.mark.asyncio
async def test_dispatcher_injection_vs_config_fallback():
    # 1. Test Injection
    mock_dispatcher = MagicMock()
    config = TrackerConfig()
    mock_adapter = MagicMock()

    tracker = BeliefTracker(
        config=config, adapter=mock_adapter, dispatcher=mock_dispatcher
    )
    assert tracker.dispatcher == mock_dispatcher

    # 2. Test Declarative Config Fallback
    config_sync = TrackerConfig(task_dispatcher_type="sync")
    tracker_sync = BeliefTracker(config=config_sync, adapter=mock_adapter)
    assert isinstance(tracker_sync.dispatcher, SyncDispatcher)

    config_async = TrackerConfig(task_dispatcher_type="asyncio")
    tracker_async = BeliefTracker(config=config_async, adapter=mock_adapter)
    assert isinstance(tracker_async.dispatcher, AsyncioDispatcher)


@pytest.mark.asyncio
async def test_asyncio_dispatcher():
    dispatcher = AsyncioDispatcher()

    mock_tracker = MagicMock()
    mock_tracker.track_async = AsyncMock()

    call = LLMCall(messages=[{"role": "user", "content": "test"}])
    response = LLMResponse(text="response text", raw_response=None)

    dispatcher.dispatch(mock_tracker, call, response, "session_123", 4)

    # Yield control to let asyncio loop run the task
    await asyncio.sleep(0.01)

    mock_tracker.track_async.assert_called_once_with(
        call.model_dump(), response.model_dump(), "session_123", 4
    )


@pytest.mark.asyncio
async def test_sync_dispatcher():
    dispatcher = SyncDispatcher()

    mock_tracker = MagicMock()
    mock_tracker.track_async = AsyncMock()

    call = LLMCall(messages=[{"role": "user", "content": "test"}])
    response = LLMResponse(text="response text", raw_response=None)

    dispatcher.dispatch(mock_tracker, call, response, "session_123", 4)

    # Yield control to let the loop run the task
    await asyncio.sleep(0.01)

    mock_tracker.track_async.assert_called_once_with(
        call.model_dump(), response.model_dump(), "session_123", 4
    )


def test_celery_dispatcher():
    mock_celery = MagicMock()
    dispatcher = CeleryDispatcher(
        celery_app=mock_celery, task_name="belief_task", custom_arg="value"
    )

    call = LLMCall(messages=[{"role": "user", "content": "test"}])
    response = LLMResponse(text="response text", raw_response=None)

    mock_tracker = MagicMock()
    dispatcher.dispatch(mock_tracker, call, response, "session_123", 4)

    mock_celery.send_task.assert_called_once_with(
        "belief_task",
        args=(call.model_dump(), response.model_dump(), "session_123", 4),
        custom_arg="value",
    )


def test_rq_dispatcher():
    mock_queue = MagicMock()
    dispatcher = RQDispatcher(queue=mock_queue, custom_arg="value")

    call = LLMCall(messages=[{"role": "user", "content": "test"}])
    response = LLMResponse(text="response text", raw_response=None)

    mock_tracker = MagicMock()
    dispatcher.dispatch(mock_tracker, call, response, "session_123", 4)

    mock_queue.enqueue.assert_called_once_with(
        execute_tracking_task,
        call.model_dump(),
        response.model_dump(),
        "session_123",
        4,
        custom_arg="value",
    )


def test_global_tracker_registration_and_worker_task():
    mock_tracker = MagicMock()
    mock_tracker.track_async = AsyncMock()

    register_global_tracker(mock_tracker)
    assert get_global_tracker() == mock_tracker

    call_dict = {
        "messages": [{"role": "user", "content": "test"}],
        "kwargs": {},
        "system": None,
        "metadata": {},
    }
    response_dict = {"text": "response text", "raw_response": None, "metadata": {}}

    # Run the worker task
    execute_tracking_task(call_dict, response_dict, "session_123", 4)

    # The async run inside execute_tracking_task should complete and call track_async
    mock_tracker.track_async.assert_called_once_with(
        call_dict, response_dict, "session_123", 4
    )
