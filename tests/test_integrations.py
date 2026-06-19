from unittest.mock import MagicMock, AsyncMock
import pytest
from typing import Optional

from beliefstate import (
    session_context,
)

# Try to import optional dependencies
try:
    from beliefstate import (
        FastAPIBeliefTrackerMiddleware,
        get_session_id,
    )
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

try:
    from beliefstate import (
        FlaskBeliefTrackerMiddleware,
        register_flask_hooks,
    )
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

try:
    from beliefstate import LlamaIndexBeliefTrackerCallback
    HAS_LLAMAINDEX = True
except ImportError:
    HAS_LLAMAINDEX = False

try:
    from beliefstate import (
        process_openai_assistant_message,
        observe_run,
    )
    HAS_OPENAI_INTEGRATION = True
except ImportError:
    HAS_OPENAI_INTEGRATION = False

try:
    from beliefstate import BeliefTrackerLangchainCallback
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False


# --- FastAPI Middleware & Dependency Tests ---


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
@pytest.mark.asyncio
async def test_fastapi_middleware_context_propagation():
    token_val = None
    state_val = None

    async def app(scope, receive, send):
        nonlocal token_val, state_val
        token_val = session_context.get()
        state_val = scope.get("state", {}).get("session_id")
        await send({"type": "http.response.start"})

    middleware = FastAPIBeliefTrackerMiddleware(app, header_name="X-Session-ID")

    scope = {"type": "http", "headers": [(b"x-session-id", b"session-fastapi-123")]}

    called = False

    async def send(event):
        nonlocal called
        called = True

    await middleware(scope, None, send)
    assert called
    assert token_val == "session-fastapi-123"
    assert state_val == "session-fastapi-123"

    # Assert context is reset afterwards
    assert session_context.get() == "default"


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
@pytest.mark.asyncio
async def test_fastapi_dependency_injection():
    # 1. Test when header is present
    mock_request = MagicMock()
    gen = get_session_id(request=mock_request, x_session_id="dep-session-789")
    sid = await gen.__anext__()
    assert sid == "dep-session-789"
    assert session_context.get() == "dep-session-789"

    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()
    assert session_context.get() == "default"

    # 2. Test fallback to request.state
    mock_request_with_state = MagicMock()
    mock_request_with_state.state.session_id = "state-session-000"
    gen_fallback = get_session_id(request=mock_request_with_state, x_session_id=None)
    sid_fallback = await gen_fallback.__anext__()
    assert sid_fallback == "state-session-000"
    assert session_context.get() == "state-session-000"

    with pytest.raises(StopAsyncIteration):
        await gen_fallback.__anext__()
    assert session_context.get() == "default"


# --- Flask Middleware & Hooks Tests ---


@pytest.mark.skipif(not HAS_FLASK, reason="Flask not installed")
def test_flask_middleware_context_propagation():
    token_val = None

    def app(environ, start_response):
        nonlocal token_val
        token_val = session_context.get()
        return [b"OK"]

    middleware = FlaskBeliefTrackerMiddleware(app, header_name="X-Session-ID")

    environ = {"HTTP_X_SESSION_ID": "session-flask-456"}

    res = middleware(environ, None)
    assert res == [b"OK"]
    assert token_val == "session-flask-456"

    # Assert context is reset afterwards
    assert session_context.get() == "default"


@pytest.mark.skipif(not HAS_FLASK, reason="Flask not installed")
def test_flask_request_hooks():
    from flask import Flask, g

    app = Flask("test_app")
    register_flask_hooks(app, header_name="X-Session-ID")

    with app.test_request_context(headers={"X-Session-ID": "flask-hook-111"}):
        app.preprocess_request()

        assert g.session_id == "flask-hook-111"
        assert session_context.get() == "flask-hook-111"

    assert session_context.get() == "default"


# --- LlamaIndex Callback Tests ---


class MockCBEventType:
    LLM = "llm"
    EMBEDDING = "embedding"


class MockMessage:
    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content


class MockResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def dict(self) -> dict:
        return {"text": self.text}


@pytest.mark.skipif(not HAS_LLAMAINDEX, reason="LlamaIndex not installed")
@pytest.mark.asyncio
async def test_llamaindex_callback_handler():
    mock_tracker = MagicMock()
    mock_tracker.config = MagicMock()
    mock_tracker.config.enable_background_tasks = True
    mock_tracker.dispatcher = MagicMock()
    mock_tracker.turn_counter = 0

    callback = LlamaIndexBeliefTrackerCallback(tracker=mock_tracker)

    # Simulating start event
    messages = [
        MockMessage(role="user", content="tell me a joke"),
        MockMessage(role="assistant", content="why did the chicken cross the road?"),
    ]
    payload_start = {"messages": messages}

    event_id = "event_llama_999"
    callback.on_event_start(
        event_type=MockCBEventType.LLM, payload=payload_start, event_id=event_id
    )

    assert event_id in callback.pending_calls
    call = callback.pending_calls[event_id]
    assert len(call.messages) == 2
    assert call.messages[0]["content"] == "tell me a joke"

    # Simulating end event
    response = MockResponse(text="to get to the other side")
    payload_end = {"response": response}

    session_context.set("session-llama-789")
    callback.on_event_end(
        event_type=MockCBEventType.LLM, payload=payload_end, event_id=event_id
    )

    assert event_id not in callback.pending_calls
    assert mock_tracker.turn_counter == 1
    mock_tracker.dispatcher.dispatch.assert_called_once()

    # Extract args passed to dispatcher.dispatch
    _, call_arg, response_arg, session_id_arg, turn_arg = (
        mock_tracker.dispatcher.dispatch.call_args[0]
    )
    assert call_arg.messages[0]["content"] == "tell me a joke"
    assert response_arg.text == "to get to the other side"
    assert session_id_arg == "session-llama-789"
    assert turn_arg == 1

    session_context.set("default")


# --- OpenAI Observer Tests ---


class MockTextObject:
    def __init__(self, value: str):
        self.value = value


class MockContentBlock:
    def __init__(self, text_val: str):
        self.type = "text"
        self.text = MockTextObject(text_val)


class MockOpenAIMessage:
    def __init__(self, role: str, content_val: str, run_id: Optional[str] = None):
        self.role = role
        self.run_id = run_id
        self.content = [MockContentBlock(content_val)]

    def model_dump(self) -> dict:
        return {
            "role": self.role,
            "run_id": self.run_id,
            "content": [
                {"type": "text", "text": {"value": self.content[0].text.value}}
            ],
        }


@pytest.mark.skipif(not HAS_OPENAI_INTEGRATION, reason="OpenAI integration not installed")
def test_process_openai_assistant_message():
    thread_messages = [
        MockOpenAIMessage(
            role="assistant",
            content_val="This is the output of the run.",
            run_id="run_1",
        ),
        MockOpenAIMessage(role="user", content_val="Who am I speaking with?"),
        MockOpenAIMessage(
            role="assistant", content_val="Older assistant reply", run_id="run_old"
        ),
        MockOpenAIMessage(role="user", content_val="Hello assistant"),
    ]

    call, response = process_openai_assistant_message(thread_messages, run_id="run_1")

    assert response.text == "This is the output of the run."
    assert len(call.messages) == 3
    assert call.messages[0]["content"] == "Hello assistant"
    assert call.messages[1]["content"] == "Older assistant reply"
    assert call.messages[2]["content"] == "Who am I speaking with?"


@pytest.mark.skipif(not HAS_OPENAI_INTEGRATION, reason="OpenAI integration not installed")
@pytest.mark.asyncio
async def test_observe_run_polling_and_dispatch():
    mock_tracker = MagicMock()
    mock_tracker.config = MagicMock()
    mock_tracker.config.enable_background_tasks = True
    mock_tracker.dispatcher = MagicMock()
    mock_tracker.turn_counter = 0

    mock_client = AsyncMock()

    mock_run_in_progress = MagicMock()
    mock_run_in_progress.status = "in_progress"

    mock_run_completed = MagicMock()
    mock_run_completed.status = "completed"

    mock_client.beta.threads.runs.retrieve = AsyncMock(
        side_effect=[mock_run_in_progress, mock_run_completed]
    )

    mock_messages_cursor = MagicMock()
    mock_messages_cursor.data = [
        MockOpenAIMessage(
            role="assistant", content_val="Final assistant reply", run_id="run_abc"
        ),
        MockOpenAIMessage(role="user", content_val="User prompt"),
    ]
    mock_client.beta.threads.messages.list = AsyncMock(
        return_value=mock_messages_cursor
    )

    await observe_run(
        tracker=mock_tracker,
        client=mock_client,
        thread_id="thread_123",
        run_id="run_abc",
        session_id="session-openai-999",
        poll_interval=0.01,
    )

    assert mock_client.beta.threads.runs.retrieve.call_count == 2
    mock_client.beta.threads.messages.list.assert_called_once_with(
        thread_id="thread_123"
    )

    assert mock_tracker.turn_counter == 1
    mock_tracker.dispatcher.dispatch.assert_called_once()

    _, call_arg, response_arg, session_id_arg, turn_arg = (
        mock_tracker.dispatcher.dispatch.call_args[0]
    )
    assert call_arg.messages[0]["content"] == "User prompt"
    assert response_arg.text == "Final assistant reply"
    assert session_id_arg == "session-openai-999"
    assert turn_arg == 1


# --- LangChain Callback Tests ---


class MockBaseMessage:
    def __init__(self, type: str, content: str) -> None:
        self.type = type
        self.content = content


class MockGeneration:
    def __init__(self, text: str) -> None:
        self.text = text


class MockLLMResult:
    def __init__(self, generations: list) -> None:
        self.generations = generations

    def dict(self) -> dict:
        return {
            "generations": [
                [{"text": g.text} for g in gen_list] for gen_list in self.generations
            ]
        }


@pytest.mark.skipif(not HAS_LANGCHAIN, reason="LangChain not installed")
@pytest.mark.asyncio
async def test_langchain_callback_handler():
    mock_tracker = MagicMock()
    mock_tracker.config = MagicMock()
    mock_tracker.config.enable_background_tasks = False
    mock_tracker.turn_counter = 0
    mock_tracker._track_background = AsyncMock()

    callback = BeliefTrackerLangchainCallback(tracker=mock_tracker)

    run_id = "langchain-run-123"
    messages = [
        [
            MockBaseMessage(type="system", content="you are helpful"),
            MockBaseMessage(type="ai", content="why did the chicken cross the road?"),
            MockBaseMessage(type="user", content="tell me a joke"),
        ]
    ]

    await callback.on_chat_model_start(
        serialized={}, messages=messages, run_id=run_id, kwargs={}
    )

    assert run_id in callback.pending_calls
    call = callback.pending_calls[run_id]
    assert len(call.messages) == 3
    assert call.messages[0]["role"] == "system"
    assert call.messages[1]["role"] == "assistant"
    assert call.messages[2]["role"] == "user"
    assert call.messages[2]["content"] == "tell me a joke"

    # Simulating end event
    response = MockLLMResult(
        generations=[[MockGeneration(text="chicken crossed the road")]]
    )

    session_context.set("session-langchain")
    await callback.on_llm_end(response=response, run_id=run_id)

    assert run_id not in callback.pending_calls
    assert mock_tracker.turn_counter == 1
    mock_tracker._track_background.assert_called_once()

    # 2. Test standard LLM start
    run_id_2 = "langchain-run-456"
    prompts = ["Tell me another joke."]
    await callback.on_llm_start(
        serialized={}, prompts=prompts, run_id=run_id_2, kwargs={}
    )
    assert run_id_2 in callback.pending_calls
    call_2 = callback.pending_calls[run_id_2]
    assert len(call_2.messages) == 1
    assert call_2.messages[0]["role"] == "user"
    assert call_2.messages[0]["content"] == "Tell me another joke."

    # 3. Test on_llm_error cleanup
    await callback.on_llm_error(error=ValueError("Run failed"), run_id=run_id_2)
    assert run_id_2 not in callback.pending_calls

    session_context.set("default")
