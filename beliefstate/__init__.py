"""BeliefState: A Universal LLM Belief State Tracker.

All adapter and integration imports are lazy — they only fail when you
*use* a class whose underlying SDK is not installed, not on ``import beliefstate``.
"""

# --- Core (always available) ---------------------------------------------------
from beliefstate.call import LLMCall, LLMResponse
from beliefstate.models import Belief
from beliefstate.config import TrackerConfig
from beliefstate.store.base import Store
from beliefstate.store.sqlite import SQLiteStore
from beliefstate.tracker import BeliefTracker, session_context
from beliefstate.extractor import BeliefExtractor
from beliefstate.detector import ContradictionDetector
from beliefstate.resolver import BeliefResolver
from beliefstate.logging_utils import TrackerEvent, log_event

# Resilience Exports
from beliefstate.resilience import (
    ResilientAdapterWrapper,
    CircuitBreaker,
    CircuitBreakerOpenException,
)

# Judge Exports
from beliefstate.judge import (
    ContradictionJudge,
    LLMJudge,
    LocalNLIJudge,
)

# Dispatcher Exports
from beliefstate.dispatcher import (
    AsyncioDispatcher,
    SyncDispatcher,
    CeleryDispatcher,
    RQDispatcher,
    register_global_tracker,
    execute_tracking_task,
)

# --- Adapters (optional SDKs) --------------------------------------------------
from beliefstate.adapters.base import ProviderAdapter

try:
    from beliefstate.adapters.openai import OpenAIAdapter
except ImportError:
    OpenAIAdapter = None  # type: ignore[assignment,misc]

try:
    from beliefstate.adapters.anthropic import AnthropicAdapter
except ImportError:
    AnthropicAdapter = None  # type: ignore[assignment,misc]

try:
    from beliefstate.adapters.gemini import GeminiAdapter
except ImportError:
    GeminiAdapter = None  # type: ignore[assignment,misc]

try:
    from beliefstate.adapters.ollama import OllamaAdapter
except ImportError:
    OllamaAdapter = None  # type: ignore[assignment,misc]

try:
    from beliefstate.adapters.litellm import LiteLLMAdapter
except ImportError:
    LiteLLMAdapter = None  # type: ignore[assignment,misc]

# --- Stores (optional backends) ------------------------------------------------
try:
    from beliefstate.store.redis import RedisStore
except ImportError:
    RedisStore = None  # type: ignore[assignment,misc]

try:
    from beliefstate.store.memory import InMemoryBeliefStore
except ImportError:
    InMemoryBeliefStore = None  # type: ignore[assignment,misc]

try:
    from beliefstate.store.postgres import PostgreSQLStore
except ImportError:
    PostgreSQLStore = None  # type: ignore[assignment,misc]

# --- Framework Integrations (optional SDKs) ------------------------------------
try:
    from beliefstate.integrations.fastapi import (
        FastAPIBeliefTrackerMiddleware,
        get_session_id,
    )
except ImportError:
    FastAPIBeliefTrackerMiddleware = None  # type: ignore[misc,assignment]
    get_session_id = None  # type: ignore[assignment]

try:
    from beliefstate.integrations.flask import (
        FlaskBeliefTrackerMiddleware,
        register_flask_hooks,
    )
except ImportError:
    FlaskBeliefTrackerMiddleware = None  # type: ignore[misc,assignment]
    register_flask_hooks = None  # type: ignore[assignment]

try:
    from beliefstate.integrations.asgi import BeliefTrackerASGIMiddleware
except ImportError:
    BeliefTrackerASGIMiddleware = None  # type: ignore[assignment,misc]

try:
    from beliefstate.integrations.langchain import BeliefTrackerLangchainCallback
except ImportError:
    BeliefTrackerLangchainCallback = None  # type: ignore[assignment,misc]

try:
    from beliefstate.integrations.llamaindex import LlamaIndexBeliefTrackerCallback
except ImportError:
    LlamaIndexBeliefTrackerCallback = None  # type: ignore[assignment,misc]

try:
    from beliefstate.integrations.openai import (
        process_openai_assistant_message,
        observe_run,
    )
except ImportError:
    process_openai_assistant_message = None  # type: ignore[assignment]
    observe_run = None  # type: ignore[assignment]


__all__ = [
    # Core
    "LLMCall",
    "LLMResponse",
    "Belief",
    "TrackerConfig",
    "Store",
    "SQLiteStore",
    "RedisStore",
    "PostgreSQLStore",
    "InMemoryBeliefStore",
    "ProviderAdapter",
    "BeliefExtractor",
    "ContradictionDetector",
    "BeliefResolver",
    "BeliefTracker",
    "session_context",
    # Resilience
    "ResilientAdapterWrapper",
    "CircuitBreaker",
    "CircuitBreakerOpenException",
    # Judges
    "ContradictionJudge",
    "LLMJudge",
    "LocalNLIJudge",
    # Dispatchers
    "AsyncioDispatcher",
    "SyncDispatcher",
    "CeleryDispatcher",
    "RQDispatcher",
    "register_global_tracker",
    "execute_tracking_task",
    # Adapters (may be None if SDK not installed)
    "OpenAIAdapter",
    "AnthropicAdapter",
    "GeminiAdapter",
    "OllamaAdapter",
    "LiteLLMAdapter",
    # Integrations (may be None if framework not installed)
    "FastAPIBeliefTrackerMiddleware",
    "get_session_id",
    "FlaskBeliefTrackerMiddleware",
    "register_flask_hooks",
    "BeliefTrackerASGIMiddleware",
    "BeliefTrackerLangchainCallback",
    "LlamaIndexBeliefTrackerCallback",
    "process_openai_assistant_message",
    "observe_run",
    # Structured logging
    "TrackerEvent",
    "log_event",
]
