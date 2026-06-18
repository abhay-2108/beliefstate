from beliefstate.call import LLMCall, LLMResponse
from beliefstate.models import Belief
from beliefstate.config import TrackerConfig
from beliefstate.store.base import Store
from beliefstate.store.sqlite import SQLiteStore
from beliefstate.store.redis import RedisStore
from beliefstate.adapters.base import ProviderAdapter
from beliefstate.adapters.openai import OpenAIAdapter
from beliefstate.adapters.anthropic import AnthropicAdapter
from beliefstate.adapters.gemini import GeminiAdapter
from beliefstate.adapters.ollama import OllamaAdapter
from beliefstate.adapters.litellm import LiteLLMAdapter
from beliefstate.extractor import BeliefExtractor
from beliefstate.detector import ContradictionDetector
from beliefstate.resolver import BeliefResolver
from beliefstate.tracker import BeliefTracker, session_context

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

# Integration Exports (optional dependencies - graceful degradation)
try:
    from beliefstate.integrations.fastapi import (
        FastAPIBeliefTrackerMiddleware,
        get_session_id,
    )
except ImportError:
    FastAPIBeliefTrackerMiddleware = None
    get_session_id = None

try:
    from beliefstate.integrations.flask import (
        FlaskBeliefTrackerMiddleware,
        register_flask_hooks,
    )
except ImportError:
    FlaskBeliefTrackerMiddleware = None
    register_flask_hooks = None

try:
    from beliefstate.integrations.llamaindex import (
        LlamaIndexBeliefTrackerCallback
    )
except ImportError:
    LlamaIndexBeliefTrackerCallback = None

try:
    from beliefstate.integrations.openai import (
        process_openai_assistant_message,
        observe_run,
    )
except ImportError:
    process_openai_assistant_message = None
    observe_run = None

try:
    from beliefstate.integrations.langchain import (
        BeliefTrackerLangchainCallback
    )
except ImportError:
    BeliefTrackerLangchainCallback = None

# Observability Exports (optional dependencies)
try:
    from beliefstate.observability import (
        setup_otel,
        trace_sync,
        trace_async,
        BeliefTrackerMetrics,
    )
except ImportError:
    setup_otel = None
    trace_sync = None
    trace_async = None
    BeliefTrackerMetrics = None

__all__ = [
    "LLMCall",
    "LLMResponse",
    "Belief",
    "TrackerConfig",
    "Store",
    "SQLiteStore",
    "RedisStore",
    "ProviderAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "GeminiAdapter",
    "OllamaAdapter",
    "LiteLLMAdapter",
    "BeliefExtractor",
    "ContradictionDetector",
    "BeliefResolver",
    "BeliefTracker",
    "session_context",
    "ResilientAdapterWrapper",
    "CircuitBreaker",
    "CircuitBreakerOpenException",
    "AsyncioDispatcher",
    "SyncDispatcher",
    "CeleryDispatcher",
    "RQDispatcher",
    "register_global_tracker",
    "execute_tracking_task",
    "ContradictionJudge",
    "LLMJudge",
    "LocalNLIJudge",
    "FastAPIBeliefTrackerMiddleware",
    "get_session_id",
    "FlaskBeliefTrackerMiddleware",
    "register_flask_hooks",
    "LlamaIndexBeliefTrackerCallback",
    "process_openai_assistant_message",
    "observe_run",
    "BeliefTrackerLangchainCallback",
    "setup_otel",
    "trace_sync",
    "trace_async",
    "BeliefTrackerMetrics",
]
