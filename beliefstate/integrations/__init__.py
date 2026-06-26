from beliefstate.integrations.asgi import BeliefTrackerASGIMiddleware
from beliefstate.integrations.wsgi import BeliefTrackerWSGIMiddleware
from beliefstate.integrations.langchain import BeliefTrackerLangchainCallback

try:
    from beliefstate.integrations.fastapi import (
        FastAPIBeliefTrackerMiddleware,
        get_session_id,
    )
except ImportError:
    FastAPIBeliefTrackerMiddleware = None  # type: ignore[assignment,misc]
    get_session_id = None  # type: ignore[assignment,misc]

try:
    from beliefstate.integrations.flask import (
        FlaskBeliefTrackerMiddleware,
        register_flask_hooks,
    )
except ImportError:
    FlaskBeliefTrackerMiddleware = None  # type: ignore[assignment,misc]
    register_flask_hooks = None  # type: ignore[assignment,misc]

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
    process_openai_assistant_message = None  # type: ignore[assignment,misc]
    observe_run = None  # type: ignore[assignment,misc]

__all__ = [
    "BeliefTrackerASGIMiddleware",
    "BeliefTrackerWSGIMiddleware",
    "BeliefTrackerLangchainCallback",
    "FastAPIBeliefTrackerMiddleware",
    "get_session_id",
    "FlaskBeliefTrackerMiddleware",
    "register_flask_hooks",
    "LlamaIndexBeliefTrackerCallback",
    "process_openai_assistant_message",
    "observe_run",
]
