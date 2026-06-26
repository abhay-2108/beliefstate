import sys
from unittest.mock import MagicMock
from typing import Optional, List, Any

import pytest

# Mock llama_index modules globally before any beliefstate imports
mock_llama_index = MagicMock()
mock_llama_index.core = MagicMock()
mock_llama_index.core.callbacks = MagicMock()


class MockCBEventType:
    LLM = "llm"
    EMBEDDING = "llm"


mock_llama_index.core.callbacks.CBEventType = MockCBEventType


class MockBaseCallbackHandler:
    def __init__(
        self,
        event_starts_to_ignore: Optional[List[Any]] = None,
        event_ends_to_ignore: Optional[List[Any]] = None,
    ) -> None:
        self.event_starts_to_ignore = event_starts_to_ignore or []
        self.event_ends_to_ignore = event_ends_to_ignore or []


mock_llama_index.core.callbacks.BaseCallbackHandler = MockBaseCallbackHandler

_original_modules = {}
_MOCK_KEYS = ["llama_index", "llama_index.core", "llama_index.core.callbacks"]

for key in _MOCK_KEYS:
    _original_modules[key] = sys.modules.get(key)

sys.modules["llama_index"] = mock_llama_index
sys.modules["llama_index.core"] = mock_llama_index.core
sys.modules["llama_index.core.callbacks"] = mock_llama_index.core.callbacks

# Now safe to import beliefstate (llamaindex.py will see the mocks)
from beliefstate import session_context  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def _restore_llama_index_modules():
    """Restore original sys.modules entries after the test session."""
    yield
    for key, original in _original_modules.items():
        if original is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = original


@pytest.fixture(autouse=True)
def _reset_session_context():
    """Reset session_context to 'default' after each test."""
    yield
    session_context.set("default")
