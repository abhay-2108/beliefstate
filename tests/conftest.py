import sys
from unittest.mock import MagicMock
from typing import Optional, List, Any

# Mock llama_index modules globally before any tests or beliefstate package modules are imported
mock_llama_index = MagicMock()
mock_llama_index.core = MagicMock()
mock_llama_index.core.callbacks = MagicMock()


class MockCBEventType:
    LLM = "llm"
    EMBEDDING = "embedding"


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

sys.modules["llama_index"] = mock_llama_index
sys.modules["llama_index.core"] = mock_llama_index.core
sys.modules["llama_index.core.callbacks"] = mock_llama_index.core.callbacks
