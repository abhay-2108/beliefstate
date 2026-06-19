"""Tests verifying lazy imports work — the package must import without optional SDKs."""
import pytest


class TestCoreImports:
    """Core classes should always be importable."""

    def test_import_beliefstate_succeeds(self):
        import beliefstate
        assert hasattr(beliefstate, "BeliefTracker")

    def test_import_belief_tracker_class(self):
        from beliefstate import BeliefTracker
        assert BeliefTracker is not None

    def test_import_tracker_config(self):
        from beliefstate import TrackerConfig
        assert TrackerConfig is not None

    def test_import_belief_model(self):
        from beliefstate import Belief
        assert Belief is not None

    def test_import_session_context(self):
        from beliefstate import session_context
        assert session_context is not None

    def test_import_sqlite_store(self):
        from beliefstate import SQLiteStore
        assert SQLiteStore is not None


class TestAdapterImportsOptional:
    """Adapter symbols should be importable (but may be None if SDK missing)."""

    def test_openai_adapter_importable(self):
        from beliefstate import OpenAIAdapter
        # May be None if openai not installed — that's fine
        assert OpenAIAdapter is None or callable(OpenAIAdapter)

    def test_anthropic_adapter_importable(self):
        from beliefstate import AnthropicAdapter
        assert AnthropicAdapter is None or callable(AnthropicAdapter)

    def test_gemini_adapter_importable(self):
        from beliefstate import GeminiAdapter
        assert GeminiAdapter is None or callable(GeminiAdapter)

    def test_ollama_adapter_importable(self):
        from beliefstate import OllamaAdapter
        assert OllamaAdapter is None or callable(OllamaAdapter)

    def test_litellm_adapter_importable(self):
        from beliefstate import LiteLLMAdapter
        assert LiteLLMAdapter is None or callable(LiteLLMAdapter)


class TestAllExports:
    """Every entry in __all__ should be importable (possibly None)."""

    def test_all_entries_accessible(self):
        import beliefstate
        for name in beliefstate.__all__:
            val = getattr(beliefstate, name, "__MISSING__")
            assert val != "__MISSING__", f"{name} listed in __all__ but not accessible"
