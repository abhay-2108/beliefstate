"""Provider setup and unified LLM call for the beliefstate test app."""

from __future__ import annotations

import os
from typing import Any

from beliefstate import BeliefTracker, TrackerConfig
from beliefstate.call import LLMCall, LLMResponse

from utils import run_async


# ── Tracker builder ────────────────────────────────────────────────────────────


def build_tracker(cfg: dict[str, Any]) -> BeliefTracker:
    """Create a BeliefTracker from sidebar configuration values.

    ``cfg`` is expected to contain keys matching the sidebar widgets:
    provider, model, embed_model, api_key, store_type, db_path,
    similarity_threshold, contradiction_threshold, entailment_threshold,
    judge_timeout, user_confidence_cap, assistant_confidence_cap,
    max_beliefs, background_tasks, and provider-specific keys.
    """
    from beliefstate.adapters import (
        AnthropicAdapter,
        GeminiAdapter,
        OllamaAdapter,
        OpenAIAdapter,
    )

    provider = cfg["provider"]
    model = cfg["model"]
    embed_model = cfg["embed_model"]
    api_key = cfg["api_key"]

    # Store config
    store_kwargs: dict[str, Any] = {}
    if cfg["store_type"] == "sqlite":
        store_kwargs = {"db_path": cfg["db_path"]}
    else:
        store_kwargs = {"db_path": ":memory:"}

    config = TrackerConfig(
        store_type="sqlite" if cfg["store_type"] == "sqlite" else "memory",
        store_kwargs=store_kwargs,
        similarity_threshold=cfg["similarity_threshold"],
        contradiction_threshold=cfg["contradiction_threshold"],
        entailment_threshold=cfg["entailment_threshold"],
        judge_timeout=cfg["judge_timeout"],
        user_confidence_cap=cfg["user_confidence_cap"],
        assistant_confidence_cap=cfg["assistant_confidence_cap"],
        max_beliefs=int(cfg["max_beliefs"]),
        enable_background_tasks=cfg["background_tasks"],
        task_dispatcher_type="sync" if not cfg["background_tasks"] else "asyncio",
        resolution_strategy=cfg.get("resolution_strategy", "overwrite"),
        min_injection_confidence=cfg.get("min_injection_confidence", 0.80),
        include_hypothetical_in_context=cfg.get(
            "include_hypothetical_in_context", False
        ),
    )

    # Adapter
    adapter: Any = None
    internal_adapter: Any = None

    if provider == "OpenAI":
        os.environ["OPENAI_API_KEY"] = api_key
        adapter = OpenAIAdapter(model=model, embed_model=embed_model)
        tracker = BeliefTracker(config=config, adapter=adapter)

    elif provider == "NVIDIA":
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=api_key, base_url="https://integrate.api.nvidia.com/v1"
        )
        embed_kwargs: dict[str, Any] = {}
        if embed_model != "nvidia/nv-embed-v1":
            embed_kwargs["extra_body"] = {"input_type": "query"}
        adapter = OpenAIAdapter(
            client=client,
            model=model,
            embed_model=embed_model,
            embed_kwargs=embed_kwargs,
        )
        tracker = BeliefTracker(config=config, adapter=adapter)

    elif provider == "Anthropic":
        os.environ["ANTHROPIC_API_KEY"] = api_key
        os.environ["OPENAI_API_KEY"] = cfg.get("embed_api_key", "")
        adapter = AnthropicAdapter(model=model)
        internal_adapter = OpenAIAdapter(
            model="gpt-4o-mini", embed_model="text-embedding-3-small"
        )
        tracker = BeliefTracker(
            config=config,
            adapter=adapter,
            internal_adapter=internal_adapter,
        )

    elif provider == "Gemini":
        os.environ["GEMINI_API_KEY"] = api_key
        adapter = GeminiAdapter(model=model, embed_model=embed_model)
        tracker = BeliefTracker(config=config, adapter=adapter)

    elif provider == "Ollama":
        adapter = OllamaAdapter(
            model=model,
            embed_model=embed_model,
            host=cfg.get("ollama_host", "http://localhost"),
            port=int(cfg.get("ollama_port", 11434)),
        )
        tracker = BeliefTracker(config=config, adapter=adapter)

    else:
        raise ValueError(f"Unknown provider: {provider}")

    return tracker


# ── LLM call helpers ──────────────────────────────────────────────────────────


def _extract_response_text(provider: str, raw: Any) -> str:
    """Pull the assistant text from a raw provider response."""
    if provider in ("OpenAI", "NVIDIA"):
        return raw.choices[0].message.content
    if provider == "Anthropic":
        # Non-streaming: content[0].text
        return raw.content[0].text
    if provider == "Gemini":
        return raw.text
    if provider == "Ollama":
        return raw["message"]["content"]
    raise ValueError(f"Unknown provider: {provider}")


def call_llm_sync(
    provider: str,
    messages: list[dict[str, str]],
    cfg: dict[str, Any],
    tracker: BeliefTracker,
    session_id: str,
    turn: int,
) -> tuple[str, float]:
    """Call the LLM, track beliefs, and return (response_text, latency_ms).

    This function uses ``tracker._dispatch`` indirectly via ``track_async``
    so that the belief pipeline runs correctly in both sync and async modes.
    """
    import time

    api_key = cfg["api_key"]
    model = cfg["model"]
    t_start = time.time()

    async def _call() -> Any:
        if provider == "OpenAI":
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=api_key)
            return await client.chat.completions.create(
                model=model, messages=messages, max_tokens=300
            )

        if provider == "NVIDIA":
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://integrate.api.nvidia.com/v1",
            )
            return await client.chat.completions.create(
                model=model, messages=messages, max_tokens=300
            )

        if provider == "Anthropic":
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(api_key=api_key)
            user_msgs = [m for m in messages if m["role"] != "system"]
            return await client.messages.create(
                model=model, max_tokens=300, messages=user_msgs
            )

        if provider == "Gemini":
            from google import genai

            client = genai.Client(api_key=api_key)
            # Build conversation history for Gemini
            contents = []
            for msg in messages[:-1]:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({"role": role, "parts": [msg["content"]]})
            contents.append({"role": "user", "parts": [messages[-1]["content"]]})
            resp = await client.aio.models.generate_content(
                model=model, contents=contents
            )
            return resp

        if provider == "Ollama":
            import ollama

            host = cfg.get("ollama_host", "http://localhost")
            port = int(cfg.get("ollama_port", 11434))
            client = ollama.AsyncClient(host=f"{host}:{port}")
            return await client.chat(model=model, messages=messages)

        raise ValueError(f"Unknown provider: {provider}")

    raw_response = run_async(_call())
    latency_ms = (time.time() - t_start) * 1000
    assistant_text = _extract_response_text(provider, raw_response)

    # Track via the belief pipeline
    llm_call = LLMCall(messages=messages, kwargs={})
    llm_response = LLMResponse(text=assistant_text, raw_response=raw_response)

    run_async(
        tracker.track_async(
            llm_call.model_dump(),
            llm_response.model_dump(),
            session_id=session_id,
            turn=turn,
        )
    )

    # Sync dispatcher needs a moment to finish
    if not cfg.get("background_tasks", False):
        import time as _time

        _time.sleep(0.2)

    return assistant_text, latency_ms
