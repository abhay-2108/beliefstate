# BeliefState: A Universal LLM Belief State Tracker

**BeliefState** is an asynchronous, zero-latency belief state tracking layer for Python applications. It seamlessly intercepts Large Language Model (LLM) chats, extracts factual beliefs, resolves contradictions via an LLM judge, and saves them to persistent storage (SQLite or Redis) — completely in the background.

It supports **OpenAI, Anthropic, Gemini, Ollama, and LiteLLM** natively, and features a highly flexible **Dual-Adapter Architecture** that allows you to use different providers for your application vs your background tracking logic.

---

## 🚀 Features

*   **Zero-Latency Tracking**: Extraction and conflict detection run in fire-and-forget background tasks.
*   **Dual-Adapter Architecture**: Use an expensive model (like Claude) for your app, and a cheap/local model (like Ollama or OpenAI) for belief extraction and embeddings.
*   **API Resilience**: Out-of-the-box exponential backoff retries (via `tenacity`) and stateful circuit breakers to fail-fast during LLM API outages.
*   **Persistent Task Queues**: Pluggable dispatcher support to run background tracking via **Celery** or **Redis Queue (RQ)** to ensure no beliefs are lost on server crashes.
*   **Embedding Batching**: Combines multiple belief embedding requests into a single API call to prevent rate limit triggers, with a robust fallback to individual requests.
*   **Smart Contradiction Resolution**: Uses semantic embeddings to group related facts, and an NLI judge to gracefully resolve contradictions (Overwrite, Keep Old, or Raise).
*   **Plug-and-Play Integrations**: Includes helpers for `LangChain` Callbacks, `FastAPI` (ASGI), and `Flask` (WSGI).

---

## 📦 Installation

To install the core package:
```bash
pip install beliefstate
```

To install with extras (e.g., Redis, Celery, RQ, or LiteLLM):
```bash
pip install "beliefstate[redis,celery,rq,litellm]"
```

---

## 🛠️ Quickstart

The easiest way to track beliefs is using the `@tracker.wrap` decorator around your existing LLM function.

```python
import asyncio
from beliefstate import BeliefTracker, TrackerConfig
from beliefstate.adapters import OpenAIAdapter

# 1. Configure the Tracker
config = TrackerConfig(
    enable_background_tasks=True,
    store_type="sqlite",
    store_kwargs={"db_path": "user_beliefs.db"}
)

# 2. Initialize the Adapter and Tracker
adapter = OpenAIAdapter(model="gpt-4o", embed_model="text-embedding-3-small")
tracker = BeliefTracker(config=config, adapter=adapter)

# 3. Wrap your standard application logic
@tracker.wrap
async def chat(messages):
    import openai
    client = openai.AsyncClient()
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return response

async def main():
    # Set the unique session/user ID
    tracker.set_session("user_123")
    
    # Run your app normally! The tracker intercepts and extracts silently.
    await chat([{"role": "user", "content": "I am a Python developer living in Tokyo."}])

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🛡️ API Resilience & Config

You can tune the retry behavior and circuit breakers directly in `TrackerConfig`:

```python
config = TrackerConfig(
    # API Retries (exponential backoff)
    retry_max_attempts=5,
    retry_min_wait=2.0,       # seconds
    retry_max_wait=30.0,      # seconds
    retry_multiplier=2.0,

    # Circuit Breakers (fail-fast to protect your app)
    enable_circuit_breaker=True,
    circuit_breaker_failure_threshold=5,
    circuit_breaker_recovery_timeout=30.0
)
```

---

## 🚂 Pluggable Background Dispatchers (Celery / RQ)

To offload tracking to a durable background worker, you can inject a pluggable `TaskDispatcher`.

### Option A: Celery Dispatcher

```python
from celery import Celery
from beliefstate import BeliefTracker, TrackerConfig
from beliefstate.dispatcher import CeleryDispatcher

celery_app = Celery("tasks", broker="redis://localhost:6379/0")

# Inject CeleryDispatcher into the tracker
tracker = BeliefTracker(
    config=TrackerConfig(),
    adapter=app_adapter,
    dispatcher=CeleryDispatcher(celery_app=celery_app)
)
```

### Option B: RQ (Redis Queue) Dispatcher

```python
from redis import Redis
from rq import Queue
from beliefstate import BeliefTracker, TrackerConfig
from beliefstate.dispatcher import RQDispatcher

redis_conn = Redis(host="localhost", port=6379)
queue = Queue("belief-state-tasks", connection=redis_conn)

# Inject RQDispatcher into the tracker
tracker = BeliefTracker(
    config=TrackerConfig(),
    adapter=app_adapter,
    dispatcher=RQDispatcher(queue=queue)
)
```

### Worker Setup
In your background worker file, register the global tracker so that tasks enqueued by name can execute the tracking synchronously on the worker process:

```python
from beliefstate.dispatcher import register_global_tracker
from my_app import tracker # Import your initialized BeliefTracker

# Register the tracker inside your celery/rq worker startup script
register_global_tracker(tracker)
```

---

## 🧠 The Dual-Adapter Architecture

If your main application uses a provider that doesn't support embeddings (like Anthropic), or if you want to use a cheaper local model for tracking to save costs, you can use the **Dual-Adapter Architecture**.

```python
from beliefstate.adapters import AnthropicAdapter, OllamaAdapter

# Your main app uses Claude 3.5 Sonnet
app_adapter = AnthropicAdapter(model="claude-3-5-sonnet-latest")

# But the background tracker uses local Llama 3 for free!
bg_adapter = OllamaAdapter(model="llama3", embed_model="nomic-embed-text")

tracker = BeliefTracker(
    config=config,
    adapter=app_adapter,             # Intercepts the Claude API payload
    internal_adapter=bg_adapter      # Runs extraction, embeddings, and judge calls
)
```

### 🔌 Multi-Provider Routing with LiteLLM

If your application relies on enterprise cloud providers (like Azure OpenAI, AWS Bedrock, or Cohere), you can leverage the `LiteLLMAdapter` to unified-route completion and embedding requests.

```python
from beliefstate.adapters import LiteLLMAdapter

# Configure dynamic routing via LiteLLM
enterprise_adapter = LiteLLMAdapter(
    model="azure/gpt-4o",
    embed_model="cohere/embed-english-v3.0"
)

tracker = BeliefTracker(
    config=config,
    adapter=enterprise_adapter
)
```

---

## 🗄️ Stores

Stores determine where the extracted facts live. You can configure them via `TrackerConfig(store_type="...")` or inject them directly.

- **`sqlite`** (`SQLiteStore`): Asynchronous, persistent single-file database. Perfect for single-server production apps. (Use `db_path=":memory:"` for transient tests).
- **`redis`** (`RedisStore`): Distributed caching. Essential if running multiple application workers (e.g., behind a load balancer).

---

## 🔌 Framework Integrations

BeliefState ships with helpers for major frameworks to handle session tracking automatically.

### FastAPI (ASGI)
```python
from fastapi import FastAPI
from beliefstate.integrations.asgi import BeliefTrackerASGIMiddleware

app = FastAPI()
app.add_middleware(
    BeliefTrackerASGIMiddleware,
    header_name="X-Session-ID"
)
# Automatically sets session_context from incoming header X-Session-ID
```

### LangChain
```python
from beliefstate import session_context
from beliefstate.integrations.langchain import BeliefTrackerLangchainCallback

# 1. Set active session ID context
session_context.set("user_123")

# 2. Initialize and attach callback
handler = BeliefTrackerLangchainCallback(tracker=tracker)
await llm.ainvoke("Hello!", config={"callbacks": [handler]})
```

---

## 📜 License
MIT License
