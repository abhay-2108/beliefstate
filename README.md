<p align="center">
  <img src="assets/data_flow.png" alt="BeliefState" width="600">
</p>

<h1 align="center">BeliefState</h1>

<p align="center">
  A production-ready belief state tracker for LLM applications.<br>
  Extract facts, detect contradictions, and persist knowledge — automatically.
</p>

<p align="center">
  <a href="https://pypi.org/project/beliefstate/"><img src="https://img.shields.io/pypi/v/beliefstate?color=blue" alt="PyPI"></a>
  <a href="https://github.com/abhay-2108/beliefstate/blob/main/LICENSE"><img src="https://img.shields.io/github/license/abhay-2108/beliefstate" alt="License"></a>
  <a href="https://pypi.org/project/beliefstate/"><img src="https://img.shields.io/pypi/pyversions/beliefstate" alt="Python"></a>
  <a href="https://github.com/abhay-2108/beliefstate/actions"><img src="https://img.shields.io/badge/tests-passing-brightgreen" alt="Tests"></a>
</p>

---

## What is BeliefState?

LLMs generate different answers every time. BeliefState gives your application **persistent memory** by intercepting LLM conversations in the background, extracting factual beliefs, resolving contradictions, and storing them — without adding latency to your request path.

```python
from beliefstate import BeliefTracker
from beliefstate.adapters import OpenAIAdapter

tracker = BeliefTracker(
    adapter=OpenAIAdapter(model="gpt-4o"),
    config={"store_type": "sqlite", "store_kwargs": {"db_path": "beliefs.db"}}
)

@tracker.wrap
async def chat(messages):
    # Your existing LLM logic — unchanged
    return await openai_client.chat.completions.create(model="gpt-4o", messages=messages)

tracker.set_session("user_123")
await chat([{"role": "user", "content": "I live in Tokyo and work at Google."}])
# BeliefState silently extracts: {subject: "user_123", predicate: "lives_in", value: "Tokyo"}
#                                        {subject: "user_123", predicate: "works_at", value: "Google"}
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Zero-latency tracking** | Extraction and contradiction detection run in fire-and-forget background tasks |
| **5 LLM providers** | OpenAI, Anthropic, Gemini, Ollama, LiteLLM (100+ providers) |
| **Dual-adapter architecture** | Use an expensive model for your app, a cheap/local model for tracking |
| **Smart contradiction resolution** | NLI judge + semantic embeddings to gracefully resolve conflicting facts |
| **Persistent stores** | SQLite, PostgreSQL, Redis, or in-memory — with full audit trails |
| **Framework integrations** | LangChain, LlamaIndex, FastAPI, Flask, OpenAI Assistants — out of the box |
| **Production resilience** | Retry with backoff, circuit breakers, health checks, structured logging |
| **Pluggable dispatchers** | Celery, Redis Queue, or in-process — survive server restarts |

---

## Installation

```bash
pip install beliefstate
```

With optional extras:

```bash
# Provider adapters
pip install "beliefstate[openai]"
pip install "beliefstate[anthropic]"
pip install "beliefstate[gemini]"
pip install "beliefstate[ollama]"
pip install "beliefstate[litellm]"

# Stores
pip install "beliefstate[redis]"        # Redis store
pip install "beliefstate[postgres]"     # PostgreSQL store

# Framework integrations
pip install "beliefstate[langchain]"
pip install "beliefstate[llamaindex]"
pip install "beliefstate[fastapi]"
pip install "beliefstate[flask]"

# Background dispatchers
pip install "beliefstate[celery]"
pip install "beliefstate[rq]"

# Everything
pip install "beliefstate[all]"
```

---

## Quick Start

### 1. Basic Usage

```python
import asyncio
from beliefstate import BeliefTracker
from beliefstate.adapters import OpenAIAdapter

async def main():
    tracker = BeliefTracker(
        adapter=OpenAIAdapter(model="gpt-4o"),
        config={"store_type": "sqlite", "store_kwargs": {"db_path": "beliefs.db"}}
    )

    @tracker.wrap
    async def chat(messages):
        import openai
        client = openai.AsyncClient()
        return await client.chat.completions.create(model="gpt-4o", messages=messages)

    tracker.set_session("user_123")
    await chat([{"role": "user", "content": "I'm a Rust developer based in Berlin."}])

    # Query stored beliefs
    beliefs = await tracker.store.get_beliefs("user_123")
    for b in beliefs:
        print(f"  {b.subject} {b.predicate} = {b.value}")

asyncio.run(main())
```

### 2. Querying Beliefs

```python
# Get all beliefs for a session
beliefs = await tracker.store.get_beliefs("user_123")

# Search by semantic similarity
results = await tracker.store.search_beliefs(
    session_id="user_123",
    query="where does the user live",
    top_k=5
)

# Get belief audit trail
audit = await tracker.store.get_audit_history(
    session_id="user_123",
    subject="user_123",
    predicate="lives_in"
)
```

### 3. Cleanup

```python
# Graceful shutdown — waits for background tasks to finish
await tracker.shutdown(grace_seconds=5.0)
```

---

## Supported Providers

### OpenAI

```python
from beliefstate.adapters import OpenAIAdapter

adapter = OpenAIAdapter(
    model="gpt-4o",
    embed_model="text-embedding-3-small",
    timeout=30.0,
    health_check_timeout=5.0,
    retry_config=RetryConfig(max_retries=3, initial_delay=1.0),
)
```

### Anthropic

```python
from beliefstate.adapters import AnthropicAdapter

adapter = AnthropicAdapter(
    model="claude-3-5-sonnet-latest",
    default_max_tokens=1024,
    timeout=30.0,
)
# Note: Use OpenAI or Ollama as internal_adapter for embeddings
```

### Google Gemini

```python
from beliefstate.adapters import GeminiAdapter

adapter = GeminiAdapter(
    model="gemini-2.0-flash",
    embed_model="text-embedding-004",
    safety_settings=[{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}],
)
```

### Ollama (Local)

```python
from beliefstate.adapters import OllamaAdapter

adapter = OllamaAdapter(
    model="llama3.2",
    embed_model="nomic-embed-text",
    host="http://localhost",
    port=11434,
)
```

### LiteLLM (Multi-Provider)

```python
from beliefstate.adapters import LiteLLMAdapter

# Route to any of 100+ providers via a unified interface
adapter = LiteLLMAdapter(model="azure/gpt-4o", embed_model="cohere/embed-english-v3.0")
```

---

## Dual-Adapter Architecture

Use an expensive model for your application and a cheap/local model for background tracking:

```python
from beliefstate import BeliefTracker
from beliefstate.adapters import AnthropicAdapter, OllamaAdapter

# Your app uses Claude
app_adapter = AnthropicAdapter(model="claude-3-5-sonnet-latest")

# Background tracker uses local Llama 3 (free, no API costs)
bg_adapter = OllamaAdapter(model="llama3", embed_model="nomic-embed-text")

tracker = BeliefTracker(
    adapter=app_adapter,
    internal_adapter=bg_adapter,  # Runs extraction, embeddings, judge calls
)
```

---

## Stores

| Store | Use Case | Configuration |
|-------|----------|---------------|
| **SQLite** | Single-server production apps | `store_kwargs={"db_path": "beliefs.db"}` |
| **PostgreSQL** | Multi-server, high-concurrency | `store_kwargs={"dsn": "postgresql://..."}` |
| **Redis** | Distributed caching, multiple workers | `store_kwargs={"redis_url": "redis://localhost:6379/0"}` |
| **Memory** | Testing, transient sessions | `store_type="memory"` |

All stores implement the same interface and include full audit trails, case-insensitive lookup, and `conversation_id` scoping.

---

## Framework Integrations

### FastAPI

```python
from fastapi import FastAPI
from beliefstate import FastAPIBeliefTrackerMiddleware

app = FastAPI()
app.add_middleware(FastAPIBeliefTrackerMiddleware, header_name="X-Session-ID")
```

### Flask

```python
from flask import Flask
from beliefstate import FlaskBeliefTrackerMiddleware, register_flask_hooks

app = Flask(__name__)
app.wsgi_app = FlaskBeliefTrackerMiddleware(app.wsgi_app, header_name="X-Session-ID")
register_flask_hooks(app, header_name="X-Session-ID")
```

### LangChain

```python
from beliefstate import session_context, BeliefTrackerLangchainCallback

session_context.set("user_123")
handler = BeliefTrackerLangchainCallback(tracker=tracker)
await llm.ainvoke("Hello!", config={"callbacks": [handler]})
```

### LlamaIndex

```python
from llama_index.core import Settings
from beliefstate import LlamaIndexBeliefTrackerCallback

Settings.callback_manager = CallbackManager([
    LlamaIndexBeliefTrackerCallback(tracker=tracker)
])
```

### OpenAI Assistants

```python
from beliefstate import observe_run

asyncio.create_task(observe_run(
    tracker=tracker, client=client,
    thread_id="thread_123", run_id="run_abc",
    session_id="user_123"
))
```

---

## Background Dispatchers

For durable background tracking that survives server restarts, use Celery or Redis Queue:

```python
# Celery
from celery import Celery
from beliefstate.dispatcher import CeleryDispatcher

celery_app = Celery("tasks", broker="redis://localhost:6379/0")
tracker = BeliefTracker(adapter=adapter, dispatcher=CeleryDispatcher(celery_app))

# Redis Queue
from redis import Redis
from rq import Queue
from beliefstate.dispatcher import RQDispatcher

queue = Queue("beliefs", connection=Redis())
tracker = BeliefTracker(adapter=adapter, dispatcher=RQDispatcher(queue=queue))
```

Register the tracker in your worker process:

```python
from beliefstate.dispatcher import register_global_tracker
from my_app import tracker
register_global_tracker(tracker)
```

---

## Configuration Reference

```python
from beliefstate import TrackerConfig
from beliefstate.adapters import RetryConfig

config = TrackerConfig(
    # Store
    store_type="sqlite",                    # sqlite | redis | postgres | memory
    store_kwargs={"db_path": "beliefs.db"},

    # Extraction
    extraction_model="gpt-4o-mini",         # model for background extraction
    embedding_model="text-embedding-3-small",
    max_beliefs_per_turn=50,
    similarity_threshold=0.7,

    # Background tasks
    enable_background_tasks=True,
    max_concurrent_extractions=10,

    # Resilience
    retry_max_attempts=5,
    retry_min_wait=2.0,
    retry_max_wait=30.0,
    retry_multiplier=2.0,

    # Circuit breaker
    enable_circuit_breaker=True,
    circuit_breaker_failure_threshold=5,
    circuit_breaker_recovery_timeout=30.0,
)
```

See `beliefstate/tracker.py` for the full `TrackerConfig` schema.

---

## API Reference

### BeliefTracker

| Method | Description |
|--------|-------------|
| `set_session(session_id, conversation_id)` | Set the active session context |
| `wrap(fn)` | Decorator that intercepts LLM calls and extracts beliefs |
| `track(call, response, session_id)` | Manually track a single LLM call/response |
| `shutdown(grace_seconds)` | Gracefully drain background tasks and close the store |

### Store Interface

| Method | Description |
|--------|-------------|
| `add_belief(session_id, belief)` | Insert or update a belief |
| `get_beliefs(session_id)` | List all beliefs for a session |
| `search_beliefs(session_id, query, top_k)` | Semantic search over beliefs |
| `get_by_key(session_id, subject, predicate)` | Get a specific belief |
| `remove_belief(session_id, subject, predicate)` | Delete a belief |
| `get_audit_history(session_id, subject, predicate)` | Full mutation history |
| `health_check()` | Verify store connectivity |

### Adapter Interface

| Method | Description |
|--------|-------------|
| `generate(call, response_format)` | Generate a completion |
| `embed(texts)` | Embed a list of texts |
| `health_check()` | Verify provider connectivity |
| `inject_context(prompt)` | Inject context into a prompt |

---

## Development

```bash
git clone https://github.com/abhay-2108/beliefstate.git
cd beliefstate
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check beliefstate/
ruff format beliefstate/

# Type check
mypy beliefstate/
```

---

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests (`pytest`) and linter (`ruff check`)
5. Submit a pull request

---

## License

MIT License. See [LICENSE](LICENSE) for details.
