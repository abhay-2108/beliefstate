# BeliefState: Developer Reference Guide

**BeliefState** is a lightweight, zero-latency, production-grade belief state tracking package for Python LLM applications. It runs silently in the background, intercepting your LLM chats to extract factual beliefs (facts), detect semantic contradictions, and persist them to durable databases.

---

## 📖 Table of Contents
1. [Core Concepts](#-core-concepts)
2. [Key Features](#-key-features)
3. [Architecture Overview](#-architecture-overview)
4. [Quickstart Guide](#-quickstart-guide)
5. [Class Reference](#-class-reference)
   - [BeliefTracker](#1-belieftracker)
   - [Provider Adapters](#2-provider-adapters)
   - [Belief Stores](#3-belief-stores)
   - [Task Dispatchers](#4-task-dispatchers)
6. [Resilience & Performance Options](#-resilience--performance-options)
   - [Exponential Backoff & Circuit Breakers](#1-exponential-backoff--circuit-breakers)
   - [Embedding Batching](#2-embedding-batching)
7. [Framework Integrations](#-framework-integrations)
   - [FastAPI / ASGI](#1-fastapi--asgi)
   - [Flask / WSGI](#2-flask--wsgi)
   - [LangChain Callback](#3-langchain-callback)
8. [Production Deployment Guide](#-production-deployment-guide)

---

## 🧠 Core Concepts

When building conversational agents, maintaining a persistent, conflict-free state of what the user has claimed (their "beliefs") is vital. Relying solely on the LLM's conversation history leads to:
*   **Context Window Pollution**: The history becomes too long to fit or becomes expensive.
*   **Silent Contradictions**: The user claims something new that contradicts an earlier claim (e.g. "I hate Python" vs "I write Python code"), and the LLM gets confused.
*   **Volatile Memory**: If the session resets, all user preferences are permanently lost.

**BeliefState** intercepts the conversation turn, extracts claims as structured triples `(Subject, Predicate, Value)` (e.g. `("USER", "likes", "Python")`), computes embeddings, checks for semantic similarities with previous claims, runs an NLI (Natural Language Inference) check to resolve contradictions, and updates the store.

---

## ✨ Key Features

1.  **Zero-Latency Design**: All belief extraction, embedding calculation, and conflict resolutions run asynchronously in the background so your user gets the LLM response instantly without blocking.
2.  **Dual-Adapter Architecture**: Inject different adapters. Run your user-facing app on Anthropic's Claude, but configure the background tracking pipeline on OpenAI or a local Ollama model to save costs.
3.  **Durable Multi-node Stores**: Supports SQLite for single-instance setups (with `:memory:` for testing) and Redis for high-scale, load-balanced worker clusters.
4.  **Bulletproof Resilience**: Powered by `tenacity` retries with exponential backoff and a stateful circuit breaker to avoid freezing your app during model API outages.
5.  **Pluggable Background Dispatchers**: Seamless integration with **Celery** or **Redis Queue (RQ)** to enqueue background tracking tasks onto persistent queues.
6.  **Embedding Batching**: Merges multiple individual embedding requests into a single batch payload to prevent rate limiting, with a robust fallback to individual requests on failure.

---

## 📐 Architecture Overview

```
                          ┌──────────────────────────┐
                          │    Main Application      │
                          └─────────────┬────────────┘
                                        │
                                        ▼ (@tracker.wrap)
                          ┌──────────────────────────┐
                          │      BeliefTracker       │
                          └─────────────┬────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
              ┌─────────────────────┐       ┌─────────────────────┐
              │   App Adapter       │       │  Internal Adapter   │
              │  (User-facing LLM)  │       │ (Resilience wrapped)│
              └─────────────────────┘       └──────────┬──────────┘
                                                       │
                                                       ▼ (Task Dispatcher)
                                            ┌─────────────────────┐
                                            │    Task Queue       │
                                            │   (Celery / RQ)     │
                                            └──────────┬──────────┘
                                                       │
                                                       ▼ (Worker execution)
                                            ┌─────────────────────┐
                                            │   BeliefExtractor   │
                                            └──────────┬──────────┘
                                                       │
                                                       ▼ (Batch Embeddings)
                                            ┌─────────────────────┐
                                            │ContradictionDetector│
                                            └──────────┬──────────┘
                                                       │
                                                       ▼
                                            ┌─────────────────────┐
                                            │   BeliefResolver    │
                                            └──────────┬──────────┘
                                                       │
                                                       ▼
                                            ┌─────────────────────┐
                                            │    BeliefStore      │
                                            │   (SQLite/Redis)    │
                                            └─────────────────────┘
```

---

## 🛠️ Quickstart Guide

### 1. Installation
Install the core package, along with any optional production extras:
```bash
# Core package only
pip install beliefstate

# Install with all extras
pip install "beliefstate[redis,celery,rq,litellm]"
```

### 2. Basic SQLite Integration
```python
import asyncio
from beliefstate import BeliefTracker, TrackerConfig
from beliefstate.adapters import OpenAIAdapter

# 1. Configuration
config = TrackerConfig(
    store_type="sqlite",
    store_kwargs={"db_path": "beliefs.db"},
    enable_background_tasks=True
)

# 2. Setup adapter & tracker
adapter = OpenAIAdapter(model="gpt-4o", embed_model="text-embedding-3-small")
tracker = BeliefTracker(config=config, adapter=adapter)

# 3. Decorate your LLM call
@tracker.wrap
async def call_assistant(messages):
    import openai
    client = openai.AsyncOpenAI()
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return response

# 4. Execute a conversation
async def main():
    tracker.set_session("session_abc_123")
    
    messages = [{"role": "user", "content": "Hello! I am John, and I prefer dark mode."}]
    res = await call_assistant(messages=messages)
    print("AI Response:", res.choices[0].message.content)
    
    # Check what beliefs were saved
    await asyncio.sleep(1.0) # wait briefly for background task to complete
    beliefs = await tracker.store.get_beliefs("session_abc_123")
    for b in beliefs:
         print(f"Stored Fact: [{b.subject}] {b.predicate} '{b.value}'")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 📜 Class Reference

### 1. `BeliefTracker`
The primary orchestrator of the package.

*   `__init__(self, config: TrackerConfig, adapter: ProviderAdapter, store: Optional[Store] = None, internal_adapter: Optional[ProviderAdapter] = None, dispatcher: Optional[TaskDispatcher] = None)`
    *   `config`: Configuration instance.
    *   `adapter`: User-facing LLM adapter.
    *   `store`: Database store instance (defaults to `SQLiteStore`).
    *   `internal_adapter`: Adapter for belief operations (defaults to `adapter`).
    *   `dispatcher`: Background task dispatcher (defaults to `AsyncioDispatcher`).
*   `wrap(self, func)`: Decorator to wrap async LLM calls.
*   `set_session(self, session_id: str)`: Sets the current session ID in the thread-local context.
*   `get_pending_conflicts(self, session_id: Optional[str] = None) -> List[str]`: Retrieves and pops queued contradiction warning strings to inject into the next user prompt.

### 2. Provider Adapters
Adapters implement the `ProviderAdapter` Protocol to standardise native SDK payloads:
*   `OpenAIAdapter`: Uses the official `AsyncOpenAI` client.
*   `GeminiAdapter`: Uses the official `google-genai` client.
*   `OllamaAdapter`: Uses local Ollama servers via `ollama.AsyncClient`.
*   `AnthropicAdapter`: Uses the `AsyncAnthropic` client. (Cannot generate embeddings natively; must be combined with OpenAI or Ollama internal adapters).
*   `LiteLLMAdapter`: Uses `litellm` library to route completion and embedding requests to any supported provider (e.g. Azure, Bedrock, Anthropic, OpenAI, Cohere, etc.). Extremely useful for unified, multi-provider enterprise architectures.


### 3. Belief Stores
Durable backends implementing `Store`:
*   `SQLiteStore`: Uses `aiosqlite` for local persistent storage. (Pass `db_path=":memory:"` for tests).
*   `RedisStore`: Uses asynchronous `redis` to store beliefs as serialized hashes. Perfect for distributed systems.

### 4. Task Dispatchers
Pluggable strategies for background execution:
*   `AsyncioDispatcher`: Runs tasks asynchronously in the current loop.
*   `SyncDispatcher`: Runs tasks blocking in the execution thread.
*   `CeleryDispatcher`: Pushes serialized payloads to a Celery queue via `send_task()`.
*   `RQDispatcher`: Pushes serialized payloads directly to an RQ Queue.

---

## 🛡️ Resilience & Performance Options

### 1. Exponential Backoff & Circuit Breakers
BeliefState wraps your internal LLM adapter calls inside a `ResilientAdapterWrapper`. When transient errors (like HTTP 429 Rate Limits, HTTP 502/503 Gateways, or DNS timeouts) occur:
*   **Tenacity Retries**: It retries the API request with exponential backoff up to `retry_max_attempts`.
*   **Fail-fast Circuit Breaker**: If the adapter fails consecutively beyond the `circuit_breaker_failure_threshold`, the circuit breaker trips to `OPEN`. Subsequent calls fail-fast immediately without invoking the network, protecting your app's capacity. After `circuit_breaker_recovery_timeout` (default `30s`), the circuit enters `HALF-OPEN` to test recovery.

Configure these in `TrackerConfig`:
```python
config = TrackerConfig(
    retry_max_attempts=5,
    retry_min_wait=2.0,
    retry_max_wait=30.0,
    enable_circuit_breaker=True,
    circuit_breaker_failure_threshold=5,
    circuit_breaker_recovery_timeout=30.0
)
```

### 2. Embedding Batching
Instead of making single HTTP requests for each extracted belief, `BeliefExtractor` collects all valid beliefs and submits them to `adapter.get_embeddings(texts)` in a single batch call.
*   **Fallback Resolution**: If the batch embedding API call raises an exception (e.g. payload too large), the extractor automatically falls back to requesting embeddings individually, ensuring no beliefs are ever lost.

---

## 🔌 Framework Integrations

### 1. FastAPI / ASGI
The `BeliefTrackerASGIMiddleware` automatically extracts a session or user ID from incoming request headers and registers it into the tracker's context.

```python
from fastapi import FastAPI
from beliefstate.integrations.asgi import BeliefTrackerASGIMiddleware

app = FastAPI()
app.add_middleware(
    BeliefTrackerASGIMiddleware, 
    header_name="X-Session-ID"
)
```

### 2. Flask / WSGI
The `BeliefTrackerWSGIMiddleware` maps session context variables from standard WSGI environ environments:

```python
from flask import Flask
from beliefstate.integrations.wsgi import BeliefTrackerWSGIMiddleware

app = Flask(__name__)
app.wsgi_app = BeliefTrackerWSGIMiddleware(
    app.wsgi_app, 
    header_name="X-Session-ID"
)
```

### 3. LangChain Callback
Allows seamless interception of LangChain chain executions without wrapping functions manually:

```python
from langchain_openai import ChatOpenAI
from beliefstate import session_context
from beliefstate.integrations.langchain import BeliefTrackerLangchainCallback

# Set session ID context
session_context.set("user_123")

callback = BeliefTrackerLangchainCallback(tracker=tracker)
model = ChatOpenAI(callbacks=[callback])
```

---

## 🚢 Production Deployment Guide

When deploying to production, we recommend using a persistent task queue to safeguard against process crashes.

### Step 1: Initialize the Tracker with Celery/RQ Dispatcher
Configure the web server to enqueue background tasks:

```python
# app.py
from celery import Celery
from beliefstate import BeliefTracker, TrackerConfig
from beliefstate.dispatcher import CeleryDispatcher

celery_app = Celery("my_app", broker="redis://localhost:6379/0")

config = TrackerConfig(store_type="redis")
tracker = BeliefTracker(
    config=config,
    adapter=app_adapter,
    dispatcher=CeleryDispatcher(celery_app=celery_app)
)
```

### Step 2: Configure the Worker Process
Background workers must be registered with the global tracker so they can successfully execute enqueued belief tasks.

```python
# tasks.py (Celery Worker Startup)
from app import celery_app, tracker
from beliefstate.dispatcher import register_global_tracker

# 1. Register the global tracker
register_global_tracker(tracker)

# 2. Define the celery task enqueued by the dispatcher
@celery_app.task(name="beliefstate.dispatcher.execute_tracking_task")
def celery_tracking_worker(call_dict, response_dict, session_id, turn):
    from beliefstate.dispatcher import execute_tracking_task
    execute_tracking_task(call_dict, response_dict, session_id, turn)
```

Start the Celery worker normally:
```bash
celery -A tasks worker --loglevel=info
```
