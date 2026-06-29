<h1 align="center">BeliefState</h1>

<p align="center">
  Persistent memory for LLM applications.<br>
  Extract facts, resolve contradictions, and recall knowledge — automatically.
</p>

<p align="center">
  <a href="https://pypi.org/project/beliefstate/"><img src="https://img.shields.io/pypi/v/beliefstate?color=blue" alt="PyPI"></a>
  <a href="https://github.com/abhay-2108/beliefstate/blob/main/LICENSE"><img src="https://img.shields.io/github/license/abhay-2108/beliefstate" alt="License"></a>
  <a href="https://pypi.org/project/beliefstate/"><img src="https://img.shields.io/pypi/pyversions/beliefstate" alt="Python"></a>
  <a href="https://github.com/abhay-2108/beliefstate/actions/workflows/ci.yml"><img src="https://github.com/abhay-2108/beliefstate/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/abhay-2108/beliefstate/actions/workflows/lint.yml"><img src="https://github.com/abhay-2108/beliefstate/actions/workflows/lint.yml/badge.svg" alt="Lint"></a>
  <a href="https://abhay-2108.github.io/beliefstate/"><img src="https://img.shields.io/badge/docs-online-brightgreen" alt="Docs"></a>
</p>

---

## The Problem

Every time a user starts a new conversation, your LLM starts from zero. It doesn't remember that the user is a Rust developer in Berlin, prefers dark mode, or already explained their project architecture three messages ago.

You can dump the full chat history into the system prompt, but that burns tokens fast — and most of it is noise. RAG helps with documents, but **user facts** (preferences, identity, context) fall through the cracks. There's no clean way to say "remember this about the user" without manual bookkeeping.

## How BeliefState Works

BeliefState sits between your LLM and your application. Every time the LLM responds, it silently:

1. **Extracts facts** — "user prefers Rust", "project uses PostgreSQL", "works at Google"
2. **Detects contradictions** — if the user says "I live in Tokyo" after saying "I live in Berlin", it flags the conflict
3. **Stores beliefs** — persisted in SQLite, PostgreSQL, or Redis with full history
4. **Injects context** — on the next call, relevant beliefs are added to the system prompt automatically

All of this happens in the background. Zero added latency to your request path.

```python
from beliefstate import BeliefTracker
from beliefstate.adapters import OpenAIAdapter

tracker = BeliefTracker(
    adapter=OpenAIAdapter(model="gpt-4o"),
    config=TrackerConfig(store_type="sqlite", store_kwargs={"db_path": "beliefs.db"})
)

@tracker.wrap
async def chat(messages):
    return await openai_client.chat.completions.create(model="gpt-4o", messages=messages)

tracker.set_session("user_123")
await chat([{"role": "user", "content": "I live in Tokyo and work at Google."}])
# BeliefState extracts: {subject: "user", predicate: "lives_in", value: "Tokyo"}
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Zero-latency** | Background extraction — no added latency to your request path |
| **5 LLM providers** | OpenAI, Anthropic, Gemini, Ollama, LiteLLM (100+ via LiteLLM) |
| **Dual-adapter** | Expensive model for your app, cheap/local model for tracking |
| **Contradiction detection** | NLI judge resolves conflicting facts gracefully |
| **Persistent stores** | SQLite, PostgreSQL, Redis — with full audit trails |
| **Framework integrations** | LangChain, LlamaIndex, FastAPI, Flask, OpenAI Assistants |
| **Production resilience** | Retry with backoff, circuit breakers, health checks |
| **Pluggable dispatchers** | Celery, Redis Queue — survives server restarts |
| **GDPR-ready** | One-call `clear_session()` with auditable deletion receipts |
| **Observability** | OpenTelemetry traces and metrics — built-in, optional |

---

## Installation

```bash
pip install beliefstate
```

With optional extras:

```bash
pip install "beliefstate[openai]"         # OpenAI adapter
pip install "beliefstate[anthropic]"      # Anthropic adapter
pip install "beliefstate[gemini]"         # Gemini adapter
pip install "beliefstate[ollama]"         # Ollama adapter (local)
pip install "beliefstate[litellm]"        # LiteLLM (100+ providers)
pip install "beliefstate[local]"          # Local embeddings (sentence-transformers)
pip install "beliefstate[redis]"          # Redis store
pip install "beliefstate[postgres]"       # PostgreSQL store
pip install "beliefstate[langchain]"      # LangChain integration
pip install "beliefstate[llamaindex]"     # LlamaIndex integration
pip install "beliefstate[fastapi]"        # FastAPI middleware
pip install "beliefstate[flask]"          # Flask middleware
pip install "beliefstate[all]"            # Everything
```

---

## Documentation

For provider setup, store configuration, framework integrations, advanced usage, and API reference:

**[https://abhay-2108.github.io/beliefstate/](https://abhay-2108.github.io/beliefstate/)**

---

## Quick Example: Dual-Adapter Setup

Use Claude for your app, Llama 3 for background tracking — zero API costs for belief extraction:

```python
from beliefstate import BeliefTracker
from beliefstate.adapters import AnthropicAdapter, OllamaAdapter

tracker = BeliefTracker(
    adapter=AnthropicAdapter(model="claude-3-5-sonnet-latest"),
    internal_adapter=OllamaAdapter(model="llama3", embed_model="nomic-embed-text"),
)
```

---

## Development

```bash
git clone https://github.com/abhay-2108/beliefstate.git
cd beliefstate
pip install -e ".[dev]"
pytest
ruff check beliefstate/
```

---

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on the development setup, coding standards, and pull request process.

This project follows a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold it.

For security vulnerabilities, please see our [Security Policy](SECURITY.md).

For version history, see the [Changelog](CHANGELOG.md).

---

## License

MIT License. See [LICENSE](LICENSE) for details.
