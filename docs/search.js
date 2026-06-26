// ── SEARCH INDEX DATABASE ──
const searchIndex = [
  // Getting Started
  {
    category: "GETTING STARTED",
    title: "Introduction",
    url: "index.html",
    desc: "Overview of beliefstate, an asynchronous, zero-latency belief state tracking layer for Python LLM conversations."
  },
  {
    category: "GETTING STARTED",
    title: "Installation",
    url: "index.html#installation",
    desc: "How to install beliefstate core and optional extras like [openai], [redis], [fastapi], [celery], or [all] using pip."
  },
  {
    category: "GETTING STARTED",
    title: "Quickstart Guide",
    url: "index.html#quickstart",
    desc: "Add a single @tracker.wrap decorator to start tracking beliefs, preferences, and facts in your LLM completions."
  },
  {
    category: "GETTING STARTED",
    title: "Why beliefstate?",
    url: "why.html",
    desc: "Detailed explanation of why stateless LLMs cause context window pollution, token inflation, and silent contradictions."
  },
  {
    category: "GETTING STARTED",
    title: "Context Window is Not Memory",
    url: "why.html#context-vs-memory",
    desc: "Why expanding context windows fail due to 'lost in the middle' recall degradation, latency increase, and truncation risks."
  },
  {
    category: "GETTING STARTED",
    title: "Design Patterns (Customer Support & Coding Agents)",
    url: "why.html#design-patterns",
    desc: "Real-world patterns showing how to keep customer support bots and developer coding agents consistent throughout long sessions."
  },
  {
    category: "GETTING STARTED",
    title: "Comparison: Vector DBs vs. BeliefState",
    url: "why.html#architecture-advantage",
    desc: "Comparison table explaining how beliefstate structured memory differs from raw vector search (RAG) and passive stores."
  },

  // Configuration
  {
    category: "CONFIGURATION",
    title: "TrackerConfig Reference",
    url: "configuration.html#tracker-config",
    desc: "Complete documentation for Pydantic BaseModel configuration options including thresholds, backoff, and dispatchers."
  },
  {
    category: "CONFIGURATION",
    title: "Store Settings",
    url: "configuration.html#tracker-config",
    desc: "Configure storage types ('sqlite' or 'redis') and store connection kwargs (db_path or redis_url)."
  },
  {
    category: "CONFIGURATION",
    title: "Similarity & Contradiction Thresholds",
    url: "configuration.html#tracker-config",
    desc: "Tune similarity_threshold, contradiction_threshold, and entailment_threshold for NLI judge and semantic gates."
  },
  {
    category: "CONFIGURATION",
    title: "Resilience & Retry Settings",
    url: "configuration.html#tracker-config",
    desc: "Configure retry_max_attempts, retry_min_wait, retry_max_wait, and circuit breaker recovery options."
  },
  {
    category: "CONFIGURATION",
    title: "Task & Dispatcher Settings",
    url: "configuration.html#tracker-config",
    desc: "Set enable_background_tasks and task_dispatcher_type ('asyncio', 'sync', 'celery', or 'rq')."
  },
  {
    category: "CONFIGURATION",
    title: "Belief TTL (Time-to-Live)",
    url: "configuration.html#tracker-config",
    desc: "Enable automatic pruning of old beliefs older than belief_max_age_seconds from the database store."
  },
  {
    category: "CONFIGURATION",
    title: "Staleness Scoring & Decay Formula",
    url: "configuration.html#tracker-config",
    desc: "Decay formula (Confidence / (Days + 1)) to deprioritize older beliefs during prompt injection without deleting them."
  },
  {
    category: "CONFIGURATION",
    title: "Token-Aware Injection Cascade",
    url: "configuration.html#tracker-config",
    desc: "Automatically filter and rank beliefs by cosine relevance with current user message when prompt exceeds token budget."
  },
  {
    category: "CONFIGURATION",
    title: "Custom Prompt Templates",
    url: "configuration.html#prompts",
    desc: "Override extract_user_prompt_template, extract_assistant_prompt_template, and NLI judge_prompt_template."
  },

  // Adapters
  {
    category: "ADAPTERS",
    title: "Choosing an Adapter",
    url: "adapters.html#adapters",
    desc: "Normalizes SDK request payloads and handles internal operations. Comparison table of OpenAI, Claude, Gemini, Ollama, LiteLLM."
  },
  {
    category: "ADAPTERS",
    title: "OpenAI Adapter",
    url: "adapters.html#openai",
    desc: "Support for GPT models and embeddings (text-embedding-3-small) with custom connection endpoints."
  },
  {
    category: "ADAPTERS",
    title: "Anthropic Claude Adapter",
    url: "adapters.html#anthropic",
    desc: "Integration with Claude SDK. Note: Anthropic does not provide an embedding API; requires decoupled embeddings."
  },
  {
    category: "ADAPTERS",
    title: "Anthropic Embeddings Exception & Resolution",
    url: "adapters.html#anthropic",
    desc: "Documenting the NotImplementedError raised when using Anthropic without an internal_adapter, and how to fix it."
  },
  {
    category: "ADAPTERS",
    title: "Google Gemini Adapter",
    url: "adapters.html#gemini",
    desc: "Support for Gemini 2.0 Generative AI models and native Google text-embedding-004."
  },
  {
    category: "ADAPTERS",
    title: "Ollama Local Adapter",
    url: "adapters.html#ollama",
    desc: "Completely offline belief tracking and embeddings using locally running Llama 3 and nomic-embed-text."
  },
  {
    category: "ADAPTERS",
    title: "LiteLLM Adapter",
    url: "adapters.html#litellm",
    desc: "Unified routing to 100+ providers like Azure OpenAI, AWS Bedrock, Cohere, and Vertex AI."
  },
  {
    category: "ADAPTERS",
    title: "RetryConfig Strategy",
    url: "adapters.html#retry-config",
    desc: "Control delay calculations using exponential base, max delay, and random jitter to prevent thundering herd."
  },
  {
    category: "ADAPTERS",
    title: "Dual-Adapter Architecture",
    url: "adapters.html#dual-adapter",
    desc: "Decouple main application SDK (premium Claude) and background tracking operations (free local Ollama Llama 3)."
  },

  // Stores
  {
    category: "STORES",
    title: "Choosing a Store",
    url: "stores.html#stores",
    desc: "Store backend comparisons. SQLite for local dev, PostgreSQL for cloud databases, Redis for multi-worker, In-Memory for tests."
  },
  {
    category: "STORES",
    title: "Store Protocol Methods",
    url: "stores.html#stores",
    desc: "CRUD operations defined in the Store protocol: add_belief, get_beliefs, search_beliefs, remove_belief, clear, count."
  },
  {
    category: "STORES",
    title: "SQLite Store Backend",
    url: "stores.html#sqlite-store",
    desc: "Async database access using aiosqlite, write-ahead logging (WAL) mode, and python-side cosine search."
  },
  {
    category: "STORES",
    title: "PostgreSQL Store Backend",
    url: "stores.html#postgresql-store",
    desc: "Multi-worker safe persistence using asyncpg and native PL/pgSQL database-side cosine vector function."
  },
  {
    category: "STORES",
    title: "Redis Store Backend",
    url: "stores.html#redis-store",
    desc: "Distributed caching using redis.asyncio, namespace hash structures, and native session TTL expiration."
  },
  {
    category: "STORES",
    title: "In-Memory Store Backend",
    url: "stores.html#memory-store",
    desc: "Process-level OrderedDict store with LRU eviction policy based on byte-size estimation."
  },

  // Integrations
  {
    category: "INTEGRATIONS",
    title: "FastAPI Middleware & Dependency",
    url: "integrations.html#fastapi",
    desc: "FastAPIBeliefTrackerMiddleware extracts session IDs from headers. get_session_id dependency propagates contextvars."
  },
  {
    category: "INTEGRATIONS",
    title: "FastAPI Manual set_session Pattern",
    url: "integrations.html#fastapi",
    desc: "How to manually call tracker.set_session(session_id) inside a route handler when the session ID is passed in request body JSON."
  },
  {
    category: "INTEGRATIONS",
    title: "Flask WSGI Middleware & Hooks",
    url: "integrations.html#flask",
    desc: "WSGI middleware wrapper and register_flask_hooks using before_request and teardown_request lifecycles."
  },
  {
    category: "INTEGRATIONS",
    title: "Generic ASGI Middleware",
    url: "integrations.html#asgi",
    desc: "Use BeliefTrackerASGIMiddleware with any ASGI-compatible framework like Starlette, Litestar, Quart, etc."
  },
  {
    category: "INTEGRATIONS",
    title: "LangChain Callback Handler",
    url: "integrations.html#langchain",
    desc: "Attach BeliefTrackerLangchainCallback to capture on_llm_end events and dispatch background extraction."
  },
  {
    category: "INTEGRATIONS",
    title: "LlamaIndex Callback Handler",
    url: "integrations.html#llamaindex",
    desc: "Attach LlamaIndexBeliefTrackerCallback to global Settings.callback_manager to track index completions."
  },
  {
    category: "INTEGRATIONS",
    title: "OpenAI Assistants Run Observer",
    url: "integrations.html#openai-assistants",
    desc: "Poll OpenAI Assistants thread runs in background and extract beliefs in bulk after run completes."
  },

  // Advanced
  {
    category: "ADVANCED",
    title: "Background Workers & Dispatchers",
    url: "advanced.html#dispatchers",
    desc: "Offload tracking tasks from standard asyncio event loop to Celery or RQ (Redis Queue) distributed workers."
  },
  {
    category: "ADVANCED",
    title: "Resilience Patterns",
    url: "advanced.html#resilience",
    desc: "Exponential backoff with random jitter and circuit breaker state machines (CLOSED, OPEN, HALF-OPEN)."
  },
  {
    category: "ADVANCED",
    title: "Observability (OTel Tracing & Metrics)",
    url: "advanced.html#observability",
    desc: "Instrument code with setup_otel() to log hierarchical spans (extract, detect, resolve) and counter metrics."
  },
  {
    category: "ADVANCED",
    title: "GDPR Compliance & Deletion",
    url: "advanced.html#gdpr",
    desc: "Erasure workflow via clear_session() which drains active background tasks first to prevent race writes."
  },

  // Help & API
  {
    category: "HELP & API",
    title: "API Reference",
    url: "reference.html#api-reference",
    desc: "Full method signatures for BeliefTracker, ProviderAdapter, Store, and Pydantic models."
  },
  {
    category: "HELP & API",
    title: "Troubleshooting & FAQ",
    url: "reference.html#faq",
    desc: "Solutions for GenericAdapter fallbacks, Ollama timeouts, GDPR deletion races, and rate-limit preventions."
  },
  {
    category: "HELP & API",
    title: "Changelog",
    url: "reference.html#changelog",
    desc: "Release history of beliefstate (v1.1.0 OSS hardening, v1.0.2 Postgres/Streaming, v1.0.1 Pronoun mapping, v1.0.0 Concurrency locks)."
  }
];

// ── SEARCH MODAL CONTROLLER ──
document.addEventListener("DOMContentLoaded", () => {
  const triggerBtn = document.getElementById("searchTriggerBtn");
  const backdrop = document.getElementById("searchBackdrop");
  const closeBtn = document.getElementById("searchCloseBtn");
  const searchInput = document.getElementById("searchInput");
  const resultsContainer = document.getElementById("searchResults");

  let selectedIndex = -1;
  let currentResults = [];

  if (!backdrop || !searchInput) return;

  // Open Modal
  const openSearch = () => {
    backdrop.classList.add("open");
    searchInput.value = "";
    resultsContainer.innerHTML = `<div class="search-no-results">Type query to search docs...</div>`;
    currentResults = [];
    selectedIndex = -1;
    setTimeout(() => searchInput.focus(), 50);
  };

  // Close Modal
  const closeSearch = () => {
    backdrop.classList.remove("open");
  };

  // Event Listeners
  if (triggerBtn) triggerBtn.addEventListener("click", openSearch);
  if (closeBtn) closeBtn.addEventListener("click", closeSearch);
  backdrop.addEventListener("click", (e) => {
    if (e.target === backdrop) closeSearch();
  });

  // Shortcut key (Ctrl + K or /)
  window.addEventListener("keydown", (e) => {
    if ((e.ctrlKey && e.key.toLowerCase() === "k") || (e.key === "/" && document.activeElement !== searchInput)) {
      e.preventDefault();
      openSearch();
    }
    if (e.key === "Escape" && backdrop.classList.contains("open")) {
      closeSearch();
    }
  });

  // Search Logic
  searchInput.addEventListener("input", () => {
    const query = searchInput.value.toLowerCase().trim();
    if (!query) {
      resultsContainer.innerHTML = `<div class="search-no-results">Type query to search docs...</div>`;
      currentResults = [];
      selectedIndex = -1;
      return;
    }

    // Simple keyword filtering
    const tokens = query.split(/\s+/);
    currentResults = searchIndex.filter(item => {
      const title = item.title.toLowerCase();
      const desc = item.desc.toLowerCase();
      const cat = item.category.toLowerCase();
      return tokens.every(token => title.includes(token) || desc.includes(token) || cat.includes(token));
    });

    renderResults();
  });

  // Render Results UI
  const renderResults = () => {
    if (currentResults.length === 0) {
      resultsContainer.innerHTML = `<div class="search-no-results">No results found. Try another query.</div>`;
      selectedIndex = -1;
      return;
    }

    resultsContainer.innerHTML = currentResults.map((item, idx) => `
      <a href="${item.url}" class="search-result-item ${idx === selectedIndex ? 'selected' : ''}" data-index="${idx}">
        <div class="search-result-category">${item.category}</div>
        <div class="search-result-title">${item.title}</div>
        <div class="search-result-desc">${item.desc}</div>
      </a>
    `).join("");

    // Add click listeners to items
    const items = resultsContainer.querySelectorAll(".search-result-item");
    items.forEach(item => {
      item.addEventListener("click", () => {
        closeSearch();
      });
    });
  };

  // Keyboard navigation inside search results
  searchInput.addEventListener("keydown", (e) => {
    if (currentResults.length === 0) return;

    if (e.key === "ArrowDown") {
      e.preventDefault();
      selectedIndex = (selectedIndex + 1) % currentResults.length;
      updateSelection();
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      selectedIndex = (selectedIndex - 1 + currentResults.length) % currentResults.length;
      updateSelection();
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (selectedIndex >= 0 && selectedIndex < currentResults.length) {
        const url = currentResults[selectedIndex].url;
        closeSearch();
        window.location.href = url;
      }
    }
  });

  const updateSelection = () => {
    const items = resultsContainer.querySelectorAll(".search-result-item");
    items.forEach((item, idx) => {
      if (idx === selectedIndex) {
        item.classList.add("selected");
        item.scrollIntoView({ block: "nearest" });
      } else {
        item.classList.remove("selected");
      }
    });
  };
});
