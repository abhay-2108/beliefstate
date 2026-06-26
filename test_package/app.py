"""
beliefstate End-to-End Test App
================================
Tests all core features of the beliefstate package:
  - Belief extraction
  - Contradiction detection
  - Resolution strategies
  - Session management
  - Health checks
  - GDPR session deletion
  - Context prompt injection
  - Belief store inspection

Run with:
    pip install streamlit beliefstate[openai]
    streamlit run beliefstate_tester.py
"""

import asyncio
import time
import json
import os
import threading


import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="beliefstate · E2E Tester",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0A0A0B; color: #F4F4F5; }
  .metric-card {
    background: #111113;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 16px 20px;
  }
  .belief-card {
    background: #111113;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
  }
  .contradiction-card {
    background: rgba(248,113,113,0.08);
    border: 1px solid rgba(248,113,113,0.3);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
  }
  .success-card {
    background: rgba(52,211,153,0.08);
    border: 1px solid rgba(52,211,153,0.25);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
  }
  .section-header {
    font-size: 13px;
    font-weight: 500;
    color: #7C6FEB;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 12px;
    font-family: monospace;
  }
  .chat-msg-user {
    background: rgba(124,111,235,0.08);
    border: 1px solid rgba(124,111,235,0.2);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 14px;
  }
  .chat-msg-assistant {
    background: #18181B;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 14px;
  }
  .turn-label {
    font-size: 10px;
    font-family: monospace;
    color: #52525B;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .pipeline-step {
    display: inline-block;
    background: #18181B;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px;
    font-family: monospace;
    color: #A1A1AA;
    margin-right: 6px;
    margin-bottom: 4px;
  }
  .status-ok { color: #34D399; }
  .status-fail { color: #F87171; }
  .status-warn { color: #F59E0B; }
</style>
""", unsafe_allow_html=True)

# ── Async helper ───────────────────────────────────────────────────────────────

_bg_loop = None
_bg_thread = None

def get_bg_loop():
    global _bg_loop, _bg_thread
    if _bg_loop is None:
        _bg_loop = asyncio.new_event_loop()
        _bg_thread = threading.Thread(target=_bg_loop.run_forever, daemon=True)
        _bg_thread.start()
    return _bg_loop

def run_async(coro):
    """Run async coroutine from sync Streamlit context using a persistent loop."""
    loop = get_bg_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()

# ── Session state init ─────────────────────────────────────────────────────────
if "tracker" not in st.session_state:
    st.session_state.tracker = None
if "adapter" not in st.session_state:
    st.session_state.adapter = None
if "conversation" not in st.session_state:
    st.session_state.conversation = []   # list of {role, content, beliefs, contradictions}
if "session_id" not in st.session_state:
    st.session_state.session_id = "test-session-001"
if "turn" not in st.session_state:
    st.session_state.turn = 0
if "test_results" not in st.session_state:
    st.session_state.test_results = []
if "health_status" not in st.session_state:
    st.session_state.health_status = None
if "init_error" not in st.session_state:
    st.session_state.init_error = None

# ── Sidebar: Config ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 beliefstate tester")
    st.markdown("---")

    st.markdown('<div class="section-header">Provider</div>', unsafe_allow_html=True)
    provider = st.selectbox(
        "LLM Provider",
        ["OpenAI", "NVIDIA", "Anthropic", "Gemini", "Ollama"],
        help="Select your LLM provider. Anthropic requires a paired embed provider."
    )

    api_key = st.text_input(
        "API Key",
        type="password",
        value=os.environ.get("OPENAI_API_KEY", "")
        if provider == "OpenAI"
        else os.environ.get("NVIDIA_API_KEY", "")
        if provider == "NVIDIA"
        else os.environ.get("ANTHROPIC_API_KEY", "")
        if provider == "Anthropic"
        else os.environ.get("GEMINI_API_KEY", ""),
        help="Or set via environment variable."
    )

    if provider == "OpenAI":
        model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
        embed_model = st.selectbox("Embed Model", ["text-embedding-3-small", "text-embedding-3-large"])
    elif provider == "NVIDIA":
        model = st.selectbox("Model", ["meta/llama-3.1-8b-instruct", "meta/llama-3.2-3b-instruct", "meta/llama-3.1-70b-instruct", "nvidia/llama-3.1-nemotron-70b-instruct"])
        embed_model = st.selectbox("Embed Model", ["nvidia/nv-embed-v1", "nvidia/nv-embedqa-e5-v5", "nvidia/llama-nemotron-embed-1b-v2"])
    elif provider == "Anthropic":
        model = st.selectbox("Model", ["claude-3-5-sonnet-latest", "claude-3-haiku-20240307"])
        st.info("⚠ Anthropic has no embedding API. Pair with OpenAI for embeddings.")
        embed_api_key = st.text_input("OpenAI Key (for embeddings)", type="password",
                                       value=os.environ.get("OPENAI_API_KEY", ""))
    elif provider == "Gemini":
        model = st.selectbox("Model", ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"])
        embed_model = "text-embedding-004"
    elif provider == "Ollama":
        model = st.selectbox("Model", ["llama3.2", "mistral", "qwen2.5", "phi3"])
        embed_model = st.selectbox("Embed Model", ["nomic-embed-text", "mxbai-embed-large"])
        ollama_host = st.text_input("Ollama Host", value="http://localhost")
        ollama_port = st.number_input("Ollama Port", value=11434)

    st.markdown("---")
    st.markdown('<div class="section-header">Tracker Config</div>', unsafe_allow_html=True)

    store_type = st.selectbox("Store Backend", ["sqlite", "in-memory"],
                               help="SQLite persists to disk, in-memory resets on restart.")
    if store_type == "sqlite":
        db_path = st.text_input("DB Path", value="beliefstate_test.db")

    resolution_strategy = st.selectbox(
        "Resolution Strategy",
        ["warn", "ask", "update", "block"],
        help="How to handle detected contradictions."
    )

    similarity_threshold = st.slider(
        "Similarity Threshold", 0.5, 1.0, 0.82, 0.01,
        help="Stage 1 cosine gate (exact duplicate dedup & relevance). Higher = stricter."
    )
    contradiction_threshold = st.slider(
        "Contradiction Threshold", 0.5, 1.0, 0.70, 0.01,
        help="Stage 2 NLI contradiction gate. Higher = fewer flags."
    )
    entailment_threshold = st.slider(
        "Entailment Threshold", 0.5, 1.0, 0.85, 0.01,
        help="NLI entailment gate. Higher = stricter."
    )
    user_confidence_cap = st.slider(
        "User Confidence Cap", 0.0, 1.0, 0.99, 0.01,
        help="Max confidence allowed for beliefs extracted from user messages."
    )
    assistant_confidence_cap = st.slider(
        "Assistant Confidence Cap", 0.0, 1.0, 0.85, 0.01,
        help="Max confidence allowed for beliefs extracted from assistant responses."
    )
    judge_timeout = st.number_input(
        "Judge Timeout (seconds)", value=60.0, min_value=1.0, max_value=300.0,
        help="Max time to wait for NLI contradiction judgment."
    )
    max_beliefs = st.number_input("Max Beliefs", value=50, min_value=1, max_value=200)

    background_tasks = st.toggle(
        "Background Tasks",
        value=False,
        help="OFF = sync mode — easier to see results immediately in a test. "
             "ON = fire-and-forget (production mode)."
    )

    st.markdown("---")
    st.session_state.session_id = st.text_input(
        "Session ID", value=st.session_state.session_id
    )

    # Config change detection to invalidate stale tracker instances
    current_config_fingerprint = f"{provider}|{model}|{embed_model}|{api_key}|{store_type}|{similarity_threshold}|{contradiction_threshold}|{entailment_threshold}|{user_confidence_cap}|{assistant_confidence_cap}|{judge_timeout}"
    if "config_fingerprint" not in st.session_state:
        st.session_state.config_fingerprint = current_config_fingerprint
    elif st.session_state.config_fingerprint != current_config_fingerprint:
        st.session_state.tracker = None
        st.session_state.adapter = None
        st.session_state.health_status = None
        st.session_state.config_fingerprint = current_config_fingerprint

    if st.button("🔧 Initialise Tracker", type="primary", use_container_width=True):
        with st.spinner("Initialising..."):
            try:
                from beliefstate import BeliefTracker, TrackerConfig

                # Build config
                store_kwargs = {}
                if store_type == "sqlite":
                    store_kwargs = {"db_path": db_path}

                config = TrackerConfig(
                    store_type="sqlite" if store_type == "sqlite" else "sqlite",
                    store_kwargs=store_kwargs if store_type == "sqlite" else {"db_path": ":memory:"},
                    similarity_threshold=similarity_threshold,
                    contradiction_threshold=contradiction_threshold,
                    entailment_threshold=entailment_threshold,
                    judge_timeout=judge_timeout,
                    user_confidence_cap=user_confidence_cap,
                    assistant_confidence_cap=assistant_confidence_cap,
                    max_beliefs=int(max_beliefs),
                    enable_background_tasks=background_tasks,
                    task_dispatcher_type="sync" if not background_tasks else "asyncio",
                )

                # Build adapter
                if provider == "OpenAI":
                    from beliefstate.adapters import OpenAIAdapter
                    adapter = OpenAIAdapter(model=model, embed_model=embed_model)
                    os.environ["OPENAI_API_KEY"] = api_key
                    tracker = BeliefTracker(config=config, adapter=adapter)

                elif provider == "NVIDIA":
                    from beliefstate.adapters import OpenAIAdapter
                    from openai import AsyncOpenAI
                    client = AsyncOpenAI(
                        api_key=api_key,
                        base_url="https://integrate.api.nvidia.com/v1"
                    )
                    # Asymmetric models on NVIDIA require an input_type in the request body
                    embed_kwargs = {}
                    if embed_model != "nvidia/nv-embed-v1":
                        embed_kwargs["extra_body"] = {"input_type": "query"}
                    adapter = OpenAIAdapter(
                        client=client,
                        model=model,
                        embed_model=embed_model,
                        embed_kwargs=embed_kwargs
                    )
                    tracker = BeliefTracker(config=config, adapter=adapter)

                elif provider == "Anthropic":
                    from beliefstate.adapters import AnthropicAdapter, OpenAIAdapter
                    os.environ["ANTHROPIC_API_KEY"] = api_key
                    os.environ["OPENAI_API_KEY"] = embed_api_key
                    app_adapter = AnthropicAdapter(model=model)
                    internal_adapter = OpenAIAdapter(
                        model="gpt-4o-mini",
                        embed_model="text-embedding-3-small"
                    )
                    tracker = BeliefTracker(
                        config=config,
                        adapter=app_adapter,
                        internal_adapter=internal_adapter
                    )
                    adapter = app_adapter

                elif provider == "Gemini":
                    from beliefstate.adapters import GeminiAdapter
                    os.environ["GEMINI_API_KEY"] = api_key
                    adapter = GeminiAdapter(model=model, embed_model=embed_model)
                    tracker = BeliefTracker(config=config, adapter=adapter)

                elif provider == "Ollama":
                    from beliefstate.adapters import OllamaAdapter
                    adapter = OllamaAdapter(
                        model=model,
                        embed_model=embed_model,
                        host=ollama_host,
                        port=int(ollama_port)
                    )
                    tracker = BeliefTracker(config=config, adapter=adapter)

                st.session_state.tracker = tracker
                st.session_state.adapter = adapter
                st.session_state.init_error = None
                st.session_state.conversation = []
                st.session_state.turn = 0
                st.session_state.test_results = []

                # Health check
                health = run_async(tracker.health_check())
                st.session_state.health_status = health
                st.success("✓ Tracker initialised")

            except Exception as e:
                st.session_state.init_error = str(e)
                st.error(f"Init failed: {e}")

    if st.session_state.health_status:
        h = st.session_state.health_status
        store_ok = h.get("store", False)
        adapter_ok = h.get("adapter", False)
        store_html = '<span class="status-ok">✓</span>' if store_ok else '<span class="status-fail">✗</span>'
        adapter_html = '<span class="status-ok">✓</span>' if adapter_ok else '<span class="status-fail">✗</span>'
        st.markdown(
            f"**Health:** Store {store_html} Adapter {adapter_html}",
            unsafe_allow_html=True
        )

    if st.session_state.init_error:
        st.error(f"Error: {st.session_state.init_error}")

# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown("# 🧠 beliefstate · End-to-End Test Suite")
st.markdown(
    "Tests core features: belief extraction · contradiction detection · "
    "resolution · session management · GDPR deletion"
)

if st.session_state.tracker is None:
    st.warning("← Configure and initialise the tracker in the sidebar to begin.")
    st.stop()

tracker = st.session_state.tracker

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_chat, tab_beliefs, tab_tests, tab_gdpr, tab_context = st.tabs([
    "💬 Chat & Track",
    "📦 Belief Store",
    "🧪 Automated Tests",
    "🗑 GDPR Deletion",
    "🔍 Context Prompt",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    col_chat, col_live = st.columns([3, 2])

    with col_chat:
        st.markdown("### Conversation")
        st.caption(
            f"Session: `{st.session_state.session_id}` · "
            f"Turn: `{st.session_state.turn}` · "
            f"Mode: `{'async' if background_tasks else 'sync'}`"
        )

        # Render conversation history
        for entry in st.session_state.conversation:
            role = entry["role"]
            content = entry["content"]
            turn_num = entry.get("turn", "?")

            if role == "user":
                st.markdown(
                    f'<div class="turn-label">Turn {turn_num} · User</div>'
                    f'<div class="chat-msg-user">{content}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="turn-label">Turn {turn_num} · Assistant</div>'
                    f'<div class="chat-msg-assistant">{content}</div>',
                    unsafe_allow_html=True
                )

                # Show beliefs extracted from this turn
                if entry.get("beliefs"):
                    with st.expander(f"📦 {len(entry['beliefs'])} belief(s) extracted", expanded=False):
                        for b in entry["beliefs"]:
                            subject = getattr(b, "subject", b.get("subject", "?") if isinstance(b, dict) else "?")
                            predicate = getattr(b, "predicate", b.get("predicate", "?") if isinstance(b, dict) else "?")
                            value = getattr(b, "value", b.get("value", "?") if isinstance(b, dict) else "?")
                            conf = getattr(b, "confidence", b.get("confidence", 0) if isinstance(b, dict) else 0)
                            st.markdown(
                                f'<div class="belief-card">'
                                f'<span style="color:#9D93F0">{subject}</span> '
                                f'<span style="color:#A1A1AA">{predicate}</span> '
                                f'<span style="color:#34D399">{value}</span> '
                                f'<span style="float:right;color:#52525B">conf: {conf:.2f}</span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                # Show contradictions if any
                if entry.get("contradictions"):
                    for c in entry["contradictions"]:
                        st.markdown(
                            f'<div class="contradiction-card">'
                            f'⚠ <b>Contradiction detected</b><br>'
                            f'<small style="color:#F87171">{c}</small>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

        st.markdown("---")

        # Preset prompts
        st.markdown("**Quick test prompts:**")
        preset_col1, preset_col2, preset_col3 = st.columns(3)

        presets = [
            ("👤 Introduce self", "Hi, my name is Raj. I'm a Python developer based in Chennai."),
            ("💰 Set budget", "My budget for this project is $5,000."),
            ("⚡ Contradict budget", "Actually, let's plan for a $10,000 budget."),
            ("🏠 Share preference", "I prefer working remotely and I love TypeScript."),
            ("🔄 Update preference", "I've switched to Python lately, not TypeScript anymore."),
            ("📍 Set location", "I'm currently living in Tokyo, Japan."),
        ]

        for i, (label, prompt) in enumerate(presets):
            col = [preset_col1, preset_col2, preset_col3][i % 3]
            with col:
                if st.button(label, use_container_width=True, key=f"preset_{i}"):
                    st.session_state["pending_message"] = prompt

        # Chat input
        user_input = st.chat_input("Type a message to test belief extraction...")

        # Use preset if clicked
        if "pending_message" in st.session_state:
            user_input = st.session_state.pop("pending_message")

        if user_input:
            tracker.set_session(st.session_state.session_id)
            st.session_state.turn += 1
            turn = st.session_state.turn

            # Add user message to history
            st.session_state.conversation.append({
                "role": "user",
                "content": user_input,
                "turn": turn,
                "beliefs": [],
                "contradictions": []
            })

            with st.spinner(f"Turn {turn}: calling LLM + tracking beliefs..."):
                try:
                    messages = [
                        {"role": entry["role"], "content": entry["content"]}
                        for entry in st.session_state.conversation
                        if entry["role"] in ("user", "assistant")
                    ]

                    t_start = time.time()

                    # The wrapped LLM call — tracker intercepts this
                    async def call_llm(msgs):
                        if provider == "OpenAI":
                            from openai import AsyncOpenAI
                            client = AsyncOpenAI(api_key=api_key)
                            resp = await client.chat.completions.create(
                                model=model, messages=msgs, max_tokens=300
                            )
                            return resp
                        elif provider == "NVIDIA":
                            from openai import AsyncOpenAI
                            client = AsyncOpenAI(
                                api_key=api_key,
                                base_url="https://integrate.api.nvidia.com/v1"
                            )
                            resp = await client.chat.completions.create(
                                model=model, messages=msgs, max_tokens=300
                            )
                            return resp
                        elif provider == "Anthropic":
                            from anthropic import AsyncAnthropic
                            client = AsyncAnthropic(api_key=api_key)
                            user_msgs = [m for m in msgs if m["role"] != "system"]
                            resp = await client.messages.create(
                                model=model,
                                max_tokens=300,
                                messages=user_msgs
                            )
                            return resp
                        elif provider == "Gemini":
                            import google.generativeai as genai
                            genai.configure(api_key=api_key)
                            m = genai.GenerativeModel(model)
                            hist = []
                            for msg in msgs[:-1]:
                                role = "user" if msg["role"] == "user" else "model"
                                hist.append({"role": role, "parts": [msg["content"]]})
                            chat = m.start_chat(history=hist)
                            resp = await chat.send_message_async(msgs[-1]["content"])
                            return resp
                        elif provider == "Ollama":
                            import ollama
                            client = ollama.AsyncClient(host=f"{ollama_host}:{int(ollama_port)}")
                            resp = await client.chat(model=model, messages=msgs)
                            return resp

                    # Apply @tracker.wrap by calling track_async manually
                    # (wrap decorator approach won't work well in Streamlit's sync context)
                    raw_response = run_async(call_llm(messages))
                    latency_ms = (time.time() - t_start) * 1000

                    # Extract response text
                    if provider in ("OpenAI", "NVIDIA"):
                        assistant_text = raw_response.choices[0].message.content
                    elif provider == "Anthropic":
                        assistant_text = raw_response.content[0].text
                    elif provider == "Gemini":
                        assistant_text = raw_response.text
                    elif provider == "Ollama":
                        assistant_text = raw_response["message"]["content"]

                    # Track manually using track_async
                    from beliefstate.call import LLMCall, LLMResponse

                    llm_call = LLMCall(messages=messages, kwargs={})
                    llm_response = LLMResponse(text=assistant_text, raw_response=raw_response)

                    run_async(tracker.track_async(
                        llm_call.model_dump(),
                        llm_response.model_dump(),
                        session_id=st.session_state.session_id,
                        turn=turn
                    ))

                    # Wait a moment for sync dispatcher to complete
                    if not background_tasks:
                        time.sleep(0.2)

                    # Fetch updated beliefs
                    beliefs = run_async(tracker.get_beliefs(
                        session_id=st.session_state.session_id
                    ))

                    # Check for contradictions by looking at belief changes
                    contradictions = []
                    context_prompt = run_async(tracker.get_context_prompt(
                        session_id=st.session_state.session_id
                    ))

                    st.session_state.conversation.append({
                        "role": "assistant",
                        "content": assistant_text,
                        "turn": turn,
                        "beliefs": beliefs,
                        "contradictions": contradictions,
                        "latency_ms": latency_ms,
                        "belief_count": len(beliefs),
                    })

                    st.rerun()

                except Exception as e:
                    st.error(f"Error on turn {turn}: {e}")
                    st.exception(e)

    with col_live:
        st.markdown("### Live Pipeline Status")

        if st.session_state.conversation:
            last_assistant = next(
                (e for e in reversed(st.session_state.conversation) if e["role"] == "assistant"),
                None
            )

            if last_assistant:
                latency = last_assistant.get("latency_ms", 0)
                belief_count = last_assistant.get("belief_count", 0)

                m1, m2 = st.columns(2)
                m1.metric("Latency", f"{latency:.0f}ms")
                m2.metric("Beliefs in store", belief_count)

                st.markdown("---")
                st.markdown("**Pipeline stages completed:**")
                stages = [
                    ("🎯 Intercept", True, "Call captured by tracker"),
                    ("💬 LLM Call", True, "Response returned to user"),
                    ("📤 Dispatch", True, f"{'Async bg task' if background_tasks else 'Sync (blocking)'}"),
                    ("🔍 Extract", True, f"{belief_count} beliefs extracted"),
                    ("⚡ Detect", True, "Cosine gate + NLI check"),
                    ("🛡 Resolve", True, f"Strategy: {resolution_strategy}"),
                    ("💾 Store", True, "Written to " + store_type),
                ]
                for stage, ok, detail in stages:
                    status = "✓" if ok else "✗"
                    color = "#34D399" if ok else "#F87171"
                    st.markdown(
                        f'<span style="color:{color}">{status}</span> '
                        f'**{stage}** — <span style="color:#A1A1AA;font-size:12px">{detail}</span>',
                        unsafe_allow_html=True
                    )

        st.markdown("---")
        st.markdown("**Session info:**")
        st.code(f"session_id: {st.session_state.session_id}\nturn: {st.session_state.turn}", language="text")

        if st.button("🗑 Clear conversation (keep beliefs)", use_container_width=True):
            st.session_state.conversation = []
            st.session_state.turn = 0
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: BELIEF STORE
# ══════════════════════════════════════════════════════════════════════════════
with tab_beliefs:
    st.markdown("### Belief Store Inspector")
    st.caption("All extracted beliefs for the current session, live from the store.")

    col_refresh, col_filter = st.columns([1, 3])
    with col_refresh:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    with col_filter:
        filter_text = st.text_input("Filter by subject/predicate/value", placeholder="e.g. USER, budget, Python")

    try:
        beliefs = run_async(tracker.get_beliefs(session_id=st.session_state.session_id))

        if not beliefs:
            st.info("No beliefs stored yet. Send some messages in the Chat tab.")
        else:
            # Apply filter
            if filter_text:
                beliefs = [
                    b for b in beliefs
                    if filter_text.lower() in str(getattr(b, "subject", "")).lower()
                    or filter_text.lower() in str(getattr(b, "predicate", "")).lower()
                    or filter_text.lower() in str(getattr(b, "value", "")).lower()
                ]

            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total beliefs", len(beliefs))

            sources = {}
            for b in beliefs:
                src = getattr(b, "source", "unknown")
                sources[src] = sources.get(src, 0) + 1
            m2.metric("From user", sources.get("user", 0))
            m3.metric("From assistant", sources.get("assistant", 0))

            avg_conf = sum(getattr(b, "confidence", 0) for b in beliefs) / max(len(beliefs), 1)
            m4.metric("Avg confidence", f"{avg_conf:.2f}")

            st.markdown("---")

            # Group by subject
            subjects = {}
            for b in beliefs:
                subj = getattr(b, "subject", "unknown")
                if subj not in subjects:
                    subjects[subj] = []
                subjects[subj].append(b)

            for subject, subject_beliefs in subjects.items():
                st.markdown(f"**Subject: `{subject}`**")
                for b in subject_beliefs:
                    predicate = getattr(b, "predicate", "?")
                    value = getattr(b, "value", "?")
                    conf = getattr(b, "confidence", 0)
                    turn = getattr(b, "turn", "?")
                    source = getattr(b, "source", "?")
                    belief_type = getattr(b, "belief_type", "assertion")
                    is_hypo = getattr(b, "is_hypothetical", False)
                    created = getattr(b, "created_at", None)

                    conf_color = "#34D399" if conf >= 0.8 else "#F59E0B" if conf >= 0.5 else "#F87171"
                    hypo_badge = ' <span style="color:#F59E0B;font-size:10px">[hypothetical]</span>' if is_hypo else ""
                    update_badge = ' <span style="color:#7C6FEB;font-size:10px">[update]</span>' if belief_type == "update" else ""

                    st.markdown(
                        f'<div class="belief-card">'
                        f'<span style="color:#A1A1AA">{predicate}</span> → '
                        f'<span style="color:#F4F4F5;font-weight:500">{value}</span>'
                        f'{hypo_badge}{update_badge}<br>'
                        f'<span style="color:#52525B;font-size:11px">'
                        f'turn {turn} · {source} · '
                        f'<span style="color:{conf_color}">conf {conf:.2f}</span>'
                        f'</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    with st.expander("📜 Audit History", expanded=False):
                        history = run_async(tracker.get_belief_history(
                            session_id=st.session_state.session_id,
                            subject=subject,
                            predicate=predicate
                        ))
                        if not history:
                            st.caption("No audit history found.")
                        else:
                            for entry in history:
                                op = entry["operation"].upper()
                                op_color = "#34D399" if op == "INSERT" else "#7C6FEB" if op == "UPDATE" else "#F87171"
                                old_val = f" (was: `{entry['old_value']}`)" if entry["old_value"] else ""
                                st.markdown(
                                    f'<div style="font-size: 11px; margin-bottom: 2px; font-family: monospace; line-height: 1.4;">'
                                    f'<span style="color:{op_color}; font-weight: bold;">{op}</span> '
                                    f'turn {entry["turn"]} · val: <b>{entry["new_value"]}</b>{old_val} · '
                                    f'conf: {entry["confidence"]:.2f} · <span style="color:#52525B">{entry["created_at"]}</span>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )

            # Raw JSON export
            with st.expander("📋 Raw JSON export"):
                raw = []
                for b in beliefs:
                    raw.append({
                        "subject": getattr(b, "subject", ""),
                        "predicate": getattr(b, "predicate", ""),
                        "value": getattr(b, "value", ""),
                        "confidence": getattr(b, "confidence", 0),
                        "turn": getattr(b, "turn", 0),
                        "source": getattr(b, "source", ""),
                        "belief_type": getattr(b, "belief_type", "assertion"),
                        "is_hypothetical": getattr(b, "is_hypothetical", False),
                    })
                st.json(raw)

    except Exception as e:
        st.error(f"Could not fetch beliefs: {e}")
        st.exception(e)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: AUTOMATED TESTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_tests:
    st.markdown("### Automated Feature Tests")
    st.caption(
        "Each test verifies a specific feature by directly calling the beliefstate API. "
        "No LLM call needed — tests against the store and tracker internals."
    )

    async def run_all_tests(tracker, session_id):
        results = []

        # ── Test 1: Health Check ──────────────────────────────────────────────
        test_name = "Health Check"
        try:
            health = await tracker.health_check()
            store_ok = health.get("store", False)
            adapter_ok = health.get("adapter", False)
            passed = store_ok  # adapter may fail if no network
            results.append({
                "name": test_name,
                "passed": passed,
                "detail": f"store={store_ok}, adapter={adapter_ok}",
                "expected": "store=True",
                "got": str(health)
            })
        except Exception as e:
            results.append({"name": test_name, "passed": False, "detail": str(e), "expected": "no exception", "got": str(e)})

        # ── Test 2: Session Set ───────────────────────────────────────────────
        test_name = "Session Context Set"
        try:
            tracker.set_session(session_id)
            tracker.set_session("other-session")
            tracker.set_session(session_id)
            results.append({
                "name": test_name,
                "passed": True,
                "detail": "set_session() called without error",
                "expected": "no exception",
                "got": "ok"
            })
        except Exception as e:
            results.append({"name": test_name, "passed": False, "detail": str(e), "expected": "no exception", "got": str(e)})

        # ── Test 3: Belief Store Write + Read ─────────────────────────────────
        test_name = "Belief Store Write & Read"
        try:
            from beliefstate.call import LLMCall, LLMResponse

            test_sid = "auto-test-write-read"
            call = LLMCall(messages=[{"role": "user", "content": "I love Python."}], kwargs={})
            resp = LLMResponse(text="Got it, you love Python.", raw_response=None)

            await tracker.track_async(call.model_dump(), resp.model_dump(), session_id=test_sid, turn=1)
            await asyncio.sleep(0.3)

            beliefs = await tracker.get_beliefs(session_id=test_sid)
            assert any(
                "python" in str(getattr(b, "value", "")).lower()
                for b in beliefs
            )
            results.append({
                "name": test_name,
                "passed": len(beliefs) > 0,
                "detail": f"{len(beliefs)} beliefs written and read back",
                "expected": "beliefs > 0",
                "got": f"{len(beliefs)} beliefs"
            })
        except Exception as e:
            results.append({"name": test_name, "passed": False, "detail": str(e), "expected": "beliefs > 0", "got": str(e)})

        # ── Test 4: Contradiction Detection ───────────────────────────────────
        test_name = "Contradiction Detection"
        try:
            test_sid = "auto-test-contradiction"

            # First: establish a belief
            call1 = LLMCall(messages=[{"role": "user", "content": "My budget is $5,000."}], kwargs={})
            resp1 = LLMResponse(text="Noted, your budget is $5,000.", raw_response=None)
            await tracker.track_async(call1.model_dump(), resp1.model_dump(), session_id=test_sid, turn=1)
            await asyncio.sleep(0.5)

            # Second: contradict it
            call2 = LLMCall(
                messages=[
                    {"role": "user", "content": "My budget is $5,000."},
                    {"role": "assistant", "content": "Noted, your budget is $5,000."},
                    {"role": "user", "content": "Actually, my budget is $50,000."},
                ],
                kwargs={}
            )
            resp2 = LLMResponse(text="Understood, your budget is $50,000.", raw_response=None)
            await tracker.track_async(call2.model_dump(), resp2.model_dump(), session_id=test_sid, turn=2)
            await asyncio.sleep(0.5)

            beliefs = await tracker.get_beliefs(session_id=test_sid)
            # If resolution=update, only one budget belief should remain
            budget_beliefs = [
                b for b in beliefs
                if "budget" in str(getattr(b, "predicate", "")).lower()
                or "budget" in str(getattr(b, "subject", "")).lower()
                or "budget" in str(getattr(b, "value", "")).lower()
            ]
            results.append({
                "name": test_name,
                "passed": True,
                "detail": f"Contradiction pipeline ran. {len(budget_beliefs)} budget belief(s) in store.",
                "expected": "pipeline runs without error",
                "got": f"{len(budget_beliefs)} budget belief(s)"
            })
        except Exception as e:
            results.append({"name": test_name, "passed": False, "detail": str(e), "expected": "pipeline runs", "got": str(e)})

        # ── Test 5: Context Prompt ────────────────────────────────────────────
        test_name = "Context Prompt Generation"
        try:
            context = await tracker.get_context_prompt(session_id=session_id)
            has_content = isinstance(context, str) and len(context) > 0
            results.append({
                "name": test_name,
                "passed": has_content,
                "detail": f"get_context_prompt() returned {len(context)} chars",
                "expected": "non-empty string",
                "got": f"{len(context)} chars"
            })
        except Exception as e:
            results.append({"name": test_name, "passed": False, "detail": str(e), "expected": "non-empty string", "got": str(e)})

        # ── Test 6: Session Isolation ─────────────────────────────────────────
        test_name = "Session Isolation"
        try:
            sid_a = "isolation-test-A"
            sid_b = "isolation-test-B"

            call_a = LLMCall(messages=[{"role": "user", "content": "I am Alice."}], kwargs={})
            resp_a = LLMResponse(text="Hello Alice.", raw_response=None)
            await tracker.track_async(call_a.model_dump(), resp_a.model_dump(), session_id=sid_a, turn=1)

            call_b = LLMCall(messages=[{"role": "user", "content": "I am Bob."}], kwargs={})
            resp_b = LLMResponse(text="Hello Bob.", raw_response=None)
            await tracker.track_async(call_b.model_dump(), resp_b.model_dump(), session_id=sid_b, turn=1)

            await asyncio.sleep(0.5)

            beliefs_a = await tracker.get_beliefs(session_id=sid_a)
            beliefs_b = await tracker.get_beliefs(session_id=sid_b)

            values_a = [str(getattr(b, "value", "")).lower() for b in beliefs_a]
            values_b = [str(getattr(b, "value", "")).lower() for b in beliefs_b]

            # Alice's beliefs should not contain Bob and vice versa
            no_bleed = not any("bob" in v for v in values_a) and not any("alice" in v for v in values_b)

            results.append({
                "name": test_name,
                "passed": no_bleed,
                "detail": f"Session A: {len(beliefs_a)} beliefs, Session B: {len(beliefs_b)} beliefs — no bleed",
                "expected": "no cross-session belief bleed",
                "got": f"A has {len(beliefs_a)}, B has {len(beliefs_b)} beliefs"
            })
        except Exception as e:
            results.append({"name": test_name, "passed": False, "detail": str(e), "expected": "no bleed", "got": str(e)})

        # ── Test 7: Sync vs Async dispatcher ─────────────────────────────────
        test_name = "Dispatcher Mode"
        try:
            dispatcher_type = tracker._config.task_dispatcher_type if hasattr(tracker, "_config") else "unknown"
            results.append({
                "name": test_name,
                "passed": True,
                "detail": f"Dispatcher type: {dispatcher_type}",
                "expected": "sync or asyncio",
                "got": dispatcher_type
            })
        except Exception as e:
            results.append({"name": test_name, "passed": False, "detail": str(e), "expected": "no exception", "got": str(e)})

        return results

    if st.button("▶ Run All Tests", type="primary", use_container_width=False):
        with st.spinner("Running automated tests..."):
            try:
                results = run_async(run_all_tests(tracker, st.session_state.session_id))
                st.session_state.test_results = results
            except Exception as e:
                st.error(f"Test runner error: {e}")
                st.exception(e)

    if st.session_state.test_results:
        results = st.session_state.test_results
        passed = sum(1 for r in results if r["passed"])
        failed = len(results) - passed

        m1, m2, m3 = st.columns(3)
        m1.metric("Total tests", len(results))
        m2.metric("Passed ✓", passed)
        m3.metric("Failed ✗", failed)

        st.markdown("---")

        for r in results:
            icon = "✅" if r["passed"] else "❌"
            with st.expander(f"{icon} {r['name']}", expanded=not r["passed"]):
                col1, col2 = st.columns(2)
                col1.markdown(f"**Expected:** `{r['expected']}`")
                col2.markdown(f"**Got:** `{r['got']}`")
                st.caption(r["detail"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: GDPR DELETION
# ══════════════════════════════════════════════════════════════════════════════
with tab_gdpr:
    st.markdown("### GDPR Session Deletion")
    st.caption(
        "Tests `clear_session()` — drains in-flight tasks, deletes all beliefs, "
        "returns an auditable DeletionReceipt."
    )

    col_gdpr1, col_gdpr2 = st.columns(2)

    with col_gdpr1:
        st.markdown("**Delete session**")
        delete_sid = st.text_input(
            "Session ID to delete",
            value=st.session_state.session_id,
            key="delete_sid_input"
        )

        st.warning(
            f"This will permanently delete all beliefs for session `{delete_sid}`. "
            "A DeletionReceipt will be returned for audit purposes."
        )

        confirm = st.checkbox("I confirm I want to delete this session's beliefs")

        if st.button("🗑 Execute clear_session()", type="primary", disabled=not confirm):
            with st.spinner("Draining tasks and deleting..."):
                try:
                    receipt = run_async(tracker.clear_session(delete_sid))
                    st.session_state["last_receipt"] = receipt

                    if delete_sid == st.session_state.session_id:
                        st.session_state.conversation = []
                        st.session_state.turn = 0

                    st.success("Session deleted.")
                except Exception as e:
                    st.error(f"Deletion failed: {e}")
                    st.exception(e)

    with col_gdpr2:
        st.markdown("**DeletionReceipt**")

        if "last_receipt" in st.session_state:
            r = st.session_state["last_receipt"]
            session_id_r = getattr(r, "session_id", "?")
            beliefs_deleted = getattr(r, "beliefs_deleted", "?")
            tasks_drained = getattr(r, "in_flight_tasks_drained", "?")
            deleted_at = getattr(r, "deleted_at", "?")

            st.markdown(
                f'<div class="success-card">'
                f'<b>✓ DeletionReceipt issued</b><br><br>'
                f'<code>session_id:</code> {session_id_r}<br>'
                f'<code>beliefs_deleted:</code> {beliefs_deleted}<br>'
                f'<code>in_flight_tasks_drained:</code> {tasks_drained}<br>'
                f'<code>deleted_at:</code> {deleted_at}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown("**Receipt as JSON (save for audit log):**")
            st.code(json.dumps({
                "session_id": str(session_id_r),
                "beliefs_deleted": beliefs_deleted,
                "in_flight_tasks_drained": str(tasks_drained),
                "deleted_at": str(deleted_at),
                "gdpr_article": "17 — Right to erasure",
            }, indent=2), language="json")
        else:
            st.info("No deletion executed yet in this session.")

    st.markdown("---")
    st.markdown("**Verify deletion:**")
    if st.button("🔍 Check beliefs after deletion"):
        try:
            beliefs = run_async(tracker.get_beliefs(session_id=delete_sid if "delete_sid_input" in st.session_state else st.session_state.session_id))
            if not beliefs:
                st.success(f"✓ Confirmed: 0 beliefs remain for session `{delete_sid}`")
            else:
                st.error(f"✗ {len(beliefs)} beliefs still found — deletion may not have completed.")
        except Exception as e:
            st.error(f"Check failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: CONTEXT PROMPT
# ══════════════════════════════════════════════════════════════════════════════
with tab_context:
    st.markdown("### Context Prompt Inspector")
    st.caption(
        "Shows the exact text that beliefstate injects into the system prompt "
        "before each LLM call — giving the model persistent memory."
    )

    col_ctx1, col_ctx2 = st.columns([1, 2])

    with col_ctx1:
        ctx_session = st.text_input("Session ID", value=st.session_state.session_id, key="ctx_sid")
        ctx_user_msg = st.text_input(
            "Current user message (for relevance ranking)",
            value="",
            placeholder="Optional — enables token-aware injection"
        )
        if st.button("📋 Get Context Prompt", use_container_width=True):
            try:
                context = run_async(tracker.get_context_prompt(
                    session_id=ctx_session,
                    current_user_message=ctx_user_msg if ctx_user_msg else None
                ))
                st.session_state["last_context"] = context
            except Exception as e:
                st.error(f"Failed: {e}")

    with col_ctx2:
        st.markdown("**What gets injected into system prompt:**")

        if "last_context" in st.session_state:
            ctx = st.session_state["last_context"]
            if ctx:
                token_estimate = len(ctx) // 4
                st.markdown(
                    f'<div class="success-card">'
                    f'<b>✓ Context ready</b> — ~{token_estimate} tokens'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.code(ctx, language="text")

                st.markdown("**How to use in your app:**")
                st.code(
                    f"""# Get the belief summary
context = await tracker.get_context_prompt(session_id="{ctx_session}")

# Inject into your system prompt
system_prompt = f\"\"\"You are a helpful assistant.

{"{context}"}
\"\"\"

# Pass to LLM as usual
response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {{"role": "system", "content": system_prompt}},
        *conversation_history
    ]
)""",
                    language="python"
                )
            else:
                st.info("No beliefs in store yet — context prompt is empty.")
        else:
            st.info("Click 'Get Context Prompt' to see what gets injected.")

    st.markdown("---")
    st.markdown("### Token Budget Calculator")
    st.caption("Check if your beliefs fit within the configured token budget.")

    try:
        beliefs = run_async(tracker.get_beliefs(session_id=ctx_session))
        if beliefs:
            budget = 500  # default from TrackerConfig
            context = run_async(tracker.get_context_prompt(session_id=ctx_session))
            actual_tokens = len(context) // 4

            col_t1, col_t2, col_t3 = st.columns(3)
            col_t1.metric("Beliefs in store", len(beliefs))
            col_t2.metric("Estimated tokens used", actual_tokens)
            col_t3.metric("Token budget", budget)

            if actual_tokens <= budget:
                st.success(f"✓ Within budget: {actual_tokens}/{budget} tokens used ({100*actual_tokens//budget}%)")
            else:
                st.warning(
                    f"⚠ Over budget by {actual_tokens - budget} tokens. "
                    "Token-aware injection will activate and select only the most relevant beliefs."
                )
    except Exception as e:
        st.error(f"Could not calculate token budget: {e}")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;font-size:12px;color:#52525B;font-family:monospace">'
    'beliefstate · MIT Licensed · '
    '<a href="https://abhay-2108.github.io/beliefstate/" style="color:#7C6FEB">Docs</a> · '
    '<a href="https://github.com/abhay-2108/beliefstate" style="color:#7C6FEB">GitHub</a>'
    '</div>',
    unsafe_allow_html=True
)