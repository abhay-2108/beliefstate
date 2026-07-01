"""Streamlit UI components: CSS, sidebar, and tab renderers."""

from __future__ import annotations

import json
import math
from typing import Any

import streamlit as st

from providers import build_tracker, call_llm_sync
from utils import (
    belief_card_html,
    belief_to_dict,
    contradiction_card_html,
    run_async,
)

# ── CSS ────────────────────────────────────────────────────────────────────────

CSS = """
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
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────


def build_sidebar() -> dict[str, Any] | None:
    """Render the sidebar and return config dict, or None if not yet initialised."""
    with st.sidebar:
        st.markdown("## 🧠 beliefstate tester")
        st.markdown("---")

        # Provider
        st.markdown(
            '<div class="section-header">Provider</div>', unsafe_allow_html=True
        )
        provider = st.selectbox(
            "LLM Provider",
            ["OpenAI", "NVIDIA", "Anthropic", "Gemini", "Ollama"],
            help="Select your LLM provider. Anthropic requires a paired embed provider.",
        )

        import os

        api_key = st.text_input(
            "API Key",
            type="password",
            value=os.environ.get("OPENAI_API_KEY", "")
            if provider == "OpenAI"
            else os.environ.get("NVIDIA_API_KEY", "")
            if provider == "NVIDIA"
            else os.environ.get("ANTHROPIC_API_KEY", "")
            if provider == "Anthropic"
            else os.environ.get("GEMINI_API_KEY", "")
            if provider == "Gemini"
            else "ollama",
            help="Or set via environment variable.",
        )

        # Model selection per provider
        if provider == "OpenAI":
            model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
            embed_model = st.selectbox(
                "Embed Model", ["text-embedding-3-small", "text-embedding-3-large"]
            )
        elif provider == "NVIDIA":
            model = st.selectbox(
                "Model",
                [
                    "meta/llama-3.1-8b-instruct",
                    "meta/llama-3.2-3b-instruct",
                    "meta/llama-3.1-70b-instruct",
                    "nvidia/llama-3.1-nemotron-70b-instruct",
                ],
            )
            embed_model = st.selectbox(
                "Embed Model",
                [
                    "nvidia/nv-embed-v1",
                    "nvidia/nv-embedqa-e5-v5",
                    "nvidia/llama-nemotron-embed-1b-v2",
                ],
            )
        elif provider == "Anthropic":
            model = st.selectbox(
                "Model", ["claude-3-5-sonnet-latest", "claude-3-haiku-20240307"]
            )
            st.info(
                "⚠ Anthropic has no embedding API. Pair with OpenAI for embeddings."
            )
            embed_api_key = st.text_input(
                "OpenAI Key (for embeddings)",
                type="password",
                value=os.environ.get("OPENAI_API_KEY", ""),
            )
            embed_model = "text-embedding-3-small"
        elif provider == "Gemini":
            model = st.selectbox(
                "Model", ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
            )
            embed_model = "text-embedding-004"
        elif provider == "Ollama":
            model = st.selectbox("Model", ["llama3.2", "mistral", "qwen2.5", "phi3"])
            embed_model = st.selectbox(
                "Embed Model", ["nomic-embed-text", "mxbai-embed-large"]
            )
            ollama_host = st.text_input("Ollama Host", value="http://localhost")
            ollama_port = st.number_input("Ollama Port", value=11434)

        st.markdown("---")
        st.markdown(
            '<div class="section-header">Tracker Config</div>', unsafe_allow_html=True
        )

        store_type = st.selectbox(
            "Store Backend",
            ["sqlite", "in-memory"],
            help="SQLite persists to disk, in-memory resets on restart.",
        )
        db_path = ""
        if store_type == "sqlite":
            db_path = st.text_input("DB Path", value="beliefstate_test.db")

        resolution_strategy = st.selectbox(
            "Resolution Strategy",
            ["overwrite", "keep_old", "raise"],
            help="overwrite: replace old belief | keep_old: keep first, discard new | raise: throw error on contradiction",
        )

        similarity_threshold = st.slider(
            "Similarity Threshold",
            0.5,
            1.0,
            0.82,
            0.01,
            help="Stage 1 cosine gate.",
        )
        contradiction_threshold = st.slider(
            "Contradiction Threshold",
            0.5,
            1.0,
            0.70,
            0.01,
            help="Stage 2 NLI contradiction gate.",
        )
        entailment_threshold = st.slider(
            "Entailment Threshold",
            0.5,
            1.0,
            0.85,
            0.01,
            help="NLI entailment gate.",
        )
        user_confidence_cap = st.slider(
            "User Confidence Cap",
            0.0,
            1.0,
            0.99,
            0.01,
        )
        assistant_confidence_cap = st.slider(
            "Assistant Confidence Cap",
            0.0,
            1.0,
            0.85,
            0.01,
        )
        judge_timeout = st.number_input(
            "Judge Timeout (seconds)",
            value=60.0,
            min_value=1.0,
            max_value=300.0,
        )
        max_beliefs = st.number_input(
            "Max Beliefs", value=50, min_value=1, max_value=200
        )

        min_injection_confidence = st.slider(
            "Min Injection Confidence",
            0.0,
            1.0,
            0.80,
            0.05,
            help="Minimum confidence for a belief to be injected into context.",
        )
        include_hypothetical_in_context = st.toggle(
            "Include Hypotheticals in Context",
            value=False,
            help="Whether to include hypothetical beliefs in the system prompt.",
        )

        background_tasks = st.toggle(
            "Background Tasks",
            value=False,
            help="OFF = sync mode, ON = fire-and-forget (production mode).",
        )

        st.markdown("---")
        session_id = st.text_input(
            "Session ID", value=st.session_state.get("session_id", "test-session-001")
        )
        st.session_state.session_id = session_id

        # Config fingerprint for invalidation
        fingerprint = f"{provider}|{model}|{embed_model}|{api_key}|{store_type}|{similarity_threshold}|{contradiction_threshold}|{entailment_threshold}|{user_confidence_cap}|{assistant_confidence_cap}|{judge_timeout}"
        if "config_fingerprint" not in st.session_state:
            st.session_state.config_fingerprint = fingerprint
        elif st.session_state.config_fingerprint != fingerprint:
            st.session_state.tracker = None
            st.session_state.adapter = None
            st.session_state.health_status = None
            st.session_state.config_fingerprint = fingerprint

        # Init button
        if st.button("🔧 Initialise Tracker", type="primary", use_container_width=True):
            with st.spinner("Initialising..."):
                try:
                    cfg = {
                        "provider": provider,
                        "model": model,
                        "embed_model": embed_model,
                        "api_key": api_key,
                        "store_type": store_type,
                        "db_path": db_path,
                        "similarity_threshold": similarity_threshold,
                        "contradiction_threshold": contradiction_threshold,
                        "entailment_threshold": entailment_threshold,
                        "judge_timeout": judge_timeout,
                        "user_confidence_cap": user_confidence_cap,
                        "assistant_confidence_cap": assistant_confidence_cap,
                        "max_beliefs": max_beliefs,
                        "background_tasks": background_tasks,
                        "resolution_strategy": resolution_strategy,
                        "min_injection_confidence": min_injection_confidence,
                        "include_hypothetical_in_context": include_hypothetical_in_context,
                    }
                    if provider == "Anthropic":
                        cfg["embed_api_key"] = embed_api_key
                    if provider == "Ollama":
                        cfg["ollama_host"] = ollama_host
                        cfg["ollama_port"] = ollama_port

                    tracker = build_tracker(cfg)
                    st.session_state.tracker = tracker
                    st.session_state.tracker_cfg = cfg
                    st.session_state.init_error = None
                    st.session_state.conversation = []
                    st.session_state.turn = 0
                    st.session_state.test_results = []

                    health = run_async(tracker.health_check())
                    st.session_state.health_status = health
                    st.success("✓ Tracker initialised")
                except Exception as e:
                    st.session_state.init_error = str(e)
                    st.error(f"Init failed: {e}")

        # Health display
        if st.session_state.get("health_status"):
            h = st.session_state.health_status
            store_ok = h.get("store", False)
            adapter_ok = h.get("adapter", False)
            store_html = (
                '<span class="status-ok">✓</span>'
                if store_ok
                else '<span class="status-fail">✗</span>'
            )
            adapter_html = (
                '<span class="status-ok">✓</span>'
                if adapter_ok
                else '<span class="status-fail">✗</span>'
            )
            st.markdown(
                f"**Health:** Store {store_html} Adapter {adapter_html}",
                unsafe_allow_html=True,
            )

        if st.session_state.get("init_error"):
            st.error(f"Error: {st.session_state.init_error}")

    # Return config if tracker is initialised
    return st.session_state.get("tracker_cfg")


# ── Tab: Chat ──────────────────────────────────────────────────────────────────


def render_chat_tab(
    tracker: Any,
    cfg: dict[str, Any],
) -> None:
    """Render the Chat & Track tab."""
    session_id = st.session_state.session_id
    conversation: list[dict] = st.session_state.get("conversation", [])
    turn = st.session_state.get("turn", 0)

    col_chat, col_live = st.columns([3, 2])

    with col_chat:
        st.markdown("### Conversation")
        st.caption(
            f"Session: `{session_id}` · "
            f"Turn: `{turn}` · "
            f"Mode: `{'async' if cfg.get('background_tasks') else 'sync'}`"
        )

        # Render history
        for entry in conversation:
            role = entry["role"]
            content = entry["content"]
            turn_num = entry.get("turn", "?")
            if role == "user":
                st.markdown(
                    f'<div class="turn-label">Turn {turn_num} · User</div>'
                    f'<div class="chat-msg-user">{content}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="turn-label">Turn {turn_num} · Assistant</div>'
                    f'<div class="chat-msg-assistant">{content}</div>',
                    unsafe_allow_html=True,
                )
                if entry.get("beliefs"):
                    with st.expander(f"📦 {len(entry['beliefs'])} belief(s) extracted"):
                        for b in entry["beliefs"]:
                            st.markdown(belief_card_html(b), unsafe_allow_html=True)
                if entry.get("contradictions"):
                    for c in entry["contradictions"]:
                        st.markdown(contradiction_card_html(c), unsafe_allow_html=True)

        st.markdown("---")

        # Preset prompts
        st.markdown("**Quick test prompts:**")
        c1, c2, c3 = st.columns(3)
        presets = [
            (
                "👤 Introduce self",
                "Hi, my name is Raj. I'm a Python developer based in Chennai.",
            ),
            ("💰 Set budget", "My budget for this project is $5,000."),
            ("⚡ Contradict budget", "Actually, let's plan for a $10,000 budget."),
            ("🏠 Share preference", "I prefer working remotely and I love TypeScript."),
            (
                "🔄 Update preference",
                "I've switched to Python lately, not TypeScript anymore.",
            ),
            ("📍 Set location", "I'm currently living in Tokyo, Japan."),
        ]
        for i, (label, prompt) in enumerate(presets):
            with [c1, c2, c3][i % 3]:
                if st.button(label, use_container_width=True, key=f"preset_{i}"):
                    st.session_state["pending_message"] = prompt

        user_input = st.chat_input("Type a message to test belief extraction...")

        if "pending_message" in st.session_state:
            user_input = st.session_state.pop("pending_message")

        if user_input:
            tracker.set_session(session_id)
            turn = st.session_state.turn + 1
            st.session_state.turn = turn

            conversation.append({"role": "user", "content": user_input, "turn": turn})

            with st.spinner(f"Turn {turn}: calling LLM + tracking beliefs..."):
                try:
                    messages = [
                        {"role": e["role"], "content": e["content"]}
                        for e in conversation
                        if e["role"] in ("user", "assistant")
                    ]

                    assistant_text, latency_ms = call_llm_sync(
                        provider=cfg["provider"],
                        messages=messages,
                        cfg=cfg,
                        tracker=tracker,
                        session_id=session_id,
                        turn=turn,
                    )

                    beliefs = run_async(tracker.get_beliefs(session_id=session_id))

                    # Fetch conflict notes from resolver
                    conflict_notes = tracker.get_pending_conflicts(session_id)

                    conversation.append(
                        {
                            "role": "assistant",
                            "content": assistant_text,
                            "turn": turn,
                            "beliefs": beliefs,
                            "contradictions": conflict_notes,
                            "latency_ms": latency_ms,
                            "belief_count": len(beliefs),
                        }
                    )
                    st.session_state.conversation = conversation

                    st.rerun()
                except Exception as e:
                    st.error(f"Error on turn {turn}: {e}")
                    st.exception(e)

    with col_live:
        st.markdown("### Live Pipeline Status")
        if conversation:
            last = next(
                (e for e in reversed(conversation) if e["role"] == "assistant"),
                None,
            )
            if last:
                m1, m2 = st.columns(2)
                m1.metric("Latency", f"{last.get('latency_ms', 0):.0f}ms")
                m2.metric("Beliefs in store", last.get("belief_count", 0))

                st.markdown("---")
                st.markdown("**Pipeline stages completed:**")
                for stage, detail in [
                    ("🎯 Intercept", "Call captured by tracker"),
                    ("💬 LLM Call", "Response returned to user"),
                    (
                        "📤 Dispatch",
                        "Async bg task"
                        if cfg.get("background_tasks")
                        else "Sync (blocking)",
                    ),
                    ("🔍 Extract", f"{last.get('belief_count', 0)} beliefs extracted"),
                    ("⚡ Detect", "Cosine gate + NLI check"),
                    (
                        "🛡 Resolve",
                        f"Strategy: {cfg.get('resolution_strategy', 'warn')}",
                    ),
                    ("💾 Store", "Written to " + cfg.get("store_type", "memory")),
                ]:
                    st.markdown(
                        f'<span style="color:#34D399">✓</span> '
                        f'**{stage}** — <span style="color:#A1A1AA;font-size:12px">{detail}</span>',
                        unsafe_allow_html=True,
                    )

        st.markdown("---")
        st.markdown("**Session info:**")
        st.code(f"session_id: {session_id}\nturn: {turn}", language="text")

        if st.button("🗑 Clear conversation (keep beliefs)", use_container_width=True):
            st.session_state.conversation = []
            st.session_state.turn = 0
            st.rerun()


# ── Tab: Beliefs ───────────────────────────────────────────────────────────────


def render_beliefs_tab(tracker: Any) -> None:
    """Render the Belief Store Inspector tab."""
    session_id = st.session_state.session_id

    st.markdown("### Belief Store Inspector")
    st.caption("All extracted beliefs for the current session, live from the store.")

    col_refresh, col_filter = st.columns([1, 3])
    with col_refresh:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    with col_filter:
        filter_text = st.text_input(
            "Filter by subject/predicate/value", placeholder="e.g. USER, budget, Python"
        )

    try:
        beliefs = run_async(tracker.get_beliefs(session_id=session_id))

        if not beliefs:
            st.info("No beliefs stored yet. Send some messages in the Chat tab.")
            return

        if filter_text:
            flt = filter_text.lower()
            beliefs = [
                b
                for b in beliefs
                if flt in str(getattr(b, "subject", "")).lower()
                or flt in str(getattr(b, "predicate", "")).lower()
                or flt in str(getattr(b, "value", "")).lower()
            ]

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total beliefs", len(beliefs))

        sources: dict[str, int] = {}
        for b in beliefs:
            src = getattr(b, "source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        m2.metric("From user", sources.get("user", 0))
        m3.metric("From assistant", sources.get("assistant", 0))

        avg_conf = sum(getattr(b, "confidence", 0) for b in beliefs) / max(
            len(beliefs), 1
        )
        m4.metric("Avg confidence", f"{avg_conf:.2f}")

        st.markdown("---")

        # Group by subject
        subjects: dict[str, list] = {}
        for b in beliefs:
            subj = getattr(b, "subject", "unknown")
            subjects.setdefault(subj, []).append(b)

        for subject, subject_beliefs in subjects.items():
            st.markdown(f"**Subject: `{subject}`**")
            for b in subject_beliefs:
                predicate = getattr(b, "predicate", "?")
                value = getattr(b, "value", "?")
                conf = getattr(b, "confidence", 0)
                turn = getattr(b, "turn", "?")
                source = getattr(b, "source", "?")
                is_hypo = getattr(b, "is_hypothetical", False)

                conf_color = (
                    "#34D399"
                    if conf >= 0.8
                    else "#F59E0B"
                    if conf >= 0.5
                    else "#F87171"
                )
                hypo_badge = (
                    ' <span style="color:#F59E0B;font-size:10px">[hypothetical]</span>'
                    if is_hypo
                    else ""
                )
                update_badge = (
                    ' <span style="color:#7C6FEB;font-size:10px">[update]</span>'
                    if getattr(b, "belief_type", "assertion") == "update"
                    else ""
                )

                st.markdown(
                    f'<div class="belief-card">'
                    f'<span style="color:#A1A1AA">{predicate}</span> → '
                    f'<span style="color:#F4F4F5;font-weight:500">{value}</span>'
                    f"{hypo_badge}{update_badge}<br>"
                    f'<span style="color:#52525B;font-size:11px">'
                    f"turn {turn} · {source} · "
                    f'<span style="color:{conf_color}">conf {conf:.2f}</span>'
                    f"</span></div>",
                    unsafe_allow_html=True,
                )

                with st.expander("📜 Audit History", expanded=False):
                    history = run_async(
                        tracker.get_belief_history(
                            session_id=session_id,
                            subject=subject,
                            predicate=predicate,
                        )
                    )
                    if not history:
                        st.caption("No audit history found.")
                    else:
                        for entry in history:
                            op = entry["operation"].upper()
                            op_color = (
                                "#34D399"
                                if op == "INSERT"
                                else "#7C6FEB"
                                if op == "UPDATE"
                                else "#F87171"
                            )
                            old_val = (
                                f" (was: `{entry['old_value']}`)"
                                if entry["old_value"]
                                else ""
                            )
                            st.markdown(
                                f'<div style="font-size:11px;margin-bottom:2px;font-family:monospace;line-height:1.4">'
                                f'<span style="color:{op_color};font-weight:bold">{op}</span> '
                                f"turn {entry['turn']} · val: <b>{entry['new_value']}</b>{old_val} · "
                                f'conf: {entry["confidence"]:.2f} · <span style="color:#52525B">{entry["created_at"]}</span>'
                                f"</div>",
                                unsafe_allow_html=True,
                            )

        with st.expander("📋 Raw JSON export"):
            st.json([belief_to_dict(b) for b in beliefs])

    except Exception as e:
        st.error(f"Could not fetch beliefs: {e}")
        st.exception(e)


# ── Tab: Tests ─────────────────────────────────────────────────────────────────


def render_tests_tab(tracker: Any) -> None:
    """Render the Automated Tests tab."""
    from tests import run_all_tests

    session_id = st.session_state.session_id

    st.markdown("### Automated Feature Tests")
    st.caption(
        "Each test verifies a specific feature by directly calling the beliefstate API. "
        "No LLM call needed — tests against the store and tracker internals."
    )

    if st.button("▶ Run All Tests", type="primary", use_container_width=False):
        with st.spinner("Running automated tests..."):
            try:
                results = run_async(run_all_tests(tracker, session_id))
                st.session_state.test_results = results
            except Exception as e:
                st.error(f"Test runner error: {e}")
                st.exception(e)

    if st.session_state.get("test_results"):
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


# ── Tab: GDPR ──────────────────────────────────────────────────────────────────


def render_gdpr_tab(tracker: Any) -> None:
    """Render the GDPR Session Deletion tab."""
    session_id = st.session_state.session_id

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
            value=session_id,
            key="delete_sid_input",
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

                    if delete_sid == session_id:
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
            st.markdown(
                f'<div class="success-card">'
                f"<b>✓ DeletionReceipt issued</b><br><br>"
                f"<code>session_id:</code> {getattr(r, 'session_id', '?')}<br>"
                f"<code>beliefs_deleted:</code> {getattr(r, 'beliefs_deleted', '?')}<br>"
                f"<code>in_flight_tasks_drained:</code> {getattr(r, 'in_flight_tasks_drained', '?')}<br>"
                f"<code>deleted_at:</code> {getattr(r, 'deleted_at', '?')}"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown("**Receipt as JSON (save for audit log):**")
            st.code(
                json.dumps(
                    {
                        "session_id": str(getattr(r, "session_id", "?")),
                        "beliefs_deleted": getattr(r, "beliefs_deleted", "?"),
                        "in_flight_tasks_drained": str(
                            getattr(r, "in_flight_tasks_drained", "?")
                        ),
                        "deleted_at": str(getattr(r, "deleted_at", "?")),
                        "gdpr_article": "17 — Right to erasure",
                    },
                    indent=2,
                ),
                language="json",
            )
        else:
            st.info("No deletion executed yet in this session.")

    st.markdown("---")
    st.markdown("**Verify deletion:**")
    if st.button("🔍 Check beliefs after deletion"):
        try:
            beliefs = run_async(
                tracker.get_beliefs(
                    session_id=delete_sid
                    if "delete_sid_input" in st.session_state
                    else session_id
                )
            )
            if not beliefs:
                st.success(f"✓ Confirmed: 0 beliefs remain for session `{delete_sid}`")
            else:
                st.error(
                    f"✗ {len(beliefs)} beliefs still found — deletion may not have completed."
                )
        except Exception as e:
            st.error(f"Check failed: {e}")


# ── Tab: Context ───────────────────────────────────────────────────────────────


def render_context_tab(tracker: Any) -> None:
    """Render the Context Prompt Inspector tab."""
    session_id = st.session_state.session_id

    st.markdown("### Context Prompt Inspector")
    st.caption(
        "Shows the exact text that beliefstate injects into the system prompt "
        "before each LLM call — giving the model persistent memory."
    )

    col_ctx1, col_ctx2 = st.columns([1, 2])

    with col_ctx1:
        ctx_session = st.text_input("Session ID", value=session_id, key="ctx_sid")
        ctx_user_msg = st.text_input(
            "Current user message (for relevance ranking)",
            value="",
            placeholder="Optional — enables token-aware injection",
        )
        if st.button("📋 Get Context Prompt", use_container_width=True):
            try:
                context = run_async(
                    tracker.get_context_prompt(
                        session_id=ctx_session,
                        current_user_message=ctx_user_msg if ctx_user_msg else None,
                    )
                )
                st.session_state["last_context"] = context
            except Exception as e:
                st.error(f"Failed: {e}")

    with col_ctx2:
        st.markdown("**What gets injected into system prompt:**")

        if "last_context" in st.session_state:
            ctx = st.session_state["last_context"]
            if ctx:
                token_estimate = math.ceil(len(ctx) / 4)
                st.markdown(
                    f'<div class="success-card"><b>✓ Context ready</b> — ~{token_estimate} tokens</div>',
                    unsafe_allow_html=True,
                )
                st.code(ctx, language="text")
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
            budget = tracker.config.belief_budget_tokens
            context = run_async(tracker.get_context_prompt(session_id=ctx_session))
            actual_tokens = math.ceil(len(context) / 4)

            col_t1, col_t2, col_t3 = st.columns(3)
            col_t1.metric("Beliefs in store", len(beliefs))
            col_t2.metric("Estimated tokens used", actual_tokens)
            col_t3.metric("Token budget", budget)

            if actual_tokens <= budget:
                st.success(
                    f"✓ Within budget: {actual_tokens}/{budget} tokens used ({100 * actual_tokens // budget}%)"
                )
            else:
                st.warning(
                    f"⚠ Over budget by {actual_tokens - budget} tokens. "
                    "Token-aware injection will activate and select only the most relevant beliefs."
                )
    except Exception as e:
        st.error(f"Could not calculate token budget: {e}")


# ── Footer ─────────────────────────────────────────────────────────────────────


def render_footer() -> None:
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;font-size:12px;color:#52525B;font-family:monospace">'
        "beliefstate · MIT Licensed · "
        '<a href="https://AltioraLabs.github.io/beliefstate/" style="color:#7C6FEB">Docs</a> · '
        '<a href="https://github.com/AltioraLabs/beliefstate" style="color:#7C6FEB">GitHub</a>'
        "</div>",
        unsafe_allow_html=True,
    )
