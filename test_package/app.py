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
    streamlit run app.py

Architecture:
    utils.py        — async helpers, belief rendering
    providers.py    — tracker builder, unified LLM call
    ui_helpers.py   — CSS, sidebar, tab renderers
    tests.py        — automated feature tests
"""

from __future__ import annotations

import os
import sys

# Ensure sibling modules are importable when run via `streamlit run`
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config & styling ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="beliefstate · E2E Tester",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui_helpers import (  # noqa: E402
    inject_css,
    build_sidebar,
    render_chat_tab,
    render_beliefs_tab,
    render_tests_tab,
    render_gdpr_tab,
    render_context_tab,
    render_footer,
)

inject_css()

# ── Session state init ─────────────────────────────────────────────────────────

for key, default in [
    ("tracker", None),
    ("adapter", None),
    ("conversation", []),
    ("session_id", "test-session-001"),
    ("turn", 0),
    ("test_results", []),
    ("health_status", None),
    ("init_error", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ────────────────────────────────────────────────────────────────────

build_sidebar()

# ── Main area ──────────────────────────────────────────────────────────────────

st.markdown("# 🧠 beliefstate · End-to-End Test Suite")
st.markdown(
    "Tests core features: belief extraction · contradiction detection · "
    "resolution · session management · GDPR deletion"
)

tracker = st.session_state.tracker

if tracker is None:
    st.warning("← Configure and initialise the tracker in the sidebar to begin.")
    st.stop()

cfg = st.session_state.get("tracker_cfg", {})

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_chat, tab_beliefs, tab_tests, tab_gdpr, tab_context = st.tabs(
    [
        "💬 Chat & Track",
        "📦 Belief Store",
        "🧪 Automated Tests",
        "🗑 GDPR Deletion",
        "🔍 Context Prompt",
    ]
)

with tab_chat:
    render_chat_tab(tracker, cfg)

with tab_beliefs:
    render_beliefs_tab(tracker)

with tab_tests:
    render_tests_tab(tracker)

with tab_gdpr:
    render_gdpr_tab(tracker)

with tab_context:
    render_context_tab(tracker)

# ── Footer ─────────────────────────────────────────────────────────────────────

render_footer()
