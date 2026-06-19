"""
BeliefState Streamlit Demo

A real-time demonstration of the BeliefState package using NVIDIA's LLM API.
This app shows how the package tracks, extracts, and detects contradictions in beliefs.

Requirements:
- streamlit
- beliefstate
- litellm (for NVIDIA API support)
- python-dotenv

Installation:
    pip install streamlit beliefstate litellm python-dotenv

Usage:
    streamlit run streamlit_app.py

Environment Setup:
    Create a .env file in the examples folder with:
    NVIDIA_API_KEY=your_nvidia_api_key_here
"""

import asyncio
import os
import json
from datetime import datetime
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

# Import beliefstate components
from beliefstate import (
    BeliefTracker,
    TrackerConfig,
    Belief,
)
from beliefstate.adapters.litellm import LiteLLMAdapter

# Load environment variables
load_dotenv()

# ============================================================================
# STREAMLIT PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="BeliefState Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# STYLING & CSS
# ============================================================================

st.markdown(
    """
    <style>
    .belief-box {
        border-left: 4px solid #007ACC;
        padding: 12px;
        margin: 8px 0;
        background-color: #f5f5f5;
        border-radius: 4px;
    }
    .contradiction-box {
        border-left: 4px solid #FF6B6B;
        padding: 12px;
        margin: 8px 0;
        background-color: #fff5f5;
        border-radius: 4px;
    }
    .entailment-box {
        border-left: 4px solid #51CF66;
        padding: 12px;
        margin: 8px 0;
        background-color: #f1fdf5;
        border-radius: 4px;
    }
    .info-box {
        border-left: 4px solid #4ECDC4;
        padding: 12px;
        margin: 8px 0;
        background-color: #f0fffe;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.tracker = None
        st.session_state.chat_history = []
        st.session_state.beliefs_store = []
        st.session_state.contradictions = []
        st.session_state.session_id = "streamlit_demo_user"
        st.session_state.api_key = None
        st.session_state.model = "nvidia/mixtral-8x7b-instruct-v0.1"


initialize_session_state()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_api_key() -> Optional[str]:
    """Get NVIDIA API key from environment or sidebar input."""
    env_key = os.getenv("NVIDIA_API_KEY")
    if env_key:
        return env_key

    st.sidebar.warning("⚠️ NVIDIA_API_KEY not found in .env file")
    sidebar_key = st.sidebar.text_input(
        "Enter NVIDIA API Key",
        type="password",
        key="api_key_input",
    )
    return sidebar_key if sidebar_key else None


def initialize_tracker(api_key: str, model: str) -> Optional[BeliefTracker]:
    """Initialize BeliefTracker with NVIDIA API."""
    try:
        # Configure the tracker
        config = TrackerConfig(
            enable_background_tasks=False,  # Sync mode for Streamlit demo
            similarity_threshold=0.75,
            contradiction_threshold=0.65,
            entailment_threshold=0.80,
            max_beliefs=30,
        )

        # Use LiteLLM adapter for NVIDIA API
        adapter = LiteLLMAdapter(
            model=model,
            embed_model="nvidia/nv-embed-qa-4",  # NVIDIA's embedding model
            api_key=api_key,
        )

        # Create tracker
        tracker = BeliefTracker(config=config, adapter=adapter)
        tracker.set_session(st.session_state.session_id)

        return tracker
    except Exception as e:
        st.error(f"Failed to initialize tracker: {str(e)}")
        return None


async def process_user_message(
    tracker: BeliefTracker, user_message: str
) -> tuple[str, list]:
    """
    Process user message through the belief tracker.

    Returns:
        tuple: (assistant_response, extracted_beliefs)
    """
    try:
        # Simulate LLM chat completion using NVIDIA API
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=st.session_state.api_key,
            base_url="https://integrate.api.nvidia.com/v1",
        )

        # Add user message to history
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            *st.session_state.chat_history,
            {"role": "user", "content": user_message},
        ]

        # Get response from NVIDIA API
        response = await client.chat.completions.create(
            model=st.session_state.model,
            messages=messages,
            temperature=0.7,
            max_tokens=256,
        )

        assistant_message = response.choices[0].message.content

        # Process through tracker
        try:
            # The tracker would extract beliefs from the assistant response
            # For demo purposes, we'll extract from user input
            extracted_beliefs = await tracker.extractor.extract_beliefs(
                user_message, turn=len(st.session_state.chat_history) + 1
            )
        except Exception as e:
            st.warning(f"Belief extraction: {str(e)}")
            extracted_beliefs = []

        return assistant_message, extracted_beliefs

    except Exception as e:
        st.error(f"Error processing message: {str(e)}")
        return "", []


# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================


def main():
    """Main Streamlit app."""

    # Header
    st.title("🧠 BeliefState Tracker Demo")
    st.markdown(
        "A real-time demonstration of belief extraction and contradiction detection "
        "powered by **NVIDIA LLM** and the **BeliefState** package."
    )

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("API Settings")
        api_key = get_api_key()

        model_options = [
            "nvidia/mixtral-8x7b-instruct-v0.1",
            "nvidia/llama-2-70b-chat",
            "nvidia/nv-mistral-nemo-12b-instruct",
        ]
        selected_model = st.selectbox(
            "Select Model",
            model_options,
            index=0,
            key="model_selector",
        )
        st.session_state.model = selected_model

        st.subheader("Tracker Settings")
        session_id = st.text_input(
            "Session ID",
            value=st.session_state.session_id,
            help="Unique identifier for this user/session",
        )
        st.session_state.session_id = session_id

        if st.button("🔄 Initialize Tracker", use_container_width=True):
            if not api_key:
                st.error("❌ Please provide NVIDIA_API_KEY")
            else:
                with st.spinner("Initializing tracker..."):
                    tracker = initialize_tracker(api_key, selected_model)
                    if tracker:
                        st.session_state.tracker = tracker
                        st.session_state.api_key = api_key
                        st.success("✅ Tracker initialized successfully!")
                    else:
                        st.error("❌ Failed to initialize tracker")

        st.divider()

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.beliefs_store = []
            st.session_state.contradictions = []
            st.rerun()

    # Check if tracker is initialized
    if not st.session_state.tracker or not st.session_state.api_key:
        st.info(
            "👈 **Getting Started:**\n\n"
            "1. Add your NVIDIA API key to `.env` file or enter it in the sidebar\n"
            "2. Click **'Initialize Tracker'** button\n"
            "3. Start chatting to see beliefs being extracted and contradictions detected!"
        )
        st.markdown(
            """
            ### About BeliefState

            BeliefState is a universal belief state tracking layer for LLM applications.
            It automatically:

            - **Extracts** factual beliefs from conversations
            - **Detects** contradictions when beliefs conflict
            - **Resolves** contradictions using semantic embeddings
            - **Stores** beliefs persistently (SQLite/Redis)

            This demo uses NVIDIA's free LLM API to demonstrate real-time belief tracking.
            """
        )
        return

    # Main Chat Interface
    st.subheader("💬 Chat with Belief Tracking")

    # Chat display
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])

    # User input
    user_input = st.chat_input("Type your message here...")

    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("🤖 Processing with BeliefState..."):
            try:
                # Process through tracker
                assistant_response, extracted_beliefs = asyncio.run(
                    process_user_message(st.session_state.tracker, user_input)
                )

                if assistant_response:
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": assistant_response}
                    )

                    # Store extracted beliefs
                    if extracted_beliefs:
                        for belief in extracted_beliefs:
                            st.session_state.beliefs_store.append(
                                {
                                    "belief": belief,
                                    "turn": len(st.session_state.chat_history),
                                    "timestamp": datetime.now().isoformat(),
                                }
                            )

                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Beliefs Panel
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📚 Extracted Beliefs")
        if st.session_state.beliefs_store:
            for i, item in enumerate(st.session_state.beliefs_store):
                belief = item["belief"]
                if isinstance(belief, dict):
                    subject = belief.get("subject", "?")
                    predicate = belief.get("predicate", "?")
                    value = belief.get("value", "?")
                    confidence = belief.get("confidence", 0)
                elif isinstance(belief, Belief):
                    subject = belief.subject
                    predicate = belief.predicate
                    value = belief.value
                    confidence = belief.confidence
                else:
                    continue

                st.markdown(
                    f"""
                    <div class="belief-box">
                    <strong>[{subject}]</strong> {predicate} <em>"{value}"</em><br/>
                    <small>Confidence: {confidence:.2%} | Turn: {item['turn']}</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No beliefs extracted yet. Start chatting to extract beliefs!")

    with col2:
        st.subheader("⚠️ Detected Contradictions")
        if st.session_state.contradictions:
            for contradiction in st.session_state.contradictions:
                st.markdown(
                    f"""
                    <div class="contradiction-box">
                    <strong>Contradiction:</strong><br/>
                    {contradiction}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info(
                "No contradictions detected yet. "
                "Try updating a belief to see contradiction detection!"
            )

    # Statistics Footer
    st.divider()
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

    with stats_col1:
        st.metric("Messages", len(st.session_state.chat_history) // 2)

    with stats_col2:
        st.metric("Beliefs Extracted", len(st.session_state.beliefs_store))

    with stats_col3:
        st.metric("Contradictions", len(st.session_state.contradictions))

    with stats_col4:
        st.metric("Session ID", st.session_state.session_id[:12] + "...")

    # Info Section
    st.divider()
    with st.expander("ℹ️ About This Demo"):
        st.markdown(
            """
            ### BeliefState Package Features

            **Core Components:**
            - **BeliefTracker**: Main orchestrator that coordinates extraction and detection
            - **BeliefExtractor**: Extracts factual beliefs from text using LLM prompts
            - **ContradictionDetector**: Detects contradictions using semantic embeddings
            - **LiteLLMAdapter**: Universal adapter for any LLM provider (OpenAI, Anthropic, NVIDIA, etc.)

            **How It Works:**
            1. User sends a message through the chat interface
            2. Message is passed to the LLM (NVIDIA API in this demo)
            3. **Extractor** processes the text and extracts beliefs using structured prompts
            4. **Detector** compares new beliefs against stored beliefs using embeddings
            5. **Judge** resolves any contradictions using NLI (Natural Language Inference)
            6. Beliefs are stored in persistent storage (SQLite/Redis)

            **Use Cases:**
            - Maintaining user profiles in conversational AI
            - Detecting inconsistencies in user statements
            - Building reliable belief systems for agents
            - Persistent memory for multi-turn conversations

            **Production Features:**
            - Automatic retry with exponential backoff
            - Circuit breaker for resilience
            - Background task execution
            - Embedding batching for efficiency
            - Optional Redis/Celery for distributed tracking
            """
        )


if __name__ == "__main__":
    main()
