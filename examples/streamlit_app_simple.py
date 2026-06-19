"""
BeliefState Streamlit Demo - Local Ollama Version

A lightweight demonstration of the BeliefState package using local Ollama.
Uses the @tracker.wrap decorator pattern for correct integration.

Requirements:
    pip install -r requirements.txt
    ollama serve (in separate terminal)

Usage:
    streamlit run examples/streamlit_app_simple.py
"""

import asyncio
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

from beliefstate import BeliefTracker, TrackerConfig
from beliefstate.adapters.ollama import OllamaAdapter

load_dotenv()

# ============================================================================
# STREAMLIT PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="BeliefState Demo - Ollama",
    page_icon="🧠",
    layout="wide",
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown(
    """
    <style>
    .belief-item {
        background-color: #e8f4f8;
        border-left: 4px solid #0066cc;
        padding: 10px;
        margin: 5px 0;
        border-radius: 4px;
    }
    .contradiction-item {
        background-color: #ffe8e8;
        border-left: 4px solid #cc0000;
        padding: 10px;
        margin: 5px 0;
        border-radius: 4px;
    }
    .success-box {
        background-color: #e8f8e8;
        border-left: 4px solid #00cc00;
        padding: 10px;
        margin: 5px 0;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# SESSION STATE
# ============================================================================


@st.cache_resource
def get_session_state():
    """Get or create session state."""
    return {
        "tracker": None,
        "model": "qwen2.5:7b",
        "embed_model": "nomic-embed-text:v1.5",
        "session_id": "demo_user",
        "messages": [],
        "beliefs": [],
        "turn": 0,
    }


state = get_session_state()


# ============================================================================
# TRACKER INITIALIZATION
# ============================================================================


def init_tracker(model: str, embed_model: str) -> Optional[BeliefTracker]:
    """Initialize BeliefTracker with Ollama adapter."""
    try:
        config = TrackerConfig(
            enable_background_tasks=False,  # Run synchronously
            similarity_threshold=0.75,
            contradiction_threshold=0.65,
            max_beliefs=20,
        )

        adapter = OllamaAdapter(
            model=model,
            embed_model=embed_model,
            host="http://localhost",
            port=11434,
        )

        tracker = BeliefTracker(config=config, adapter=adapter)
        tracker.set_session(state["session_id"])
        
        # Test the adapter to make sure it works
        try:
            import asyncio
            result = asyncio.run(adapter.health_check())
            if result:
                st.write("✅ Ollama adapter health check passed")
            else:
                st.warning("⚠️ Ollama health check failed - extraction may not work")
        except Exception as hc_error:
            st.warning(f"⚠️ Could not verify Ollama: {str(hc_error)}")
        
        st.write("✅ Tracker initialized successfully")
        return tracker
    except Exception as e:
        st.error(f"❌ Tracker initialization failed: {str(e)}")
        import traceback
        st.write(traceback.format_exc())
        return None


# ============================================================================
# MAIN APP
# ============================================================================


def main():
    st.title("🧠 BeliefState Package Demo")
    st.markdown("Belief extraction and contradiction detection with Local Ollama")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        st.success("✅ Using Local Ollama - No API key needed!")

        # Model Selection
        models = ["qwen2.5:7b", "llama2", "neural-chat", "mistral"]
        state["model"] = st.selectbox("Chat Model", models, index=0)

        embed_models = ["nomic-embed-text:v1.5", "nomic-embed-text"]
        state["embed_model"] = st.selectbox("Embedding Model", embed_models, index=0)

        # Initialize Button
        if st.button("🚀 Initialize Tracker", use_container_width=True):
            with st.spinner("Initializing Ollama tracker..."):
                tracker = init_tracker(state["model"], state["embed_model"])
                if tracker:
                    state["tracker"] = tracker
                    st.success("✅ Tracker ready!")
                    st.rerun()
                else:
                    st.error("❌ Failed to initialize")

        st.divider()

        # Clear Data
        if st.button("🗑️ Clear All Data", use_container_width=True):
            state["messages"] = []
            state["beliefs"] = []
            state["turn"] = 0
            st.success("✅ Data cleared")
            st.rerun()

        st.divider()
        st.info(
            "**Instructions:**\n"
            "1. Make sure Ollama is running\n"
            "2. Initialize the tracker\n"
            "3. Type a message\n"
            "4. View extracted beliefs\n"
            "5. Try contradictions"
        )

    # Main Content
    if not state["tracker"]:
        st.warning("👈 Initialize tracker in the sidebar first")
        return

    # Chat History
    st.subheader("💬 Chat")
    for msg in state["messages"]:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])

    # User Input
    user_input = st.chat_input("Tell me something about yourself...")

    if user_input:
        state["messages"].append({"role": "user", "content": user_input})
        state["turn"] += 1

        with st.spinner("Processing with BeliefState..."):
            try:
                st.write(f"📤 Turn {state['turn']}: Sending to LLM...")
                
                # Define the LLM call function
                async def chat_function():
                    """LLM call to be wrapped with BeliefState tracking."""
                    from openai import AsyncOpenAI

                    client = AsyncOpenAI(
                        api_key="ollama",
                        base_url="http://localhost:11434/v1",
                    )

                    system_msg = (
                        "You are a helpful assistant. Keep responses brief and natural. "
                        "Engage naturally with the user about their interests and background."
                    )
                    messages = [
                        {"role": "system", "content": system_msg}
                    ] + state["messages"]

                    response = await client.chat.completions.create(
                        model=state["model"],
                        messages=messages,
                        temperature=0.7,
                        max_tokens=256,
                    )

                    return response

                # ✅ Apply tracker.wrap and execute
                st.write("🔍 Applying @tracker.wrap...")
                wrapped = state["tracker"].wrap(chat_function)
                st.write("⏳ Executing wrapped function (belief extraction happens here)...")
                response_obj = asyncio.run(wrapped())
                st.write("✅ LLM response received and processing complete")

                assistant_message = response_obj.choices[0].message.content
                st.write(f"💬 AI Response: {assistant_message[:100]}...")
                state["messages"].append(
                    {"role": "assistant", "content": assistant_message}
                )

                # ✅ Fetch beliefs from store AFTER wrap execution
                st.write("🔍 Querying belief store...")
                beliefs_list = asyncio.run(
                    state["tracker"].store.get_beliefs(state["session_id"])
                )
                st.write(f"📊 Store query returned {len(beliefs_list)} total beliefs")

                # Update beliefs display
                old_count = len(state["beliefs"])
                state["beliefs"] = []
                for i, belief in enumerate(beliefs_list):
                    b_item = {
                        "subject": belief.subject,
                        "predicate": belief.predicate,
                        "value": belief.value,
                        "confidence": belief.confidence,
                        "turn": belief.turn,
                    }
                    state["beliefs"].append(b_item)
                    st.write(f"  Belief {i+1}: [{belief.subject}] {belief.predicate} '{belief.value}' (conf: {belief.confidence:.0%}, turn: {belief.turn})")
                
                new_count = len(state["beliefs"])
                if new_count > old_count:
                    st.success(f"✅ {new_count - old_count} new belief(s) extracted this turn")
                else:
                    st.info(f"ℹ️ No new beliefs extracted (total: {new_count})")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback

                with st.expander("Debug Info"):
                    st.code(traceback.format_exc())

        st.rerun()

    # Beliefs Display
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📚 Extracted Beliefs")
        if state["beliefs"]:
            for belief in state["beliefs"]:
                st.markdown(
                    f"""
                    <div class="belief-item">
                    <strong>[{belief['subject']}]</strong> {belief['predicate']} 
                    <em>"{belief['value']}"</em><br/>
                    <small>Confidence: {belief['confidence']:.0%} | Turn: {belief['turn']}</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No beliefs extracted yet")

    with col2:
        st.subheader("📊 Statistics")
        st.metric("Messages", len(state["messages"]))
        st.metric("Beliefs", len(state["beliefs"]))
        st.metric("Turns", state["turn"])

    # Footer
    st.divider()
    with st.expander("ℹ️ About BeliefState"):
        st.markdown(
            """
            **BeliefState** automatically:
            - Extracts facts from conversations
            - Detects contradictions
            - Tracks persistent beliefs
            - Resolves conflicts

            **Test contradiction detection:**
            1. Say "I love Python"
            2. Then say "I hate Python"
            3. See the contradiction detected!
            """
        )


if __name__ == "__main__":
    main()
