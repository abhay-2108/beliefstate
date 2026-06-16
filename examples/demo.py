import asyncio
import ollama
from beliefstate.config import TrackerConfig
from beliefstate.adapters.ollama import OllamaAdapter
from beliefstate.tracker import BeliefTracker

# 1. Setup Tracker
config = TrackerConfig(
    enable_background_tasks=False,  # Run synchronously for the demo to see immediate console prints
    similarity_threshold=0.7,
    contradiction_threshold=0.6,
)

# Use the local Ollama models you have installed
adapter = OllamaAdapter(model="qwen2.5:7b", embed_model="nomic-embed-text:v1.5")
tracker = BeliefTracker(config=config, adapter=adapter)


# 2. Decorate your standard LLM call
@tracker.wrap
async def chat_with_ollama(messages):
    client = ollama.AsyncClient()
    response = await client.chat(model="qwen2.5:7b", messages=messages)
    return response


async def main():
    tracker.set_session("demo_user_001")

    print("==================================================")
    print(" Starting Belief State Tracker Demo (Ollama) ")
    print("==================================================\n")

    history = [
        {
            "role": "system",
            "content": "You are a friendly assistant. Keep your answers brief.",
        }
    ]

    # --- Turn 1 ---
    prompt1 = "Hi! My name is Alice and I strictly write code in Python. I'm building an AI app."
    print(f"User: {prompt1}")
    history.append({"role": "user", "content": prompt1})

    # Check pending conflicts (should be empty on turn 1)
    for c in tracker.get_pending_conflicts():
        history.append({"role": "system", "content": c})

    print("Assistant is thinking...")
    response1 = await chat_with_ollama(messages=history)
    ai_text1 = response1["message"]["content"]
    print(f"AI: {ai_text1}\n")
    history.append({"role": "assistant", "content": ai_text1})

    print(">>> Checking Belief Store...")
    beliefs = await tracker.store.get_beliefs("demo_user_001")
    for b in beliefs:
        print(
            f"  - [{b.subject}] {b.predicate} '{b.value}' (Confidence: {b.confidence})"
        )
    print()

    # --- Turn 2 ---
    prompt2 = (
        "Actually, I changed my mind. I hate Python, I'm rewriting everything in Rust."
    )
    print(f"User: {prompt2}")
    history.append({"role": "user", "content": prompt2})

    # Check pending conflicts (should see the contradiction!)
    conflicts = tracker.get_pending_conflicts()
    for c in conflicts:
        print(f"!! INJECTING CONFLICT INTO PROMPT: {c}")
        history.append({"role": "system", "content": c})

    print("\nAssistant is thinking...")
    response2 = await chat_with_ollama(messages=history)
    ai_text2 = response2["message"]["content"]
    print(f"AI: {ai_text2}\n")
    history.append({"role": "assistant", "content": ai_text2})

    print(">>> Checking Belief Store...")
    beliefs = await tracker.store.get_beliefs("demo_user_001")
    for b in beliefs:
        print(
            f"  - [{b.subject}] {b.predicate} '{b.value}' (Confidence: {b.confidence})"
        )
    print()

    print("\n==================================================")
    print("Demo completed successfully!")
    print("==================================================")


if __name__ == "__main__":
    asyncio.run(main())
