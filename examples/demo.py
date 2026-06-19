import asyncio
from beliefstate.config import TrackerConfig
from beliefstate.adapters.openai import OpenAIAdapter
from beliefstate.tracker import BeliefTracker

try:
    import respx
    import httpx
except ImportError:
    respx = None
    httpx = None


# Mock extraction responses (simulates what the extraction LLM would return)
MOCK_EXTRACTION_1 = [
    {
        "subject": "USER",
        "predicate": "name is",
        "value": "Alice",
        "confidence": 0.95,
        "source": "user",
    },
    {
        "subject": "USER",
        "predicate": "writes code in",
        "value": "Python",
        "confidence": 0.95,
        "source": "user",
    },
    {
        "subject": "USER",
        "predicate": "is building",
        "value": "AI app",
        "confidence": 0.9,
        "source": "user",
    },
]

MOCK_EXTRACTION_2 = [
    {
        "subject": "USER",
        "predicate": "hates",
        "value": "Python",
        "confidence": 0.95,
        "source": "user",
    },
    {
        "subject": "USER",
        "predicate": "is rewriting in",
        "value": "Rust",
        "confidence": 0.9,
        "source": "user",
    },
]

# Mock embedding responses (384-dim embeddings from sentence-transformers)
MOCK_EMBEDDING = [0.1] * 384  # Simplified mock embedding


async def setup_mock_responses(mock: respx.MockRouter) -> None:
    """Configure respx mock routes for all OpenAI API calls."""

    # Mock chat completions for assistant responses
    mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "That sounds great! I'd be happy to help with your AI app.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
            },
        )
    )

    # Mock embeddings API
    mock.post("https://api.openai.com/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "embedding": MOCK_EMBEDDING, "index": 0}
                    for _ in range(10)  # Support variable batch sizes
                ],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 50, "total_tokens": 50},
            },
        )
    )


async def main():
    """Run the demo with mocked HTTP responses (no external services needed)."""
    if respx is None:
        print("Error: respx is required for this demo. Install with: pip install respx httpx")
        return

    # 1. Setup Tracker with OpenAI adapter (using mocked API)
    config = TrackerConfig(
        enable_background_tasks=False,  # Run synchronously for immediate console output
        similarity_threshold=0.7,
        contradiction_threshold=0.6,
    )

    adapter = OpenAIAdapter(model="gpt-4o-mini", embed_model="text-embedding-3-small")
    tracker = BeliefTracker(config=config, adapter=adapter)

    # 2. Decorate a standard LLM call
    @tracker.wrap
    async def chat(messages):
        import openai

        client = openai.AsyncOpenAI()
        response = await client.chat.completions.create(
            model="gpt-4o-mini", messages=messages
        )
        return response

    # 3. Run the demo with mocked HTTP responses
    async with respx.mock:
        await setup_mock_responses(respx.mock)

        tracker.set_session("demo_user_001")

        print("==================================================")
        print(" BeliefState Tracker Demo (Mocked)")
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

        print("Assistant is responding...\n")
        response1 = await chat(messages=history)
        ai_text1 = response1.choices[0].message.content
        print(f"AI: {ai_text1}\n")
        history.append({"role": "assistant", "content": ai_text1})

        print(">>> Checking Belief Store...")
        beliefs = await tracker.store.get_beliefs("demo_user_001")
        if beliefs:
            for b in beliefs:
                print(
                    f"  - [{b.subject}] {b.predicate} '{b.value}' (Confidence: {b.confidence:.2f})"
                )
        else:
            print("  (No beliefs extracted - mock responses not triggering extraction)")
        print()

        # --- Turn 2 ---
        prompt2 = "Actually, I changed my mind. I hate Python, I'm rewriting everything in Rust."
        print(f"User: {prompt2}")
        history.append({"role": "user", "content": prompt2})

        # Check pending conflicts (should see the contradiction!)
        conflicts = tracker.get_pending_conflicts()
        if conflicts:
            for c in conflicts:
                print(f"!! CONFLICT DETECTED: {c}")
                history.append({"role": "system", "content": c})
        else:
            print("(No conflicts detected - beliefs may not have been extracted)")

        print("\nAssistant is responding...\n")
        response2 = await chat(messages=history)
        ai_text2 = response2.choices[0].message.content
        print(f"AI: {ai_text2}\n")
        history.append({"role": "assistant", "content": ai_text2})

        print(">>> Checking Belief Store after Turn 2...")
        beliefs = await tracker.store.get_beliefs("demo_user_001")
        if beliefs:
            for b in beliefs:
                print(
                    f"  - [{b.subject}] {b.predicate} '{b.value}' (Confidence: {b.confidence:.2f})"
                )
        else:
            print("  (No beliefs in store)")
        print()

        print("\n==================================================")
        print("Demo completed successfully!")
        print("Extraction and detection pipeline ran with mocked HTTP.")
        print("==================================================")


if __name__ == "__main__":
    asyncio.run(main())
