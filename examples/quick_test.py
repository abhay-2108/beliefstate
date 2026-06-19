#!/usr/bin/env python3
"""
Quick test of BeliefState with Ollama
Run: python examples/quick_test.py
"""

import asyncio
from beliefstate import BeliefTracker, TrackerConfig
from beliefstate.adapters.ollama import OllamaAdapter


async def main():
    print("🧠 BeliefState + Ollama Quick Test")
    print("=" * 50)

    # Initialize tracker
    print("\n1️⃣ Initializing tracker...")
    config = TrackerConfig(
        enable_background_tasks=False,
        similarity_threshold=0.75,
        contradiction_threshold=0.65,
        max_beliefs=20,
    )

    adapter = OllamaAdapter(
        model="qwen2.5:7b",
        embed_model="nomic-embed-text:v1.5",
        host="http://localhost",
        port=11434,
    )

    tracker = BeliefTracker(config=config, adapter=adapter)
    tracker.set_session("test_session")
    print("✅ Tracker initialized!")

    # Test wrapper
    print("\n2️⃣ Testing @tracker.wrap decorator...")

    @tracker.wrap
    async def test_chat():
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key="ollama",
            base_url="http://localhost:11434/v1",
        )

        messages = [
            {"role": "system", "content": "You are helpful. Be brief."},
            {"role": "user", "content": "I love Python programming"},
        ]

        response = await client.chat.completions.create(
            model="qwen2.5:7b",
            messages=messages,
            temperature=0.7,
            max_tokens=256,
        )

        return response

    response = await test_chat()
    print(f"✅ LLM Response: {response.choices[0].message.content[:100]}...")

    # Check beliefs
    print("\n3️⃣ Checking extracted beliefs...")
    beliefs = await tracker.store.get_beliefs("test_session")
    print(f"✅ Beliefs extracted: {len(beliefs)}")
    for belief in beliefs:
        print(
            f"   - [{belief.subject}] {belief.predicate} '{belief.value}' "
            f"(confidence: {belief.confidence:.2f})"
        )

    # Test contradiction
    print("\n4️⃣ Testing contradiction detection...")

    @tracker.wrap
    async def contradict_chat():
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key="ollama",
            base_url="http://localhost:11434/v1",
        )

        messages = [
            {"role": "system", "content": "You are helpful. Be brief."},
            {"role": "user", "content": "Actually, I hate Python"},
        ]

        response = await client.chat.completions.create(
            model="qwen2.5:7b",
            messages=messages,
            temperature=0.7,
            max_tokens=256,
        )

        return response

    response = await contradict_chat()
    print(f"✅ LLM Response: {response.choices[0].message.content[:100]}...")

    # Final beliefs
    print("\n5️⃣ Final beliefs after contradiction:")
    beliefs = await tracker.store.get_beliefs("test_session")
    print(f"✅ Total beliefs: {len(beliefs)}")
    for belief in beliefs:
        print(
            f"   - [{belief.subject}] {belief.predicate} '{belief.value}' "
            f"(confidence: {belief.confidence:.2f})"
        )

    print("\n" + "=" * 50)
    print("✅ All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
