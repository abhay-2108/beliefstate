import os
import asyncio
from beliefstate import BeliefTracker, TrackerConfig, OllamaAdapter

try:
    from ollama import AsyncClient
except ImportError:
    print("Error: The 'ollama' library is not installed.")
    print("Install it with: pip install ollama")
    exit(1)

import logging

logging.basicConfig(level=logging.INFO)


async def main():
    model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    print("Checking connection to local Ollama server...")
    client = AsyncClient()

    try:
        # Check if Ollama is running and has the model
        tags = await client.list()
        models = [m.get("model") for m in tags.get("models", [])]
        print(f"Available local models in Ollama: {models}")

        # Check if the requested models are present
        if model_name not in models and f"{model_name}:latest" not in models:
            print(f"Warning: Model '{model_name}' might not be pulled yet in Ollama.")
            print(f"You can pull it by running: ollama pull {model_name}")

        if embed_model not in models and f"{embed_model}:latest" not in models:
            print(
                f"Warning: Embedding model '{embed_model}' might not be pulled yet in Ollama."
            )
            print(f"You can pull it by running: ollama pull {embed_model}")

    except Exception as e:
        print(f"Error connecting to local Ollama: {e}")
        print(
            "Please ensure that Ollama is running locally (default: http://localhost:11434)."
        )
        return

    print(
        f"\nInitializing OllamaAdapter with model='{model_name}', embed_model='{embed_model}'..."
    )
    adapter = OllamaAdapter(client=client, model=model_name, embed_model=embed_model)

    config = TrackerConfig(
        store_type="sqlite",
        store_kwargs={"db_path": ":memory:"},
        enable_background_tasks=False,  # Run synchronously to track directly in the script
    )

    tracker = BeliefTracker(config=config, adapter=adapter)
    tracker.set_session("ollama-test-session")

    @tracker.wrap
    async def chat_with_ollama(messages):
        print(f"\nSending chat prompt to local Ollama ({model_name})...")
        response = await client.chat(model=model_name, messages=messages)
        return response

    # 1. User chat
    user_prompt = "Hey there! I am Abhay, and I prefer dark mode interface."
    messages = [{"role": "user", "content": user_prompt}]

    try:
        response = await chat_with_ollama(messages)
        content = response.get("message", {}).get("content", "")
        print(f"Ollama Response: {content}")

        # Retrieve and print saved beliefs
        beliefs = await tracker.store.get_beliefs("ollama-test-session")
        print("\n--- Extracted Beliefs in SQLite ---")
        if not beliefs:
            print("No beliefs extracted.")
        for b in beliefs:
            print(f"Fact: [{b.subject}] {b.predicate} '{b.value}'")

    except Exception as e:
        print(f"\nAn error occurred during Ollama generation/tracking: {e}")


if __name__ == "__main__":
    asyncio.run(main())
