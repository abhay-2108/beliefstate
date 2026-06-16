import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from beliefstate import BeliefTracker, TrackerConfig, OpenAIAdapter

# Load env variables from .env
load_dotenv()


async def main():
    api_key = os.getenv("NVIDIA_API_KEY")
    model_name = os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")
    api_url = os.getenv("NVIDIA_API_URL", "https://integrate.api.nvidia.com/v1")

    if not api_key:
        print("Error: NVIDIA_API_KEY is not set in the environment or .env file.")
        return

    print("Configuring NVIDIA NIM client...")
    print(f"API URL: {api_url}")
    print(f"Model: {model_name}")

    # NVIDIA NIM runs on an OpenAI-compatible API
    # We use nvidia/llama-nemotron-embed-1b-v2 as the embedding model
    client = AsyncOpenAI(api_key=api_key, base_url=api_url)
    adapter = OpenAIAdapter(
        client=client,
        model=model_name,
        embed_model="nvidia/llama-nemotron-embed-1b-v2",
        embed_kwargs={"extra_body": {"input_type": "passage"}},
    )

    config = TrackerConfig(
        store_type="sqlite",
        store_kwargs={"db_path": ":memory:"},
        enable_background_tasks=False,  # Run synchronously to wait for results in the script
    )

    tracker = BeliefTracker(config=config, adapter=adapter)
    tracker.set_session("nvidia-test-session")

    @tracker.wrap
    async def chat_with_assistant(messages):
        print("\nSending request to NVIDIA NIM...")
        response = await client.chat.completions.create(
            model=model_name, messages=messages
        )
        return response

    # 1. User introduces themselves
    user_prompt = "Hello! My name is Abhay, and I prefer dark mode interface."
    messages = [{"role": "user", "content": user_prompt}]

    response = await chat_with_assistant(messages)
    assistant_reply = response.choices[0].message.content
    print(f"NVIDIA NIM Response: {assistant_reply}")

    # Retrieve and print saved beliefs
    beliefs = await tracker.store.get_beliefs("nvidia-test-session")
    print("\n--- Extracted Beliefs in SQLite ---")
    if not beliefs:
        print(
            "No beliefs extracted. Check if the prompt generated output that allows extraction."
        )
    for b in beliefs:
        print(f"Fact: [{b.subject}] {b.predicate} '{b.value}'")


if __name__ == "__main__":
    asyncio.run(main())
