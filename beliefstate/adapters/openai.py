from typing import Any, Dict, List, Optional
from beliefstate.adapters.base import ProviderAdapter
from beliefstate.call import LLMCall, LLMResponse

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = Any


class OpenAIAdapter(ProviderAdapter):
    """Adapter for OpenAI API."""

    def __init__(
        self,
        client: Optional[Any] = None,
        model: str = "gpt-4o-mini",
        embed_model: str = "text-embedding-3-small",
        embed_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.embed_model = embed_model
        self.embed_kwargs = embed_kwargs or {}

        if client:
            self.client = client
        else:
            try:
                from openai import AsyncOpenAI

                self.client = AsyncOpenAI()  # Uses OPENAI_API_KEY from environment
            except ImportError:
                self.client = None

    def to_llm_call(self, *args, **kwargs) -> LLMCall:
        messages = kwargs.get("messages", [])
        if not messages and len(args) > 0 and isinstance(args[0], list):
            messages = args[0]

        system_prompt = None
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "system":
                system_prompt = m.get("content")
            elif hasattr(m, "role") and m.role == "system":
                system_prompt = m.content

        # Clean kwargs of things we shouldn't persist directly if needed, or keep all
        # For universal LLMCall, we just want to know what the user sent
        return LLMCall(
            messages=messages,
            kwargs=kwargs,
            system=system_prompt,
            metadata={"model": kwargs.get("model", self.model)},
        )

    def to_llm_response(self, response: Any) -> LLMResponse:
        # Handle dict (like from REST API) or ChatCompletion object
        if isinstance(response, dict):
            text = (
                response.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
        else:
            text = response.choices[0].message.content

        return LLMResponse(text=text, raw_response=response)

    async def generate(
        self, call: LLMCall, response_format: Optional[Any] = None
    ) -> LLMResponse:
        if not self.client:
            raise RuntimeError(
                "OpenAI client not installed or configured. Install with `pip install openai`."
            )

        kwargs = call.kwargs.copy()
        kwargs["messages"] = call.messages
        if "model" not in kwargs:
            kwargs["model"] = self.model

        if response_format:
            try:
                response = await self.client.beta.chat.completions.parse(
                    response_format=response_format, **kwargs
                )
            except AttributeError:
                kwargs["response_format"] = response_format
                response = await self.client.chat.completions.create(**kwargs)
        else:
            response = await self.client.chat.completions.create(**kwargs)

        return self.to_llm_response(response)

    async def get_embedding(self, text: str) -> List[float]:
        res = await self.get_embeddings([text])
        return res[0]

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not self.client:
            raise RuntimeError(
                "OpenAI client not installed or configured. Install with `pip install openai`."
            )
        if not texts:
            return []

        kwargs = {"input": texts, "model": self.embed_model}
        if self.embed_kwargs:
            kwargs.update(self.embed_kwargs)

        response = await self.client.embeddings.create(**kwargs)
        return [item.embedding for item in response.data]
