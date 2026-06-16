from typing import Any, List, Optional
from beliefstate.adapters.base import ProviderAdapter
from beliefstate.call import LLMCall, LLMResponse

try:
    import litellm
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False

class LiteLLMAdapter(ProviderAdapter):
    """Adapter for LiteLLM API, routing to any provider (Azure, Bedrock, OpenAI, etc.) via LiteLLM."""
    
    def __init__(self, model: str = "gpt-4o-mini", embed_model: str = "text-embedding-3-small", **kwargs: Any):
        if not HAS_LITELLM:
            raise ImportError(
                "LiteLLM is not installed. Install with `pip install beliefstate[litellm]` or `pip install litellm`."
            )
        self.model = model
        self.embed_model = embed_model
        self.kwargs = kwargs

    def to_llm_call(self, *args, **kwargs) -> LLMCall:
        messages = kwargs.get("messages", [])
        if not messages and len(args) > 0 and isinstance(args[0], list):
            messages = args[0]
            
        system_prompt = kwargs.get("system", None)
        if not system_prompt:
            for m in messages:
                if isinstance(m, dict) and m.get("role") == "system":
                    system_prompt = m.get("content")
                    break
                    
        return LLMCall(
            messages=messages,
            kwargs=kwargs,
            system=system_prompt,
            metadata={"model": kwargs.get("model", self.model)}
        )

    def to_llm_response(self, response: Any) -> LLMResponse:
        # LiteLLM's response supports standard openai response format/attributes
        text = ""
        if hasattr(response, "choices") and len(response.choices) > 0:
            text = response.choices[0].message.content or ""
        elif isinstance(response, dict):
            if "choices" in response and len(response["choices"]) > 0:
                text = response["choices"][0].get("message", {}).get("content", "")
                
        return LLMResponse(
            text=text,
            raw_response=response
        )

    async def generate(self, call: LLMCall, response_format: Optional[Any] = None) -> LLMResponse:
        if not HAS_LITELLM:
            raise ImportError("LiteLLM is not installed.")
            
        kwargs = self.kwargs.copy()
        kwargs.update(call.kwargs)
        kwargs["messages"] = call.messages
        if "model" not in kwargs:
            kwargs["model"] = self.model
            
        if call.system and "system" not in kwargs:
            # LiteLLM handles system instruction either via system keyword or standard message list
            # Injecting it into messages for general compatibility
            has_system = any(m.get("role") == "system" for m in kwargs["messages"])
            if not has_system:
                kwargs["messages"] = [{"role": "system", "content": call.system}] + kwargs["messages"]
                
        if response_format:
            kwargs["response_format"] = response_format
                
        response = await litellm.acompletion(**kwargs)
        return self.to_llm_response(response)

    async def get_embedding(self, text: str) -> List[float]:
        res = await self.get_embeddings([text])
        return res[0]

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not HAS_LITELLM:
            raise ImportError("LiteLLM is not installed.")
        if not texts:
            return []
            
        kwargs = self.kwargs.copy()
        response = await litellm.aembedding(
            model=self.embed_model,
            input=texts,
            **kwargs
        )
        # In LiteLLM, response.data has list of dicts/objects containing embedding keys
        embeddings = []
        for item in response.data:
            if isinstance(item, dict):
                embeddings.append(item["embedding"])
            else:
                embeddings.append(getattr(item, "embedding"))
        return embeddings
