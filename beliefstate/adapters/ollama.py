from typing import Any, Dict, List, Optional
from beliefstate.adapters.base import ProviderAdapter
from beliefstate.call import LLMCall, LLMResponse

try:
    from ollama import AsyncClient
except ImportError:
    AsyncClient = Any

class OllamaAdapter(ProviderAdapter):
    """Adapter for Ollama API."""
    
    def __init__(self, client: Optional[Any] = None, model: str = "llama3.2", embed_model: str = "nomic-embed-text"):
        self.model = model
        self.embed_model = embed_model
        
        if client:
            self.client = client
        else:
            try:
                from ollama import AsyncClient
                self.client = AsyncClient() # Defaults to local Ollama server
            except (ImportError, Exception):
                self.client = None

    def to_llm_call(self, *args, **kwargs) -> LLMCall:
        messages = kwargs.get("messages", [])
        if not messages and len(args) > 1 and isinstance(args[1], list):
            messages = args[1]
            
        system_prompt = None
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "system":
                system_prompt = m.get("content")
                
        return LLMCall(
            messages=messages,
            kwargs=kwargs,
            system=system_prompt,
            metadata={"model": kwargs.get("model", self.model)}
        )
        
    def to_llm_response(self, response: Any) -> LLMResponse:
        if isinstance(response, dict):
            text = response.get("message", {}).get("content", "")
        else:
            text = getattr(response, "message", {}).get("content", "")
            if not text and hasattr(response, "message"):
                text = getattr(response.message, "content", "")
                
        return LLMResponse(
            text=text,
            raw_response=response
        )

    async def generate(self, call: LLMCall, response_format: Optional[Any] = None) -> LLMResponse:
        if not self.client:
            raise RuntimeError("Ollama client not installed. Install with `pip install ollama`.")
            
        kwargs = call.kwargs.copy()
        kwargs["messages"] = call.messages
        if "model" not in kwargs:
            kwargs["model"] = self.model
            
        if response_format:
            try:
                kwargs["format"] = response_format.model_json_schema()
            except Exception:
                kwargs["format"] = "json"
            
        response = await self.client.chat(**kwargs)
        return self.to_llm_response(response)

    async def get_embedding(self, text: str) -> List[float]:
        if not self.client:
            raise RuntimeError("Ollama client not installed. Install with `pip install ollama`.")
            
        response = await self.client.embeddings(
            model=self.embed_model,
            prompt=text
        )
        return response.get("embedding", [])

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not self.client:
            raise RuntimeError("Ollama client not installed. Install with `pip install ollama`.")
        if not texts:
            return []
            
        import asyncio
        tasks = [self.get_embedding(text) for text in texts]
        return list(await asyncio.gather(*tasks))
