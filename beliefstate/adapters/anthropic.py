from typing import Any, Dict, List, Optional
from beliefstate.adapters.base import ProviderAdapter
from beliefstate.call import LLMCall, LLMResponse

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = Any

class AnthropicAdapter(ProviderAdapter):
    """Adapter for Anthropic API."""
    
    def __init__(self, client: Optional[Any] = None, model: str = "claude-3-5-sonnet-latest", embed_model: str = "voyage-large-2"):
        self.model = model
        self.embed_model = embed_model
        
        if client:
            self.client = client
        else:
            try:
                from anthropic import AsyncAnthropic
                self.client = AsyncAnthropic()
            except (ImportError, Exception):
                self.client = None

    def to_llm_call(self, *args, **kwargs) -> LLMCall:
        messages = kwargs.get("messages", [])
        if not messages and len(args) > 0 and isinstance(args[0], list):
            messages = args[0]
            
        system_prompt = kwargs.get("system", None)
        
        return LLMCall(
            messages=messages,
            kwargs=kwargs,
            system=system_prompt,
            metadata={"model": kwargs.get("model", self.model)}
        )
        
    def to_llm_response(self, response: Any) -> LLMResponse:
        # Handle generic dict or anthropic Message object
        if isinstance(response, dict):
            content = response.get("content", [])
            text = content[0].get("text", "") if content else ""
        else:
            text = response.content[0].text
            
        return LLMResponse(
            text=text,
            raw_response=response
        )

    async def generate(self, call: LLMCall, response_format: Optional[Any] = None) -> LLMResponse:
        if not self.client:
            raise RuntimeError("Anthropic client not installed. Install with `pip install anthropic`.")
            
        import json
        kwargs = call.kwargs.copy()
        
        # Prompt-based fallback formatting instructions if the provider does not support native schema validation
        messages = call.messages.copy()
        if response_format:
            try:
                schema_json = json.dumps(response_format.model_json_schema())
            except Exception:
                schema_json = "{}"
            instruction = f"\n\nIMPORTANT: You must return a valid JSON object or JSON array conforming strictly to the following JSON Schema: {schema_json}. Do NOT include any explanations, markdown code blocks, or preamble in your response. Output only raw JSON."
            if messages:
                last_m = messages[-1].copy()
                last_m["content"] = last_m.get("content", "") + instruction
                messages[-1] = last_m
            else:
                messages.append({"role": "user", "content": instruction})
                
        kwargs["messages"] = messages
        if call.system and "system" not in kwargs:
            kwargs["system"] = call.system
        if "model" not in kwargs:
            kwargs["model"] = self.model
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 1024 # Anthropic requires max_tokens
            
        response = await self.client.messages.create(**kwargs)
        return self.to_llm_response(response)

    async def get_embedding(self, text: str) -> List[float]:
        # Anthropic does not provide native embeddings via their main models, they recommend Voyage AI.
        # So we cannot natively call client.embeddings.create.
        raise NotImplementedError(
            "Anthropic does not natively provide embeddings. "
            "Please use the OpenAI or Ollama adapter for the internal tracker pipeline, "
            "or implement a custom embedding function."
        )

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError(
            "Anthropic does not natively provide embeddings. "
            "Please use the OpenAI or Ollama adapter for the internal tracker pipeline, "
            "or implement a custom embedding function."
        )
