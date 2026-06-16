from typing import Any, Dict, List, Optional
from beliefstate.adapters.base import ProviderAdapter
from beliefstate.call import LLMCall, LLMResponse

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = Any
    types = Any

class GeminiAdapter(ProviderAdapter):
    """Adapter for Google GenAI API (google-genai)."""
    
    def __init__(self, client: Optional[Any] = None, model: str = "gemini-2.0-flash", embed_model: str = "text-embedding-004"):
        self.model = model
        self.embed_model = embed_model
        
        if client:
            self.client = client
        else:
            try:
                from google import genai
                self.client = genai.Client()
            except (ImportError, Exception):
                self.client = None

    def to_llm_call(self, *args, **kwargs) -> LLMCall:
        contents = kwargs.get("contents", [])
        if not contents and len(args) > 0:
            contents = args[0]
            
        messages = []
        if isinstance(contents, str):
            messages.append({"role": "user", "content": contents})
        elif isinstance(contents, list):
            for m in contents:
                if isinstance(m, dict):
                    role = "user" if m.get("role") == "user" else "assistant"
                    messages.append({"role": role, "content": str(m.get("parts", m.get("content", "")))})
                elif hasattr(m, "role"):
                    role = "user" if m.role == "user" else "assistant"
                    text = m.parts[0].text if getattr(m, "parts", None) else ""
                    messages.append({"role": role, "content": text})
                elif isinstance(m, str):
                    messages.append({"role": "user", "content": m})
        
        config = kwargs.get("config", {})
        system_instruction = None
        if hasattr(config, "system_instruction"):
            system_instruction = str(config.system_instruction)
        elif isinstance(config, dict) and "system_instruction" in config:
            system_instruction = str(config["system_instruction"])

        return LLMCall(
            messages=messages,
            kwargs=kwargs,
            system=system_instruction,
            metadata={"model": kwargs.get("model", self.model)}
        )
        
    def to_llm_response(self, response: Any) -> LLMResponse:
        if isinstance(response, dict):
            text = response.get("text", "")
        else:
            text = getattr(response, "text", "")
            
        return LLMResponse(
            text=text,
            raw_response=response
        )

    async def generate(self, call: LLMCall, response_format: Optional[Any] = None) -> LLMResponse:
        if not self.client:
            raise RuntimeError("Google GenAI client not installed. Install with `pip install google-genai`.")
            
        from google.genai import types
        
        # Combine messages into a simple string for internal tracker calls (like json extraction)
        formatted_contents = ""
        for m in call.messages:
            formatted_contents += f"{m.get('role', 'user')}: {m.get('content', '')}\n"
            
        config_args = {}
        if call.system:
            config_args["system_instruction"] = call.system
            
        if response_format:
            config_args["response_mime_type"] = "application/json"
            config_args["response_schema"] = response_format
            
        generate_config = types.GenerateContentConfig(**config_args) if config_args else None
            
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=formatted_contents,
            config=generate_config
        )
        return self.to_llm_response(response)

    async def get_embedding(self, text: str) -> List[float]:
        res = await self.get_embeddings([text])
        return res[0]

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not self.client:
            raise RuntimeError("Google GenAI client not installed. Install with `pip install google-genai`.")
        if not texts:
            return []
            
        response = await self.client.aio.models.embed_content(
            model=self.embed_model,
            contents=texts
        )
        return [list(emb.values) for emb in response.embeddings]
