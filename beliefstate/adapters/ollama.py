from typing import Any, Dict, List, Optional
from beliefstate.adapters.base import ProviderAdapter
from beliefstate.call import LLMCall, LLMResponse

try:
    from ollama import AsyncClient
except ImportError:
    AsyncClient = Any


def _dereference_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Inlines all $ref/definitions inside the schema to make it compatible with Ollama."""
    if not isinstance(schema, dict):
        return schema

    defs = schema.get("$defs", schema.get("definitions", {}))

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_path = node["$ref"]
                parts = ref_path.split("/")
                ref_name = parts[-1]
                if ref_name in defs:
                    resolved = resolve(defs[ref_name])
                    merged = {k: v for k, v in node.items() if k != "$ref"}
                    merged.update(resolved)
                    return merged
            return {k: resolve(v) for k, v in node.items()}
        elif isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    new_schema = resolve(schema)
    new_schema.pop("$defs", None)
    new_schema.pop("definitions", None)
    return new_schema


class OllamaAdapter(ProviderAdapter):
    """Adapter for Ollama API."""

    def __init__(
        self,
        client: Optional[Any] = None,
        model: str = "llama3.2",
        embed_model: str = "nomic-embed-text",
        embed_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.embed_model = embed_model
        self.embed_kwargs = embed_kwargs or {}

        if client:
            self.client = client
        else:
            try:
                from ollama import AsyncClient

                self.client = AsyncClient()  # Defaults to local Ollama server
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
            metadata={"model": kwargs.get("model", self.model)},
        )

    def to_llm_response(self, response: Any) -> LLMResponse:
        if isinstance(response, dict):
            text = response.get("message", {}).get("content", "")
        else:
            text = getattr(response, "message", {}).get("content", "")
            if not text and hasattr(response, "message"):
                text = getattr(response.message, "content", "")

        return LLMResponse(text=text, raw_response=response)

    async def generate(
        self, call: LLMCall, response_format: Optional[Any] = None
    ) -> LLMResponse:
        if not self.client:
            raise RuntimeError(
                "Ollama client not installed. Install with `pip install ollama`."
            )

        kwargs = call.kwargs.copy()
        kwargs["messages"] = call.messages
        if "model" not in kwargs:
            kwargs["model"] = self.model

        if response_format:
            if isinstance(response_format, dict):
                kwargs["format"] = _dereference_schema(response_format)
            else:
                try:
                    schema = response_format.model_json_schema()
                    kwargs["format"] = _dereference_schema(schema)
                except Exception:
                    kwargs["format"] = "json"

        response = await self.client.chat(**kwargs)
        return self.to_llm_response(response)

    async def get_embedding(self, text: str) -> List[float]:
        if not self.client:
            raise RuntimeError(
                "Ollama client not installed. Install with `pip install ollama`."
            )

        emb_args = {"model": self.embed_model, "prompt": text}
        if self.embed_kwargs:
            emb_args.update(self.embed_kwargs)

        response = await self.client.embeddings(**emb_args)
        return response.get("embedding", [])

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not self.client:
            raise RuntimeError(
                "Ollama client not installed. Install with `pip install ollama`."
            )
        if not texts:
            return []

        if hasattr(self.client, "embed"):
            try:
                embed_args = {"model": self.embed_model, "input": texts}
                if self.embed_kwargs:
                    embed_args.update(self.embed_kwargs)
                response = await self.client.embed(**embed_args)
                if "embeddings" in response:
                    return response["embeddings"]
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(
                    f"Ollama batch embed failed: {e}. Falling back to individual embeddings."
                )

        import asyncio

        tasks = [self.get_embedding(text) for text in texts]
        return list(await asyncio.gather(*tasks))
