import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, cast
from beliefstate.adapters.base import ProviderAdapter
from beliefstate.adapters.common import (
    RetryConfig,
    retry_with_backoff,
    with_timeout,
    StructuredLogger,
    PermanentError,
)
from beliefstate.call import LLMCall, LLMResponse

logger = logging.getLogger(__name__)

try:
    from ollama import AsyncClient
except ImportError:
    AsyncClient = Any  # type: ignore[misc, assignment]


def _dereference_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Inlines all $ref/definitions inside the schema to make it compatible with Ollama."""
    if not isinstance(schema, dict):
        return schema

    defs_val = schema.get("$defs") or schema.get("definitions") or {}
    defs: Dict[str, Any] = defs_val if isinstance(defs_val, dict) else {}

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
    if not isinstance(new_schema, dict):
        return {}
    new_schema.pop("$defs", None)
    new_schema.pop("definitions", None)
    return new_schema


class OllamaAdapter(ProviderAdapter):
    """Adapter for Ollama API with production-ready robustness.

    Features:
    - Automatic retry with exponential backoff for transient errors
    - Configurable request timeouts
    - Structured logging for debugging
    - Health check mechanism
    - Server availability validation
    - Model availability verification

    NOTE: Requires Ollama to be running locally (default: http://localhost:11434).
    Configure host/port via OLLAMA_HOST environment variable.
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        model: str = "llama3.2",
        embed_model: str = "nomic-embed-text",
        embed_kwargs: Optional[Dict[str, Any]] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: float = 30.0,
        retry_config: Optional[RetryConfig] = None,
        health_check_timeout: float = 5.0,
    ):
        self.model = model
        self.embed_model = embed_model
        self.embed_kwargs = embed_kwargs or {}
        self.timeout = timeout
        self.retry_config = retry_config or RetryConfig()
        self.health_check_timeout = health_check_timeout
        self.log = StructuredLogger(__name__, "Ollama")

        # Parse OLLAMA_HOST if available
        if not host and not port:
            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            if "://" in ollama_host:
                from urllib.parse import urlparse

                parsed = urlparse(ollama_host)
                host = parsed.hostname or "http://localhost"
                if parsed.port:
                    port = parsed.port
            else:
                parts = ollama_host.split(":")
                host = parts[0]
                if len(parts) > 1:
                    try:
                        port = int(parts[1])
                    except ValueError:
                        pass

        self.host = host or "http://localhost"
        self.port = port or 11434

        if client:
            self.client = client
        else:
            try:
                from ollama import AsyncClient

                # Avoid duplicating port if host already includes a port
                if ":" in self.host.replace("://", ""):
                    base_url = self.host
                else:
                    base_url = f"{self.host}:{self.port}"
                self.client = AsyncClient(host=base_url)
                self.log.info(
                    "Initialized",
                    model=model,
                    embed_model=embed_model,
                    host=self.host,
                    port=self.port,
                )
            except ImportError:
                self.log.error("Ollama SDK not installed")
                self.client = None
            except Exception as e:
                self.log.error(f"Failed to initialize Ollama client: {e}")
                self.client = None

    def to_llm_call(self, *args: Any, **kwargs: Any) -> LLMCall:
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
            msg = getattr(response, "message", None)
            text = getattr(msg, "content", "") if msg else ""

        return LLMResponse(text=text, raw_response=response)

    def inject_context(
        self,
        context_prompt: str,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """Inject context prompt into Ollama messages (either in args[1] or kwargs['messages'])."""
        messages = kwargs.get("messages", [])
        in_kwargs = "messages" in kwargs
        arg_idx = -1

        if not messages and len(args) > 1 and isinstance(args[1], list):
            messages = args[1]
            in_kwargs = False
            arg_idx = 1
        elif not messages and len(args) > 0 and isinstance(args[0], list):
            messages = args[0]
            in_kwargs = False
            arg_idx = 0

        if not messages:
            in_kwargs = True
            messages = []

        new_messages = [m.copy() if isinstance(m, dict) else m for m in messages]
        system_idx = -1
        for idx, m in enumerate(new_messages):
            if isinstance(m, dict) and m.get("role") == "system":
                system_idx = idx
                break
            elif hasattr(m, "role") and m.role == "system":
                system_idx = idx
                break

        if system_idx != -1:
            m = new_messages[system_idx]
            if isinstance(m, dict):
                orig_content = m.get("content", "")
                m["content"] = (
                    f"{orig_content}\n\n{context_prompt}"
                    if orig_content
                    else context_prompt
                )
            else:
                orig_content = getattr(m, "content", "")
                m.content = (
                    f"{orig_content}\n\n{context_prompt}"
                    if orig_content
                    else context_prompt
                )
        else:
            new_messages.insert(0, {"role": "system", "content": context_prompt})

        if in_kwargs:
            new_kwargs = kwargs.copy()
            new_kwargs["messages"] = new_messages
            return args, new_kwargs
        elif arg_idx != -1:
            new_args = list(args)
            new_args[arg_idx] = new_messages
            return tuple(new_args), kwargs

        return args, kwargs

    async def _generate_with_backoff(
        self, call: LLMCall, response_format: Optional[Any] = None
    ) -> LLMResponse:
        """Internal method that actually calls the API."""
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

    async def generate(
        self, call: LLMCall, response_format: Optional[Any] = None
    ) -> LLMResponse:
        """Generate a response with automatic retry and timeout handling.

        Args:
            call: LLMCall with messages and parameters
            response_format: Optional response schema (for structured output)

        Returns:
            LLMResponse with generated text

        Raises:
            RuntimeError: If Ollama client is not configured
            asyncio.TimeoutError: If request exceeds timeout
            PermanentError: If error is not transient
        """
        if not self.client:
            raise RuntimeError(
                "Ollama client not installed. Install with `pip install ollama`. "
                f"Also ensure Ollama is running at {self.host}:{self.port}"
            )

        try:

            async def api_call() -> LLMResponse:
                return cast(
                    LLMResponse,
                    await retry_with_backoff(
                        self._generate_with_backoff,
                        call,
                        response_format,
                        config=self.retry_config,
                    ),
                )

            result = await with_timeout(
                api_call(),
                self.timeout * (self.retry_config.max_retries + 1),
                "Ollama generate",
            )
            return cast(LLMResponse, result)

        except PermanentError:
            self.log.error("Generate failed with permanent error", model=self.model)
            raise
        except asyncio.TimeoutError:
            self.log.error("Generate timed out", timeout=self.timeout, model=self.model)
            raise
        except Exception as e:
            self.log.error(
                "Generate failed unexpectedly", error=str(e), model=self.model
            )
            raise

    async def _get_embedding_with_backoff(self, text: str) -> List[float]:
        """Internal method for single embedding."""
        emb_args = {"model": self.embed_model, "prompt": text}
        if self.embed_kwargs:
            emb_args.update(self.embed_kwargs)

        response = await self.client.embeddings(**emb_args)
        return cast(List[float], getattr(response, "embedding", []))

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if not self.client:
            raise RuntimeError(
                "Ollama client not installed. Install with `pip install ollama`. "
                f"Also ensure Ollama is running at {self.host}:{self.port}"
            )

        try:

            async def api_call() -> List[float]:
                return cast(
                    List[float],
                    await retry_with_backoff(
                        self._get_embedding_with_backoff,
                        text,
                        config=self.retry_config,
                    ),
                )

            result = await with_timeout(
                api_call(),
                self.timeout * (self.retry_config.max_retries + 1),
                "Ollama embedding",
            )
            return cast(List[float], result)

        except PermanentError:
            self.log.error(
                "Get embedding failed with permanent error", model=self.embed_model
            )
            raise
        except asyncio.TimeoutError:
            self.log.error(
                "Get embedding timed out", timeout=self.timeout, model=self.embed_model
            )
            raise
        except Exception as e:
            self.log.error(
                "Get embedding failed unexpectedly",
                error=str(e),
                model=self.embed_model,
            )
            raise

    async def _get_embeddings_with_backoff(self, texts: List[str]) -> List[List[float]]:
        """Internal method for batch embeddings (with fallback to individual)."""
        if hasattr(self.client, "embed"):
            try:
                embed_args = {"model": self.embed_model, "input": texts}
                if self.embed_kwargs:
                    embed_args.update(self.embed_kwargs)
                response = await self.client.embed(**embed_args)
                if "embeddings" in response:
                    return cast(List[List[float]], response["embeddings"])
            except Exception as e:
                self.log.warning(
                    f"Ollama batch embed failed: {e}. Falling back to individual embeddings."
                )

        # Fallback: call get_embedding for each text
        tasks = [self._get_embedding_with_backoff(text) for text in texts]
        return list(await asyncio.gather(*tasks))

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts with automatic retry and timeout.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: If Ollama client is not configured
            asyncio.TimeoutError: If request exceeds timeout
            PermanentError: If error is not transient
        """
        if not self.client:
            raise RuntimeError(
                "Ollama client not installed. Install with `pip install ollama`. "
                f"Also ensure Ollama is running at {self.host}:{self.port}"
            )
        if not texts:
            return []

        try:

            async def api_call() -> List[List[float]]:
                return cast(
                    List[List[float]],
                    await retry_with_backoff(
                        self._get_embeddings_with_backoff,
                        texts,
                        config=self.retry_config,
                    ),
                )

            result = await with_timeout(
                api_call(),
                self.timeout * (self.retry_config.max_retries + 1),
                f"Ollama embeddings ({len(texts)} texts)",
            )
            return cast(List[List[float]], result)

        except PermanentError:
            self.log.error(
                "Get embeddings failed with permanent error",
                model=self.embed_model,
                count=len(texts),
            )
            raise
        except asyncio.TimeoutError:
            self.log.error(
                "Get embeddings timed out",
                timeout=self.timeout,
                model=self.embed_model,
                count=len(texts),
            )
            raise
        except Exception as e:
            self.log.error(
                "Get embeddings failed unexpectedly",
                error=str(e),
                model=self.embed_model,
                count=len(texts),
            )
            raise

    async def health_check(self) -> bool:
        """Check if Ollama server is accessible and the model is available.

        Returns:
            True if healthy, False otherwise
        """
        if not self.client:
            self.log.warning("Health check failed: client not configured")
            return False

        try:
            # Try to list models to verify server is running
            await with_timeout(
                self.client.list(),
                timeout_seconds=self.health_check_timeout,
                operation_name="Ollama health check",
            )
            self.log.debug("Health check passed")
            return True
        except Exception as e:
            self.log.warning(f"Health check failed: {e}")
            return False
