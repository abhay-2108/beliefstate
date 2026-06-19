import asyncio
import logging
import os
from typing import Any, Dict, List, Optional
from beliefstate.adapters.base import ProviderAdapter
from beliefstate.adapters.common import (
    RetryConfig,
    retry_with_backoff,
    with_timeout,
    validate_api_key,
    StructuredLogger,
    PermanentError,
)
from beliefstate.call import LLMCall, LLMResponse

logger = logging.getLogger(__name__)

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = Any  # type: ignore[misc, assignment]


class OpenAIAdapter(ProviderAdapter):
    """Adapter for OpenAI API with production-ready robustness.
    
    Features:
    - Automatic retry with exponential backoff for transient errors
    - Configurable request timeouts
    - Structured logging for debugging
    - Health check mechanism
    - API key validation at initialization
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        model: str = "gpt-4o-mini",
        embed_model: str = "text-embedding-3-small",
        embed_kwargs: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
        retry_config: Optional[RetryConfig] = None,
    ):
        self.model = model
        self.embed_model = embed_model
        self.embed_kwargs = embed_kwargs or {}
        self.timeout = timeout
        self.retry_config = retry_config or RetryConfig()
        self.log = StructuredLogger(__name__, "OpenAI")

        if client:
            self.client = client
        else:
            try:
                from openai import AsyncOpenAI

                api_key = os.getenv("OPENAI_API_KEY")
                validate_api_key(api_key, "OpenAI")
                self.client = AsyncOpenAI(api_key=api_key)
                self.log.info("Initialized", model=model, embed_model=embed_model)
            except ImportError:
                self.log.error("OpenAI SDK not installed")
                self.client = None
            except ValueError as e:
                self.log.error(f"Configuration error: {e}")

    def to_llm_call(self, *args: Any, **kwargs: Any) -> LLMCall:
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

    async def _generate_with_backoff(
        self, call: LLMCall, response_format: Optional[Any] = None
    ) -> LLMResponse:
        """Internal method that actually calls the API (without timeout wrapper)."""
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
            RuntimeError: If OpenAI client is not configured
            asyncio.TimeoutError: If request exceeds timeout
            PermanentError: If error is not transient
        """
        if not self.client:
            raise RuntimeError(
                "OpenAI client not installed or configured. Install with `pip install openai`."
            )

        try:
            # Wrap the retry logic with timeout
            async def api_call() -> LLMResponse:
                return await retry_with_backoff(
                    self._generate_with_backoff,
                    call,
                    response_format,
                    config=self.retry_config,
                )

            result = await with_timeout(
                api_call(),
                self.timeout * (self.retry_config.max_retries + 1),  # Allow time for retries
                "OpenAI generate",
            )
            return result

        except PermanentError:
            self.log.error("Generate failed with permanent error", model=self.model)
            raise
        except asyncio.TimeoutError:
            self.log.error("Generate timed out", timeout=self.timeout, model=self.model)
            raise
        except Exception as e:
            self.log.error("Generate failed unexpectedly", error=str(e), model=self.model)
            raise

    async def _get_embeddings_with_backoff(self, texts: List[str]) -> List[List[float]]:
        """Internal method that actually calls the embeddings API."""
        kwargs = {"input": texts, "model": self.embed_model}
        if self.embed_kwargs:
            kwargs.update(self.embed_kwargs)

        response = await self.client.embeddings.create(**kwargs)
        return [item.embedding for item in response.data]

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        res = await self.get_embeddings([text])
        return res[0]

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts with automatic retry and timeout.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            RuntimeError: If OpenAI client is not configured
            asyncio.TimeoutError: If request exceeds timeout
            PermanentError: If error is not transient
        """
        if not self.client:
            raise RuntimeError(
                "OpenAI client not installed or configured. Install with `pip install openai`."
            )
        if not texts:
            return []

        try:
            async def api_call() -> List[List[float]]:
                return await retry_with_backoff(
                    self._get_embeddings_with_backoff,
                    texts,
                    config=self.retry_config,
                )

            result = await with_timeout(
                api_call(),
                self.timeout * (self.retry_config.max_retries + 1),
                f"OpenAI embeddings ({len(texts)} texts)",
            )
            return result

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
        """Check if OpenAI API is accessible and healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        if not self.client:
            self.log.warning("Health check failed: client not configured")
            return False

        try:
            # Try to list models with a short timeout as a health check
            await with_timeout(
                self.client.models.list(),
                timeout_seconds=5.0,
                operation_name="OpenAI health check",
            )
            self.log.debug("Health check passed")
            return True
        except Exception as e:
            self.log.warning(f"Health check failed: {e}")
            return False
