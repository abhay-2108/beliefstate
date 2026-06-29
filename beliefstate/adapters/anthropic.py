import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, cast
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
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = Any  # type: ignore[misc, assignment]


class AnthropicAdapter(ProviderAdapter):
    """Adapter for Anthropic API with production-ready robustness.

    Features:
    - Automatic retry with exponential backoff for transient errors
    - Configurable request timeouts
    - Structured logging for debugging
    - Health check mechanism
    - API key validation at initialization
    - Informative error message for embeddings (Anthropic limitation)

    NOTE: Anthropic does not provide native embeddings. Use OpenAI, Ollama, or
    configure an external embedding service for the internal tracker pipeline.
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        model: str = "claude-3-5-sonnet-latest",
        embed_model: str = "voyage-large-2",
        embed_kwargs: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
        retry_config: Optional[RetryConfig] = None,
        health_check_timeout: float = 5.0,
        default_max_tokens: int = 1024,
    ):
        self.model = model
        self.embed_model = embed_model
        self.embed_kwargs = embed_kwargs or {}
        self.timeout = timeout
        self.retry_config = retry_config or RetryConfig()
        self.health_check_timeout = health_check_timeout
        self.default_max_tokens = default_max_tokens
        self.log = StructuredLogger(__name__, "Anthropic")

        if client:
            self.client = client
        else:
            try:
                from anthropic import AsyncAnthropic

                api_key = os.getenv("ANTHROPIC_API_KEY")
                validate_api_key(api_key, "Anthropic")
                self.client = AsyncAnthropic(api_key=api_key)
                self.log.info("Initialized", model=model)
            except ImportError:
                self.log.error("Anthropic SDK not installed")
                self.client = None
            except ValueError as e:
                self.log.error(f"Configuration error: {e}")
                self.client = None

    def to_llm_call(self, *args: Any, **kwargs: Any) -> LLMCall:
        messages = kwargs.get("messages", [])
        if not messages and len(args) > 0 and isinstance(args[0], list):
            messages = args[0]

        system_prompt = kwargs.get("system", None)

        return LLMCall(
            messages=messages,
            kwargs=kwargs,
            system=system_prompt,
            metadata={"model": kwargs.get("model", self.model)},
        )

    def to_llm_response(self, response: Any) -> LLMResponse:
        # Handle generic dict or anthropic Message object
        if isinstance(response, dict):
            content = response.get("content", [])
            text = content[0].get("text", "") if content else ""
        else:
            text = response.content[0].text

        return LLMResponse(text=text, raw_response=response)

    def inject_context(
        self,
        context_prompt: str,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """Inject context prompt into Anthropic kwargs['system']."""
        new_kwargs = kwargs.copy()
        system = new_kwargs.get("system", "")
        if system:
            new_kwargs["system"] = f"{system}\n\n{context_prompt}"
        else:
            new_kwargs["system"] = context_prompt
        return args, new_kwargs

    async def _generate_with_backoff(
        self, call: LLMCall, response_format: Optional[Any] = None
    ) -> LLMResponse:
        """Internal method that actually calls the API."""
        import json

        kwargs = call.kwargs.copy()

        # Prompt-based fallback formatting instructions if the provider does not support native schema validation
        messages = call.messages.copy()
        if response_format:
            if isinstance(response_format, dict):
                schema_json = json.dumps(response_format)
            else:
                try:
                    schema_json = json.dumps(response_format.model_json_schema())
                except Exception:
                    schema_json = "{}"
            instruction = f"\n\nIMPORTANT: You must return a valid JSON object or JSON array conforming strictly to the following JSON Schema: {schema_json}. Do NOT include any explanations, markdown code blocks, or preamble in your response. Output only raw JSON."
            if messages:
                last_m = messages[-1].copy()
                existing_content = last_m.get("content", "")
                if isinstance(existing_content, list):
                    # Anthropic API supports list content blocks
                    last_m["content"] = existing_content + [
                        {"type": "text", "text": instruction}
                    ]
                else:
                    last_m["content"] = str(existing_content) + instruction
                messages[-1] = last_m
            else:
                messages.append({"role": "user", "content": instruction})

        kwargs["messages"] = messages
        if call.system and "system" not in kwargs:
            kwargs["system"] = call.system
        if "model" not in kwargs:
            kwargs["model"] = self.model
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = (
                self.default_max_tokens
            )  # Anthropic requires max_tokens

        response = await self.client.messages.create(**kwargs)
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
            RuntimeError: If Anthropic client is not configured
            asyncio.TimeoutError: If request exceeds timeout
            PermanentError: If error is not transient
        """
        if not self.client:
            raise RuntimeError(
                "Anthropic client not installed or configured. Install with `pip install anthropic`."
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
                "Anthropic generate",
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

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text.

        NOTE: Anthropic does not provide native embeddings.

        Raises:
            NotImplementedError: Always, as Anthropic doesn't support embeddings
        """
        raise NotImplementedError(
            "Anthropic does not natively provide embeddings. "
            "Recommendation: Configure an external embedding provider by:\n"
            "  1. Use OpenAI adapter (configure OPENAI_API_KEY)\n"
            "  2. Use Ollama adapter (run Ollama locally)\n"
            "  3. Use Voyage AI client directly (configure VOYAGE_API_KEY)\n"
            "  4. Configure tracker to use a different internal_adapter for embeddings\n"
            "\nFor more details, see beliefstate/config.py:Config.internal_adapter"
        )

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts.

        NOTE: Anthropic does not provide native embeddings.

        Raises:
            NotImplementedError: Always, as Anthropic doesn't support embeddings
        """
        raise NotImplementedError(
            "Anthropic does not natively provide embeddings. "
            "Recommendation: Configure an external embedding provider by:\n"
            "  1. Use OpenAI adapter (configure OPENAI_API_KEY)\n"
            "  2. Use Ollama adapter (run Ollama locally)\n"
            "  3. Use Voyage AI client directly (configure VOYAGE_API_KEY)\n"
            "  4. Configure tracker to use a different internal_adapter for embeddings\n"
            "\nFor more details, see beliefstate/config.py:Config.internal_adapter"
        )

    async def health_check(self) -> bool:
        """Check if Anthropic API is accessible and healthy.

        Returns:
            True if healthy, False otherwise
        """
        if not self.client:
            self.log.warning("Health check failed: client not configured")
            return False

        try:
            # Try a minimal API call with a short timeout
            await with_timeout(
                self.client.messages.create(
                    model=self.model,
                    max_tokens=10,
                    messages=[{"role": "user", "content": "ok"}],
                ),
                timeout_seconds=self.health_check_timeout,
                operation_name="Anthropic health check",
            )
            self.log.debug("Health check passed")
            return True
        except Exception as e:
            self.log.warning(f"Health check failed: {e}")
            return False
