import asyncio
import logging
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
    import litellm

    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False


class LiteLLMAdapter(ProviderAdapter):
    """Adapter for LiteLLM API with production-ready robustness.

    Routes to any provider (Azure, Bedrock, OpenAI, Anthropic, etc.) via LiteLLM.

    Features:
    - Automatic retry with exponential backoff for transient errors
    - Configurable request timeouts
    - Structured logging for debugging
    - Health check mechanism
    - Support for any LiteLLM-supported provider

    LiteLLM supports 100+ providers including:
    - OpenAI, Azure OpenAI
    - Anthropic Claude
    - Google Gemini
    - AWS Bedrock
    - Mistral, Groq, and more

    Configure the model using format: "provider/model-name" or
    use LiteLLM aliases like "gpt-4", "claude-3-sonnet", etc.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        embed_model: str = "text-embedding-3-small",
        embed_kwargs: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
        retry_config: Optional[RetryConfig] = None,
        health_check_timeout: float = 5.0,
        **kwargs: Any,
    ):
        if not HAS_LITELLM:
            raise ImportError(
                "LiteLLM is not installed. Install with `pip install beliefstate[litellm]` or `pip install litellm`."
            )
        self.model = model
        self.embed_model = embed_model
        self.embed_kwargs = embed_kwargs or {}
        self.timeout = timeout
        self.retry_config = retry_config or RetryConfig()
        self.health_check_timeout = health_check_timeout
        self.kwargs = kwargs
        self.log = StructuredLogger(__name__, "LiteLLM")
        self.log.info("Initialized", model=model, embed_model=embed_model)

    def to_llm_call(self, *args: Any, **kwargs: Any) -> LLMCall:
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
            metadata={"model": kwargs.get("model", self.model)},
        )

    def to_llm_response(self, response: Any) -> LLMResponse:
        # LiteLLM's response supports standard openai response format/attributes
        text = ""
        if hasattr(response, "choices") and len(response.choices) > 0:
            text = response.choices[0].message.content or ""
        elif isinstance(response, dict):
            if "choices" in response and len(response["choices"]) > 0:
                text = response["choices"][0].get("message", {}).get("content", "")

        return LLMResponse(text=text, raw_response=response)

    def inject_context(
        self,
        context_prompt: str,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """Inject context prompt into LiteLLM arguments."""
        if "system" in kwargs:
            new_kwargs = kwargs.copy()
            system = new_kwargs.get("system", "")
            new_kwargs["system"] = (
                f"{system}\n\n{context_prompt}" if system else context_prompt
            )
            return args, new_kwargs

        messages = kwargs.get("messages", [])
        in_kwargs = "messages" in kwargs
        arg_idx = -1

        if not messages and len(args) > 0 and isinstance(args[0], list):
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
                kwargs["messages"] = [
                    {"role": "system", "content": call.system}
                ] + kwargs["messages"]

        if response_format:
            kwargs["response_format"] = response_format

        response = await litellm.acompletion(**kwargs)
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
            ImportError: If LiteLLM is not installed
            asyncio.TimeoutError: If request exceeds timeout
            PermanentError: If error is not transient
        """
        if not HAS_LITELLM:
            raise ImportError("LiteLLM is not installed.")

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
                f"LiteLLM generate via {self.model}",
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

    async def _get_embeddings_with_backoff(self, texts: List[str]) -> List[List[float]]:
        """Internal method that actually calls the embeddings API."""
        kwargs = self.kwargs.copy()
        if self.embed_kwargs:
            kwargs.update(self.embed_kwargs)

        response = await litellm.aembedding(
            model=self.embed_model, input=texts, **kwargs
        )
        # In LiteLLM, response.data has list of dicts/objects containing embedding keys
        embeddings = []
        for item in response.data:
            if isinstance(item, dict):
                embeddings.append(item["embedding"])
            else:
                embeddings.append(getattr(item, "embedding", []))
        return embeddings

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
            ImportError: If LiteLLM is not installed
            asyncio.TimeoutError: If request exceeds timeout
            PermanentError: If error is not transient
        """
        if not HAS_LITELLM:
            raise ImportError("LiteLLM is not installed.")
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
                f"LiteLLM embeddings via {self.embed_model} ({len(texts)} texts)",
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
        """Check if LiteLLM routing and the underlying provider are healthy.

        Returns:
            True if healthy, False otherwise
        """
        if not HAS_LITELLM:
            self.log.warning("Health check failed: LiteLLM not installed")
            return False

        try:
            # Try to route a minimal request
            await with_timeout(
                litellm.acompletion(
                    model=self.model,
                    messages=[{"role": "user", "content": "ok"}],
                    max_tokens=5,
                ),
                timeout_seconds=self.health_check_timeout,
                operation_name=f"LiteLLM health check via {self.model}",
            )
            self.log.debug("Health check passed")
            return True
        except Exception as e:
            self.log.warning(f"Health check failed: {e}")
            return False
