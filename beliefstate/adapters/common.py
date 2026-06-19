"""Shared utilities for all adapters."""

import asyncio
import logging
from typing import Any, Callable, Optional, TypeVar
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryConfig:
    """Configuration for retry logic with exponential backoff."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number (0-indexed)."""
        delay = self.initial_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            import random

            delay *= random.uniform(0.5, 1.0)

        return delay


class TransientError(Exception):
    """Indicates an error that might succeed on retry."""

    pass


class PermanentError(Exception):
    """Indicates an error that won't succeed on retry."""

    pass


def is_transient_error(error: Exception) -> bool:
    """Determine if an error is transient (worth retrying)."""
    error_msg = str(error).lower()
    error_type = type(error).__name__

    # Common transient errors
    transient_indicators = [
        "rate limit",
        "timeout",
        "connection reset",
        "connection refused",
        "temporarily unavailable",
        "service unavailable",
        "gateway",
        "503",
        "429",
        "504",
        "ephemeral",
        "transient",
    ]

    # OpenAI-specific
    if "APIConnectionError" in error_type or "APITimeoutError" in error_type:
        return True

    # Anthropic-specific
    if "APIConnectionError" in error_type or "APITimeoutError" in error_type:
        return True

    # Generic checks
    if any(indicator in error_msg for indicator in transient_indicators):
        return True

    return False


async def retry_with_backoff(
    coro_func: Callable[..., Any],
    *args: Any,
    config: Optional[RetryConfig] = None,
    **kwargs: Any,
) -> Any:
    """Execute an async function with retry logic and exponential backoff.

    Args:
        coro_func: Async function to call
        config: RetryConfig for backoff behavior
        *args, **kwargs: Arguments to pass to coro_func

    Returns:
        Result from coro_func

    Raises:
        PermanentError: If error is determined to be permanent
        Original exception: If max retries exceeded on transient error
    """
    config = config or RetryConfig()

    last_error: Optional[Exception] = None

    for attempt in range(config.max_retries + 1):
        try:
            logger.debug(f"Attempt {attempt + 1}/{config.max_retries + 1}: {coro_func.__name__}")
            result = await coro_func(*args, **kwargs)
            if attempt > 0:
                logger.info(f"Recovered after {attempt} retries: {coro_func.__name__}")
            return result

        except Exception as e:
            last_error = e

            # Check if error is permanent
            if not is_transient_error(e):
                logger.error(f"Permanent error in {coro_func.__name__}: {e}")
                raise PermanentError(f"Permanent error: {e}") from e

            # Check if we've exhausted retries
            if attempt >= config.max_retries:
                logger.error(
                    f"Max retries ({config.max_retries}) exceeded for {coro_func.__name__}: {e}"
                )
                raise

            # Calculate and wait for backoff
            delay = config.get_delay(attempt)
            logger.warning(
                f"Transient error in {coro_func.__name__} (attempt {attempt + 1}): {e}. Retrying in {delay:.2f}s"
            )
            await asyncio.sleep(delay)

    # Should not reach here, but just in case
    if last_error:
        raise last_error


def async_retry(config: Optional[RetryConfig] = None) -> Callable:
    """Decorator for async functions to add retry logic.

    Usage:
        @async_retry(RetryConfig(max_retries=3))
        async def call_api():
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await retry_with_backoff(func, *args, config=config or RetryConfig(), **kwargs)

        return wrapper

    return decorator


async def with_timeout(
    coro: Any,
    timeout_seconds: float,
    operation_name: str = "operation",
) -> Any:
    """Execute an async operation with a timeout.

    Args:
        coro: Coroutine to execute
        timeout_seconds: Timeout in seconds
        operation_name: Human-readable name for logging

    Returns:
        Result from coro

    Raises:
        asyncio.TimeoutError: If timeout is exceeded
    """
    try:
        logger.debug(f"Starting {operation_name} with {timeout_seconds}s timeout")
        result = await asyncio.wait_for(coro, timeout=timeout_seconds)
        logger.debug(f"Completed {operation_name}")
        return result
    except asyncio.TimeoutError:
        logger.error(f"Timeout after {timeout_seconds}s for {operation_name}")
        raise


def validate_api_key(api_key: Optional[str], provider: str) -> None:
    """Validate that an API key is configured.

    Args:
        api_key: API key to validate
        provider: Provider name for error messages

    Raises:
        ValueError: If API key is missing or empty
    """
    if not api_key or not api_key.strip():
        raise ValueError(
            f"{provider} API key is not configured. "
            f"Set the appropriate environment variable or pass it explicitly."
        )


async def validate_model_availability(
    list_models_func: Callable,
    model_name: str,
    provider: str,
    timeout: float = 5.0,
) -> bool:
    """Validate that a model is available from a provider.

    Args:
        list_models_func: Async function that returns available models
        model_name: Model to check
        provider: Provider name for logging
        timeout: Timeout for availability check

    Returns:
        True if model is available, False otherwise
    """
    try:
        models = await with_timeout(list_models_func(), timeout, f"list models from {provider}")
        if isinstance(models, dict):
            models = models.get("data", [])
        available = any(
            (isinstance(m, dict) and m.get("id") == model_name) or (hasattr(m, "id") and m.id == model_name)
            for m in models
        )
        if available:
            logger.info(f"Model {model_name} is available on {provider}")
        else:
            logger.warning(f"Model {model_name} not found on {provider}")
        return available
    except Exception as e:
        logger.warning(f"Could not verify model availability on {provider}: {e}")
        return False


class StructuredLogger:
    """Structured logging wrapper for consistency across adapters."""

    def __init__(self, name: str, provider: str):
        self.logger = logging.getLogger(name)
        self.provider = provider

    def _log(self, level: str, operation: str, **metadata: Any) -> None:
        """Log with structured metadata."""
        msg_parts = [f"[{self.provider}]", operation]
        if metadata:
            msg_parts.append(f"metadata={metadata}")
        message = " ".join(msg_parts)

        getattr(self.logger, level)(message, extra={"provider": self.provider, **metadata})

    def debug(self, operation: str, **metadata: Any) -> None:
        self._log("debug", operation, **metadata)

    def info(self, operation: str, **metadata: Any) -> None:
        self._log("info", operation, **metadata)

    def warning(self, operation: str, **metadata: Any) -> None:
        self._log("warning", operation, **metadata)

    def error(self, operation: str, **metadata: Any) -> None:
        self._log("error", operation, **metadata)
