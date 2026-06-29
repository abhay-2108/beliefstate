import time
import logging
import threading
from typing import Any, List, Callable, Coroutine, Optional, cast
from tenacity import (
    AsyncRetrying,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception,
)

from beliefstate.config import TrackerConfig
from beliefstate.call import LLMCall, LLMResponse
from beliefstate.adapters.base import ProviderAdapter

logger = logging.getLogger("beliefstate.resilience")


class CircuitBreakerOpenException(Exception):
    """Raised when the circuit breaker is OPEN and rejecting calls."""

    pass


class CircuitBreaker:
    """A simple stateful circuit breaker (CLOSED, OPEN, HALF-OPEN)."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN
        self.last_state_change = time.time()
        self._lock = threading.Lock()

    def record_success(self) -> None:
        with self._lock:
            if self.state != "CLOSED":
                logger.info("Circuit breaker recovered. State changed to CLOSED.")
            self.failure_count = 0
            self.state = "CLOSED"
            self.last_state_change = time.time()

    def record_failure(self) -> None:
        with self._lock:
            self.failure_count += 1
            logger.warning(
                f"Recorded failure {self.failure_count}/{self.failure_threshold}."
            )
            if self.failure_count >= self.failure_threshold:
                if self.state != "OPEN":
                    logger.error(
                        f"Circuit breaker tripped. State changed to OPEN. Cooldown: {self.recovery_timeout}s."
                    )
                self.state = "OPEN"
                self.last_state_change = time.time()

    def allow_request(self) -> bool:
        with self._lock:
            if self.state == "CLOSED":
                return True
            if self.state == "OPEN":
                now = time.time()
                if now - self.last_state_change > self.recovery_timeout:
                    logger.info(
                        "Circuit breaker entered HALF-OPEN state (cooldown expired)."
                    )
                    self.state = "HALF-OPEN"
                    self.last_state_change = now
                    return True
                return False
            if self.state == "HALF-OPEN":
                return True
            return False


def is_transient_error(exc: BaseException) -> bool:
    """Identify if an exception is transient and worth retrying."""
    # 1. Do not retry on common developer errors
    if isinstance(
        exc, (ValueError, TypeError, KeyError, AttributeError, NameError, ImportError)
    ):
        return False

    # 2. Check for Pydantic validation errors
    try:
        from pydantic import ValidationError

        if isinstance(exc, ValidationError):
            return False
    except ImportError:
        pass

    # 3. Check for specific non-transient status codes/messages in standard exceptions
    class_name = exc.__class__.__name__.lower()

    # 4. Check for httpx / request exceptions
    try:
        import httpx

        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            return status == 429 or status >= 500
        if isinstance(exc, httpx.RequestError):
            return True
    except ImportError:
        pass

    # 5. Check attributes like status_code (common in SDK exceptions)
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        try:
            status = int(status_code)
            return status == 429 or status >= 500
        except (ValueError, TypeError):
            pass

    # Check response status code
    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if status_code is not None:
            try:
                status = int(status_code)
                return status == 429 or status >= 500
            except (ValueError, TypeError):
                pass

    # 6. Specific SDK non-retryable errors
    if any(
        k in class_name
        for k in ["auth", "permission", "invalid", "badrequest", "signature"]
    ):
        return False

    # 7. Check class name for typical retryable issues
    if any(
        k in class_name
        for k in [
            "rate",
            "limit",
            "timeout",
            "connection",
            "transient",
            "unavailable",
            "server",
            "gateway",
        ]
    ):
        return True

    # Default: do not retry unrecognized errors (fail fast)
    return False


def before_sleep_log(retry_state: Any) -> None:
    logger.warning(
        f"API call failed: {retry_state.outcome.exception()}. "
        f"Attempt {retry_state.attempt_number} failed. Retrying..."
    )


class ResilientAdapterWrapper(ProviderAdapter):
    """Wrapper that wraps any LLM adapter with retries and circuit breaker checks."""

    def __init__(self, adapter: ProviderAdapter, config: TrackerConfig):
        self.adapter = adapter
        self.config = config
        self.llm_breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker_failure_threshold,
            recovery_timeout=config.circuit_breaker_recovery_timeout,
        )
        self.embed_breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker_failure_threshold,
            recovery_timeout=config.circuit_breaker_recovery_timeout,
        )

    def to_llm_call(self, *args: Any, **kwargs: Any) -> LLMCall:
        return self.adapter.to_llm_call(*args, **kwargs)

    def to_llm_response(self, response: Any) -> LLMResponse:
        return self.adapter.to_llm_response(response)

    async def generate(
        self, call: LLMCall, response_format: Optional[Any] = None
    ) -> LLMResponse:
        return cast(
            LLMResponse,
            await self._execute_with_resilience(
                lambda: self.adapter.generate(call, response_format=response_format),
                self.llm_breaker,
                "generate",
            ),
        )

    async def get_embedding(self, text: str) -> List[float]:
        return cast(
            List[float],
            await self._execute_with_resilience(
                lambda: self.adapter.get_embedding(text),
                self.embed_breaker,
                "get_embedding",
            ),
        )

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return cast(
            List[List[float]],
            await self._execute_with_resilience(
                lambda: self.adapter.get_embeddings(texts),
                self.embed_breaker,
                "get_embeddings",
            ),
        )

    def inject_context(self, context_prompt: str, *args: Any, **kwargs: Any) -> Any:
        return self.adapter.inject_context(context_prompt, *args, **kwargs)

    async def _execute_with_resilience(
        self,
        operation: Callable[[], Coroutine[Any, Any, Any]],
        breaker: CircuitBreaker,
        op_name: str,
    ) -> Any:
        # Check circuit breaker
        if self.config.enable_circuit_breaker and not breaker.allow_request():
            logger.error(
                f"Circuit breaker is OPEN for operation '{op_name}'. Fail fast."
            )
            raise CircuitBreakerOpenException(
                f"Circuit breaker is OPEN for '{op_name}'"
            )

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.config.retry_max_attempts),
                wait=wait_exponential(
                    multiplier=self.config.retry_multiplier,
                    min=self.config.retry_min_wait,
                    max=self.config.retry_max_wait,
                ),
                retry=retry_if_exception(is_transient_error),
                before_sleep=before_sleep_log,
                reraise=True,
            ):
                with attempt:
                    result = await operation()
                    if self.config.enable_circuit_breaker:
                        breaker.record_success()
                    return result
        except CircuitBreakerOpenException:
            raise
        except Exception as e:
            # Entire operation failed after all retries
            if self.config.enable_circuit_breaker:
                breaker.record_failure()
            logger.error(f"All retry attempts failed for operation '{op_name}': {e}")
            raise e

    async def health_check(self) -> bool:
        """Delegate health check to wrapped adapter."""
        if hasattr(self.adapter, "health_check"):
            return await self.adapter.health_check()
        return False
