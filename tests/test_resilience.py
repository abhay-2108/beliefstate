import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from beliefstate.config import TrackerConfig
from beliefstate.call import LLMCall, LLMResponse
from beliefstate.resilience import (
    ResilientAdapterWrapper,
    CircuitBreaker,
    CircuitBreakerOpenException,
    is_transient_error,
)


def test_transient_error_detection():
    # Developer/Python built-ins should NOT be transient
    assert not is_transient_error(ValueError("Invalid argument"))
    assert not is_transient_error(TypeError("Wrong type"))
    assert not is_transient_error(KeyError("missing key"))

    # Specific API-related error names
    class FakeRateLimitError(Exception):
        pass

    assert is_transient_error(FakeRateLimitError("Rate limit exceeded"))

    class FakeTimeoutError(Exception):
        pass

    assert is_transient_error(FakeTimeoutError("Timeout occurred"))

    # Generic exceptions are retryable by default
    assert is_transient_error(Exception("Some unknown issue"))


@pytest.mark.asyncio
async def test_resilient_wrapper_retries():
    config = TrackerConfig(
        retry_max_attempts=3,
        retry_min_wait=0.01,  # Keep it fast for tests
        retry_max_wait=0.05,
        retry_multiplier=1.1,
        enable_circuit_breaker=False,
    )

    mock_adapter = MagicMock()
    # Mock calls: raise two transient errors, then succeed
    mock_adapter.generate = AsyncMock(
        side_effect=[
            Exception("Transient error 1"),
            Exception("Transient error 2"),
            LLMResponse(text="Success response", raw_response=None),
        ]
    )

    wrapper = ResilientAdapterWrapper(mock_adapter, config)
    call = LLMCall(messages=[])

    result = await wrapper.generate(call)
    assert result.text == "Success response"
    assert mock_adapter.generate.call_count == 3


@pytest.mark.asyncio
async def test_resilient_wrapper_circuit_breaker_trips():
    config = TrackerConfig(
        retry_max_attempts=1,  # Fail fast inside the retry block
        retry_min_wait=0.001,
        retry_max_wait=0.002,
        enable_circuit_breaker=True,
        circuit_breaker_failure_threshold=3,  # Trip after 3 failures
        circuit_breaker_recovery_timeout=10.0,
    )

    mock_adapter = MagicMock()
    mock_adapter.generate = AsyncMock(side_effect=Exception("API Outage"))

    wrapper = ResilientAdapterWrapper(mock_adapter, config)
    call = LLMCall(messages=[])

    # Perform 3 failing calls to trip the circuit breaker
    for _ in range(3):
        with pytest.raises(Exception, match="API Outage"):
            await wrapper.generate(call)

    # The 4th call should fail immediately with CircuitBreakerOpenException
    with pytest.raises(CircuitBreakerOpenException, match="Circuit breaker is OPEN"):
        await wrapper.generate(call)

    # The underlying adapter should NOT have been called a 4th time
    assert mock_adapter.generate.call_count == 3


@pytest.mark.asyncio
async def test_circuit_breaker_recovery():
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05)

    # Starts CLOSED
    assert breaker.allow_request() is True

    # Fail 2 times -> trips to OPEN
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == "OPEN"
    assert breaker.allow_request() is False

    # Wait for recovery timeout to pass
    await asyncio.sleep(0.06)

    # Should enter HALF-OPEN and allow request
    assert breaker.allow_request() is True
    assert breaker.state == "HALF-OPEN"

    # Successful request resets state to CLOSED
    breaker.record_success()
    assert breaker.state == "CLOSED"
    assert breaker.allow_request() is True
