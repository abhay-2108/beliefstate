import pytest
from unittest.mock import MagicMock, patch
from beliefstate.observability import setup_otel, trace_sync, trace_async, BeliefTrackerMetrics, OTEL_AVAILABLE

def test_setup_otel_disabled():
    # Setup with enabled=False should run without issues
    setup_otel(enabled=False)

def test_setup_otel_enabled_not_available():
    # Mock OTEL_AVAILABLE as False
    with patch("beliefstate.observability.OTEL_AVAILABLE", False):
        setup_otel(enabled=True)

def test_trace_sync_disabled():
    @trace_sync("test_op", {"attr": "val"})
    def my_func(x):
        return x + 1

    assert my_func(5) == 6

@pytest.mark.asyncio
async def test_trace_async_disabled():
    @trace_async("test_op", {"attr": "val"})
    async def my_async_func(x):
        return x + 1

    assert await my_async_func(5) == 6

def test_metrics_disabled():
    metrics = BeliefTrackerMetrics()
    assert not metrics.enabled
    # Calling methods on disabled metrics should not raise exceptions
    metrics.record_beliefs_extracted(5)
    metrics.record_contradiction()
    metrics.record_deduplication()
    metrics.record_extraction_latency(10.5)
    metrics.record_detection_latency(20.5)
    metrics.record_store_search_latency(5.0)
    metrics.record_adapter_generate_latency(150.0)
    metrics.record_adapter_embedding_latency(50.0)

@pytest.mark.skipif(not OTEL_AVAILABLE, reason="opentelemetry is not installed")
def test_trace_sync_enabled():
    with patch("beliefstate.observability._otel_enabled", True), \
         patch("beliefstate.observability._tracer") as mock_tracer:
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

        @trace_sync("test_op", {"attr": "val"})
        def my_func(x):
            return x + 1

        assert my_func(5) == 6
        mock_tracer.start_as_current_span.assert_called_once_with("test_op")
        mock_span.set_attribute.assert_any_call("attr", "val")
        mock_span.set_attribute.assert_any_call("status", "success")

@pytest.mark.skipif(not OTEL_AVAILABLE, reason="opentelemetry is not installed")
def test_trace_sync_error():
    with patch("beliefstate.observability._otel_enabled", True), \
         patch("beliefstate.observability._tracer") as mock_tracer:
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

        @trace_sync("test_op", {"attr": "val"})
        def my_func():
            raise ValueError("oops")

        with pytest.raises(ValueError, match="oops"):
            my_func()

        mock_span.set_attribute.assert_any_call("status", "error")
        mock_span.set_attribute.assert_any_call("error.type", "ValueError")

@pytest.mark.skipif(not OTEL_AVAILABLE, reason="opentelemetry is not installed")
@pytest.mark.asyncio
async def test_trace_async_enabled():
    with patch("beliefstate.observability._otel_enabled", True), \
         patch("beliefstate.observability._tracer") as mock_tracer:
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

        @trace_async("test_op", {"attr": "val"})
        async def my_async_func(x):
            return x + 1

        assert await my_async_func(5) == 6
        mock_tracer.start_as_current_span.assert_called_once_with("test_op")
        mock_span.set_attribute.assert_any_call("attr", "val")
        mock_span.set_attribute.assert_any_call("status", "success")
