"""Tests for TrackerConfig defaults and overrides."""

from beliefstate.config import TrackerConfig


class TestTrackerConfigDefaults:
    """Verify all default values are as documented."""

    def test_default_store_type(self):
        c = TrackerConfig()
        assert c.store_type == "sqlite"

    def test_default_store_kwargs_empty(self):
        c = TrackerConfig()
        assert c.store_kwargs == {}

    def test_default_similarity_threshold(self):
        c = TrackerConfig()
        assert c.similarity_threshold == 0.82

    def test_default_contradiction_threshold(self):
        c = TrackerConfig()
        assert c.contradiction_threshold == 0.70

    def test_default_entailment_threshold(self):
        c = TrackerConfig()
        assert c.entailment_threshold == 0.85

    def test_default_background_tasks_enabled(self):
        c = TrackerConfig()
        assert c.enable_background_tasks is True

    def test_default_retry_settings(self):
        c = TrackerConfig()
        assert c.retry_max_attempts == 5
        assert c.retry_min_wait == 2.0
        assert c.retry_max_wait == 30.0
        assert c.retry_multiplier == 2.0

    def test_default_circuit_breaker_settings(self):
        c = TrackerConfig()
        assert c.enable_circuit_breaker is True
        assert c.circuit_breaker_failure_threshold == 5
        assert c.circuit_breaker_recovery_timeout == 30.0

    def test_default_dispatcher_type(self):
        c = TrackerConfig()
        assert c.task_dispatcher_type == "asyncio"

    def test_default_max_beliefs(self):
        c = TrackerConfig()
        assert c.max_beliefs == 50

    def test_default_belief_sort_strategy(self):
        c = TrackerConfig()
        assert c.belief_sort_strategy == "confidence_recency"

    def test_default_staleness_scoring(self):
        c = TrackerConfig()
        assert c.enable_staleness_scoring is True
        assert c.staleness_threshold == 0.1

    def test_default_token_aware_injection(self):
        c = TrackerConfig()
        assert c.enable_token_aware_injection is True
        assert c.belief_budget_tokens == 300


class TestTrackerConfigOverrides:
    """Verify custom overrides are accepted."""

    def test_custom_store_type(self):
        c = TrackerConfig(store_type="redis")
        assert c.store_type == "redis"

    def test_custom_thresholds(self):
        c = TrackerConfig(similarity_threshold=0.5, contradiction_threshold=0.9)
        assert c.similarity_threshold == 0.5
        assert c.contradiction_threshold == 0.9

    def test_custom_retry_settings(self):
        c = TrackerConfig(retry_max_attempts=10, retry_min_wait=0.5)
        assert c.retry_max_attempts == 10
        assert c.retry_min_wait == 0.5

    def test_disable_circuit_breaker(self):
        c = TrackerConfig(enable_circuit_breaker=False)
        assert c.enable_circuit_breaker is False

    def test_custom_dispatcher_type(self):
        c = TrackerConfig(task_dispatcher_type="sync")
        assert c.task_dispatcher_type == "sync"

    def test_custom_max_beliefs(self):
        c = TrackerConfig(max_beliefs=100)
        assert c.max_beliefs == 100

    def test_disable_background_tasks(self):
        c = TrackerConfig(enable_background_tasks=False)
        assert c.enable_background_tasks is False

    def test_custom_belief_ttl(self):
        c = TrackerConfig(enable_belief_ttl=True, belief_max_age_seconds=3600)
        assert c.enable_belief_ttl is True
        assert c.belief_max_age_seconds == 3600
