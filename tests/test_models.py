"""Tests for Belief and DeletionReceipt Pydantic models."""
import pytest
from datetime import datetime, timezone, timedelta
from pydantic import ValidationError

from beliefstate.models import Belief, DeletionReceipt


# --- Belief Model Tests ---


class TestBeliefDefaults:
    """Test that Belief model has correct default values."""

    def test_minimal_belief_creation(self):
        b = Belief(
            subject="USER",
            predicate="likes",
            value="Python",
            confidence=0.9,
            turn=1,
            source="user",
        )
        assert b.subject == "USER"
        assert b.predicate == "likes"
        assert b.value == "Python"
        assert b.confidence == 0.9
        assert b.turn == 1
        assert b.source == "user"

    def test_default_embedding_empty_list(self):
        b = Belief(subject="U", predicate="p", value="v", confidence=0.5, turn=1, source="user")
        assert b.embedding == []

    def test_default_embedding_model_empty_string(self):
        b = Belief(subject="U", predicate="p", value="v", confidence=0.5, turn=1, source="user")
        assert b.embedding_model == ""

    def test_default_embedding_dim_zero(self):
        b = Belief(subject="U", predicate="p", value="v", confidence=0.5, turn=1, source="user")
        assert b.embedding_dim == 0

    def test_default_belief_type_assertion(self):
        b = Belief(subject="U", predicate="p", value="v", confidence=0.5, turn=1, source="user")
        assert b.belief_type == "assertion"

    def test_default_is_hypothetical_false(self):
        b = Belief(subject="U", predicate="p", value="v", confidence=0.5, turn=1, source="user")
        assert b.is_hypothetical is False

    def test_default_session_id_none(self):
        b = Belief(subject="U", predicate="p", value="v", confidence=0.5, turn=1, source="user")
        assert b.session_id is None

    def test_default_conversation_id_none(self):
        b = Belief(subject="U", predicate="p", value="v", confidence=0.5, turn=1, source="user")
        assert b.conversation_id is None

    def test_default_created_at_is_recent_utc(self):
        before = datetime.now(timezone.utc)
        b = Belief(subject="U", predicate="p", value="v", confidence=0.5, turn=1, source="user")
        after = datetime.now(timezone.utc)
        # created_at should be between before and after
        assert before <= b.created_at.replace(tzinfo=timezone.utc) <= after

    def test_default_last_referenced_at_is_recent_utc(self):
        before = datetime.now(timezone.utc)
        b = Belief(subject="U", predicate="p", value="v", confidence=0.5, turn=1, source="user")
        after = datetime.now(timezone.utc)
        assert before <= b.last_referenced_at.replace(tzinfo=timezone.utc) <= after


class TestBeliefValidation:
    """Test Pydantic validation rules on Belief model."""

    def test_confidence_lower_bound(self):
        """Confidence must be >= 0.0."""
        with pytest.raises(ValidationError):
            Belief(subject="U", predicate="p", value="v", confidence=-0.1, turn=1, source="user")

    def test_confidence_upper_bound(self):
        """Confidence must be <= 1.0."""
        with pytest.raises(ValidationError):
            Belief(subject="U", predicate="p", value="v", confidence=1.1, turn=1, source="user")

    def test_confidence_zero_valid(self):
        b = Belief(subject="U", predicate="p", value="v", confidence=0.0, turn=1, source="user")
        assert b.confidence == 0.0

    def test_confidence_one_valid(self):
        b = Belief(subject="U", predicate="p", value="v", confidence=1.0, turn=1, source="user")
        assert b.confidence == 1.0

    def test_missing_required_field_raises(self):
        """All required fields must be present."""
        with pytest.raises(ValidationError):
            Belief(subject="U", predicate="p")  # type: ignore[call-arg]


class TestBeliefSerialization:
    """Test Pydantic serialization roundtrips."""

    def test_model_dump_roundtrip(self):
        b = Belief(
            subject="USER", predicate="likes", value="Python",
            confidence=0.9, turn=1, source="user",
            embedding=[0.1, 0.2, 0.3],
        )
        d = b.model_dump()
        b2 = Belief(**d)
        assert b2.subject == b.subject
        assert b2.embedding == b.embedding

    def test_model_dump_json_roundtrip(self):
        b = Belief(
            subject="USER", predicate="likes", value="Python",
            confidence=0.9, turn=1, source="user",
        )
        json_str = b.model_dump_json()
        b2 = Belief.model_validate_json(json_str)
        assert b2.subject == b.subject
        assert b2.value == b.value

    def test_belief_with_all_fields(self):
        """Ensure all optional fields serialize correctly."""
        b = Belief(
            subject="USER", predicate="works at", value="Google",
            confidence=0.95, turn=5, source="user",
            embedding=[0.1, 0.2],
            embedding_model="text-embedding-3-small",
            embedding_dim=1536,
            belief_type="update",
            is_hypothetical=True,
            session_id="s123",
            conversation_id="c456",
        )
        d = b.model_dump()
        assert d["embedding_model"] == "text-embedding-3-small"
        assert d["embedding_dim"] == 1536
        assert d["belief_type"] == "update"
        assert d["is_hypothetical"] is True
        assert d["session_id"] == "s123"
        assert d["conversation_id"] == "c456"


# --- DeletionReceipt Tests ---


class TestDeletionReceipt:
    def test_creation(self):
        r = DeletionReceipt(
            session_id="s123",
            beliefs_deleted=10,
            deleted_at=datetime.now(timezone.utc),
        )
        assert r.session_id == "s123"
        assert r.beliefs_deleted == 10
        assert r.in_flight_tasks_drained == 0  # default

    def test_with_drained_tasks(self):
        r = DeletionReceipt(
            session_id="s123",
            beliefs_deleted=5,
            deleted_at=datetime.now(timezone.utc),
            in_flight_tasks_drained=3,
        )
        assert r.in_flight_tasks_drained == 3
