"""Structured logging helpers for BeliefState.

Provides ``TrackerEvent`` — a lightweight dataclass that produces
JSON-parseable log entries with consistent fields. Works with any
stdlib ``logging.Formatter`` (plain text falls back to f-string repr).

Usage:
    from beliefstate.logging_utils import TrackerEvent, log_event

    log_event(TrackerEvent(
        session_id="user-123",
        operation="extract_beliefs",
        turn=5,
        detail="Extracted 3 beliefs from user message",
        latency_ms=142.5,
    ))
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

logger = logging.getLogger("beliefstate.events")


@dataclass
class TrackerEvent:
    """Structured event for belief tracking operations.

    Attributes:
        session_id: Session identifier
        operation: Operation name (e.g., 'extract_beliefs', 'detect_contradiction')
        turn: Turn number in the conversation
        detail: Human-readable description of what happened
        latency_ms: Operation latency in milliseconds (optional)
        extra: Additional key-value metadata (optional)
    """

    session_id: str = ""
    operation: str = ""
    turn: int = 0
    detail: str = ""
    latency_ms: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a flat dict suitable for structured logging."""
        d = asdict(self)
        # Remove None values for cleaner output
        return {k: v for k, v in d.items() if v is not None and v != {} and v != ""}

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), default=str)


def log_event(event: TrackerEvent, level: int = logging.INFO) -> None:
    """Emit a structured log event.

    Uses ``logger.info()`` with ``extra`` dict so JSON formatters
    (Datadog, CloudWatch, etc.) can parse the fields automatically.
    Falls back to readable f-string for plain-text formatters.

    Args:
        event: TrackerEvent to log
        level: Logging level (default: INFO)
    """
    logger.log(
        level,
        "[%s] %s (session=%s, turn=%d%s)",
        event.operation,
        event.detail,
        event.session_id,
        event.turn,
        f", latency={event.latency_ms:.1f}ms" if event.latency_ms else "",
        extra={"beliefstate_event": event.to_dict()},
    )
