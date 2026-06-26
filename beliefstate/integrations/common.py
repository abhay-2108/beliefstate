"""Shared utilities for all integrations."""

import logging
from typing import Any, Optional


class IntegrationLogger:
    """Structured logging for integrations."""

    def __init__(self, name: str, integration_type: str):
        self.logger = logging.getLogger(name)
        self.integration_type = integration_type

    def _log(self, level: str, operation: str, **metadata: Any) -> None:
        """Log with structured metadata."""
        msg = f"[{self.integration_type}] {operation}"
        getattr(self.logger, level)(
            msg, extra={"integration": self.integration_type, **metadata}
        )

    def debug(self, operation: str, **metadata: Any) -> None:
        self._log("debug", operation, **metadata)

    def info(self, operation: str, **metadata: Any) -> None:
        self._log("info", operation, **metadata)

    def warning(self, operation: str, **metadata: Any) -> None:
        self._log("warning", operation, **metadata)

    def error(self, operation: str, **metadata: Any) -> None:
        self._log("error", operation, **metadata)


def validate_session_id(session_id: Optional[str]) -> str:
    """Validate and normalize a session ID.

    Args:
        session_id: Session ID to validate

    Returns:
        Valid session ID

    Raises:
        ValueError: If session ID is invalid
    """
    if not session_id or not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("Session ID must be a non-empty string")
    return session_id.strip()
