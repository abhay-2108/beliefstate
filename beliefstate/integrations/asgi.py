from typing import Any
from beliefstate.tracker import session_context
from beliefstate.integrations.common import IntegrationLogger, validate_session_id


class BeliefTrackerASGIMiddleware:
    """
    ASGI Middleware (works with FastAPI, Starlette, Litestar, Quart, etc.)
    to automatically extract a session ID from a request header and set it in the tracker's context.

    Features:
    - Automatic session ID extraction from configurable header
    - Request-scoped context propagation
    - Structured logging
    - Error handling with graceful degradation
    - Support for HTTP and WebSocket connections

    Usage:
        app = Starlette(...)
        app.add_middleware(BeliefTrackerASGIMiddleware)
    """

    def __init__(self, app: Any, header_name: str = "x-session-id") -> None:
        self.app = app
        self.header_name = header_name.lower().encode("latin1")
        self.log = IntegrationLogger(__name__, "ASGI")

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        session_id = None
        try:
            for name, value in scope.get("headers", []):
                if name == self.header_name:
                    try:
                        session_id = value.decode("latin1")
                        # Validate session ID
                        validate_session_id(session_id)
                        break
                    except (UnicodeDecodeError, ValueError) as e:
                        self.log.warning("Invalid session ID in header", error=str(e))
                        session_id = None
        except Exception as e:
            self.log.error("Error extracting session ID", error=str(e))
            session_id = None

        if session_id:
            # Set the context variable for this specific request
            token = session_context.set(session_id)
            try:
                self.log.debug(
                    "Session context set",
                    session_id=session_id,
                    scope_type=scope.get("type"),
                )
                await self.app(scope, receive, send)
            except Exception as e:
                self.log.error(
                    "Error in middleware",
                    session_id=session_id,
                    error=str(e),
                )
                raise
            finally:
                session_context.reset(token)
                self.log.debug("Session context reset", session_id=session_id)
        else:
            # Proceed without session context (optional session)
            self.log.debug("No session ID found in request")
            await self.app(scope, receive, send)
