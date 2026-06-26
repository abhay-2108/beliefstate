import logging
from typing import Optional, AsyncGenerator, Any

try:
    from fastapi import Header, Request

    HAS_FASTAPI = True
except ImportError:
    Header = Request = Any  # type: ignore
    HAS_FASTAPI = False
from beliefstate.tracker import session_context
from beliefstate.integrations.asgi import BeliefTrackerASGIMiddleware
from beliefstate.integrations.common import IntegrationLogger, validate_session_id

logger = logging.getLogger(__name__)


class FastAPIBeliefTrackerMiddleware(BeliefTrackerASGIMiddleware):
    """
    FastAPI-branded ASGI middleware to automatically extract a session ID
    from request headers, set it in the tracker's context, and expose it
    via request.state.session_id.

    Features:
    - Automatic session ID extraction from X-Session-ID header
    - Request-scoped context propagation
    - Structured logging
    - Error handling with graceful degradation

    Usage:
        app = FastAPI()
        app.add_middleware(FastAPIBeliefTrackerMiddleware)
    """

    def __init__(self, app: Any, header_name: str = "x-session-id"):
        super().__init__(app, header_name)
        self.log = IntegrationLogger(__name__, "FastAPI")

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
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
            if "state" not in scope:
                scope["state"] = {}
            scope["state"]["session_id"] = session_id

            token = session_context.set(session_id)
            try:
                self.log.debug("Session context set", session_id=session_id)
                await self.app(scope, receive, send)
            except Exception as e:
                self.log.error(
                    "Error in middleware", session_id=session_id, error=str(e)
                )
                raise
            finally:
                session_context.reset(token)
                self.log.debug("Session context reset", session_id=session_id)
        else:
            # Proceed without session context (optional session)
            self.log.debug("No session ID found in request")
            await self.app(scope, receive, send)


if HAS_FASTAPI:

    async def get_session_id(
        request: Request, x_session_id: Optional[str] = Header(None, alias="X-Session-ID")
    ) -> AsyncGenerator[Optional[str], None]:
        """
        FastAPI dependency injection helper to extract the session ID from the
        X-Session-ID header (or fallback to request.state) and bind it to the
        tracker context.

        Features:
        - Graceful fallback if session ID is missing (allows optional sessions)
        - Request-scoped context propagation
        - Structured logging

        Usage:
            @app.post("/chat")
            async def chat(message: str, session_id: str = Depends(get_session_id)):
                # session_id is automatically set in tracker context
                ...

        Raises:
            ValueError: If session ID validation fails (can be caught by FastAPI error handlers)
        """
        log = IntegrationLogger(__name__, "FastAPI")

        # Try to get session ID from header, then from request state
        sid = x_session_id or getattr(request.state, "session_id", None)

        if sid:
            try:
                # Validate session ID
                sid = validate_session_id(sid)
                token = session_context.set(sid)
                try:
                    log.debug("Session context set in dependency", session_id=sid)
                    yield sid
                finally:
                    session_context.reset(token)
                    log.debug("Session context reset in dependency", session_id=sid)
            except ValueError as e:
                log.error("Invalid session ID in dependency", error=str(e))
                raise
        else:
            # Session ID is optional
            log.debug("No session ID provided in dependency (optional)")
            yield None
