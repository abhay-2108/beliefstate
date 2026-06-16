from typing import Optional, AsyncGenerator, Any
from fastapi import Header, Request
from beliefstate.tracker import session_context
from beliefstate.integrations.asgi import BeliefTrackerASGIMiddleware


class FastAPIBeliefTrackerMiddleware(BeliefTrackerASGIMiddleware):
    """
    FastAPI-branded ASGI middleware to automatically extract a session ID
    from request headers, set it in the tracker's context, and expose it
    via request.state.session_id.
    """

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        session_id = None
        for name, value in scope.get("headers", []):
            if name == self.header_name:
                session_id = value.decode("latin1")
                break

        if session_id:
            if "state" not in scope:
                scope["state"] = {}
            scope["state"]["session_id"] = session_id

            token = session_context.set(session_id)
            try:
                await self.app(scope, receive, send)
            finally:
                session_context.reset(token)
        else:
            await self.app(scope, receive, send)


async def get_session_id(
    request: Request, x_session_id: Optional[str] = Header(None, alias="X-Session-ID")
) -> AsyncGenerator[Optional[str], None]:
    """
    FastAPI dependency injection helper to extract the session ID from the
    X-Session-ID header (or fallback to request.state) and bind it to the
    tracker context.

    Usage:
        @app.post("/chat")
        async def chat(message: str, session_id: str = Depends(get_session_id)):
            ...
    """
    sid = x_session_id or getattr(request.state, "session_id", None)
    if sid:
        token = session_context.set(sid)
        try:
            yield sid
        finally:
            session_context.reset(token)
    else:
        yield None
