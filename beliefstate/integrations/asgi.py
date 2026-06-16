from beliefstate.tracker import session_context


class BeliefTrackerASGIMiddleware:
    """
    ASGI Middleware (works with FastAPI, Starlette, Litestar, Quart, etc.)
    to automatically extract a session ID from a request header and set it in the tracker's context.
    """

    def __init__(self, app, header_name: str = "x-session-id"):
        self.app = app
        self.header_name = header_name.lower().encode("latin1")

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        session_id = None
        for name, value in scope.get("headers", []):
            if name == self.header_name:
                session_id = value.decode("latin1")
                break

        if session_id:
            # Set the context variable for this specific request
            token = session_context.set(session_id)
            try:
                await self.app(scope, receive, send)
            finally:
                session_context.reset(token)
        else:
            await self.app(scope, receive, send)
