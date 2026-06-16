from typing import Any
from beliefstate.tracker import session_context


class BeliefTrackerWSGIMiddleware:
    """
    WSGI Middleware (works with Flask, Django, etc.)
    to automatically extract a session ID from a request header and set it in the tracker's context.
    """

    def __init__(self, app: Any, header_name: str = "X-Session-ID") -> None:
        self.app = app
        self.header_name = header_name

    def __call__(self, environ: Any, start_response: Any) -> Any:
        # WSGI standardizes headers to HTTP_UPPER_CASE_WITH_UNDERSCORES
        wsgi_header = "HTTP_" + self.header_name.upper().replace("-", "_")
        session_id = environ.get(wsgi_header)

        if session_id:
            token = session_context.set(session_id)
            try:
                return self.app(environ, start_response)
            finally:
                session_context.reset(token)

        return self.app(environ, start_response)
