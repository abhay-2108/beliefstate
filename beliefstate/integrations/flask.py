from typing import Optional
from flask import Flask, request, g
from beliefstate.tracker import session_context
from beliefstate.integrations.wsgi import BeliefTrackerWSGIMiddleware


class FlaskBeliefTrackerMiddleware(BeliefTrackerWSGIMiddleware):
    """
    Flask-branded WSGI middleware to automatically extract a session ID
    from an incoming request header and set it in the tracker's context.
    """

    pass


def register_flask_hooks(app: Flask, header_name: str = "X-Session-ID") -> None:
    """
    Helper to register request hooks directly on a Flask application instance.
    This sets the session ID globally within the request context using flask.g
    and binds the ContextVar for the duration of the request.

    Usage:
        app = Flask(__name__)
        register_flask_hooks(app)
    """

    @app.before_request
    def set_session_id() -> None:
        session_id = request.headers.get(header_name)
        if session_id:
            g.session_id = session_id
            g.session_token = session_context.set(session_id)

    @app.teardown_request
    def reset_session_id(exception: Optional[BaseException] = None) -> None:
        token = getattr(g, "session_token", None)
        if token:
            session_context.reset(token)
