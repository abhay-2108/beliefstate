from typing import Optional, Any

try:
    from flask import Flask, request, g

    HAS_FLASK = True
except ImportError:
    Flask = request = g = Any  # type: ignore
    HAS_FLASK = False
from beliefstate.tracker import session_context
from beliefstate.integrations.wsgi import BeliefTrackerWSGIMiddleware
from beliefstate.integrations.common import IntegrationLogger, validate_session_id


class FlaskBeliefTrackerMiddleware(BeliefTrackerWSGIMiddleware):
    """
    Flask-branded WSGI middleware to automatically extract a session ID
    from an incoming request header and set it in the tracker's context.

    Features:
    - Automatic session ID extraction from X-Session-ID header
    - Request-scoped context propagation using Flask g
    - Structured logging
    - Error handling with graceful degradation

    Usage:
        app = Flask(__name__)
        app.wsgi_app = FlaskBeliefTrackerMiddleware(app.wsgi_app)
    """

    def __init__(self, app: Any, header_name: str = "X-Session-ID"):
        # Pass header_name as string, parent will handle conversion
        super().__init__(app, header_name)
        self.log = IntegrationLogger(__name__, "Flask")


def register_flask_hooks(app: Flask, header_name: str = "X-Session-ID") -> None:
    """
    Helper to register request hooks directly on a Flask application instance.
    This sets the session ID globally within the request context using flask.g
    and binds the ContextVar for the duration of the request.

    Features:
    - Automatic session ID extraction from headers
    - Request-scoped context propagation
    - Structured logging
    - Graceful error handling
    - Thread-safe using Flask's g object

    Usage:
        app = Flask(__name__)
        register_flask_hooks(app)

        @app.route("/chat", methods=["POST"])
        def chat():
            # Session context is automatically available
            ...
    """
    log = IntegrationLogger(__name__, "Flask")

    @app.before_request
    def set_session_id() -> None:
        """Extract and set session ID at the start of request."""
        try:
            session_id = request.headers.get(header_name)
            if session_id:
                try:
                    # Validate session ID
                    session_id = validate_session_id(session_id)
                    g.session_id = session_id
                    g.session_token = session_context.set(session_id)
                    log.debug("Session context set", session_id=session_id)
                except ValueError as e:
                    log.warning("Invalid session ID in header", error=str(e))
                    g.session_id = None
                    g.session_token = None
            else:
                log.debug("No session ID found in request headers")
                g.session_id = None
                g.session_token = None
        except Exception as e:
            log.error("Error setting session ID", error=str(e))
            g.session_id = None
            g.session_token = None

    @app.teardown_request
    def reset_session_id(exception: Optional[BaseException] = None) -> None:
        """Reset session context at the end of request."""
        try:
            token = getattr(g, "session_token", None)
            session_id = getattr(g, "session_id", None)
            if token:
                session_context.reset(token)
                log.debug("Session context reset", session_id=session_id)

            if exception:
                log.error(
                    "Request ended with exception",
                    session_id=session_id,
                    error=str(exception),
                )
        except Exception as e:
            log.error("Error resetting session ID", error=str(e))
