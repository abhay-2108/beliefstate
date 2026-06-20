import pytest
from beliefstate.integrations.asgi import BeliefTrackerASGIMiddleware
from beliefstate.tracker import session_context

class MockASGIApp:
    async def __call__(self, scope, receive, send):
        # Verify that context is set inside the application call
        self.session_id = session_context.get()
        return "called"

@pytest.mark.asyncio
async def test_asgi_middleware_non_http():
    app = MockASGIApp()
    middleware = BeliefTrackerASGIMiddleware(app)
    scope = {"type": "lifespan"}
    
    await middleware(scope, None, None)
    assert not hasattr(app, "session_id") or app.session_id == "default"

@pytest.mark.asyncio
async def test_asgi_middleware_no_header():
    app = MockASGIApp()
    middleware = BeliefTrackerASGIMiddleware(app)
    scope = {
        "type": "http",
        "headers": []
    }
    
    await middleware(scope, None, None)
    assert app.session_id == "default"

@pytest.mark.asyncio
async def test_asgi_middleware_valid_header():
    app = MockASGIApp()
    middleware = BeliefTrackerASGIMiddleware(app)
    scope = {
        "type": "http",
        "headers": [
            (b"x-session-id", b"user-123")
        ]
    }
    
    await middleware(scope, None, None)
    assert app.session_id == "user-123"
    # Out of middleware, context should be reset to default
    assert session_context.get() == "default"

@pytest.mark.asyncio
async def test_asgi_middleware_invalid_header_value():
    app = MockASGIApp()
    middleware = BeliefTrackerASGIMiddleware(app)
    scope = {
        "type": "http",
        "headers": [
            (b"x-session-id", b"")  # Invalid session ID (empty)
        ]
    }
    
    await middleware(scope, None, None)
    assert app.session_id == "default"
