import asyncio
from typing import Any, Dict, Protocol, runtime_checkable

from beliefstate.call import LLMCall, LLMResponse

_global_tracker = None


def register_global_tracker(tracker: Any) -> None:
    """Register the global tracker instance to be used by background workers."""
    global _global_tracker
    _global_tracker = tracker


def get_global_tracker() -> Any:
    """Retrieve the registered global tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        raise RuntimeError(
            "Global BeliefTracker is not registered. "
            "Please call `register_global_tracker(tracker)` in your worker startup script."
        )
    return _global_tracker


def execute_tracking_task(
    call_dict: Dict[str, Any], response_dict: Dict[str, Any], session_id: str, turn: int
) -> None:
    """Worker entrypoint for processing belief tracking tasks."""
    tracker = get_global_tracker()

    # Run the async tracking task in a new event loop on the worker thread/process
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            tracker.track_async(call_dict, response_dict, session_id, turn)
        )
    finally:
        loop.close()


@runtime_checkable
class TaskDispatcher(Protocol):
    """Protocol defining the interface for enqueuing background tasks."""

    def dispatch(
        self,
        tracker: Any,
        call: LLMCall,
        response: LLMResponse,
        session_id: str,
        turn: int,
    ) -> None:
        """Dispatch the belief tracking task."""
        ...


class AsyncioDispatcher:
    """Default dispatcher that runs tasks in the background using asyncio.create_task."""

    def dispatch(
        self,
        tracker: Any,
        call: LLMCall,
        response: LLMResponse,
        session_id: str,
        turn: int,
    ) -> None:
        asyncio.create_task(
            tracker.track_async(
                call.model_dump(), response.model_dump(), session_id, turn
            )
        )


class SyncDispatcher:
    """Dispatcher that runs tasks synchronously (useful for testing or single-threaded environments)."""

    def dispatch(
        self,
        tracker: Any,
        call: LLMCall,
        response: LLMResponse,
        session_id: str,
        turn: int,
    ) -> None:
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.create_task(
                    tracker.track_async(
                        call.model_dump(), response.model_dump(), session_id, turn
                    )
                )
                return
        except RuntimeError:
            pass

        asyncio.run(
            tracker.track_async(
                call.model_dump(), response.model_dump(), session_id, turn
            )
        )


class CeleryDispatcher:
    """Dispatcher that serializes payloads and enqueues them via Celery."""

    def __init__(
        self,
        celery_app: Any = None,
        task_name: str = "beliefstate.dispatcher.execute_tracking_task",
        **kwargs: Any,
    ):
        self.celery_app = celery_app
        self.task_name = task_name
        self.kwargs = kwargs

    def dispatch(
        self,
        tracker: Any,
        call: LLMCall,
        response: LLMResponse,
        session_id: str,
        turn: int,
    ) -> None:
        if not self.celery_app:
            raise RuntimeError(
                "CeleryDispatcher requires a configured celery_app instance."
            )

        call_dict = call.model_dump()
        response_dict = response.model_dump()

        self.celery_app.send_task(
            self.task_name,
            args=(call_dict, response_dict, session_id, turn),
            **self.kwargs,
        )


class RQDispatcher:
    """Dispatcher that serializes payloads and enqueues them via Redis Queue (rq)."""

    def __init__(
        self,
        queue: Any = None,
        queue_name: str = "default",
        connection: Any = None,
        **kwargs: Any,
    ):
        self.kwargs = kwargs
        if queue is not None:
            self.queue = queue
        else:
            try:
                from rq import Queue
                from redis import Redis
            except ImportError:
                raise ImportError(
                    "rq and redis packages are required to use RQDispatcher. "
                    "Install them with `pip install rq redis`."
                )

            conn = connection or Redis()
            self.queue = Queue(name=queue_name, connection=conn)

    def dispatch(
        self,
        tracker: Any,
        call: LLMCall,
        response: LLMResponse,
        session_id: str,
        turn: int,
    ) -> None:
        call_dict = call.model_dump()
        response_dict = response.model_dump()

        self.queue.enqueue(
            execute_tracking_task,
            call_dict,
            response_dict,
            session_id,
            turn,
            **self.kwargs,
        )
