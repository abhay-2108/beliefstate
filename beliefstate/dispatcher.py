import asyncio
import logging
import threading
from typing import Any, Dict, Protocol, runtime_checkable
from asyncio import Task as AsyncTask

from beliefstate.call import LLMCall, LLMResponse
from beliefstate.tracker import session_context

logger = logging.getLogger(__name__)

_global_tracker = None
_global_tracker_lock = threading.Lock()


def _log_task_error(task: asyncio.Task[None]) -> None:
    """Log exceptions from fire-and-forget background tasks."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error(f"Background tracking task failed: {exc}", exc_info=exc)


def register_global_tracker(tracker: Any) -> None:
    """Register the global tracker instance to be used by background workers.

    IMPORTANT: Call this in your worker startup script BEFORE processing tasks.
    ContextVar (session_context) does not cross process boundaries, so session_id
    must be explicitly passed through task payloads.

    Example:
        # In your celery worker startup (tasks.py or celery config)
        from my_app import tracker
        from beliefstate.dispatcher import register_global_tracker

        register_global_tracker(tracker)
    """
    global _global_tracker
    with _global_tracker_lock:
        _global_tracker = tracker


def get_global_tracker() -> Any:
    """Retrieve the registered global tracker instance."""
    with _global_tracker_lock:
        if _global_tracker is None:
            raise RuntimeError(
                "Global BeliefTracker is not registered. "
                "Please call `register_global_tracker(tracker)` in your worker startup script."
            )
        return _global_tracker


def execute_tracking_task(
    call_dict: Dict[str, Any], response_dict: Dict[str, Any], session_id: str, turn: int
) -> None:
    """Worker entrypoint for processing belief tracking tasks.

    This function is called by Celery/RQ workers in a separate process.
    Session context does NOT propagate across process boundaries,
    so session_id is explicitly passed as a parameter.

    Args:
        call_dict: Serialized LLMCall (from model_dump())
        response_dict: Serialized LLMResponse (from model_dump())
        session_id: Session ID (explicitly passed, not from ContextVar)
        turn: Turn number in conversation
    """
    tracker = get_global_tracker()

    # Set session context in the worker process so internal code can access it
    token = session_context.set(session_id)

    # Run the async tracking task in a new event loop on the worker thread/process
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            tracker.track_async(call_dict, response_dict, session_id, turn)
        )
    finally:
        loop.close()
        session_context.reset(token)


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
    """Default dispatcher that runs tasks in the background using asyncio.create_task.

    ⚠️ WARNING: This dispatcher is NOT DURABLE - in-flight tasks are lost on process restart.

    Suitable for: Development, testing, single-process deployments
    NOT suitable for: Production, multi-worker setups, applications requiring durability

    For production use, consider using:
    - CeleryDispatcher: Distributed task queue with RabbitMQ/Redis backend
    - RQDispatcher: Simpler queue with Redis backend

    Tracks in-flight tasks for graceful draining during GDPR deletion.
    """

    def __init__(self, log_warning_in_production: bool = True):
        """Initialize AsyncioDispatcher.

        Args:
            log_warning_in_production: If True, logs a warning if BELIEFSTATE_ENV=production is detected
        """
        self._in_flight_tasks: Dict[
            str, list[AsyncTask[Any]]
        ] = {}  # session_id -> [tasks]

        # Warn if running in production
        if log_warning_in_production:
            import os

            env = os.getenv("BELIEFSTATE_ENV", "").lower()
            if env == "production" or env == "prod":
                logger.warning(
                    "AsyncioDispatcher is NOT DURABLE: in-flight belief extraction tasks are lost on process restart. "
                    "For production deployments, use CeleryDispatcher or RQDispatcher instead. "
                    "Set BELIEFSTATE_ENV=development to suppress this warning."
                )

    def dispatch(
        self,
        tracker: Any,
        call: LLMCall,
        response: LLMResponse,
        session_id: str,
        turn: int,
    ) -> None:
        task = asyncio.create_task(
            tracker.track_async(
                call.model_dump(), response.model_dump(), session_id, turn
            )
        )

        # Track task for this session
        if session_id not in self._in_flight_tasks:
            self._in_flight_tasks[session_id] = []
        self._in_flight_tasks[session_id].append(task)

        # Clean up completed tasks
        def cleanup_task(t: AsyncTask[Any]) -> None:
            if session_id in self._in_flight_tasks:
                try:
                    self._in_flight_tasks[session_id].remove(task)
                except (ValueError, KeyError):
                    pass
                # Sweep: remove empty session entries
                if not self._in_flight_tasks[session_id]:
                    self._in_flight_tasks.pop(session_id, None)

        task.add_done_callback(cleanup_task)

    async def drain_session(self, session_id: str) -> int:
        """Wait for all in-flight tasks for a session to complete.

        Returns:
            Number of tasks that were drained
        """
        if session_id not in self._in_flight_tasks:
            return 0

        tasks = self._in_flight_tasks.get(session_id, [])
        count = len(tasks)

        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
                logger.debug(
                    f"Drained {count} in-flight tasks for session {session_id}"
                )
            except Exception as e:
                logger.warning(f"Error draining tasks: {e}")

        # Clean up
        self._in_flight_tasks.pop(session_id, None)
        return count


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
                task = loop.create_task(
                    tracker.track_async(
                        call.model_dump(), response.model_dump(), session_id, turn
                    )
                )
                task.add_done_callback(_log_task_error)
                return
        except RuntimeError:
            pass

        asyncio.run(
            tracker.track_async(
                call.model_dump(), response.model_dump(), session_id, turn
            )
        )


class CeleryDispatcher:
    """Dispatcher that serializes payloads and enqueues them via Celery.

    SESSION CONTEXT PROPAGATION:
    ContextVar (session_context) does NOT propagate across process boundaries.
    This dispatcher explicitly serializes session_id into the task payload,
    ensuring it reaches the worker process intact.

    Setup:
        1. Create dispatcher with celery app:
           dispatcher = CeleryDispatcher(celery_app=celery_app)

        2. Register tracker in worker startup:
           # tasks.py or celery config
           from beliefstate.dispatcher import register_global_tracker
           register_global_tracker(tracker)

        3. Define the Celery task:
           @celery_app.task(name="beliefstate.dispatcher.execute_tracking_task")
           def celery_tracking_worker(call_dict, response_dict, session_id, turn):
               from beliefstate.dispatcher import execute_tracking_task
               execute_tracking_task(call_dict, response_dict, session_id, turn)
    """

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

        # Explicitly pass session_id as a task argument (not relying on ContextVar)
        self.celery_app.send_task(
            self.task_name,
            args=(call_dict, response_dict, session_id, turn),
            **self.kwargs,
        )
        logger.debug(
            f"Enqueued belief tracking task for session={session_id}, turn={turn}"
        )


class RQDispatcher:
    """Dispatcher that serializes payloads and enqueues them via Redis Queue (rq).

    SESSION CONTEXT PROPAGATION:
    ContextVar (session_context) does NOT propagate across process boundaries.
    This dispatcher explicitly serializes session_id into the task payload,
    ensuring it reaches the worker process intact.

    Setup:
        1. Create dispatcher with queue:
           from rq import Queue
           from redis import Redis
           queue = Queue('belief-tracking', connection=Redis())
           dispatcher = RQDispatcher(queue=queue)

        2. Register tracker in worker startup:
           # worker_startup.py or before running worker
           from beliefstate.dispatcher import register_global_tracker
           register_global_tracker(tracker)

        3. Start RQ worker:
           rq worker belief-tracking
    """

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

        # Explicitly pass session_id as a task argument (not relying on ContextVar)
        self.queue.enqueue(
            execute_tracking_task,
            call_dict,
            response_dict,
            session_id,
            turn,
            **self.kwargs,
        )
        logger.debug(
            f"Enqueued belief tracking task for session={session_id}, turn={turn}"
        )
