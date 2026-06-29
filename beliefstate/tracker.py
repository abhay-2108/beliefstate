import asyncio
import logging
import math
import os
import warnings
from typing import Any, Callable, Coroutine, Dict, List, Optional, TypeVar, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from contextvars import ContextVar
from collections import deque

from beliefstate.config import TrackerConfig
from beliefstate.call import LLMCall, LLMResponse
from beliefstate.adapters.base import ProviderAdapter
from beliefstate.models import Belief, DeletionReceipt
from beliefstate.store.base import Store, summary_for_prompt
from beliefstate.store.sqlite import SQLiteStore
from beliefstate.extractor import BeliefExtractor
from beliefstate.detector import ContradictionDetector
from beliefstate.resolver import BeliefResolver

logger = logging.getLogger(__name__)

session_context: ContextVar[str] = ContextVar("session_id", default="default")
conversation_context: ContextVar[Optional[str]] = ContextVar(
    "conversation_id", default=None
)

T = TypeVar("T")

# Per-session async locks for coordinating concurrent writes
_session_locks: Dict[str, asyncio.Lock] = {}


class ConfigurationWarning(UserWarning):
    pass


def _validate_deployment_config(config: TrackerConfig) -> None:
    """Warn if SQLite is used with multiple workers."""
    worker_count = max(
        int(os.environ.get("WEB_CONCURRENCY", 1)),
        int(os.environ.get("GUNICORN_WORKERS", 1)),
        int(os.environ.get("NUM_WORKERS", 1)),
    )
    if worker_count > 1 and config.store_type == "sqlite":
        warnings.warn(
            f"beliefstate: SQLite store detected with {worker_count} workers. "
            "Each worker has its own belief store — beliefs are NOT shared "
            "across workers and contradiction detection will not work correctly. "
            "Set store_type='redis' for multi-worker deployments.",
            ConfigurationWarning,
            stacklevel=3,
        )


def _ensure_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def calculate_staleness_score(belief: Any) -> float:
    if not hasattr(belief, "last_referenced_at") or (belief.last_referenced_at is None):
        ref_time = (
            belief.created_at
            if hasattr(belief, "created_at")
            else datetime.now(timezone.utc)
        )
    else:
        ref_time = belief.last_referenced_at
    ref_time = _ensure_aware(ref_time)
    days_since_referenced = (datetime.now(timezone.utc) - ref_time).days
    confidence = getattr(belief, "confidence", 1.0)
    return confidence / (days_since_referenced + 1)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return math.ceil(len(text) / 4)


class GenericAdapter(ProviderAdapter):
    """Fallback adapter for unknown LLM providers. Cannot generate."""

    def to_llm_call(self, *args: Any, **kwargs: Any) -> LLMCall:
        messages = kwargs.get("messages", [])
        if not messages and len(args) > 0 and isinstance(args[0], list):
            messages = args[0]
        return LLMCall(messages=messages, kwargs=kwargs)

    def to_llm_response(self, response: Any) -> LLMResponse:
        text = ""
        if hasattr(response, "content"):
            text = response.content
        elif hasattr(response, "text"):
            text = response.text
        elif hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message"):
                text = choice.message.content
            elif hasattr(choice, "text"):
                text = choice.text
        elif isinstance(response, dict):
            text = response.get("content") or response.get("text", "")
        return LLMResponse(text=text, raw_response=response)

    async def generate(
        self, call: LLMCall, response_format: Optional[Any] = None
    ) -> LLMResponse:
        raise NotImplementedError("Generic adapter cannot generate.")

    async def get_embedding(self, text: str) -> list[float]:
        raise NotImplementedError("Generic adapter cannot generate embeddings.")

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Generic adapter cannot generate embeddings.")

    def inject_context(self, context_prompt: str, *args: Any, **kwargs: Any) -> Any:
        return kwargs

    async def health_check(self) -> bool:
        return False


# Cache adapter instances to avoid re-instantiation on every call
_adapter_cache: Dict[str, ProviderAdapter] = {}


def _get_cached_adapter(
    cache_key: str, module_path: str, class_name: str
) -> ProviderAdapter:
    """Get or create a cached adapter instance."""
    if cache_key in _adapter_cache:
        return _adapter_cache[cache_key]
    try:
        import importlib

        module = importlib.import_module(module_path)
        adapter_cls = getattr(module, class_name)
        instance: ProviderAdapter = adapter_cls()
        _adapter_cache[cache_key] = instance
        return instance
    except ImportError:
        raise RuntimeError(
            f"{class_name} selected but SDK not installed. "
            f"Install with: pip install {cache_key}"
        )


def _detect_adapter(result: Any) -> ProviderAdapter:
    # Try isinstance checks first (most reliable), fall back to type name
    try:
        from openai.types.chat import ChatCompletion

        if isinstance(result, ChatCompletion):
            return _get_cached_adapter(
                "openai", "beliefstate.adapters.openai", "OpenAIAdapter"
            )
    except (ImportError, AttributeError, TypeError):
        pass

    try:
        from anthropic.types import Message

        if isinstance(result, Message):
            return _get_cached_adapter(
                "anthropic", "beliefstate.adapters.anthropic", "AnthropicAdapter"
            )
    except (ImportError, AttributeError, TypeError):
        pass

    try:
        from google.generativeai.types.generation_types import (
            GenerateContentResponse,
        )

        if isinstance(result, GenerateContentResponse):
            return _get_cached_adapter(
                "gemini", "beliefstate.adapters.gemini", "GeminiAdapter"
            )
    except (ImportError, AttributeError, TypeError):
        pass

    try:
        from ollama._types import ChatResponse

        if isinstance(result, ChatResponse):
            return _get_cached_adapter(
                "ollama", "beliefstate.adapters.ollama", "OllamaAdapter"
            )
    except (ImportError, AttributeError, TypeError):
        pass

    # Fallback to type name matching
    type_name = type(result).__module__ + "." + type(result).__name__
    type_lower = type_name.lower()

    if "openai" in type_lower:
        return _get_cached_adapter(
            "openai", "beliefstate.adapters.openai", "OpenAIAdapter"
        )
    if "anthropic" in type_lower:
        return _get_cached_adapter(
            "anthropic", "beliefstate.adapters.anthropic", "AnthropicAdapter"
        )
    if "google" in type_lower or "genai" in type_lower:
        return _get_cached_adapter(
            "gemini", "beliefstate.adapters.gemini", "GeminiAdapter"
        )
    if "ollama" in type_lower:
        return _get_cached_adapter(
            "ollama", "beliefstate.adapters.ollama", "OllamaAdapter"
        )

    logger.warning(
        f"Could not auto-detect adapter from type: {type_name}. Using generic adapter."
    )
    return GenericAdapter()


def _get_session_lock(session_id: str) -> asyncio.Lock:
    lock = _session_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        existing = _session_locks.setdefault(session_id, lock)
        if existing is not lock:
            lock = existing
    return lock


# --- Stats tracking ---


@dataclass
class TrackerStats:
    total_turns_processed: int = 0
    total_beliefs_extracted: int = 0
    total_contradictions_detected: int = 0
    total_duplicates_skipped: int = 0
    extraction_errors: int = 0
    last_error: str = ""
    last_successful_extraction: Optional[datetime] = None
    _recent_outcomes: deque[int] = field(default_factory=lambda: deque(maxlen=100))

    @property
    def extraction_success_rate(self) -> float:
        if not self._recent_outcomes:
            return 1.0
        return float(sum(self._recent_outcomes)) / len(self._recent_outcomes)

    def record_success(self) -> None:
        self._recent_outcomes.append(1)
        self.last_successful_extraction = datetime.now(timezone.utc)
        self.total_turns_processed += 1

    def record_error(self, error: str) -> None:
        self._recent_outcomes.append(0)
        self.extraction_errors += 1
        self.last_error = error


class AsyncStreamWrapper:
    def __init__(
        self,
        stream_gen: Any,
        tracker: "BeliefTracker",
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        session_id: str,
        turn: int,
    ):
        self.stream_gen = stream_gen
        self.tracker = tracker
        self.args = args
        self.kwargs = kwargs
        self.session_id = session_id
        self.turn = turn
        self.accumulated_text = ""
        self.first_chunk = None

    def __aiter__(self) -> "AsyncStreamWrapper":
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self.stream_gen.__anext__()
        except StopAsyncIteration:
            await self._finalize_tracking()
            raise StopAsyncIteration
        if self.first_chunk is None:
            self.first_chunk = chunk
        chunk_text = ""
        if hasattr(chunk, "choices") and chunk.choices:
            if hasattr(chunk.choices[0], "delta") and hasattr(
                chunk.choices[0].delta, "content"
            ):
                chunk_text = chunk.choices[0].delta.content or ""
        elif isinstance(chunk, dict):
            if "choices" in chunk and chunk["choices"]:
                choice = chunk["choices"][0]
                if "delta" in choice and "content" in choice["delta"]:
                    chunk_text = choice["delta"]["content"] or ""
        elif hasattr(chunk, "content"):
            chunk_text = chunk.content or ""
        elif hasattr(chunk, "text"):
            chunk_text = chunk.text or ""
        self.accumulated_text += chunk_text
        return chunk

    async def _finalize_tracking(self) -> None:
        try:
            if self.first_chunk is not None and hasattr(self.first_chunk, "choices"):
                reconstructed = {
                    "id": getattr(self.first_chunk, "id", "stream-accumulated"),
                    "object": "chat.completion",
                    "created": getattr(self.first_chunk, "created", 0),
                    "model": getattr(self.first_chunk, "model", "unknown"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": self.accumulated_text,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }
            else:
                reconstructed = {
                    "content": self.accumulated_text,
                    "text": self.accumulated_text,
                }

            if self.tracker._auto_detect_adapter and self.tracker.app_adapter is None:
                self.tracker.app_adapter = _detect_adapter(reconstructed)
                from beliefstate.resilience import ResilientAdapterWrapper

                self.tracker.internal_adapter = ResilientAdapterWrapper(
                    self.tracker.app_adapter, self.tracker.config
                )
                self.tracker._ensure_initialized()

            llm_call = self.tracker.app_adapter.to_llm_call(*self.args, **self.kwargs)
            llm_response = self.tracker.app_adapter.to_llm_response(reconstructed)
            self.tracker._ensure_initialized()
            self.tracker._dispatch(
                self.tracker._track_background(
                    llm_call, llm_response, self.session_id, self.turn
                )
            )
        except Exception as e:
            logger.error(
                f"Tracker stream finalization error for session {self.session_id} "
                f"turn {self.turn}: {e}",
                exc_info=True,
            )


class BeliefTracker:
    def __init__(
        self,
        config: TrackerConfig,
        adapter: Optional[ProviderAdapter] = None,
        store: Optional[Store] = None,
        internal_adapter: Optional[ProviderAdapter] = None,
        dispatcher: Optional[Any] = None,
        judge: Optional[Any] = None,
    ):
        _validate_deployment_config(config)
        self.config = config

        self._auto_detect_adapter = adapter is None
        if adapter is not None:
            self.app_adapter = adapter
        else:
            self.app_adapter = None  # type: ignore[assignment]

        from beliefstate.resilience import ResilientAdapterWrapper

        if internal_adapter is not None:
            raw_internal = internal_adapter
        elif adapter is not None:
            raw_internal = adapter
        else:
            raw_internal = None

        self.internal_adapter = (
            ResilientAdapterWrapper(raw_internal, self.config)
            if raw_internal is not None
            else None
        )

        self.store: Store
        if store is None:
            stype = self.config.store_type.lower()
            if stype == "postgres":
                from beliefstate.store.postgres import PostgreSQLStore

                self.store = PostgreSQLStore(**self.config.store_kwargs)
            elif stype == "redis":
                from beliefstate.store.redis import RedisStore

                if RedisStore is None:
                    raise RuntimeError(
                        "Redis SDK is not installed. Run `pip install redis`"
                    )
                self.store = RedisStore(**self.config.store_kwargs)
            else:
                self.store = SQLiteStore(
                    db_path=self.config.store_kwargs.get("db_path", "beliefstate.db")
                )
        else:
            self.store = store

        self.extractor: Optional[BeliefExtractor] = None
        self.detector: Optional[ContradictionDetector] = None
        self.resolver = BeliefResolver(store=self.store, strategy="overwrite")
        self._session_turn_counters: Dict[str, int] = {}
        self._session_turn_states: Dict[str, int] = {}
        self._session_providers: Dict[str, str] = {}
        self._stats = TrackerStats()
        self._pending_tasks: Set[asyncio.Task[None]] = set()

        if dispatcher is not None:
            self.dispatcher = dispatcher
        else:
            from beliefstate.dispatcher import (
                AsyncioDispatcher,
                SyncDispatcher,
                CeleryDispatcher,
                RQDispatcher,
            )

            dtype = self.config.task_dispatcher_type.lower()
            kwargs = self.config.dispatcher_kwargs
            if dtype == "asyncio":
                self.dispatcher = AsyncioDispatcher()
            elif dtype == "sync":
                self.dispatcher = SyncDispatcher()
            elif dtype == "celery":
                self.dispatcher = CeleryDispatcher(**kwargs)
            elif dtype == "rq":
                self.dispatcher = RQDispatcher(**kwargs)
            else:
                self.dispatcher = AsyncioDispatcher()

    @property
    def turn_counter(self) -> int:
        if not self._session_turn_counters:
            return 0
        return max(self._session_turn_counters.values())

    @turn_counter.setter
    def turn_counter(self, value: int) -> None:
        sid = session_context.get()
        self._session_turn_counters[sid] = value

    def get_session_turn(self, session_id: Optional[str] = None) -> int:
        sid = session_id or session_context.get()
        return self._session_turn_counters.get(sid, 0)

    def get_stats(self) -> TrackerStats:
        return self._stats

    def _dispatch(self, coro: Any) -> None:
        """Dispatch a coroutine as a tracked background task."""
        task = asyncio.create_task(coro)
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        self._sweep_stale_locks()

    def _sweep_stale_locks(self) -> None:
        """Remove session locks for sessions no longer tracked."""
        active_sessions = set(self._session_turn_counters.keys())
        stale = set(_session_locks.keys()) - active_sessions
        for sid in stale:
            lock = _session_locks[sid]
            if not lock.locked():
                del _session_locks[sid]

    async def shutdown(self, grace_seconds: float = 5.0) -> None:
        """Gracefully drain pending background tasks and close the store.

        Call this in FastAPI lifespan on_shutdown or signal handler.
        """
        if self._pending_tasks:
            logger.info(
                f"beliefstate_shutdown: draining {len(self._pending_tasks)} pending tasks"
            )
            await asyncio.wait(self._pending_tasks, timeout=grace_seconds)
            for task in list(self._pending_tasks):
                if not task.done():
                    task.cancel()
        if hasattr(self.store, "close"):
            await self.store.close()

    async def __aenter__(self) -> "BeliefTracker":
        if hasattr(self.store, "open"):
            await self.store.open()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if hasattr(self.store, "close"):
            await self.store.close()

    def _ensure_initialized(self) -> None:
        if self.internal_adapter is not None:
            raw_adapter = getattr(
                self.internal_adapter, "adapter", self.internal_adapter
            )
            is_generic = isinstance(raw_adapter, GenericAdapter) or (
                raw_adapter.__class__.__name__ == "GenericAdapter"
            )
            if is_generic:
                raise ValueError(
                    "Auto-detected LLM provider adapter does not support belief extraction/generation. "
                    "Please explicitly configure a generation-capable 'internal_provider' "
                    "or pass an adapter explicitly to TrackerConfig."
                )

        if self.extractor is None:
            if self.app_adapter is None:
                raise RuntimeError(
                    "Adapter not initialized. Wrap a decorated function first "
                    "or pass adapter explicitly."
                )
            self.extractor = BeliefExtractor(
                adapter=self.internal_adapter,  # type: ignore[arg-type]
                config=self.config,
            )

        if self.detector is None:
            if self.internal_adapter is None:
                raise RuntimeError(
                    "Internal adapter not initialized. Wrap a decorated "
                    "function first or pass adapter explicitly."
                )
            self.detector = ContradictionDetector(
                adapter=self.internal_adapter,
                store=self.store,
                config=self.config,
            )

    def set_session(self, session_id: str) -> None:
        session_context.set(session_id)

    def get_pending_conflicts(self, session_id: Optional[str] = None) -> list[str]:
        sid = session_id or session_context.get()
        return self.resolver.pop_pending_conflicts(sid)

    async def clear_session(
        self, session_id: Optional[str] = None
    ) -> "DeletionReceipt":
        sid = session_id or session_context.get()

        drained_count = 0
        if hasattr(self.dispatcher, "drain_session"):
            try:
                drained_count = await self.dispatcher.drain_session(sid)
                logger.info(
                    f"Drained {drained_count} in-flight tasks for session {sid}"
                )
            except Exception as e:
                logger.warning(f"Error draining tasks for session {sid}: {e}")

        beliefs = await self.store.get_beliefs(sid)
        beliefs_deleted = len(beliefs)
        await self.store.clear(sid)
        logger.info(f"Deleted {beliefs_deleted} beliefs for session {sid}")

        self.resolver.clear_session(sid)
        logger.info(f"Cleared conflict history for session {sid}")

        self._session_turn_counters.pop(sid, None)
        self._session_turn_states.pop(sid, None)
        self._session_providers.pop(sid, None)
        _session_locks.pop(sid, None)

        receipt = DeletionReceipt(
            session_id=sid,
            beliefs_deleted=beliefs_deleted,
            deleted_at=datetime.now(timezone.utc),
            in_flight_tasks_drained=drained_count,
        )
        logger.info(f"GDPR deletion receipt: {receipt}")
        return receipt

    async def get_beliefs(self, session_id: Optional[str] = None) -> List[Any]:
        sid = session_id or session_context.get()
        return await self.store.get_beliefs(sid)

    async def get_stats_dict(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        sid = session_id or session_context.get()
        beliefs = await self.store.get_beliefs(sid)
        if not beliefs:
            return {
                "total_beliefs": 0,
                "by_subject": {},
                "by_source": {},
                "avg_confidence": 0.0,
                "contradictions_detected": self.resolver.conflict_history.get(
                    sid, {}
                ).__len__(),
            }
        by_subject: Dict[str, int] = {}
        for b in beliefs:
            by_subject[b.subject] = by_subject.get(b.subject, 0) + 1
        by_source: Dict[str, int] = {}
        for b in beliefs:
            by_source[b.source] = by_source.get(b.source, 0) + 1
        avg_confidence = sum(b.confidence for b in beliefs) / len(beliefs)
        contradictions_count = len(self.resolver.conflict_history.get(sid, {}))
        return {
            "total_beliefs": len(beliefs),
            "by_subject": by_subject,
            "by_source": by_source,
            "avg_confidence": avg_confidence,
            "contradictions_detected": contradictions_count,
        }

    async def get_summary(
        self,
        session_id: Optional[str] = None,
        max_beliefs: Optional[int] = None,
        format_template: Optional[str] = None,
    ) -> str:
        max_b = max_beliefs or self.config.max_beliefs
        sid = session_id or session_context.get()
        beliefs = await self.store.get_beliefs(sid)
        if not beliefs:
            return ""
        return summary_for_prompt(beliefs, max_beliefs=max_b)

    async def clear_beliefs(self, session_id: Optional[str] = None) -> None:
        sid = session_id or session_context.get()
        await self.store.clear(sid)

    async def remove_belief(
        self, session_id: Optional[str], subject: str, predicate: str
    ) -> None:
        sid = session_id or session_context.get()
        await self.store.remove_belief(sid, subject, predicate)

    async def set_session_ttl(
        self, session_id: Optional[str], ttl_seconds: int
    ) -> None:
        sid = session_id or session_context.get()
        if hasattr(self.store, "set_session_ttl"):
            await self.store.set_session_ttl(sid, ttl_seconds)
        else:
            logger.warning(
                "Store does not support native TTL "
                "(use prune_expired_beliefs for SQLite)"
            )

    async def prune_expired_beliefs(
        self, session_id: Optional[str] = None, max_age_seconds: Optional[int] = None
    ) -> int:
        if not hasattr(self.store, "prune_expired_beliefs"):
            logger.warning("Store does not support belief pruning")
            return 0
        max_age = max_age_seconds or self.config.belief_max_age_seconds
        sid = session_id or session_context.get()
        if session_id:
            return int(await self.store.prune_expired_beliefs(max_age, sid))
        else:
            return int(await self.store.prune_expired_beliefs(max_age, None))

    async def update_belief_reference(
        self, session_id: Optional[str], subject: str, predicate: str
    ) -> None:
        sid = session_id or session_context.get()
        beliefs = await self.store.get_beliefs(sid)
        for b in beliefs:
            if b.subject == subject and b.predicate == predicate:
                b.last_referenced_at = datetime.now(timezone.utc)
                await self.store.update_belief(sid, b)
                break

    async def export_beliefs(
        self, session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        sid = session_id or session_context.get()
        beliefs = await self.store.get_beliefs(sid)
        return [b.model_dump(mode="json") for b in beliefs]

    async def import_beliefs(
        self,
        session_id: Optional[str] = None,
        beliefs_data: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        if not beliefs_data:
            return 0
        sid = session_id or session_context.get()
        imported = 0
        for item in beliefs_data:
            try:
                belief = Belief.model_validate(item)
                await self.store.add_belief(sid, belief)
                imported += 1
            except Exception as e:
                logger.warning(f"Skipping invalid belief during import: {e}")
        logger.info(
            f"Imported {imported}/{len(beliefs_data)} beliefs for session {sid}"
        )
        return imported

    async def health_check(self) -> Dict[str, bool]:
        result: Dict[str, bool] = {"store": False, "adapter": False}
        try:
            if hasattr(self.store, "health_check"):
                result["store"] = await self.store.health_check()
            else:
                await self.store.get_beliefs("__health_check__")
                result["store"] = True
        except Exception as e:
            logger.warning(f"Store health check failed: {e}")
        try:
            if self.internal_adapter and hasattr(self.internal_adapter, "health_check"):
                result["adapter"] = await self.internal_adapter.health_check()
            elif self.app_adapter and hasattr(self.app_adapter, "health_check"):
                result["adapter"] = await self.app_adapter.health_check()
        except Exception as e:
            logger.warning(f"Adapter health check failed: {e}")
        return result

    async def get_belief_history(
        self,
        session_id: str,
        subject: str,
        predicate: str,
    ) -> List[Dict[str, Any]]:
        """Return audit trail for a specific belief."""
        return await self.store.get_audit_history(session_id, subject, predicate)

    async def get_context_prompt(
        self,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        format_template: Optional[str] = None,
        current_user_message: Optional[str] = None,
    ) -> str:
        sid = session_id or session_context.get()
        cid = conversation_id or conversation_context.get()
        beliefs = await self.store.get_beliefs(sid, cid)
        if not beliefs:
            return ""

        if self.config.enable_staleness_scoring:
            filtered_beliefs = [
                b
                for b in beliefs
                if calculate_staleness_score(b) >= self.config.staleness_threshold
            ]
            if not filtered_beliefs:
                filtered_beliefs = sorted(
                    beliefs, key=lambda b: (b.confidence, b.turn), reverse=True
                )[:5]
        else:
            filtered_beliefs = beliefs

        filtered_beliefs = [
            b for b in filtered_beliefs if not getattr(b, "is_hypothetical", False)
        ]
        if not filtered_beliefs and beliefs:
            filtered_beliefs = beliefs

        strategy = self.config.belief_sort_strategy
        if strategy == "confidence_recency":
            sorted_beliefs = sorted(
                filtered_beliefs, key=lambda b: (b.confidence, b.turn), reverse=True
            )
        elif strategy == "recency":
            sorted_beliefs = sorted(
                filtered_beliefs, key=lambda b: b.turn, reverse=True
            )
        elif strategy == "confidence":
            sorted_beliefs = sorted(
                filtered_beliefs, key=lambda b: b.confidence, reverse=True
            )
        else:
            sorted_beliefs = sorted(
                filtered_beliefs, key=lambda b: (b.confidence, b.turn), reverse=True
            )

        if self.config.enable_token_aware_injection and current_user_message:
            sample_facts = [
                f"- {b.subject} {b.predicate} {b.value}"
                for b in sorted_beliefs[: self.config.max_beliefs]
            ]
            sample_summary = "\n".join(sample_facts)
            estimated_tokens = estimate_tokens(sample_summary)

            if estimated_tokens > self.config.belief_budget_tokens:
                try:
                    if self.extractor and hasattr(self.extractor, "adapter"):
                        user_msg_embedding = await self.extractor.adapter.get_embedding(
                            current_user_message
                        )
                        from beliefstate.detector import cosine_similarity

                        scored_beliefs = []
                        for b in sorted_beliefs:
                            if b.embedding:
                                similarity = cosine_similarity(
                                    user_msg_embedding, b.embedding
                                )
                                scored_beliefs.append((b, similarity))
                        scored_beliefs.sort(key=lambda x: x[1], reverse=True)
                        beliefs_per_token = len(sample_facts) / max(estimated_tokens, 1)
                        max_beliefs_in_budget = max(
                            1, int(beliefs_per_token * self.config.belief_budget_tokens)
                        )
                        sorted_beliefs = [
                            b for b, _ in scored_beliefs[:max_beliefs_in_budget]
                        ]
                except Exception as e:
                    logger.warning(
                        f"Token-aware injection fallback: {e}. Using default selection."
                    )

        capped_beliefs = sorted_beliefs[: self.config.max_beliefs]
        facts_list = [f"- {b.subject} {b.predicate} {b.value}" for b in capped_beliefs]
        facts_str = "\n".join(facts_list)
        if format_template:
            return format_template.format(facts=facts_str)
        return summary_for_prompt(capped_beliefs, max_beliefs=self.config.max_beliefs)

    async def inject_context(
        self,
        messages: list[Dict[str, Any]],
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        format_template: Optional[str] = None,
        current_user_message: Optional[str] = None,
    ) -> list[Dict[str, Any]]:
        context_prompt = await self.get_context_prompt(
            session_id=session_id,
            conversation_id=conversation_id,
            format_template=format_template,
            current_user_message=current_user_message,
        )
        if not context_prompt:
            return messages
        new_messages = [m.copy() for m in messages]
        system_idx = -1
        for idx, m in enumerate(new_messages):
            if isinstance(m, dict) and m.get("role") == "system":
                system_idx = idx
                break
        if system_idx != -1:
            orig_content = new_messages[system_idx].get("content", "")
            if orig_content:
                new_messages[system_idx]["content"] = (
                    f"{orig_content}\n\n{context_prompt}"
                )
            else:
                new_messages[system_idx]["content"] = context_prompt
        else:
            new_messages.insert(0, {"role": "system", "content": context_prompt})
        return new_messages

    def _generic_inject_context(
        self,
        context_prompt: str,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        if "messages" in kwargs:
            messages = kwargs["messages"]
            if isinstance(messages, list):
                new_kwargs = kwargs.copy()
                new_kwargs["messages"] = self._fallback_inject_messages(
                    messages, context_prompt
                )
                return args, new_kwargs
        if "contents" in kwargs:
            new_kwargs = kwargs.copy()
            if "config" in new_kwargs:
                new_kwargs["config"] = self._fallback_inject_config(
                    new_kwargs["config"], context_prompt
                )
            else:
                new_kwargs["config"] = {"system_instruction": context_prompt}
            return args, new_kwargs
        if "system" in kwargs or any(k.lower() == "system" for k in kwargs):
            key = next(k for k in kwargs if k.lower() == "system")
            new_kwargs = kwargs.copy()
            orig = new_kwargs.get(key, "")
            new_kwargs[key] = f"{orig}\n\n{context_prompt}" if orig else context_prompt
            return args, new_kwargs
        new_args = list(args)
        for idx, arg in enumerate(new_args[:2]):
            if isinstance(arg, list):
                new_args[idx] = self._fallback_inject_messages(arg, context_prompt)
                return tuple(new_args), kwargs
        return args, kwargs

    def _fallback_inject_messages(
        self, messages: List[Any], context_prompt: str
    ) -> List[Any]:
        new_messages = [m.copy() if isinstance(m, dict) else m for m in messages]
        system_idx = -1
        for idx, m in enumerate(new_messages):
            if isinstance(m, dict) and m.get("role") == "system":
                system_idx = idx
                break
            elif hasattr(m, "role") and m.role == "system":
                system_idx = idx
                break
        if system_idx != -1:
            m = new_messages[system_idx]
            if isinstance(m, dict):
                orig_content = m.get("content", "")
                m["content"] = (
                    f"{orig_content}\n\n{context_prompt}"
                    if orig_content
                    else context_prompt
                )
            else:
                orig_content = getattr(m, "content", "")
                m.content = (
                    f"{orig_content}\n\n{context_prompt}"
                    if orig_content
                    else context_prompt
                )
        else:
            new_messages.insert(0, {"role": "system", "content": context_prompt})
        return new_messages

    def _fallback_inject_config(self, config: Any, context_prompt: str) -> Any:
        if isinstance(config, dict):
            new_config = config.copy()
            orig = new_config.get("system_instruction", "")
            new_config["system_instruction"] = (
                f"{orig}\n\n{context_prompt}" if orig else context_prompt
            )
            return new_config
        elif hasattr(config, "system_instruction"):
            orig = getattr(config, "system_instruction", "")
            new_system = f"{orig}\n\n{context_prompt}" if orig else context_prompt
            if hasattr(config, "model_copy"):
                return config.model_copy(update={"system_instruction": new_system})
            else:
                import copy

                new_config = copy.copy(config)
                setattr(new_config, "system_instruction", new_system)
                return new_config
        return config

    async def _track_background(
        self, call: LLMCall, response: LLMResponse, session_id: str, turn: int
    ) -> None:
        """The background pipeline for extracting, detecting, and resolving beliefs.

        Uses per-session async lock to coordinate writes.
        """
        try:
            self._ensure_initialized()

            # Extract from BOTH user message and assistant response
            last_user_msg = ""
            for m in reversed(call.messages):
                if m.get("role") == "user":
                    last_user_msg = m.get("content", "")
                    break

            new_beliefs = await self.extractor.process_turn(  # type: ignore[union-attr]
                last_user_msg,
                response.text,
                session_id,
                turn,
            )

            self._stats.total_beliefs_extracted += len(new_beliefs)

            if not new_beliefs:
                self._stats.record_success()
                return

            assert self.detector is not None
            contradictions, duplicates = await self.detector.detect_with_deduplication(
                session_id, new_beliefs
            )

            self._stats.total_contradictions_detected += len(contradictions)
            self._stats.total_duplicates_skipped += len(duplicates)

            # Write phase is serialised per session
            async with _get_session_lock(session_id):
                last_processed_turn = self._session_turn_states.get(session_id, -1)
                if turn > last_processed_turn + 1:
                    logger.warning(
                        f"Out-of-order turn completion for session "
                        f"{session_id}: turn {turn} completed before turn "
                        f"{last_processed_turn + 1}."
                    )
                elif turn <= last_processed_turn:
                    logger.warning(
                        f"Retransmitted or replayed turn for session "
                        f"{session_id}: turn {turn} completed after turn "
                        f"{last_processed_turn}. Skipping."
                    )
                    return

                self._session_turn_states[session_id] = turn
                await self.resolver.resolve(session_id, contradictions)

                contradicting_new_beliefs = [c[1] for c in contradictions]
                for b in new_beliefs:
                    if b not in contradicting_new_beliefs and b not in duplicates:
                        await self.store.add_belief(session_id, b)

            self._stats.record_success()

        except Exception as e:
            self._stats.record_error(str(e))
            logger.error(f"Background tracking error: {e}", exc_info=True)

    async def track_async(
        self,
        call_dict: Dict[str, Any],
        response_dict: Dict[str, Any],
        session_id: str,
        turn: int,
    ) -> None:
        call = LLMCall.model_validate(call_dict)
        response = LLMResponse.model_validate(response_dict)
        await self._track_background(call, response, session_id, turn)

    def track_sync(
        self,
        call_dict: Dict[str, Any],
        response_dict: Dict[str, Any],
        session_id: str,
        turn: int,
    ) -> None:
        def _log_task_error(task: asyncio.Task[None]) -> None:
            if task.cancelled():
                return
            exc = task.exception()
            if exc is not None:
                logger.error(
                    f"Background tracking task failed for session {session_id} "
                    f"turn {turn}: {exc}",
                    exc_info=exc,
                )

        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                task = loop.create_task(
                    self.track_async(call_dict, response_dict, session_id, turn)
                )
                task.add_done_callback(_log_task_error)
                return
        except RuntimeError:
            pass
        asyncio.run(self.track_async(call_dict, response_dict, session_id, turn))

    def wrap(
        self,
        func: Optional[Callable[..., Coroutine[Any, Any, Any]]] = None,
        *,
        stream: bool = False,
        auto_inject: bool = True,
    ) -> Any:
        def decorator(
            f: Callable[..., Coroutine[Any, Any, Any]],
        ) -> Callable[..., Coroutine[Any, Any, Any]]:
            @wraps(f)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                session_id = session_context.get()
                # NOTE: Dict operations are atomic under CPython's GIL.
                # For other Python implementations, this would need a lock.
                self._session_turn_counters[session_id] = (
                    self._session_turn_counters.get(session_id, 0) + 1
                )
                current_turn = self._session_turn_counters[session_id]

                if auto_inject:
                    last_user_msg = ""
                    messages_list = None
                    if "messages" in kwargs and isinstance(kwargs["messages"], list):
                        messages_list = kwargs["messages"]
                    elif "contents" in kwargs and isinstance(kwargs["contents"], list):
                        messages_list = kwargs["contents"]
                    else:
                        for arg in args:
                            if isinstance(arg, list):
                                messages_list = arg
                                break
                    if messages_list:
                        for m in reversed(messages_list):
                            if isinstance(m, dict) and m.get("role") == "user":
                                last_user_msg = str(m.get("content", ""))
                                break
                            elif hasattr(m, "role") and m.role == "user":
                                last_user_msg = str(getattr(m, "content", ""))
                                break
                            elif isinstance(m, str):
                                last_user_msg = m
                                break

                    conversation_id = conversation_context.get()
                    context_prompt = await self.get_context_prompt(
                        session_id=session_id,
                        conversation_id=conversation_id,
                        current_user_message=last_user_msg if last_user_msg else None,
                    )

                    if context_prompt:
                        injected = False
                        if self.app_adapter and hasattr(
                            self.app_adapter, "inject_context"
                        ):
                            try:
                                res = self.app_adapter.inject_context(
                                    context_prompt, *args, **kwargs
                                )
                                if isinstance(res, tuple) and len(res) == 2:
                                    args, kwargs = res
                                    injected = True
                            except Exception as e:
                                logger.debug(
                                    f"inject_context failed, using fallback: {e}"
                                )
                        if not injected:
                            args, kwargs = self._generic_inject_context(
                                context_prompt, *args, **kwargs
                            )

                native_response = await f(*args, **kwargs)

                if stream:
                    return AsyncStreamWrapper(
                        native_response,
                        self,
                        args,
                        kwargs,
                        session_id,
                        current_turn,
                    )

                if self._auto_detect_adapter and self.app_adapter is None:
                    self.app_adapter = _detect_adapter(native_response)
                    from beliefstate.resilience import ResilientAdapterWrapper

                    self.internal_adapter = ResilientAdapterWrapper(
                        self.app_adapter, self.config
                    )
                    self._ensure_initialized()

                try:
                    llm_call = self.app_adapter.to_llm_call(*args, **kwargs)
                    llm_response = self.app_adapter.to_llm_response(native_response)
                    self._ensure_initialized()

                    provider_name = self.app_adapter.__class__.__name__
                    if session_id in self._session_providers:
                        if self._session_providers[session_id] != provider_name:
                            logger.warning(
                                f"Mid-session provider change for session "
                                f"{session_id}: switched from "
                                f"{self._session_providers[session_id]} to "
                                f"{provider_name}."
                            )
                    else:
                        self._session_providers[session_id] = provider_name

                    if self.config.enable_background_tasks:
                        self._dispatch(
                            self._track_background(
                                llm_call, llm_response, session_id, current_turn
                            )
                        )
                    else:
                        await self._track_background(
                            llm_call, llm_response, session_id, current_turn
                        )
                except Exception as e:
                    logger.error(f"Tracker wrapper error: {e}", exc_info=True)

                return native_response

            return wrapper

        if func is None:
            return decorator
        return decorator(func)
