import asyncio
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional, TypeVar
from datetime import datetime, timezone
from functools import wraps
from contextvars import ContextVar

from beliefstate.config import TrackerConfig
from beliefstate.call import LLMCall, LLMResponse
from beliefstate.adapters.base import ProviderAdapter
from beliefstate.models import DeletionReceipt
from beliefstate.store.base import Store
from beliefstate.store.sqlite import SQLiteStore
from beliefstate.extractor import BeliefExtractor
from beliefstate.detector import ContradictionDetector
from beliefstate.resolver import BeliefResolver

logger = logging.getLogger(__name__)

# Context variable for session management
session_context: ContextVar[str] = ContextVar("session_id", default="default")
conversation_context: ContextVar[Optional[str]] = ContextVar(
    "conversation_id", default=None
)

T = TypeVar("T")

# Per-session async locks for coordinating concurrent writes
_session_locks: Dict[str, asyncio.Lock] = {}


def _ensure_aware(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware (UTC).
    
    Existing beliefs stored before the utcnow→now(utc) migration may have
    naive timestamps.  This helper normalises them so arithmetic works.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def calculate_staleness_score(belief: Any) -> float:
    """Calculate staleness score for a belief.
    
    Score = confidence / (days_since_referenced + 1)
    
    This ensures:
    - Recent beliefs (referenced today) have max score
    - Older beliefs have linearly decreasing scores
    - Score is capped by confidence value
    """
    if not hasattr(belief, 'last_referenced_at') or (
        belief.last_referenced_at is None
    ):
        # Fallback to created_at if last_referenced_at not set
        ref_time = (
            belief.created_at
            if hasattr(belief, 'created_at')
            else datetime.now(timezone.utc)
        )
    else:
        ref_time = belief.last_referenced_at
    
    ref_time = _ensure_aware(ref_time)
    days_since_referenced = (datetime.now(timezone.utc) - ref_time).days
    confidence = getattr(belief, 'confidence', 1.0)
    
    return confidence / (days_since_referenced + 1)


def estimate_tokens(text: str) -> int:
    """Estimate token count for text.
    
    Uses simple heuristic: approximately 1 token per 4 characters.
    This is approximate but works well for quick estimates.
    """
    if not text:
        return 0
    return len(text) // 4


def _detect_adapter(result: Any) -> ProviderAdapter:
    """Auto-detect the appropriate adapter from an LLM response object.
    
    Checks the type of the response object to determine which provider was used.
    Falls back to generic adapter if type cannot be determined.
    
    Args:
        result: The native SDK response object
    
    Returns:
        Appropriate ProviderAdapter instance
    
    Raises:
        RuntimeError: If adapter cannot be determined and SDK is not installed
    """
    type_name = type(result).__module__ + "." + type(result).__name__
    
    # Check for OpenAI
    if "openai" in type_name.lower():
        try:
            from beliefstate.adapters.openai import OpenAIAdapter
            logger.debug("Detected OpenAI response type")
            return OpenAIAdapter()
        except ImportError:
            raise RuntimeError(
                "OpenAI adapter selected but openai SDK not installed. "
                "Install with: pip install openai"
            )
    
    # Check for Anthropic
    if "anthropic" in type_name.lower():
        try:
            from beliefstate.adapters.anthropic import AnthropicAdapter
            logger.debug("Detected Anthropic response type")
            return AnthropicAdapter()
        except ImportError:
            raise RuntimeError(
                "Anthropic adapter selected but anthropic SDK not installed. "
                "Install with: pip install anthropic"
            )
    
    # Check for Google Gemini
    if "google" in type_name.lower() or "genai" in type_name.lower():
        try:
            from beliefstate.adapters.gemini import GeminiAdapter
            logger.debug("Detected Google Gemini response type")
            return GeminiAdapter()
        except ImportError:
            raise RuntimeError(
                "Gemini adapter selected but google-generativeai SDK not installed. "
                "Install with: pip install google-generativeai"
            )
    
    # Check for Ollama
    if "ollama" in type_name.lower():
        try:
            from beliefstate.adapters.ollama import OllamaAdapter
            logger.debug("Detected Ollama response type")
            return OllamaAdapter()
        except ImportError:
            raise RuntimeError(
                "Ollama adapter selected but ollama SDK not installed. "
                "Install with: pip install ollama"
            )
    
    # Fallback: try to extract text generically
    logger.warning(
        f"Could not auto-detect adapter from type: {type_name}. "
        "Using generic adapter. Consider specifying adapter explicitly."
    )
    
    # Create a generic adapter that tries common patterns
    class GenericAdapter(ProviderAdapter):
        """Fallback adapter that tries common response patterns."""
        
        def to_llm_call(self, *args: Any, **kwargs: Any) -> LLMCall:
            messages = kwargs.get("messages", [])
            if not messages and len(args) > 0 and isinstance(args[0], list):
                messages = args[0]
            return LLMCall(messages=messages, kwargs=kwargs)
        
        def to_llm_response(self, response: Any) -> LLMResponse:
            # Try common text extraction patterns
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
            raise NotImplementedError(
                "Generic adapter cannot generate. Please specify adapter explicitly."
            )
        
        async def get_embedding(self, text: str) -> list[float]:
            raise NotImplementedError(
                "Generic adapter cannot generate embeddings. "
                "Please specify adapter explicitly."
            )
        
        async def get_embeddings(
            self, texts: list[str]
        ) -> list[list[float]]:
            raise NotImplementedError(
                "Generic adapter cannot generate embeddings. "
                "Please specify adapter explicitly."
            )
    
    return GenericAdapter()


def _get_session_lock(session_id: str) -> asyncio.Lock:
    """Get or create an async lock for a session.
    
    Used to coordinate concurrent writes to the belief store from multiple
    LLM calls happening in parallel for the same session.
    
    Args:
        session_id: The session ID
    
    Returns:
        An asyncio.Lock instance unique to this session
    """
    if session_id not in _session_locks:
        _session_locks[session_id] = asyncio.Lock()
    return _session_locks[session_id]


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
        self.config = config
        
        # Use provided adapter or will be set during first wrap call
        self._auto_detect_adapter = adapter is None
        if adapter is not None:
            self.app_adapter = adapter
        else:
            self.app_adapter = None  # type: ignore[assignment]

        from beliefstate.resilience import ResilientAdapterWrapper

        # Internal adapter defaults to app_adapter if not provided
        if internal_adapter is not None:
            raw_internal = internal_adapter
        elif adapter is not None:
            raw_internal = adapter
        else:
            raw_internal = None  # type: ignore[assignment]
        
        self.internal_adapter = (
            ResilientAdapterWrapper(raw_internal, self.config)
            if raw_internal is not None
            else None
        )

        # Initialize store based on config if not provided
        self.store: Store
        if store is None:
            self.store = SQLiteStore(
                db_path=self.config.store_kwargs.get("db_path", "beliefstate.db")
            )
        else:
            self.store = store

        self.extractor: Optional[BeliefExtractor] = None
        self.detector: Optional[ContradictionDetector] = None
        self.resolver = BeliefResolver(store=self.store, strategy="overwrite")
        self.turn_counter = 0
        
        # Track latest processed turn per session for optimistic concurrency
        # control. This helps detect out-of-order completions
        # (turn N+1 finishing before turn N)
        self._session_turn_states: Dict[str, int] = {}
        
        # Track provider history for each session to detect mid-session
        # provider changes
        self._session_providers: Dict[str, str] = {}

        # Resolve dispatcher
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

    def _ensure_initialized(self) -> None:
        """Ensure extractor and detector are initialized (lazy init for auto-detect)."""
        if self.extractor is None:
            if self.app_adapter is None:
                raise RuntimeError(
                    "Adapter not initialized. Wrap a decorated function first "
                    "or pass adapter explicitly."
                )
            self.extractor = BeliefExtractor(
                adapter=self.internal_adapter, config=self.config
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
        """Set the session ID for the current execution context."""
        session_context.set(session_id)

    def get_pending_conflicts(self, session_id: Optional[str] = None) -> list[str]:
        """Get any pending belief conflicts for the session."""
        sid = session_id or session_context.get()
        return self.resolver.pop_pending_conflicts(sid)

    async def clear_session(
        self, session_id: Optional[str] = None
    ) -> "DeletionReceipt":
        """Clear all tracking data for a session (beliefs, conflicts, history).
        
        Implements GDPR-compliant data deletion with in-flight task draining:
        1. Drains any in-flight belief extraction tasks (AsyncioDispatcher only)
        2. Deletes all beliefs from the store
        3. Clears resolver conflict tracking
        4. Returns auditable DeletionReceipt with timestamp and counts
        
        Args:
            session_id: Session ID (defaults to current context)
        
        Returns:
            DeletionReceipt with session_id, beliefs_deleted, deleted_at,
            in_flight_tasks_drained
        
        Example:
            receipt = await tracker.clear_session("user-123")
            print(f"Deleted {receipt.beliefs_deleted} beliefs")
            print(f"Drained {receipt.in_flight_tasks_drained} in-flight tasks")
        """
        from beliefstate.models import DeletionReceipt
        
        sid = session_id or session_context.get()
        
        # 1. Drain in-flight tasks if using AsyncioDispatcher
        drained_count = 0
        if hasattr(self.dispatcher, 'drain_session'):
            try:
                drained_count = await self.dispatcher.drain_session(sid)
                logger.info(
                    f"Drained {drained_count} in-flight tasks for "
                    f"session {sid}"
                )
            except Exception as e:
                logger.warning(
                    f"Error draining tasks for session {sid}: {e}"
                )
        
        # 2. Delete all beliefs from the store
        beliefs = await self.store.get_beliefs(sid)
        beliefs_deleted = len(beliefs)
        await self.store.clear(sid)
        logger.info(f"Deleted {beliefs_deleted} beliefs for session {sid}")
        
        # 3. Clear resolver conflict tracking
        self.resolver.clear_session(sid)
        logger.info(f"Cleared conflict history for session {sid}")
        
        # 4. Return auditable receipt
        receipt = DeletionReceipt(
            session_id=sid,
            beliefs_deleted=beliefs_deleted,
            deleted_at=datetime.now(timezone.utc),
            in_flight_tasks_drained=drained_count,
        )
        logger.info(f"GDPR deletion receipt: {receipt}")
        return receipt

    async def get_beliefs(self, session_id: Optional[str] = None) -> List[Any]:
        """Retrieve all beliefs for a session.
        
        Args:
            session_id: Session ID (defaults to current context)
        
        Returns:
            List of Belief objects for the session
        
        Example:
            beliefs = await tracker.get_beliefs("user-123")
            for b in beliefs:
                confidence = b.confidence
                print(f"{b.subject} {b.predicate} {b.value} "
                      f"(confidence: {confidence})")
        """
        sid = session_id or session_context.get()
        return await self.store.get_beliefs(sid)

    async def get_stats(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about beliefs for a session.
        
        Args:
            session_id: Session ID (defaults to current context)
        
        Returns:
            Dictionary with stats:
            - total_beliefs: Total number of beliefs
            - by_subject: Count of beliefs per subject
            - by_source: Count of beliefs from "user" vs "assistant"
            - avg_confidence: Average confidence score
            - contradictions_detected: Number of contradictions found
        
        Example:
            stats = await tracker.get_stats("user-123")
            print(f"Total beliefs: {stats['total_beliefs']}")
        """
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

        # Count by subject
        by_subject: Dict[str, int] = {}
        for b in beliefs:
            by_subject[b.subject] = by_subject.get(b.subject, 0) + 1

        # Count by source
        by_source: Dict[str, int] = {}
        for b in beliefs:
            by_source[b.source] = by_source.get(b.source, 0) + 1

        # Average confidence
        avg_confidence = sum(b.confidence for b in beliefs) / len(beliefs)

        # Contradiction count
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
        """Get a formatted summary of beliefs for display or prompt injection.
        
        Respects max_beliefs config (default 50) but can be overridden per call.
        Sorts by confidence + recency strategy.
        
        Args:
            session_id: Session ID (defaults to current context)
            max_beliefs: Override max_beliefs config for this call
            format_template: Custom template (default: "Known user facts &
                preferences:\\n{facts}")
        
        Returns:
            Formatted string ready for display or prompt injection
        
        Example:
            summary = await tracker.get_summary("user-123", max_beliefs=10)
            print(summary)
            # Output:
            # Known user facts & preferences:
            # - USER likes Python
            # - USER works in Tokyo
            # - USER prefers dark mode
        """
        # Use provided max_beliefs or fall back to config
        max_b = max_beliefs or self.config.max_beliefs

        sid = session_id or session_context.get()
        beliefs = await self.store.get_beliefs(sid)

        if not beliefs:
            return ""

        # Sort beliefs by strategy (same as get_context_prompt)
        strategy = self.config.belief_sort_strategy
        if strategy == "confidence_recency":
            sorted_beliefs = sorted(
                beliefs, key=lambda b: (b.confidence, b.turn), reverse=True
            )
        elif strategy == "recency":
            sorted_beliefs = sorted(beliefs, key=lambda b: b.turn, reverse=True)
        elif strategy == "confidence":
            sorted_beliefs = sorted(beliefs, key=lambda b: b.confidence, reverse=True)
        else:
            sorted_beliefs = sorted(
                beliefs, key=lambda b: (b.confidence, b.turn), reverse=True
            )

        # Cap at max_beliefs
        capped_beliefs = sorted_beliefs[:max_b]

        facts_list = [f"- {b.subject} {b.predicate} {b.value}" for b in capped_beliefs]
        facts_str = "\n".join(facts_list)

        template = format_template or "Known user facts & preferences:\n{facts}"
        return template.format(facts=facts_str)

    async def clear_beliefs(self, session_id: Optional[str] = None) -> None:
        """Clear all beliefs for a session (but keep conflict history).
        
        Args:
            session_id: Session ID (defaults to current context)
        
        Example:
            await tracker.clear_beliefs("user-123")
        """
        sid = session_id or session_context.get()
        await self.store.clear(sid)

    async def remove_belief(
        self, session_id: Optional[str], subject: str, predicate: str
    ) -> None:
        """Remove a specific belief from the store.
        
        Args:
            session_id: Session ID (defaults to current context)
            subject: Subject of the belief (e.g., "USER", "ASSISTANT")
            predicate: Predicate of the belief (e.g., "likes", "works in")
        
        Example:
            await tracker.remove_belief("user-123", "USER", "likes")
            # Removes all beliefs where subject=USER and predicate=likes
        """
        sid = session_id or session_context.get()
        await self.store.remove_belief(sid, subject, predicate)

    async def set_session_ttl(
        self, session_id: Optional[str], ttl_seconds: int
    ) -> None:
        """Set time-to-live (expiration) for all beliefs in a session.
        
        For Redis: Uses native Redis EXPIRE. For SQLite: Not supported
        (use prune_expired_beliefs instead).
        
        Args:
            session_id: Session ID (defaults to current context)
            ttl_seconds: Time in seconds before beliefs expire
        
        Example:
            await tracker.set_session_ttl("user-123", 86400)  # 24 hours
        """
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
        """Prune beliefs older than specified age (SQLite only).
        
        For Redis: Use set_session_ttl() for native expiration.
        For SQLite: Manually prune old beliefs.
        
        Args:
            session_id: Session ID (defaults to all sessions if None)
            max_age_seconds: Age threshold (defaults to config.belief_max_age_seconds)
        
        Returns:
            Number of beliefs deleted
        
        Example:
            deleted = await tracker.prune_expired_beliefs("user-123", 86400)
        """
        if not hasattr(self.store, "prune_expired_beliefs"):
            logger.warning("Store does not support belief pruning")
            return 0
        
        max_age = max_age_seconds or self.config.belief_max_age_seconds
        sid = session_id or session_context.get()
        
        # If session_id provided, prune only that session; else prune all
        if session_id:
            return await self.store.prune_expired_beliefs(max_age, sid)
        else:
            return await self.store.prune_expired_beliefs(max_age, None)

    async def update_belief_reference(
        self, session_id: Optional[str], subject: str, predicate: str
    ) -> None:
        """Update last_referenced_at timestamp for a belief.
        
        Should be called when a belief is actively used/injected into a prompt.
        Helps with staleness scoring for session resumption.
        
        Args:
            session_id: Session ID (defaults to current context)
            subject: Belief subject
            predicate: Belief predicate
        """
        sid = session_id or session_context.get()
        beliefs = await self.store.get_beliefs(sid)
        for b in beliefs:
            if b.subject == subject and b.predicate == predicate:
                b.last_referenced_at = datetime.now(timezone.utc)
                await self.store.update_belief(sid, b)
                logger.debug(
                    f"Updated reference time for belief: {subject} "
                    f"{predicate}"
                )
                break

    async def get_context_prompt(
        self,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        format_template: Optional[str] = None,
        current_user_message: Optional[str] = None,
    ) -> str:
        """
        Retrieve active, non-contradictory beliefs for the session and
        format them into a structured text block for the system prompt.
        
        If conversation_id is provided, returns beliefs only from that
        conversation thread. Otherwise returns beliefs from all conversations
        in the session.
        
        If token-aware injection enabled and belief summary exceeds budget,
        uses cosine similarity to select only the most relevant beliefs based
        on the current user message.
        
        Enforces max_beliefs limit and sorts by confidence + recency.
        Applies staleness scoring if enabled: only includes beliefs with
        staleness_score >= staleness_threshold.
        """
        sid = session_id or session_context.get()
        cid = conversation_id or conversation_context.get()
        beliefs = await self.store.get_beliefs(sid, cid)
        if not beliefs:
            return ""

        # Apply staleness filter if enabled
        if self.config.enable_staleness_scoring:
            filtered_beliefs = [
                b for b in beliefs 
                if calculate_staleness_score(b) >= 
                self.config.staleness_threshold
            ]
            if not filtered_beliefs:
                # If all beliefs are too stale, include at least the most
                # confident recent ones
                filtered_beliefs = sorted(
                    beliefs,
                    key=lambda b: (b.confidence, b.turn),
                    reverse=True
                )[:5]
        else:
            filtered_beliefs = beliefs
        
        # Filter out hypothetical beliefs (not suitable for injection)
        filtered_beliefs = [
            b for b in filtered_beliefs
            if not getattr(b, 'is_hypothetical', False)
        ]
        if not filtered_beliefs and beliefs:
            # If all beliefs are hypothetical, don't filter
            # (use at least something)
            filtered_beliefs = beliefs

        # Sort beliefs by strategy
        strategy = self.config.belief_sort_strategy
        if strategy == "confidence_recency":
            # Primary: confidence (descending), Secondary: turn (descending)
            sorted_beliefs = sorted(
                filtered_beliefs,
                key=lambda b: (b.confidence, b.turn),
                reverse=True
            )
        elif strategy == "recency":
            # Sort by turn only (most recent first)
            sorted_beliefs = sorted(
                filtered_beliefs, key=lambda b: b.turn, reverse=True
            )
        elif strategy == "confidence":
            # Sort by confidence only (highest first)
            sorted_beliefs = sorted(
                filtered_beliefs, key=lambda b: b.confidence, reverse=True
            )
        else:
            # Default to confidence_recency
            sorted_beliefs = sorted(
                filtered_beliefs,
                key=lambda b: (b.confidence, b.turn),
                reverse=True
            )

        # Token-aware filtering: if belief summary would be too large,
        # use relevance-based selection
        if self.config.enable_token_aware_injection and current_user_message:
            # Generate a sample summary and check token count
            sample_facts = [
                f"- {b.subject} {b.predicate} {b.value}"
                for b in sorted_beliefs[: self.config.max_beliefs]
            ]
            sample_summary = "\n".join(sample_facts)
            estimated_tokens = estimate_tokens(sample_summary)
            
            if estimated_tokens > self.config.belief_budget_tokens:
                logger.debug(
                    f"Belief summary ({estimated_tokens} tokens) exceeds "
                    f"budget ({self.config.belief_budget_tokens}). "
                    f"Using relevance-based filtering."
                )
                # Embed current user message and rank beliefs by relevance
                try:
                    if (
                        self.extractor
                        and hasattr(self.extractor, 'adapter')
                    ):
                        user_msg_embedding = (
                            await self.extractor.adapter.get_embedding(
                                current_user_message
                            )
                        )
                        
                        # Score beliefs by relevance
                        # (cosine similarity with user message)
                        from beliefstate.detector import cosine_similarity
                        scored_beliefs = []
                        for b in sorted_beliefs:
                            if b.embedding:
                                similarity = cosine_similarity(
                                    user_msg_embedding, b.embedding
                                )
                                scored_beliefs.append((b, similarity))
                        
                        # Sort by relevance and select top-K to fit budget
                        scored_beliefs.sort(key=lambda x: x[1], reverse=True)
                        
                        # Estimate how many beliefs fit in the budget
                        beliefs_per_token = len(sample_facts) / max(
                            estimated_tokens, 1
                        )
                        max_beliefs_in_budget = max(
                            1,
                            int(
                                beliefs_per_token
                                * self.config.belief_budget_tokens
                            )
                        )
                        
                        sorted_beliefs = [
                            b for b, _
                            in scored_beliefs[:max_beliefs_in_budget]
                        ]
                        logger.debug(
                            f"Selected {len(sorted_beliefs)} most relevant "
                            f"beliefs to fit token budget"
                        )
                except Exception as e:
                    logger.warning(
                        f"Token-aware injection fallback: {e}. "
                        f"Using default selection."
                    )

        # Enforce max_beliefs limit
        capped_beliefs = sorted_beliefs[: self.config.max_beliefs]

        facts_list = [f"- {b.subject} {b.predicate} {b.value}" for b in capped_beliefs]
        facts_str = "\n".join(facts_list)

        template = format_template or "Known user facts & preferences:\n{facts}"
        return template.format(facts=facts_str)

    async def inject_context(
        self,
        messages: list[Dict[str, Any]],
        session_id: Optional[str] = None,
        format_template: Optional[str] = None,
    ) -> list[Dict[str, Any]]:
        """
        Inject the current session belief state context into a list of messages.
        If a system message is already present, appends the belief context to it.
        Otherwise, prepends a new system message containing the belief context.
        """
        context_prompt = await self.get_context_prompt(session_id, format_template)
        if not context_prompt:
            return messages

        new_messages = [m.copy() for m in messages]

        # Look for existing system message
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

    async def _track_background(
        self, call: LLMCall, response: LLMResponse, session_id: str, turn: int
    ) -> None:
        """The background pipeline for extracting, detecting, and resolving beliefs.
        
        Uses per-session async lock to coordinate writes when multiple LLM calls happen
        in parallel for the same session. This prevents race conditions in:
        - Contradiction resolution (updating/removing beliefs)
        - Adding new beliefs to the store
        
        Implements optimistic concurrency control for out-of-order completions:
        - Tracks the latest processed turn per session
        - Warns if turns complete out of order (turn N+1 before turn N)
        - Proceeds anyway (beliefs are still added, but order may be jumbled)
        """
        try:
            # Ensure components are initialized
            self._ensure_initialized()
            
            new_beliefs = []

            # 1a. Extract from the user's latest message
            last_user_msg = ""
            for m in reversed(call.messages):
                if m.get("role") == "user":
                    last_user_msg = m.get("content", "")
                    break

            if last_user_msg:
                user_beliefs = await self.extractor.extract(
                    last_user_msg, turn=turn, source="user"
                )
                new_beliefs.extend(user_beliefs)

            # 1b. Extract from the assistant's response
            assistant_beliefs = await self.extractor.extract(
                response.text, turn=turn, source="assistant"
            )
            new_beliefs.extend(assistant_beliefs)

            if not new_beliefs:
                return

            # 2. Detect contradictions and deduplicate entailed beliefs
            contradictions, duplicates = await self.detector.detect_with_deduplication(
                session_id, new_beliefs
            )

            # 3-4. Write phase: use per-session lock to coordinate concurrent writes
            session_lock = _get_session_lock(session_id)
            async with session_lock:
                # Check for out-of-order turn completion
                # (optimistic concurrency control)
                last_processed_turn = (
                    self._session_turn_states.get(session_id, -1)
                )
                if turn > last_processed_turn + 1:
                    logger.warning(
                        f"Out-of-order turn completion for session "
                        f"{session_id}: turn {turn} completed before turn "
                        f"{last_processed_turn + 1}. This can happen with "
                        f"parallel requests. Belief order may be jumbled."
                    )
                elif turn <= last_processed_turn:
                    logger.warning(
                        f"Retransmitted or replayed turn for session "
                        f"{session_id}: turn {turn} completed after turn "
                        f"{last_processed_turn}. Skipping to avoid "
                        f"duplicates."
                    )
                    return
                
                # Update turn state
                self._session_turn_states[session_id] = turn
                
                # 3. Resolve (updates the store for contradictory beliefs
                # based on strategy)
                await self.resolver.resolve(session_id, contradictions)

                # 4. Save non-contradictory, non-duplicate beliefs
                contradicting_new_beliefs = [c[1] for c in contradictions]

                for b in new_beliefs:
                    if b not in contradicting_new_beliefs and b not in duplicates:
                        await self.store.add_belief(session_id, b)

        except Exception as e:
            # Prevent tracker exceptions from bubbling up in background tasks
            logger.error(f"Background tracking error: {e}", exc_info=True)

    async def track_async(
        self,
        call_dict: Dict[str, Any],
        response_dict: Dict[str, Any],
        session_id: str,
        turn: int,
    ) -> None:
        """Asynchronously process a tracking payload (useful for background workers)."""
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
        """Synchronously process a tracking payload (blocking)."""
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.create_task(
                    self.track_async(call_dict, response_dict, session_id, turn)
                )
                return
        except RuntimeError:
            pass
        asyncio.run(self.track_async(call_dict, response_dict, session_id, turn))

    def wrap(
        self, func: Callable[..., Coroutine[Any, Any, Any]], stream: bool = False
    ) -> Callable[..., Coroutine[Any, Any, Any]]:
        """Decorator to wrap an async LLM function and track beliefs.
        
        If adapter was not provided during initialization, auto-detects from first call.
        
        Args:
            func: Async function that calls an LLM
            stream: If True, expects func to return an async generator
                   (streaming response). Accumulates chunks and runs
                   extraction after stream is exhausted.
        
        Example:
            # Non-streaming (default)
            @tracker.wrap
            async def call_llm():
                return await client.chat.completions.create(...)
            
            # Streaming
            @tracker.wrap(stream=True)
            async def call_llm_streaming():
                return client.chat.completions.create(stream=True)  # yields chunks
            
            # Usage - same API
            response = await call_llm()
            response = await call_llm_streaming()  # Automatically accumulates stream
        """

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            session_id = session_context.get()
            self.turn_counter += 1
            current_turn = self.turn_counter

            # 1. Execute the user's actual LLM call (blocks until finished)
            native_response = await func(*args, **kwargs)

            # Handle streaming: accumulate chunks into full response
            if stream:
                native_response = await self._accumulate_stream(native_response)

            # 2. Auto-detect adapter on first call if needed
            if self._auto_detect_adapter and self.app_adapter is None:
                logger.info("Auto-detecting adapter from response type...")
                self.app_adapter = _detect_adapter(native_response)
                
                # Initialize internal adapter and wrapped components
                from beliefstate.resilience import ResilientAdapterWrapper
                self.internal_adapter = ResilientAdapterWrapper(
                    self.app_adapter, self.config
                )
                self._ensure_initialized()

            # 3. Normalize inputs and outputs for the background tracker
            try:
                llm_call = self.app_adapter.to_llm_call(*args, **kwargs)
                llm_response = self.app_adapter.to_llm_response(native_response)

                # Ensure extractor and detector are initialized
                self._ensure_initialized()
                
                # Track provider info for this session
                provider_name = self.app_adapter.__class__.__name__
                if session_id in self._session_providers:
                    # Check for mid-session provider change
                    if self._session_providers[session_id] != provider_name:
                        logger.warning(
                            f"Mid-session provider change for session "
                            f"{session_id}: switched from "
                            f"{self._session_providers[session_id]} to "
                            f"{provider_name}. This may cause "
                            f"embedding/extraction inconsistencies."
                        )
                else:
                    # First call in this session
                    self._session_providers[session_id] = provider_name
                
                # Warn if internal_adapter not set with premium models
                premium_models = [
                    "gpt-4", "gpt-4-turbo", "claude-3", "gemini-pro"
                ]
                if (
                    not self.internal_adapter
                    and any(
                        model_hint in provider_name.lower()
                        for model_hint in premium_models
                    )
                ):
                    logger.warning(
                        f"No internal_adapter configured for premium "
                        f"provider {provider_name}. Belief extraction will "
                        f"use the same premium provider, increasing costs. "
                        f"Consider setting internal_adapter to a cheaper "
                        f"model (e.g., GPT-3.5, Claude Instant)."
                    )

                # 4. Dispatch background tracking
                if self.config.enable_background_tasks:
                    self.dispatcher.dispatch(
                        self, llm_call, llm_response, session_id, current_turn
                    )
                else:
                    # Run synchronously (useful for testing)
                    await self._track_background(
                        llm_call, llm_response, session_id, current_turn
                    )
            except Exception as e:
                # Never fail the user's main application due to tracker parsing errors
                logger.error(f"Tracker wrapper error: {e}", exc_info=True)

            return native_response

        return wrapper

    async def _accumulate_stream(self, stream_generator: Any) -> Any:
        """Accumulate chunks from a streaming response into a complete response object.
        
        Handles different streaming formats:
        - OpenAI streaming (yields ChatCompletionChunk objects)
        - Generic async generator (yields dicts or objects with text/content)
        
        Args:
            stream_generator: Async generator yielding response chunks
        
        Returns:
            Accumulated response object (dict or native SDK response)
        """
        accumulated_text = ""
        first_chunk = None
        
        # Iterate through stream chunks
        async for chunk in stream_generator:
            if first_chunk is None:
                first_chunk = chunk
            
            # Extract text from chunk
            chunk_text = ""
            
            # OpenAI format: chunk.choices[0].delta.content
            if hasattr(chunk, "choices") and chunk.choices:
                if (
                    hasattr(chunk.choices[0], "delta")
                    and hasattr(chunk.choices[0].delta, "content")
                ):
                    chunk_text = chunk.choices[0].delta.content or ""
            
            # Dict format
            elif isinstance(chunk, dict):
                if "choices" in chunk and chunk["choices"]:
                    choice = chunk["choices"][0]
                    if "delta" in choice and "content" in choice["delta"]:
                        chunk_text = choice["delta"]["content"] or ""
            
            # Generic object with content/text
            elif hasattr(chunk, "content"):
                chunk_text = chunk.content or ""
            elif hasattr(chunk, "text"):
                chunk_text = chunk.text or ""
            
            accumulated_text += chunk_text
            logger.debug(
                f"Accumulated chunk: {len(chunk_text)} chars "
                f"(total: {len(accumulated_text)})"
            )
        
        # Construct accumulated response
        if first_chunk is None:
            # Empty stream - return empty response
            return {"choices": [{"message": {"content": "", "role": "assistant"}}]}
        
        # OpenAI format - reconstruct as complete response
        if hasattr(first_chunk, "choices"):
            # Create a dict-like response object
            return {
                "id": getattr(first_chunk, "id", "stream-accumulated"),
                "object": "chat.completion",
                "created": getattr(first_chunk, "created", 0),
                "model": getattr(first_chunk, "model", "unknown"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": accumulated_text,
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
        
        # Generic fallback
        return {
            "content": accumulated_text,
            "text": accumulated_text,
        }
