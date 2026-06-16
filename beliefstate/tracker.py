import asyncio
import logging
from typing import Any, Callable, Coroutine, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)
from functools import wraps
from contextvars import ContextVar

from beliefstate.config import TrackerConfig
from beliefstate.call import LLMCall, LLMResponse
from beliefstate.adapters.base import ProviderAdapter
from beliefstate.store.base import Store
from beliefstate.store.sqlite import SQLiteStore
from beliefstate.extractor import BeliefExtractor
from beliefstate.detector import ContradictionDetector
from beliefstate.resolver import BeliefResolver

# Context variable for session management
session_context: ContextVar[str] = ContextVar("session_id", default="default")

T = TypeVar("T")

class BeliefTracker:
    def __init__(
        self, 
        config: TrackerConfig, 
        adapter: ProviderAdapter, 
        store: Optional[Store] = None,
        internal_adapter: Optional[ProviderAdapter] = None,
        dispatcher: Optional[Any] = None,
        judge: Optional[Any] = None
    ):
        self.config = config
        self.app_adapter = adapter
        
        from beliefstate.resilience import ResilientAdapterWrapper
        raw_internal = internal_adapter or adapter
        self.internal_adapter = ResilientAdapterWrapper(raw_internal, self.config)
        
        # Initialize store based on config if not provided
        if store is None:
            self.store = SQLiteStore(db_path=self.config.store_kwargs.get("db_path", "beliefstate.db"))
        else:
            self.store = store
            
        self.extractor = BeliefExtractor(adapter=self.internal_adapter, config=self.config)
        self.detector = ContradictionDetector(
            adapter=self.internal_adapter, 
            store=self.store, 
            config=self.config,
            judge=judge
        )
        self.resolver = BeliefResolver(store=self.store, strategy="overwrite")
        self.turn_counter = 0

        # Resolve dispatcher
        if dispatcher is not None:
            self.dispatcher = dispatcher
        else:
            from beliefstate.dispatcher import AsyncioDispatcher, SyncDispatcher, CeleryDispatcher, RQDispatcher
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

    def set_session(self, session_id: str):
        """Set the session ID for the current execution context."""
        session_context.set(session_id)

    def get_pending_conflicts(self, session_id: Optional[str] = None) -> list[str]:
        """Get any pending belief conflicts for the session."""
        sid = session_id or session_context.get()
        return self.resolver.pop_pending_conflicts(sid)

    async def get_context_prompt(self, session_id: Optional[str] = None, format_template: Optional[str] = None) -> str:
        """
        Retrieve all active, non-contradictory beliefs for the session 
        and format them into a structured text block for the system prompt.
        """
        sid = session_id or session_context.get()
        beliefs = await self.store.get_beliefs(sid)
        if not beliefs:
            return ""
            
        facts_list = [f"- {b.subject} {b.predicate} {b.value}" for b in beliefs]
        facts_str = "\n".join(facts_list)
        
        template = format_template or "Known user facts & preferences:\n{facts}"
        return template.format(facts=facts_str)

    async def inject_context(
        self, 
        messages: list[Dict[str, Any]], 
        session_id: Optional[str] = None, 
        format_template: Optional[str] = None
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
                new_messages[system_idx]["content"] = f"{orig_content}\n\n{context_prompt}"
            else:
                new_messages[system_idx]["content"] = context_prompt
        else:
            new_messages.insert(0, {"role": "system", "content": context_prompt})
            
        return new_messages

    async def _track_background(self, call: LLMCall, response: LLMResponse, session_id: str, turn: int):
        """The background pipeline for extracting, detecting, and resolving beliefs."""
        try:
            new_beliefs = []
            
            # 1a. Extract from the user's latest message
            last_user_msg = ""
            for m in reversed(call.messages):
                if m.get("role") == "user":
                    last_user_msg = m.get("content", "")
                    break
                    
            if last_user_msg:
                user_beliefs = await self.extractor.extract(last_user_msg, turn=turn, source="user")
                new_beliefs.extend(user_beliefs)
                
            # 1b. Extract from the assistant's response
            assistant_beliefs = await self.extractor.extract(response.text, turn=turn, source="assistant")
            new_beliefs.extend(assistant_beliefs)
            
            if not new_beliefs:
                return
                
            # 2. Detect
            contradictions = await self.detector.detect(session_id, new_beliefs)
            
            # 3. Resolve (updates the store for contradictory beliefs based on strategy)
            await self.resolver.resolve(session_id, contradictions)
            
            # 4. Save non-contradictory beliefs
            contradicting_new_beliefs = [c[1] for c in contradictions]
            
            for b in new_beliefs:
                if b not in contradicting_new_beliefs:
                    await self.store.add_belief(session_id, b)
                    
        except Exception as e:
            # Prevent tracker exceptions from bubbling up in background tasks
            logger.error(f"Background tracking error: {e}", exc_info=True)

    async def track_async(self, call_dict: Dict[str, Any], response_dict: Dict[str, Any], session_id: str, turn: int) -> None:
        """Asynchronously process a tracking payload (useful for background workers)."""
        call = LLMCall.model_validate(call_dict)
        response = LLMResponse.model_validate(response_dict)
        await self._track_background(call, response, session_id, turn)

    def track_sync(self, call_dict: Dict[str, Any], response_dict: Dict[str, Any], session_id: str, turn: int) -> None:
        """Synchronously process a tracking payload (blocking)."""
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.create_task(self.track_async(call_dict, response_dict, session_id, turn))
                return
        except RuntimeError:
            pass
        asyncio.run(self.track_async(call_dict, response_dict, session_id, turn))

    def wrap(self, func: Callable[..., Coroutine[Any, Any, Any]]):
        """Decorator to wrap an async LLM function and track beliefs."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            session_id = session_context.get()
            self.turn_counter += 1
            current_turn = self.turn_counter
            
            # 1. Execute the user's actual LLM call (blocks until finished)
            native_response = await func(*args, **kwargs)
            
            # 2. Normalize inputs and outputs for the background tracker
            try:
                llm_call = self.app_adapter.to_llm_call(*args, **kwargs)
                llm_response = self.app_adapter.to_llm_response(native_response)
                
                # 3. Dispatch background tracking
                if self.config.enable_background_tasks:
                    self.dispatcher.dispatch(self, llm_call, llm_response, session_id, current_turn)
                else:
                    # Run synchronously (useful for testing)
                    await self._track_background(llm_call, llm_response, session_id, current_turn)
            except Exception as e:
                # Never fail the user's main application due to tracker parsing errors
                logger.error(f"Tracker wrapper error: {e}", exc_info=True)
                
            return native_response
            
        return wrapper
