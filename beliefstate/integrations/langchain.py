import asyncio
from typing import Any, Dict, List, Optional
from uuid import UUID
from beliefstate.tracker import BeliefTracker, session_context
from beliefstate.call import LLMCall, LLMResponse

try:
    from langchain_core.callbacks import AsyncCallbackHandler
    from langchain_core.outputs import LLMResult
    from langchain_core.messages import BaseMessage
except ImportError:
    AsyncCallbackHandler = object  # type: ignore[misc, assignment]
    LLMResult = Any  # type: ignore[misc, assignment]
    BaseMessage = Any  # type: ignore[misc, assignment]


class BeliefTrackerLangchainCallback(AsyncCallbackHandler):
    """
    LangChain callback handler to automatically track beliefs from chat generation.
    It hooks into LangChain's event system so you don't need to manually wrap functions.
    """

    def __init__(self, tracker: BeliefTracker):
        self.tracker = tracker
        self.pending_calls: Dict[str, LLMCall] = {}

    async def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        universal_msgs = []
        if prompts and len(prompts) > 0:
            for p in prompts:
                universal_msgs.append({"role": "user", "content": p})
        self.pending_calls[str(run_id)] = LLMCall(
            messages=universal_msgs, kwargs=kwargs
        )

    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        # Translate LangChain messages into our universal format
        universal_msgs = []
        if messages and len(messages) > 0:
            for m in messages[0]:
                role = "user"
                m_type = getattr(m, "type", "")
                if m_type == "system":
                    role = "system"
                elif m_type == "ai":
                    role = "assistant"

                universal_msgs.append(
                    {"role": role, "content": getattr(m, "content", "")}
                )

        self.pending_calls[str(run_id)] = LLMCall(
            messages=universal_msgs, kwargs=kwargs
        )

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        call = self.pending_calls.pop(str(run_id), None)
        if not call:
            return

        # Parse LangChain generation output
        if not response.generations or not response.generations[0]:
            return

        text = response.generations[0][0].text
        # Safely dump response to dict if it's a pydantic model (supporting both v1 and v2)
        raw: Any
        if hasattr(response, "model_dump"):
            raw = response.model_dump()
        elif hasattr(response, "dict"):
            raw = response.dict()
        else:
            raw = response

        llm_response = LLMResponse(text=text, raw_response=raw)

        session_id = session_context.get()
        self.tracker.turn_counter += 1
        current_turn = self.tracker.turn_counter

        if self.tracker.config.enable_background_tasks:
            asyncio.create_task(
                self.tracker._track_background(
                    call, llm_response, session_id, current_turn
                )
            )
        else:
            await self.tracker._track_background(
                call, llm_response, session_id, current_turn
            )

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        # Prevent memory leaks by cleaning up pending calls on error
        self.pending_calls.pop(str(run_id), None)
