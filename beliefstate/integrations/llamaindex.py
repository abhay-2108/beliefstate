import asyncio
from typing import Any, Dict, List, Optional
from beliefstate.tracker import BeliefTracker, session_context
from beliefstate.call import LLMCall, LLMResponse

try:
    from llama_index.core.callbacks import BaseCallbackHandler, CBEventType

    HAS_LLAMAINDEX = True
except ImportError:

    class BaseCallbackHandler:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    CBEventType = Any
    HAS_LLAMAINDEX = False


class LlamaIndexBeliefTrackerCallback(BaseCallbackHandler):
    """
    LlamaIndex callback handler to automatically track beliefs from chat generations.
    Hooks into LlamaIndex's event system to intercept completions and embeddings.
    """

    def __init__(
        self,
        tracker: BeliefTracker,
        event_starts_to_ignore: Optional[List[Any]] = None,
        event_ends_to_ignore: Optional[List[Any]] = None,
    ) -> None:
        self.tracker = tracker
        self.pending_calls: Dict[str, LLMCall] = {}
        super().__init__(
            event_starts_to_ignore=event_starts_to_ignore or [],
            event_ends_to_ignore=event_ends_to_ignore or [],
        )

    def on_event_start(
        self,
        event_type: Any,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> str:
        if not HAS_LLAMAINDEX:
            raise ImportError(
                "llama-index-core is not installed. "
                "Install it via `pip install beliefstate[llamaindex]` to use LlamaIndex callbacks."
            )

        if event_type == CBEventType.LLM and payload:
            messages = []
            if "messages" in payload:
                for m in payload["messages"]:
                    role = getattr(m, "role", "user")
                    content = getattr(m, "content", "")
                    messages.append({"role": str(role), "content": str(content)})
            elif "prompts" in payload:
                for p in payload["prompts"]:
                    messages.append({"role": "user", "content": str(p)})

            self.pending_calls[event_id] = LLMCall(messages=messages, kwargs=kwargs)
        return event_id

    def on_event_end(
        self,
        event_type: Any,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        if not HAS_LLAMAINDEX:
            return

        if event_type == CBEventType.LLM:
            call = self.pending_calls.pop(event_id, None)
            if not call:
                call = LLMCall(messages=[])

            if payload and "response" in payload:
                response = payload["response"]
                text = ""
                if hasattr(response, "message") and hasattr(
                    response.message, "content"
                ):
                    text = response.message.content
                elif hasattr(response, "text"):
                    text = response.text
                else:
                    text = str(response)

                raw = response.dict() if hasattr(response, "dict") else response
                llm_response = LLMResponse(text=text, raw_response=raw)

                session_id = session_context.get()
                self.tracker.turn_counter += 1
                current_turn = self.tracker.turn_counter

                if self.tracker.config.enable_background_tasks:
                    self.tracker.dispatcher.dispatch(
                        self.tracker, call, llm_response, session_id, current_turn
                    )
                else:
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(
                            self.tracker._track_background(
                                call, llm_response, session_id, current_turn
                            )
                        )
                    except RuntimeError:
                        asyncio.run(
                            self.tracker._track_background(
                                call, llm_response, session_id, current_turn
                            )
                        )
