"""Shared utilities for the beliefstate test app."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

# ── Async helper ───────────────────────────────────────────────────────────────

_bg_loop: asyncio.AbstractEventLoop | None = None
_bg_thread: threading.Thread | None = None


def get_bg_loop() -> asyncio.AbstractEventLoop:
    global _bg_loop, _bg_thread
    if _bg_loop is None:
        _bg_loop = asyncio.new_event_loop()
        _bg_thread = threading.Thread(target=_bg_loop.run_forever, daemon=True)
        _bg_thread.start()
    return _bg_loop


def run_async(coro: Any) -> Any:
    """Run an async coroutine from Streamlit's sync context."""
    loop = get_bg_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


# ── Belief / contradiction rendering ───────────────────────────────────────────


def belief_to_dict(b: Any) -> dict:
    """Convert a belief object (or dict) to a plain dict."""
    if isinstance(b, dict):
        return b
    return {
        "subject": getattr(b, "subject", ""),
        "predicate": getattr(b, "predicate", ""),
        "value": getattr(b, "value", ""),
        "confidence": getattr(b, "confidence", 0),
        "turn": getattr(b, "turn", 0),
        "source": getattr(b, "source", ""),
        "belief_type": getattr(b, "belief_type", "assertion"),
        "is_hypothetical": getattr(b, "is_hypothetical", False),
    }


def belief_card_html(b: Any) -> str:
    """Render a single belief as an HTML card."""
    d = belief_to_dict(b)
    conf = d["confidence"]
    hypo = (
        ' <span style="color:#F59E0B;font-size:10px">[hypothetical]</span>'
        if d["is_hypothetical"]
        else ""
    )
    update = (
        ' <span style="color:#7C6FEB;font-size:10px">[update]</span>'
        if d["belief_type"] == "update"
        else ""
    )
    return (
        f'<div class="belief-card">'
        f'<span style="color:#9D93F0">{d["subject"]}</span> '
        f'<span style="color:#A1A1AA">{d["predicate"]}</span> '
        f'<span style="color:#34D399">{d["value"]}</span> '
        f"{hypo}{update} "
        f'<span style="float:right;color:#52525B">conf: {conf:.2f}</span>'
        f"</div>"
    )


def contradiction_card_html(msg: str) -> str:
    """Render a contradiction warning as an HTML card."""
    return (
        f'<div class="contradiction-card">'
        f"⚠ <b>Contradiction detected</b><br>"
        f'<small style="color:#F87171">{msg}</small>'
        f"</div>"
    )
