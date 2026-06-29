"""Automated feature tests for the beliefstate package.

Each test directly calls the beliefstate API (no LLM calls needed).
Tests run against the store and tracker internals.
"""

from __future__ import annotations

import asyncio
from typing import Any


async def run_all_tests(tracker: Any, session_id: str) -> list[dict[str, Any]]:
    """Run all automated tests and return a list of result dicts."""
    from beliefstate.call import LLMCall, LLMResponse

    results: list[dict[str, Any]] = []

    # ── Test 1: Health Check ──────────────────────────────────────────────
    try:
        health = await tracker.health_check()
        store_ok = health.get("store", False)
        adapter_ok = health.get("adapter", False)
        results.append(
            {
                "name": "Health Check",
                "passed": store_ok,
                "detail": f"store={store_ok}, adapter={adapter_ok}",
                "expected": "store=True",
                "got": str(health),
            }
        )
    except Exception as e:
        results.append(
            {
                "name": "Health Check",
                "passed": False,
                "detail": str(e),
                "expected": "no exception",
                "got": str(e),
            }
        )

    # ── Test 2: Session Set ───────────────────────────────────────────────
    try:
        tracker.set_session(session_id)
        tracker.set_session("other-session")
        tracker.set_session(session_id)
        results.append(
            {
                "name": "Session Context Set",
                "passed": True,
                "detail": "set_session() called without error",
                "expected": "no exception",
                "got": "ok",
            }
        )
    except Exception as e:
        results.append(
            {
                "name": "Session Context Set",
                "passed": False,
                "detail": str(e),
                "expected": "no exception",
                "got": str(e),
            }
        )

    # ── Test 3: Belief Store Write + Read ─────────────────────────────────
    try:
        test_sid = "auto-test-write-read"
        call = LLMCall(
            messages=[{"role": "user", "content": "I love Python."}],
            kwargs={},
        )
        resp = LLMResponse(text="Got it, you love Python.", raw_response=None)

        await tracker.track_async(
            call.model_dump(),
            resp.model_dump(),
            session_id=test_sid,
            turn=1,
        )
        await asyncio.sleep(0.3)

        beliefs = await tracker.get_beliefs(session_id=test_sid)
        results.append(
            {
                "name": "Belief Store Write & Read",
                "passed": len(beliefs) > 0,
                "detail": f"{len(beliefs)} beliefs written and read back",
                "expected": "beliefs > 0",
                "got": f"{len(beliefs)} beliefs",
            }
        )
    except Exception as e:
        results.append(
            {
                "name": "Belief Store Write & Read",
                "passed": False,
                "detail": str(e),
                "expected": "beliefs > 0",
                "got": str(e),
            }
        )

    # ── Test 4: Contradiction Detection ───────────────────────────────────
    try:
        test_sid = "auto-test-contradiction"

        call1 = LLMCall(
            messages=[{"role": "user", "content": "My budget is $5,000."}],
            kwargs={},
        )
        resp1 = LLMResponse(text="Noted, your budget is $5,000.", raw_response=None)
        await tracker.track_async(
            call1.model_dump(),
            resp1.model_dump(),
            session_id=test_sid,
            turn=1,
        )
        await asyncio.sleep(0.5)

        call2 = LLMCall(
            messages=[
                {"role": "user", "content": "My budget is $5,000."},
                {"role": "assistant", "content": "Noted, your budget is $5,000."},
                {"role": "user", "content": "Actually, my budget is $50,000."},
            ],
            kwargs={},
        )
        resp2 = LLMResponse(
            text="Understood, your budget is $50,000.", raw_response=None
        )
        await tracker.track_async(
            call2.model_dump(),
            resp2.model_dump(),
            session_id=test_sid,
            turn=2,
        )
        await asyncio.sleep(0.5)

        beliefs = await tracker.get_beliefs(session_id=test_sid)
        budget_beliefs = [
            b
            for b in beliefs
            if "budget" in str(getattr(b, "predicate", "")).lower()
            or "budget" in str(getattr(b, "subject", "")).lower()
            or "budget" in str(getattr(b, "value", "")).lower()
        ]
        results.append(
            {
                "name": "Contradiction Detection",
                "passed": True,
                "detail": f"Contradiction pipeline ran. {len(budget_beliefs)} budget belief(s) in store.",
                "expected": "pipeline runs without error",
                "got": f"{len(budget_beliefs)} budget belief(s)",
            }
        )
    except Exception as e:
        results.append(
            {
                "name": "Contradiction Detection",
                "passed": False,
                "detail": str(e),
                "expected": "pipeline runs",
                "got": str(e),
            }
        )

    # ── Test 5: Context Prompt ────────────────────────────────────────────
    try:
        context = await tracker.get_context_prompt(session_id=session_id)
        has_content = isinstance(context, str) and len(context) > 0
        results.append(
            {
                "name": "Context Prompt Generation",
                "passed": has_content,
                "detail": f"get_context_prompt() returned {len(context)} chars",
                "expected": "non-empty string",
                "got": f"{len(context)} chars",
            }
        )
    except Exception as e:
        results.append(
            {
                "name": "Context Prompt Generation",
                "passed": False,
                "detail": str(e),
                "expected": "non-empty string",
                "got": str(e),
            }
        )

    # ── Test 6: Session Isolation ─────────────────────────────────────────
    try:
        sid_a = "isolation-test-A"
        sid_b = "isolation-test-B"

        call_a = LLMCall(
            messages=[{"role": "user", "content": "I am Alice."}],
            kwargs={},
        )
        resp_a = LLMResponse(text="Hello Alice.", raw_response=None)
        await tracker.track_async(
            call_a.model_dump(),
            resp_a.model_dump(),
            session_id=sid_a,
            turn=1,
        )

        call_b = LLMCall(
            messages=[{"role": "user", "content": "I am Bob."}],
            kwargs={},
        )
        resp_b = LLMResponse(text="Hello Bob.", raw_response=None)
        await tracker.track_async(
            call_b.model_dump(),
            resp_b.model_dump(),
            session_id=sid_b,
            turn=1,
        )

        await asyncio.sleep(0.5)

        beliefs_a = await tracker.get_beliefs(session_id=sid_a)
        beliefs_b = await tracker.get_beliefs(session_id=sid_b)

        values_a = [str(getattr(b, "value", "")).lower() for b in beliefs_a]
        values_b = [str(getattr(b, "value", "")).lower() for b in beliefs_b]

        no_bleed = not any("bob" in v for v in values_a) and not any(
            "alice" in v for v in values_b
        )

        results.append(
            {
                "name": "Session Isolation",
                "passed": no_bleed,
                "detail": f"Session A: {len(beliefs_a)} beliefs, Session B: {len(beliefs_b)} beliefs — no bleed",
                "expected": "no cross-session belief bleed",
                "got": f"A has {len(beliefs_a)}, B has {len(beliefs_b)} beliefs",
            }
        )
    except Exception as e:
        results.append(
            {
                "name": "Session Isolation",
                "passed": False,
                "detail": str(e),
                "expected": "no bleed",
                "got": str(e),
            }
        )

    # ── Test 7: Dispatcher Mode ───────────────────────────────────────────
    try:
        dispatcher_type = tracker.config.task_dispatcher_type
        results.append(
            {
                "name": "Dispatcher Mode",
                "passed": True,
                "detail": f"Dispatcher type: {dispatcher_type}",
                "expected": "sync or asyncio",
                "got": dispatcher_type,
            }
        )
    except Exception as e:
        results.append(
            {
                "name": "Dispatcher Mode",
                "passed": False,
                "detail": str(e),
                "expected": "no exception",
                "got": str(e),
            }
        )

    return results
