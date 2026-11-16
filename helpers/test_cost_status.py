"""Self-check for the cost estimate's visibility (SHOW_COST).

Bug being guarded: with the default ``SHOW_TOKEN_COUNT`` of ``Off`` the pipe did
not tally usage at all (``include_usage`` was False), so ``SHOW_COST`` produced
nothing: no ``💵`` in the status line and no ``cost_usd`` in the usage that
OpenWebUI shows in the message info tooltip.

Drives a whole ``pipe()`` turn of the built artifact against a fake streaming
client built from the SDK's own event types, and reads what the pipe emits.
Needs the pipe's dependencies (anthropic, pydantic, fastapi, cryptography) but
no OpenWebUI and no network.

Usage:
    python helpers/test_cost_status.py
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

ARTIFACT = Path(__file__).resolve().parents[1] / "anthropic_pipe.py"
MODEL = "claude-opus-5-5"
# 1,000 uncached input + 2,000 output tokens on Opus 5.5 ($4 / $20 per MTok):
# 1000 * 4 / 1e6 + 2000 * 20 / 1e6 = 0.004 + 0.04
EXPECTED_COST_USD = 0.044


def load_module():
    spec = importlib.util.spec_from_file_location("_anthropic_pipe_artifact", ARTIFACT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_anthropic_pipe_artifact"] = module
    spec.loader.exec_module(module)
    return module


def sdk_events(input_tokens: int, output_tokens: int):
    """One plain-text answer, as the SDK's stream yields it, plus its final snapshot."""
    from anthropic.types.beta import (
        BetaMessage,
        BetaRawContentBlockDeltaEvent,
        BetaRawContentBlockStartEvent,
        BetaRawContentBlockStopEvent,
        BetaRawMessageDeltaEvent,
        BetaRawMessageStartEvent,
        BetaRawMessageStopEvent,
    )

    def message(text: str, stop_reason, out: int):
        return BetaMessage.model_validate(
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": MODEL,
                "content": [{"type": "text", "text": text}] if text else [],
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": out,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            }
        )

    events = [
        BetaRawMessageStartEvent(type="message_start", message=message("", None, 1)),
        BetaRawContentBlockStartEvent.model_validate(
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
        ),
        BetaRawContentBlockDeltaEvent.model_validate(
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello there."}}
        ),
        BetaRawContentBlockStopEvent(type="content_block_stop", index=0),
        BetaRawMessageDeltaEvent.model_validate(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            }
        ),
        BetaRawMessageStopEvent(type="message_stop"),
    ]
    return events, message("Hello there.", "end_turn", output_tokens)


class FakeStream:
    def __init__(self, events: List[Any], snapshot: Any):
        self._events = events
        self.current_message_snapshot = snapshot
        self.response = SimpleNamespace(headers={})

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        async def gen():
            for event in self._events:
                yield event

        return gen()


class FakeClient:
    """Stands in for the Anthropic client: streams one scripted answer."""

    def __init__(self, input_tokens: int, output_tokens: int):
        events, snapshot = sdk_events(input_tokens, output_tokens)
        self.beta = SimpleNamespace(
            messages=SimpleNamespace(stream=lambda **_request: FakeStream(events, snapshot))
        )

    def with_options(self, **_options):
        return self


async def run_turn(
    module, *, show_token_count: str, show_cost: bool, item_mode: bool = False
) -> Dict[str, Any]:
    pipe = module.Pipe()
    pipe.valves = pipe.Valves(ANTHROPIC_API_KEY="sk-ant-test")
    pipe._build_anthropic_client = lambda *a, **k: FakeClient(1000, 2000)

    user_valves = pipe.UserValves(SHOW_TOKEN_COUNT=show_token_count, SHOW_COST=show_cost)
    events: List[Dict[str, Any]] = []

    async def emitter(event: Dict[str, Any]) -> None:
        events.append(event)

    result = await pipe.pipe(
        body={
            # OpenWebUI strips the manifold prefix before calling a pipe.
            "model": MODEL,
            "stream": True,
            "messages": [{"role": "user", "content": "Say hello."}],
        },
        __user__={"id": "u1", "valves": user_valves},
        __event_emitter__=emitter,
        # A chat_id and message_id make the pipe stream output items (OpenWebUI
        # 0.12.2+); without them it answers in text mode.
        __metadata__={"chat_id": "c1", "message_id": "m1"} if item_mode else {},
        __request__=None,
    )
    item_usage = None
    if hasattr(result, "__aiter__"):  # item mode returns the event generator
        async for item_event in result:
            if item_event.get("type") == "response.completed":
                item_usage = item_event["response"].get("usage")

    statuses = [
        e["data"]["description"]
        for e in events
        if e.get("type") == "status" and isinstance(e.get("data"), dict) and e["data"].get("description")
    ]
    completions = [e["data"] for e in events if e.get("type") == "chat:completion"]
    return {
        "events": events,
        "final_status": next((s for s in reversed(statuses) if s.startswith("✅")), None),
        "usage": item_usage
        if item_mode
        else next((c["usage"] for c in reversed(completions) if c.get("usage")), None),
    }


FAILURES: List[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok  {label}")
    else:
        print(f"FAIL: {label} {detail}")
        FAILURES.append(label)


async def main() -> int:
    module = load_module()

    # The reported case: valve on, token display at its default of Off.
    r = await run_turn(module, show_token_count="Off", show_cost=True)
    check(
        "cost shows in the status line with the token display Off",
        bool(r["final_status"]) and "💵 ≈$0.044" in r["final_status"],
        f"(status: {r['final_status']!r})",
    )
    check(
        "status line has no token gauge when the token display is Off",
        bool(r["final_status"]) and "[" not in r["final_status"],
        f"(status: {r['final_status']!r})",
    )
    check(
        "usage reaches OpenWebUI with cost_usd and its breakdown",
        bool(r["usage"]) and abs(r["usage"].get("cost_usd", 0) - EXPECTED_COST_USD) < 1e-9
        and r["usage"].get("cost_breakdown_usd") == {"input": 0.004, "output": 0.04},
        f"(usage: {r['usage']!r})",
    )

    # Valve off and token display Off: nothing to show, as before.
    r = await run_turn(module, show_token_count="Off", show_cost=False)
    check("SHOW_COST off and token display Off: no final status line", r["final_status"] is None,
          f"(status: {r['final_status']!r})")
    check("SHOW_COST off and token display Off: no usage emitted", not r["usage"],
          f"(usage: {r['usage']!r})")

    # Token display on: the cost is appended to the gauge, once.
    r = await run_turn(module, show_token_count="On", show_cost=True)
    check(
        "cost is appended to the gauge when the token display is On",
        bool(r["final_status"]) and "[" in r["final_status"] and r["final_status"].count("💵") == 1
        and "≈$0.044" in r["final_status"],
        f"(status: {r['final_status']!r})",
    )

    # Token display on, cost off: gauge only.
    r = await run_turn(module, show_token_count="On", show_cost=False)
    check(
        "SHOW_COST off keeps the gauge and drops the cost",
        bool(r["final_status"]) and "[" in r["final_status"] and "💵" not in r["final_status"]
        and "cost_usd" not in (r["usage"] or {}),
        f"(status: {r['final_status']!r}, usage: {r['usage']!r})",
    )

    # OpenWebUI 0.12.2+ streams output items; usage rides on response.completed.
    r = await run_turn(module, show_token_count="Off", show_cost=True, item_mode=True)
    check(
        "item mode: cost shows in the status line with the token display Off",
        bool(r["final_status"]) and "💵 ≈$0.044" in r["final_status"],
        f"(status: {r['final_status']!r})",
    )
    check(
        "item mode: response.completed carries cost_usd",
        bool(r["usage"]) and abs(r["usage"].get("cost_usd", 0) - EXPECTED_COST_USD) < 1e-9,
        f"(usage: {r['usage']!r})",
    )
    r = await run_turn(module, show_token_count="Off", show_cost=False, item_mode=True)
    check("item mode: SHOW_COST off and token display Off: no final status line",
          r["final_status"] is None, f"(status: {r['final_status']!r})")

    if FAILURES:
        print(f"\n{len(FAILURES)} check(s) failed")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
