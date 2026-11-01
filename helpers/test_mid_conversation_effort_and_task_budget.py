"""Self-check for per-message effort (cache-preserving) and Task Budgets.

Two independent features, tested together because both live in the same
``output_config`` assembly in ``create_request_payload``:

- Per-message effort (beta ``mid-conversation-output-config-2026-07-01``):
  OpenWebUI's ``reasoning_effort`` is a per-message control. Changing the
  top-level ``output_config.effort`` every time it differs from the user's
  static ``EFFORT`` valve would invalidate the whole prompt cache. On models
  that support it, the top level stays pinned to the valve baseline and the
  deviation goes out as an effort-only ``role: "system"`` message inserted
  right before the upcoming user turn instead.
- Task Budgets (beta ``task-budgets-2026-03-13``): an advisory token ceiling
  for the whole tool loop, set only when the admin/user configures a nonzero
  ``TASK_BUDGET_TOKENS``.

Runs without OpenWebUI and without network access: ``create_request_payload``
is exercised directly, mirroring the harness in ``test_fast_mode_header.py``.

Usage:
    python helpers/test_mid_conversation_effort_and_task_budget.py
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "anthropic_pipe" / "request" / "payload.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("_payload", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_payload"] = module
    spec.loader.exec_module(module)
    module.FILES_AVAILABLE = False
    status_path = MODULE_PATH.parents[1] / "response" / "status_events.py"
    status_spec = importlib.util.spec_from_file_location("_status_events", status_path)
    status_module = importlib.util.module_from_spec(status_spec)
    status_spec.loader.exec_module(status_module)
    module.StatusEmitter = status_module.StatusEmitter
    return module


class FakeValves:
    def __init__(self, **kwargs):
        self.DATA_RESIDENCY = "global"
        self.ENABLE_FAST_MODE = False
        self.CACHE_CONTROL = "cache disabled"
        self.ENABLE_PROGRAMMATIC_TOOL_CALLING = False
        self.ENABLE_INTERLEAVED_THINKING = False
        self.WEB_FETCH = False
        self.ANTHROPIC_API_KEY = "sk-admin-key"
        self.REFUSAL_FALLBACK = "off"
        self.ENABLE_CACHE_DIAGNOSTICS = False
        self.__dict__.update(kwargs)


class FakeUserValves:
    def __init__(self, **kwargs):
        self.ENABLE_THINKING = False
        self.THINKING_BUDGET_TOKENS = 8192
        self.THINKING_DISPLAY = "omitted"
        self.EFFORT = "high"
        self.TASK_BUDGET_TOKENS = 0
        self.SKILLS: List[str] = []
        self.USE_FILES_API = False
        self.USE_PDF_NATIVE_UPLOAD = False
        self.ENABLE_TOOL_SEARCH = False
        self.ENABLE_ADVISOR_TOOL = False
        self.CONTEXT_EDITING_STRATEGY = "none"
        self.ENABLE_COMPACTION = False
        self.ANTHROPIC_API_KEY = ""
        self.__dict__.update(kwargs)


# A sonnet-5.5-shaped capability set: thinking on by default, adaptive,
# effort-capable, and supporting both new features.
SONNET_5_5_INFO: Dict[str, Any] = {
    "max_tokens": 64000,
    "context_length": 1000000,
    "supports_effort": True,
    "supports_effort_xhigh": True,
    "supports_effort_max": True,
    "supports_thinking": True,
    "supports_adaptive_thinking": True,
    "thinking_on_by_default": True,
    "thinking_off_type": "between_tools",
    "supports_mid_conversation_effort": True,
    "supports_task_budget": True,
    "supports_fast_mode": False,
}

# A sonnet-5-shaped set (predecessor): same effort/thinking surface, but
# neither new feature is supported.
SONNET_5_INFO: Dict[str, Any] = {
    **SONNET_5_5_INFO,
    "supports_mid_conversation_effort": False,
    "supports_task_budget": False,
}


class FakePipe:
    """Minimal stand-in exposing what create_request_payload touches on the
    effort/thinking/output_config path: no files, no tools, no skills."""

    API_VERSION = "2026-10-21"

    def __init__(self, valves: FakeValves, model_info: Dict[str, Any], history: List[Dict[str, Any]]):
        self.valves = valves
        self._model_info = model_info
        self._history = history

    def get_model_info(self, actual_model_name: str) -> Dict[str, Any]:
        return dict(self._model_info)

    def _convert_messages_to_claude_format(self, raw_messages):
        # (system_messages, processed_messages, previous_marker_metadata)
        return [], list(self._history), []

    async def _get_full_context_texts(self, *args, **kwargs):
        return {}, [], []

    def _convert_tools_to_claude_format(self, *args, **kwargs):
        return [], []

    def _canonicalize_block(self, block):
        return block


async def _noop_event_emitter(event: Dict[str, Any]) -> None:
    """Async no-op standing in for OpenWebUI's __event_emitter__."""


# A one-turn history: just the upcoming user message, nothing to protect yet.
FIRST_TURN_HISTORY = [
    {"role": "user", "content": [{"type": "text", "text": "hello"}]},
]

# A later-turn history: a full prior exchange, plus the upcoming user message
# as the last element -- the shape create_request_payload actually sees.
LATER_TURN_HISTORY = [
    {"role": "user", "content": [{"type": "text", "text": "hello"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "hi there"}]},
    {"role": "user", "content": [{"type": "text", "text": "now do something harder"}]},
]


async def _call(
    module,
    pipe: FakePipe,
    *,
    reasoning_effort: Optional[str] = None,
    user_valves: Optional[FakeUserValves] = None,
):
    body: Dict[str, Any] = {"model": "anthropic/claude-test", "messages": []}
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    user = {"valves": user_valves or FakeUserValves()}
    return await module.create_request_payload(
        pipe, body, {}, user, None, _noop_event_emitter, None
    )


def _effort_only_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        m for m in messages
        if isinstance(m, dict) and m.get("role") == "system" and m.get("content") == []
    ]


async def test_first_turn_no_override_even_with_per_message_effort(module):
    """No history to protect yet -- just set the top-level value directly,
    same as before this feature existed."""
    pipe = FakePipe(FakeValves(), SONNET_5_5_INFO, FIRST_TURN_HISTORY)
    uv = FakeUserValves(ENABLE_THINKING=True, EFFORT="high")
    payload, headers, _marker_metadata, _api_tool_names = await _call(
        module, pipe, reasoning_effort="low", user_valves=uv
    )
    assert payload["output_config"]["effort"] == "low", payload["output_config"]
    assert not _effort_only_messages(payload["messages"]), payload["messages"]
    beta = headers.get("anthropic-beta", "")
    assert "mid-conversation-output-config-2026-07-01" not in beta.split(","), beta


async def test_later_turn_override_preserves_baseline_and_cache(module):
    """History exists, thinking is adaptive, model supports the beta, and the
    per-message effort differs from the EFFORT valve baseline: top-level stays
    pinned to the baseline, and the deviation goes out as an inserted
    effort-only message right before the upcoming user turn."""
    pipe = FakePipe(FakeValves(), SONNET_5_5_INFO, LATER_TURN_HISTORY)
    uv = FakeUserValves(ENABLE_THINKING=True, EFFORT="high")
    payload, headers, _marker_metadata, _api_tool_names = await _call(
        module, pipe, reasoning_effort="low", user_valves=uv
    )
    assert payload["output_config"]["effort"] == "high", (
        f"top-level output_config must stay pinned to the baseline, got {payload['output_config']}"
    )
    overrides = _effort_only_messages(payload["messages"])
    assert len(overrides) == 1, payload["messages"]
    assert overrides[0]["output_config"] == {"effort": "low"}, overrides[0]
    # Inserted immediately before the final (upcoming) user turn.
    assert payload["messages"][-1]["role"] == "user"
    assert payload["messages"][-2] is overrides[0] or payload["messages"][-2] == overrides[0]
    beta = headers.get("anthropic-beta", "")
    assert "mid-conversation-output-config-2026-07-01" in beta.split(","), beta


async def test_later_turn_no_override_when_effort_matches_baseline(module):
    """No per-message deviation -- behaves exactly as before: a single
    top-level value, no synthetic message, no beta header."""
    pipe = FakePipe(FakeValves(), SONNET_5_5_INFO, LATER_TURN_HISTORY)
    uv = FakeUserValves(ENABLE_THINKING=True, EFFORT="high")
    payload, headers, _marker_metadata, _api_tool_names = await _call(
        module, pipe, reasoning_effort=None, user_valves=uv
    )
    assert payload["output_config"]["effort"] == "high", payload["output_config"]
    assert not _effort_only_messages(payload["messages"]), payload["messages"]
    beta = headers.get("anthropic-beta", "")
    assert "mid-conversation-output-config-2026-07-01" not in beta.split(","), beta


async def test_unsupported_model_falls_back_to_top_level_change(module):
    """A model without supports_mid_conversation_effort (e.g. Sonnet 5) must
    keep the pre-existing behaviour: top-level effort tracks the per-message
    request directly, cache cost and all."""
    pipe = FakePipe(FakeValves(), SONNET_5_INFO, LATER_TURN_HISTORY)
    uv = FakeUserValves(ENABLE_THINKING=True, EFFORT="high")
    payload, headers, _marker_metadata, _api_tool_names = await _call(
        module, pipe, reasoning_effort="low", user_valves=uv
    )
    assert payload["output_config"]["effort"] == "low", payload["output_config"]
    assert not _effort_only_messages(payload["messages"]), payload["messages"]
    beta = headers.get("anthropic-beta", "")
    assert "mid-conversation-output-config-2026-07-01" not in beta.split(","), beta


async def test_thinking_off_disables_override(module):
    """between_tools (Sonnet 5.5's thinking-off setting) is not adaptive --
    the beta 400s there, so the override must not be attempted even though
    the model otherwise supports it and the efforts differ."""
    pipe = FakePipe(FakeValves(), SONNET_5_5_INFO, LATER_TURN_HISTORY)
    uv = FakeUserValves(ENABLE_THINKING=False, EFFORT="high")
    payload, headers, _marker_metadata, _api_tool_names = await _call(
        module, pipe, reasoning_effort="low", user_valves=uv
    )
    assert payload.get("thinking", {}).get("type") == "between_tools", payload.get("thinking")
    # thinking off clamps xhigh/max to high, but "low" isn't touched by that
    # rule -- the override is skipped purely because thinking isn't adaptive.
    assert not _effort_only_messages(payload["messages"]), payload["messages"]
    beta = headers.get("anthropic-beta", "")
    assert "mid-conversation-output-config-2026-07-01" not in beta.split(","), beta


async def test_task_budget_disabled_by_default(module):
    pipe = FakePipe(FakeValves(), SONNET_5_5_INFO, FIRST_TURN_HISTORY)
    payload, headers, _marker_metadata, _api_tool_names = await _call(module, pipe)
    assert "task_budget" not in payload.get("output_config", {}), payload.get("output_config")
    beta = headers.get("anthropic-beta", "")
    assert "task-budgets-2026-03-13" not in beta.split(","), beta


async def test_task_budget_enabled_on_supported_model(module):
    pipe = FakePipe(FakeValves(), SONNET_5_5_INFO, FIRST_TURN_HISTORY)
    uv = FakeUserValves(TASK_BUDGET_TOKENS=64000)
    payload, headers, _marker_metadata, _api_tool_names = await _call(
        module, pipe, user_valves=uv
    )
    assert payload["output_config"]["task_budget"] == {"type": "tokens", "total": 64000}, (
        payload["output_config"]
    )
    beta = headers.get("anthropic-beta", "")
    assert "task-budgets-2026-03-13" in beta.split(","), beta


async def test_task_budget_ignored_on_unsupported_model(module):
    pipe = FakePipe(FakeValves(), SONNET_5_INFO, FIRST_TURN_HISTORY)
    uv = FakeUserValves(TASK_BUDGET_TOKENS=64000)
    payload, headers, _marker_metadata, _api_tool_names = await _call(
        module, pipe, user_valves=uv
    )
    assert "task_budget" not in payload.get("output_config", {}), payload.get("output_config")
    beta = headers.get("anthropic-beta", "")
    assert "task-budgets-2026-03-13" not in beta.split(","), beta


async def test_task_budget_remaining_never_sent(module):
    """The pipe resends full history on every tool-loop request, so the server
    tracks the countdown itself. A client-decremented ``remaining`` under-reports
    the budget and changes the rendered prompt every request (cache miss), which
    is why v0.10.3 was corrected in v0.10.4. ``remaining`` is only for client-side
    history rewrites, which this pipe never does mid-turn. The orchestrator is not
    exercised end to end here, so guard the sources: no code path may write a
    ``"remaining"`` key into the payload."""
    root = MODULE_PATH.parents[1]
    for rel in ("pipe_orchestrator.py", "request/payload.py"):
        source = (root / rel).read_text(encoding="utf-8")
        assert '"remaining"' not in source and "'remaining'" not in source, (
            f"{rel} writes a task_budget 'remaining' key"
        )


async def main():
    module = _load_module()
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        await test(module)
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    asyncio.run(main())
