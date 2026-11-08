"""Self-check for tool call / tool result pairing in replayed history.

OpenWebUI hands the pipe its own tool calls as OpenAI-style messages: an
assistant message carrying ``tool_calls`` followed by ``role: "tool"`` answers.
That is how its hidden preloads arrive (Open Terminal's skill manifest, memory
recall, attached knowledge, all with ``fc_...`` ids), and how a turn that ran on
another model in the same chat is replayed. The pipe used to drop the assistant
half as empty and forward the answers as orphaned ``tool_result`` blocks, which
the API rejects with "unexpected tool_use_id found in tool_result blocks".

Loads the built artifact, so it tests what ships. Needs the pipe's own
dependencies (anthropic, pydantic, fastapi, cryptography) but no OpenWebUI and
no network.

Usage:
    python helpers/test_tool_history_pairing.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ARTIFACT = Path(__file__).resolve().parents[1] / "anthropic_pipe.py"


def load_pipe():
    spec = importlib.util.spec_from_file_location("_anthropic_pipe_artifact", ARTIFACT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_anthropic_pipe_artifact"] = module
    spec.loader.exec_module(module)
    return module.Pipe()


def blocks(msg, block_type):
    content = msg["content"]
    return [b for b in content if isinstance(b, dict) and b.get("type") == block_type]


def assert_api_valid(messages):
    """The pairing rule the API enforces, written independently of the code under test."""
    for i, msg in enumerate(messages):
        if msg["role"] == "user":
            previous = messages[i - 1] if i else None
            valid = {b["id"] for b in blocks(previous, "tool_use")} if previous and previous["role"] == "assistant" else set()
            for b in blocks(msg, "tool_result"):
                assert b["tool_use_id"] in valid, f"orphaned tool_result {b['tool_use_id']!r} in message {i}"
            kinds = [b.get("type") for b in msg["content"] if isinstance(b, dict)]
            seen_other = False
            for kind in kinds:
                assert not (seen_other and kind == "tool_result"), f"tool_result must lead the message: {kinds}"
                seen_other = seen_other or kind != "tool_result"
        elif msg["role"] == "assistant":
            following = messages[i + 1] if i + 1 < len(messages) else None
            answered = {b["tool_use_id"] for b in blocks(following, "tool_result")} if following and following["role"] == "user" else set()
            for b in blocks(msg, "tool_use"):
                assert b["id"] in answered, f"unanswered tool_use {b['id']!r} in message {i}"


def convert(pipe, raw):
    _, processed, _ = pipe._convert_messages_to_claude_format(raw)
    return processed


def call(call_id, name="list_available_skills", arguments="{}"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def test_hidden_preload_pairs(pipe):
    """The reported failure: a hidden fc_ call after the user message."""
    out = convert(pipe, [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "list my files"},
        {"role": "assistant", "content": "", "tool_calls": [call("fc_4b6ddb3943fa4568bf4ebce0")]},
        {"role": "tool", "tool_call_id": "fc_4b6ddb3943fa4568bf4ebce0", "content": "<available_skills/>"},
    ])
    assert_api_valid(out)
    assert [m["role"] for m in out] == ["user", "assistant", "user"], [m["role"] for m in out]
    use = blocks(out[1], "tool_use")[0]
    assert use == {
        "type": "tool_use",
        "id": "fc_4b6ddb3943fa4568bf4ebce0",
        "name": "list_available_skills",
        "input": {},
    }, use
    assert blocks(out[2], "tool_result")[0]["content"] == "<available_skills/>"


def test_parallel_calls_share_one_result_message(pipe):
    out = convert(pipe, [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [call("fc_a", "read_file", '{"path": "/x"}'), call("fc_b", "run_command", '{"command": "ls"}')],
        },
        {"role": "tool", "tool_call_id": "fc_a", "content": "A"},
        {"role": "tool", "tool_call_id": "fc_b", "content": "B"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "thanks"},
    ])
    assert_api_valid(out)
    assert [m["role"] for m in out] == ["user", "assistant", "user", "assistant", "user"]
    assert [b["type"] for b in out[1]["content"]] == ["text", "tool_use", "tool_use"]
    assert blocks(out[1], "tool_use")[0]["input"] == {"path": "/x"}
    assert [b["tool_use_id"] for b in blocks(out[2], "tool_result")] == ["fc_a", "fc_b"]


def test_orphaned_result_is_kept_as_text(pipe):
    out = convert(pipe, [
        {"role": "user", "content": "hi"},
        {"role": "tool", "tool_call_id": "fc_gone", "content": "leftover output"},
    ])
    assert_api_valid(out)
    texts = [b["text"] for m in out for b in blocks(m, "text")]
    assert any("leftover output" in t for t in texts), texts


def test_unanswered_call_is_dropped(pipe):
    out = convert(pipe, [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "trying", "tool_calls": [call("fc_nobody", "read_file")]},
        {"role": "user", "content": "never mind"},
    ])
    assert_api_valid(out)
    assert not any(blocks(m, "tool_use") for m in out)
    assert any(b.get("text") == "trying" for b in blocks(out[1], "text"))


def test_bad_arguments_and_ids(pipe):
    out = convert(pipe, [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [call("call:odd.id", "read_file", "{not json")]},
        {"role": "tool", "tool_call_id": "call:odd.id", "content": "ok"},
    ])
    assert_api_valid(out)
    use = blocks(out[1], "tool_use")[0]
    assert use["id"] == "call_odd_id" and use["input"] == {}, use
    assert blocks(out[2], "tool_result")[0]["tool_use_id"] == "call_odd_id"
    # An entry without a name or id is skipped instead of producing an invalid block.
    out = convert(pipe, [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "text", "tool_calls": [{"id": "", "function": {"name": "x"}}]},
    ])
    assert not any(blocks(m, "tool_use") for m in out)


def test_valid_history_is_untouched(pipe):
    """The reconcile pass must not disturb a correct history (cache prefix stability)."""
    history = [
        {"role": "user", "content": [{"type": "text", "text": "q"}]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "ok"},
            {"type": "tool_use", "id": "toolu_1", "name": "t", "input": {"a": 1}},
        ]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "r"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "fin"}]},
    ]
    before = json.dumps(history, sort_keys=True)
    out = pipe._reconcile_tool_pairs(history)
    assert json.dumps(out, sort_keys=True) == before
    assert all(a is b for a, b in zip(out, history)), "valid messages must pass through as the same objects"


TESTS = [
    test_hidden_preload_pairs,
    test_parallel_calls_share_one_result_message,
    test_orphaned_result_is_kept_as_text,
    test_unanswered_call_is_dropped,
    test_bad_arguments_and_ids,
    test_valid_history_is_untouched,
]


def main() -> int:
    pipe = load_pipe()
    failures = 0
    for test in TESTS:
        try:
            test(pipe)
            print(f"PASS {test.__name__}")
        except Exception as exc:  # noqa: BLE001 - report every failure, not just the first
            failures += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(TESTS) - failures}/{len(TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
