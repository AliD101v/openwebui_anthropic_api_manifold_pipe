"""Self-check: LEGACY_CHAT_MIGRATION converts legacy turns OpenWebUI 0.13 stores behind a hidden call.

Before 0.11 a turn was one string of <details> HTML; pipe 0.11 stores output items,
and _migrate_legacy_message rewrites old turns into items. OpenWebUI 0.13 puts a
hidden call (the skill list) and its result in front of every stored turn's output.
The migration must keep that prelude: the returned items are the FULL new output,
prelude first, and the replay sent to the API must stay identical.

Loads the built artifact, so it tests what ships. Needs the pipe's own
dependencies (anthropic, pydantic, fastapi, cryptography) but no OpenWebUI and
no network.

Usage:
    python helpers/test_legacy_migration_hidden_calls.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ARTIFACT = Path(__file__).resolve().parents[1] / "anthropic_pipe.py"
MODULE = "_anthropic_pipe_artifact"


def load_pipe():
    spec = importlib.util.spec_from_file_location(MODULE, ARTIFACT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE] = module
    spec.loader.exec_module(module)
    return module.Pipe()


def legacy_turn(pipe, with_search=True):
    """One assistant turn as the pre-0.11 pipe stored it, built with the pipe's own formatters."""
    use = {"type": "server_tool_use", "id": "srvtoolu_01", "name": "web_search", "input": {"query": "nvidia earnings"}}
    result = {
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_01",
        "content": [{"type": "web_search_result", "url": "https://example.com/a", "title": "A", "encrypted_content": "ENC" * 50}],
    }
    enc = pipe._encode_block_payload
    text = pipe._format_thinking_block("", duration=9, signature="SIGNATURE_ONE" * 20)
    if with_search:
        text += (
            f'<details type="tool_calls" done="true" data-block-kind="server_tool_use" '
            f'data-payload-b64="{enc(use)}" data-result-payload-b64="{enc(result)}">\n'
            f"<summary>Searched 1 site</summary>\n</details>\n"
        )
    return text + "\nThe answer is 42.\n"


def convert_output_to_messages(output):
    """Port of open_webui.utils.misc.convert_output_to_messages (raw=True, no reasoning_format)."""
    messages, calls, content, details, outputs = [], [], [], [], []

    def flush_pending():
        nonlocal calls, content, details
        if not (content or calls or details):
            return
        m = {"role": "assistant", "content": "\n".join(content) if content else ""}
        if calls:
            m["tool_calls"] = calls
        if details:
            m["reasoning_details"] = details
        messages.append(m)
        calls, content, details = [], [], []

    def flush_outputs():
        nonlocal outputs
        if not outputs:
            return
        outputs = [o for o in outputs if any(c["id"] == o.get("call_id") for c in calls)]
        flush_pending()
        for o in outputs:
            text = "".join(p.get("text", "") for p in o.get("output", []) if p.get("type") == "input_text")
            messages.append({"role": "tool", "tool_call_id": o.get("call_id", ""), "content": text})
        outputs = []

    for item in output:
        t = item.get("type", "")
        if t != "function_call_output":
            flush_outputs()
        if t == "message":
            text = "".join(p.get("text", "") for p in item.get("content", []) if p.get("type") == "output_text")
            if text:
                content.append(text)
        elif t == "function_call":
            if not item.get("call_id") or item.get("status") not in {"completed", "failed", "rejected"}:
                continue
            fn = item.get("provider_function")
            fn = dict(fn) if isinstance(fn, dict) else item
            args = fn.get("arguments")
            calls.append({
                "id": item["call_id"], "type": "function",
                "function": {"name": fn.get("name", ""),
                             "arguments": args if isinstance(args, str) else json.dumps(args)},
            })
        elif t == "function_call_output":
            outputs.append(item)
        elif t == "reasoning":
            rd = item.get("reasoning_details")
            if rd:
                rd = rd if isinstance(rd, list) else [rd]
                details.extend(d for d in rd if isinstance(d, dict)
                               and (d.get("format") != "anthropic-claude-v1" or d.get("signature")))
    flush_outputs()
    flush_pending()
    return messages


def frame(middle: list) -> list:
    return [{"role": "user", "content": "."}, *middle, {"role": "user", "content": "."}]


def hidden_pair(call_id: str, name: str, text: str) -> list:
    """The hidden call OpenWebUI 0.13 stores at the start of a turn, plus its result."""
    return [
        {"type": "function_call", "id": f"it_{call_id}", "call_id": call_id, "name": name,
         "arguments": "{}", "hidden": True, "status": "completed"},
        {"type": "function_call_output", "id": f"it_{call_id}_out", "call_id": call_id, "status": "completed",
         "output": [{"type": "input_text", "text": text}]},
    ]


def message_item(text: str) -> dict:
    return {"type": "message", "id": "it_msg", "status": "completed", "role": "assistant",
            "content": [{"type": "output_text", "text": text}]}


def stored_message(text: str, output) -> dict:
    return {"role": "assistant", "model": "anthropic_pipe.test-model", "content": text, "output": output}


def test_converts_turn_behind_hidden_call(pipe):
    """A legacy turn behind the hidden skill list converts, prelude kept in front."""
    html = legacy_turn(pipe)
    hidden = hidden_pair("fc_8a42", "list_available_skills", "<available_skills/>")
    stored = stored_message(html, [*hidden, message_item(html)])
    outcome, new_output = pipe._migrate_legacy_message(stored, convert_output_to_messages)
    assert outcome == "converted", outcome
    assert new_output[0] == hidden[0] and new_output[1] == hidden[1]
    rest = new_output[2:]
    assert [i["type"] for i in rest] == ["reasoning", "function_call", "function_call_output", "message"], [i.get("type") for i in rest]
    assert "provider_function" in rest[1]
    for item in new_output:
        assert "<details" not in json.dumps(item), item.get("type")


def test_api_messages_unchanged_after_migration(pipe):
    """The converted turn replays to the API exactly as the stored turn does."""
    html = legacy_turn(pipe)
    hidden = hidden_pair("fc_8a42", "list_available_skills", "<available_skills/>")
    stored = stored_message(html, [*hidden, message_item(html)])
    outcome, new_output = pipe._migrate_legacy_message(stored, convert_output_to_messages)
    assert outcome == "converted", outcome
    old = pipe._convert_messages_to_claude_format(frame(convert_output_to_messages(stored["output"])))[1]
    new = pipe._convert_messages_to_claude_format(frame(convert_output_to_messages(new_output)))[1]
    assert old == new


def test_converts_turn_without_hidden_call(pipe):
    """A turn with no hidden call migrates exactly as before."""
    html = legacy_turn(pipe)
    stored = stored_message(html, [message_item(html)])
    outcome, new_output = pipe._migrate_legacy_message(stored, convert_output_to_messages)
    assert outcome == "converted", outcome
    assert new_output[0]["type"] == "reasoning"
    for item in new_output:
        if item.get("type") == "function_call":
            assert "provider_function" in item
        assert "<details" not in json.dumps(item)


def test_converts_turn_behind_two_hidden_calls(pipe):
    """Skill list and memory recall both stay in front, in the original order."""
    html = legacy_turn(pipe)
    pair_1 = hidden_pair("fc_skills", "list_available_skills", "<available_skills/>")
    pair_2 = hidden_pair("fc_memory", "recall_memories", "<memories/>")
    stored = stored_message(html, [*pair_1, *pair_2, message_item(html)])
    outcome, new_output = pipe._migrate_legacy_message(stored, convert_output_to_messages)
    assert outcome == "converted", outcome
    assert new_output[:4] == [*pair_1, *pair_2]


def test_skips_turn_with_real_item_after_prelude(pipe):
    """A turn that already has real items (a reasoning item) stays untouched."""
    html = legacy_turn(pipe)
    hidden = hidden_pair("fc_8a42", "list_available_skills", "<available_skills/>")
    stored = stored_message(html, [*hidden, {"type": "reasoning", "id": "it_reasoning", "status": "completed"}, message_item(html)])
    assert pipe._migrate_legacy_message(stored, convert_output_to_messages) == ("skip", None)


def test_skips_plain_text_behind_prelude(pipe):
    """A plain answer behind the prelude has nothing to convert."""
    plain = "The answer is 42."
    hidden = hidden_pair("fc_8a42", "list_available_skills", "<available_skills/>")
    stored = stored_message(plain, [*hidden, message_item(plain)])
    assert pipe._migrate_legacy_message(stored, convert_output_to_messages) == ("skip", None)


def test_skips_hidden_call_without_result(pipe):
    """A hidden call whose result is missing would replay unpaired: skip."""
    html = legacy_turn(pipe)
    stored = stored_message(html, [hidden_pair("fc_8a42", "list_available_skills", "<available_skills/>")[0]])
    assert pipe._migrate_legacy_message(stored, convert_output_to_messages) == ("skip", None)


def test_skips_prelude_after_message(pipe):
    """The hidden pair must come before the turn, not after: skip."""
    html = legacy_turn(pipe)
    hidden = hidden_pair("fc_8a42", "list_available_skills", "<available_skills/>")
    stored = stored_message(html, [message_item(html), *hidden])
    assert pipe._migrate_legacy_message(stored, convert_output_to_messages) == ("skip", None)


def test_no_output_keeps_old_behaviour(pipe):
    """output None, [] or absent behaves exactly as before: details convert, plain skips."""
    html = legacy_turn(pipe)
    for shape in (None, []):
        for content, expected in ((html, "converted"), ("The answer is 42.", "skip")):
            stored = stored_message(content, list(shape) if shape is not None else None)
            outcome, items = pipe._migrate_legacy_message(stored, convert_output_to_messages)
            assert outcome == expected, (shape, outcome)
            if expected == "converted":
                assert items
            else:
                assert items is None
    for content, expected in ((html, "converted"), ("The answer is 42.", "skip")):
        stored = {"role": "assistant", "model": "anthropic_pipe.test-model", "content": content}  # no output key
        outcome, items = pipe._migrate_legacy_message(stored, convert_output_to_messages)
        assert outcome == expected, outcome
        if expected == "converted":
            assert items
        else:
            assert items is None


TESTS = [
    test_converts_turn_behind_hidden_call,
    test_api_messages_unchanged_after_migration,
    test_converts_turn_without_hidden_call,
    test_converts_turn_behind_two_hidden_calls,
    test_skips_turn_with_real_item_after_prelude,
    test_skips_plain_text_behind_prelude,
    test_skips_hidden_call_without_result,
    test_skips_prelude_after_message,
    test_no_output_keeps_old_behaviour,
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
