"""Self-check: a turn written before 0.11 is replayed as API blocks, not as raw HTML.

Before 0.11 the pipe stored a turn as one string with ``<details>`` HTML:
signed thinking, server tool carriers (base64 payloads), tool call cards.
OpenWebUI 0.13 records a hidden call (the skill list) at the start of every
turn, so such a turn replays as an assistant message with ``tool_calls``, its
``tool`` result and then the old HTML text. The item-run path took that text
into the run and sent it to the API verbatim: thinking signatures and base64
carriers as plain prose, which Anthropic refused as "reasoning extraction"
(and ~10x the tokens). The legacy parser has to see that text instead.

Loads the built artifact, so it tests what ships. Needs the pipe's own
dependencies (anthropic, pydantic, fastapi, cryptography) but no OpenWebUI and
no network.

Usage:
    python helpers/test_legacy_turn_replay.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ARTIFACT = Path(__file__).resolve().parents[1] / "anthropic_pipe.py"
MODULE = "_anthropic_pipe_artifact"
LEAKS = ("<details", "data-signature", "data-payload-b64", "<summary>", "[anthropic-hidden")


def load_pipe():
    spec = importlib.util.spec_from_file_location(MODULE, ARTIFACT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE] = module
    spec.loader.exec_module(module)
    return module.Pipe()


def blocks(msg, block_type=None):
    content = msg["content"]
    return [b for b in content if isinstance(b, dict) and (block_type is None or b.get("type") == block_type)]


def sent_text(messages):
    return "".join(b.get("text", "") for m in messages for b in blocks(m, "text"))


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


HIDDEN_CALL = {
    "role": "assistant",
    "content": "",
    "tool_calls": [{"id": "fc_skills", "type": "function", "function": {"name": "list_available_skills", "arguments": "{}"}}],
}
HIDDEN_RESULT = {"role": "tool", "tool_call_id": "fc_skills", "content": "<available_skills/>"}


def convert(pipe, raw):
    _, processed, _ = pipe._convert_messages_to_claude_format(raw)
    return processed


def test_legacy_turn_behind_hidden_call_is_parsed(pipe):
    """The reported failure: hidden call + result + legacy HTML text."""
    out = convert(pipe, [
        {"role": "user", "content": "first"},
        HIDDEN_CALL,
        HIDDEN_RESULT,
        {"role": "assistant", "content": legacy_turn(pipe)},
        {"role": "user", "content": "second"},
    ])
    text = sent_text(out)
    for leak in LEAKS:
        assert leak not in text, f"{leak!r} reached the API as plain text"
    # hidden call and its result stay a valid pair, then the turn comes back as real blocks
    assert [m["role"] for m in out] == ["user", "assistant", "user", "assistant", "user"], [m["role"] for m in out]
    assert [b["type"] for b in out[1]["content"]] == ["tool_use"]
    assert [b["type"] for b in out[2]["content"]] == ["tool_result"]
    kinds = [b["type"] for b in out[3]["content"]]
    assert kinds == ["thinking", "server_tool_use", "web_search_tool_result", "text"], kinds
    assert out[3]["content"][0]["signature"] == "SIGNATURE_ONE" * 20
    assert out[3]["content"][-1]["text"].strip() == "The answer is 42."


def test_matches_the_turn_without_a_hidden_call(pipe):
    """Same turn, with and without the hidden call: the turn's own blocks must be identical."""
    turn = legacy_turn(pipe)
    plain = convert(pipe, [{"role": "user", "content": "q"}, {"role": "assistant", "content": turn}, {"role": "user", "content": "n"}])
    hidden = convert(pipe, [
        {"role": "user", "content": "q"}, HIDDEN_CALL, HIDDEN_RESULT,
        {"role": "assistant", "content": turn}, {"role": "user", "content": "n"},
    ])
    assert plain[1] == hidden[3], (plain[1], hidden[3])


def test_every_hidden_preload_kind_is_handled(pipe):
    """Memory recall and attached knowledge are hidden calls too, not only the skill list."""
    out = convert(pipe, [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "fc_a", "type": "function", "function": {"name": "list_available_skills", "arguments": "{}"}},
            {"id": "fc_b", "type": "function", "function": {"name": "recall_memories", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "fc_a", "content": "skills"},
        {"role": "tool", "tool_call_id": "fc_b", "content": "memories"},
        {"role": "assistant", "content": legacy_turn(pipe, with_search=False)},
        {"role": "user", "content": "n"},
    ])
    for leak in LEAKS:
        assert leak not in sent_text(out), f"{leak!r} reached the API as plain text"
    assert [b["type"] for b in out[-2]["content"]] == ["thinking", "text"], [b["type"] for b in out[-2]["content"]]


def test_item_turns_still_form_a_run(pipe):
    """A turn the pipe wrote as output items has no <details>: it must stay one run, untouched."""
    group_item_runs = sys.modules[MODULE].group_item_runs
    messages = [
        {"role": "user", "content": "q"},
        HIDDEN_CALL,
        HIDDEN_RESULT,
        {"role": "assistant", "content": "Plain answer after the hidden call."},
        {"role": "user", "content": "n"},
    ]
    grouped = group_item_runs(messages)
    assert [m["role"] for m in grouped] == ["user", "__item_run__", "user"], [m["role"] for m in grouped]
    assert len(grouped[1]["messages"]) == 3
    out = convert(pipe, messages)
    assert [b["type"] for b in out[-2]["content"]] == ["text"]
    assert out[-2]["content"][0]["text"] == "Plain answer after the hidden call."


def test_prose_about_details_is_not_a_legacy_turn(pipe):
    """The marker is the pipe's own HTML, not any mention of the tag."""
    is_legacy = sys.modules[MODULE].is_legacy_turn_text
    assert not is_legacy({"role": "assistant", "content": "Use <details> and <summary> for a disclosure widget."})
    assert not is_legacy({"role": "assistant", "content": '<details open><summary>x</summary>y</details>'})
    assert not is_legacy({"role": "user", "content": legacy_turn(pipe)})
    assert is_legacy({"role": "assistant", "content": legacy_turn(pipe)})
    # a message that also carries calls or thinking is a run start, never plain legacy text
    assert not is_legacy({**HIDDEN_CALL, "content": legacy_turn(pipe)})


TESTS = [
    test_legacy_turn_behind_hidden_call_is_parsed,
    test_matches_the_turn_without_a_hidden_call,
    test_every_hidden_preload_kind_is_handled,
    test_item_turns_still_form_a_run,
    test_prose_about_details_is_not_a_legacy_turn,
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
