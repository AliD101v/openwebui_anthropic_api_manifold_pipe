---
name: sync-upstream
description: Bring this fork and its working branch up to date with Podden's upstream - fetch, analyse what upstream already ships, merge everything (the pipe, the toggle filters, the companion filter, helpers, docs), resolve conflicts, rebuild the artifacts, re-release fork-only work as the next patch version, run the checks, fast-forward main and report including the toggle changes. Use when asked to sync, update or merge upstream / Podden, or to check what upstream has shipped.
argument-hint: "[dry-run]"
---

# sync-upstream

This repo is a fork of `Podden/openwebui_anthropic_api_manifold_pipe`. Upstream moves fast (new Claude models, OpenWebUI versions, streaming rewrites) and the fork carries a small set of fork-only features on top. A sync is **merge upstream, keep upstream's version of everything it ships, keep only what upstream still lacks**. Read `AGENTS.md` first: the build, the generated artifacts and the release checklist are defined there and this skill does not repeat them.

Remotes: `upstream` (Podden, merge source), `origin` (the GitHub fork), `forgejo` (a mirror). `main` mirrors the working branch; GitHub's "N commits behind Podden" badge counts `origin/main`, not the feature branch.

**Always sync everything**, not only the pipe: the toggle filters (`anthropic_pipe_*_toggle.py`), the companion filter (`anthropic_manifold_companion_filter.py`), `helpers/`, `AGENTS.md`, `README.md` and any file upstream adds. The final report always covers the toggles (step 10), even when they did not change.

With the argument `dry-run`: do steps 1-2 only (including the toggle/filter listing), report what a merge would involve, change nothing.

## Hard rules

- Never push, force-push or delete a branch without asking. Pushing is outward-facing; "bring it up to date" does not authorise it.
- Never hand-merge `anthropic_pipe.py` / `anthropic_pipe.min.py`. They are generated: resolve the sources, then rebuild.
- No AI attribution in commit messages, the changelog or code comments.
- Do not edit upstream's own tests to make them pass. If a test fails, first prove whether it fails on a clean `upstream/main` (step 6).

## Steps

### 1. Preflight and measure

```bash
git status --short                      # a dirty README must be committed or stashed before merging
git fetch upstream && git fetch origin
for b in HEAD main origin/main; do printf "%s: " $b; git rev-list --left-right --count $b...upstream/main; done   # ours, theirs
git log --oneline HEAD..upstream/main
MB=$(git merge-base HEAD upstream/main)
comm -12 <(git diff --name-only $MB upstream/main | sort) <(git diff --name-only $MB HEAD | sort)   # changed on both sides
git diff --stat $MB upstream/main -- 'anthropic_pipe_*toggle.py' anthropic_manifold_companion_filter.py   # toggles + companion filter
git ls-tree --name-only upstream/main | grep -E 'toggle|filter'   # catches filters upstream added since last sync
```

Note `$MB` for the report: it is the "previous sync point" the toggle comparison in step 10 is measured from.

Nothing new upstream and `main` equal to the branch means there is nothing to do: say so and stop. Pending uncommitted work in the tree: commit it as its own commit first (ask if it is not obviously the user's README/provenance edits).

### 2. Decide what the fork still owns

Work out the fork-only delta from the code, not from commit messages: `git diff upstream/main HEAD --stat -- . ':!anthropic_pipe.py' ':!anthropic_pipe.min.py'`. Earlier local commits are often redundant with upstream's later releases under other version numbers.

For each fork-only feature, check whether upstream has caught up:

```bash
git grep -n -c "<distinctive symbol or beta header>" upstream/main -- src helpers
```

Then read how upstream does it before keeping our code. Upstream refactors can make fork code *dead* without a textual conflict: e.g. upstream's item-run replay (`group_item_runs` / `rebuild_run` in `shared/output_items.py`) runs before the message loop and already converts OpenAI-style `tool_calls`, so a fork branch for the same case never fires. Keep only the parts upstream does not cover, delete the rest, and drop the matching `(added in this fork)` tags.

Fork-only features at the time of writing (verify, this list ages): per-message effort that preserves the prompt cache (`supports_mid_conversation_effort`, beta `mid-conversation-output-config-2026-07-01`), Task Budgets (`TASK_BUDGET_TOKENS`, `supports_task_budget`, beta `task-budgets-2026-03-13`; `task_budget` goes out unchanged, never a decremented `remaining`), and `_reconcile_tool_pairs` (tool_use / tool_result pairing safety net and tool-id mapping in `request/messages.py`).

### 3. Merge

```bash
git merge --no-ff --no-commit upstream/main
```

Resolve by hand only in `src/anthropic_pipe/**` and `README.md`. Typical conflicts:

| File | Resolution |
|---|---|
| `pipe_template.py` | `version:` line and the changelog docstring. Take upstream's side, add the fork's entry on top (step 4). |
| `request/messages.py` etc. | Upstream often moves or deletes code the fork edited nearby. Keep only the fork's still-needed additions; the block upstream deleted stays deleted. |
| `README.md` | Version line and "Recommended OpenWebUI": upstream's, with our version number. Changelog: upstream's entries untouched, ours on top. |
| `anthropic_pipe.py`, `anthropic_pipe.min.py` | Do not resolve. Rebuild (below). |

After the sources are clean: `python3 helpers/build_anthropic_pipe.py`.

### 3a. Toggles, companion filter and everything else

These are standalone files users install separately in OpenWebUI. The build does not touch them, so they come through the merge as ordinary files and conflicts are hand-resolved (upstream's side unless the fork edited the same lines on purpose).

For each file in `anthropic_pipe_*_toggle.py` and `anthropic_manifold_companion_filter.py`, plus any filter added upstream:

- Read the upstream diff (`git diff $MB upstream/main -- <file>`), not just the stat. Separate **metadata** changes (description, `version:`) from **code** changes (what the filter writes into `__metadata__`, which models or valves it reads).
- Check that the filter still matches the pipe: the keys a toggle sets (e.g. `anthropic_fast`) must be what `src/anthropic_pipe/request/payload.py` reads, and any model names a filter hard-codes must agree with the capability table in `pipe_template.py`. A pipe change that renames a metadata key or valve needs the filter changed in the same sync.
- Compare the post-merge file to upstream: `git diff upstream/main HEAD -- <file>`. Empty means no fork edits; otherwise list them in the report.
- Record each file's `version:` line before and after (`grep -n '^version:' <files>`).
- Also look over the remaining changed files (`helpers/`, `AGENTS.md`, `.gitignore`, `LICENSE`) for anything that changes how the sync itself works, e.g. a new helper test to run in step 6 or a changed build command.

### 4. Version and changelog

- Keep upstream's entries exactly as published. If upstream shipped a version number the fork also used (it has happened for 0.10.2 and 0.10.3), re-release the fork's work as the **next patch after upstream's current version**, as one combined entry that describes the final behaviour (fold later fixes of the same feature into it rather than listing history).
- Template docstring (`pipe_template.py`): entry untagged. `README.md`: same entry with `### \`vX.Y.Z\` **(added in this fork)**`, plus the "Current pipe version" line.
- Fork-only features/valves in the README Highlights and valve tables carry a bold `**(added in this fork)**` at the end of the cell. A fork-only fix the README body does not mention elsewhere gets a tagged bullet in "Current status".
- Update model lists in these entries when step 5 extends support.

### 5. Gap audit

- New models upstream added: per-model capability flags live in `MODEL_CAPABILITY_OVERRIDES` in `pipe_template.py`. Fork features that gate on a flag stay silently off for a new model until the flag is added. Check the live docs on platform.claude.com for each flag (the bundled `claude-api` skill lags the live docs; do not trust it for prices, versions or support matrices). The `web-fetch` agent can read the pages. Enable only what the docs list.
- Genuinely new Anthropic features that neither side has adopted: report them, and ask before implementing speculative or optional ones (they add real design complexity).

### 6. Verify

The sandbox has no project dependencies. Throwaway venv, versions from the template's `requirements:` line:

```bash
uv venv /tmp/pipe-venv -q
uv pip install -q --python /tmp/pipe-venv/bin/python "anthropic~=2.7" pydantic fastapi cryptography pillow pillow-heif
for t in helpers/test_*.py; do echo "== $t"; /tmp/pipe-venv/bin/python -I $t 2>&1 | tail -8; done
git grep -n -E '^(<<<<<<<|>>>>>>>) ' -- . ':!*.min.py'      # quote the globs: zsh aborts on unmatched ones
python3 helpers/minify_pipe.py anthropic_pipe.py -o /tmp/min_check.py --check && cmp /tmp/min_check.py anthropic_pipe.min.py
```

Baseline: `helpers/test_advisor_pairing.py` fails 3 checks on a clean `upstream/main` as of v0.11.1 (its Opus 5.5 / Fable expectations predate upstream's own advisor-table change). Prove it for any failing test before blaming the merge:

```bash
rm -rf /tmp/up && mkdir /tmp/up && git archive upstream/main | tar -x -C /tmp/up && /tmp/pipe-venv/bin/python -I /tmp/up/helpers/<test>.py
```

Report such failures to the user as upstream's, and leave the test alone. Everything here is offline: no live API call is made, so say that in the summary.

### 7. Commit

Stage `src/`, `README.md` and **both** regenerated artifacts, then commit the merge. Message style: `Merge upstream/main: keep upstream vX, local work becomes vY`, with a body listing what upstream shipped, what the fork kept and what it dropped as redundant.

### 8. Fast-forward main

```bash
git merge-base --is-ancestor main HEAD && git branch -f main HEAD    # local only, no checkout; refuse if main is not an ancestor
```

If `main` diverged from the branch, stop and ask.

### 9. Push (ask first)

Offer: plain `git push` of the branch and `main` to `origin` and `forgejo` (never force). `origin` only moves when the user pushes.

### 10. Report

Short: what upstream shipped, what the fork kept / dropped / extended, the version it is now, test results (with the upstream-baseline failures called out), and what is not verified (live API, live OpenWebUI).

**Always include a "Toggles and filters" section**, every time, one line per file (thinking, web search, code execution, files, fast, companion filter, and any new filter): `unchanged`, or what changed upstream (metadata only vs code, with the version before -> after), and whether the fork edits it. When a filter's description or code changed, add one line telling the user to re-import that file in OpenWebUI, because installed copies do not update from the repo. If a filter had to change because of a pipe change, say which.

End with the push question.
