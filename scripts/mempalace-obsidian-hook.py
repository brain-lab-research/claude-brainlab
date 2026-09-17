#!/usr/bin/env python3
"""Stop hook: one interruption, three places the session gets written down.

Replaces `python3 -m mempalace hook run --hook stop --harness claude-code`.

MemPalace blocks the end of a turn every few exchanges and asks the agent to save what
happened. That design is the reason it works: the hook does not try to understand the
session, it interrupts and demands, and the agent — which has the whole context — decides
what is worth keeping and says so out loud.

Obsidian and the lab knowledge base ride the same interruption instead of adding their
own. Two hooks blocking on two cadences would double the noise, and a second mechanism
that writes on its own turned out not to work at all: the previous lab hook parsed the
session for markup nobody was documented to write, so in real use it never fired.

WHY THIS COUNTS ITS OWN EXCHANGES

The interval is meant to be "every N messages from the human". The upstream counter takes
every entry with role=user, and in an agentic session almost all of those are tool
results: in one long session here, 5283 such entries were 4722 tool results, 177 hook
replies and only 384 real messages. At an interval of ten that fired 552 times instead of
38, and each extra turn re-reads the whole context — 8.6% of the session's output tokens
and 8.0% of its cache reads went to saving. So the counting happens here, over genuine
turns only, and the interval means what it says.

Saving before compaction is the other obvious idea and it does not work: a blocking
PreCompact hook cancels the compaction instead of deferring it. The cadence stays on Stop.

The lab section appears only when a lab-knowledge MCP server is configured, so an install
without lab access is not nagged about an unconfigured base. Configuration is not a
reachability check. Checkpoint markers track requests only; the agent must read back writes.
"""

import json
import sys
from pathlib import Path

import mempalace.hooks_cli as hooks_cli

SETTINGS = Path.home() / ".claude" / "settings.json"
CODEX_CONFIG = Path.home() / ".codex" / "config.toml"

#: Настоящих сообщений человека между сохранениями.
SAVE_INTERVAL = 10

OBSIDIAN_ADDENDUM = """
4. obsidian — preserve durable results in the existing mapped project notes:
   Resolve cwd through ~/.Codex/obsidian-projects.json (or the client's configured mapping).
   Reuse canonical files and current vault conventions, including project-local tooling notes.
   Keep private drafts private. Do not invent a general/ destination for a mapped project.
   Save the protocol, results, decisions and remaining questions, with useful artifact links.
   Read the written note back and report its path. Skip if nothing durable changed.
"""

LAB_ADDENDUM = """
5. lab knowledge — save meaningful shared findings within the user's existing authorization:
   Load lab-knowledge and its references/record-contract.md, then read the live tool schema.
   Resolve the existing project by stable ID and aliases; never guess a project or its board.
   Hypotheses carry falsification_criteria and their actual empirical/theoretical kind.
   Experiments preserve actual run state; evidence carries metrics, sources and limitations.
   Mathematical arguments use derivations. Decisions preserve operational/scientific kind
   and established support. Literature claims may be paraphrased; quotations are optional
   source material, not a requirement to manufacture text or modify the original source.
   If the server still requires a quote, keep that claim pending and report the incompatibility.
   Operational observations and resources use their own types, not invented hypotheses.
   Personal tasks stay in Operon. Shared tasks use the bound Yonote board; a Hermes board
   projection must keep the original task ID and one authoritative status source.
   Search before writing, preserve unrelated data, and reuse stable idempotency keys.
   Read back each saved object and report its ID. Report partial failures as pending;
   a checkpoint request or a local note is not proof of a successful MCP write.
   Skip shared writes when nothing qualifies or publication was not authorized.
"""


def lab_base_configured(harness: str = "claude-code", cwd: str = "") -> bool:
    """Check this client's configuration, without reading or printing credentials."""
    if harness == "codex":
        try:
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib
            config = tomllib.loads(CODEX_CONFIG.read_text(encoding="utf-8"))
        except (ImportError, OSError, ValueError):
            return False
        server = config.get("mcp_servers", {}).get("lab-knowledge")
        return isinstance(server, dict) and server.get("enabled", True) is not False
    paths = [SETTINGS, Path.home() / ".claude.json"]
    if cwd:
        paths.append(Path(cwd) / ".mcp.json")
    for path in paths:
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(config, dict):
            continue
        server = config.get("mcpServers", {}).get("lab-knowledge")
        if isinstance(server, dict) and server.get("disabled", False) is not True:
            return True
    return False


def transcript_harness(transcript_path: str) -> str:
    path = Path(transcript_path).expanduser()
    if path.is_file():
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if isinstance(entry, dict):
                    if entry.get("type") in ("session_meta", "response_item"):
                        return "codex"
                    if isinstance(entry.get("message"), dict):
                        return "claude-code"
    return "claude-code"


def _text_of(content: object) -> str:
    """Plain text of a message, whatever shape the harness wrote it in."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") != "tool_result"
        )
    return ""


def human_turns(transcript_path: str) -> int:
    """How many times the person actually said something.

    Tool results and the hook's own reminders wear role=user too, and counting them is what
    turned an interval of ten into a save every other tool call.
    """
    path = Path(transcript_path).expanduser()
    if not path.is_file():
        return 0
    count = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue
            message = entry.get("message")
            if entry.get("type") == "response_item":
                message = entry.get("payload")
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, list) and any(
                isinstance(block, dict) and block.get("type") == "tool_result"
                for block in content
            ):
                continue
            text = _text_of(content)
            if text.lstrip().startswith(("<environment_context>", "<turn_aborted>",
                                         "# AGENTS.md instructions", "<permissions instructions>")):
                continue
            if "<command-message>" in text:
                continue
            if "AUTO-SAVE checkpoint" in text or "Stop hook feedback" in text:
                continue
            count += 1
    return count


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    parsed = hooks_cli._parse_harness_input(data, "claude-code")  # noqa: SLF001

    # Уже внутри цикла сохранения: пропустить, иначе получится петля.
    if str(parsed["stop_hook_active"]).lower() in ("true", "1", "yes"):
        print(json.dumps({}))
        return

    harness = transcript_harness(parsed["transcript_path"])
    turns = human_turns(parsed["transcript_path"])
    state_dir = hooks_cli.STATE_DIR
    state_dir.mkdir(parents=True, exist_ok=True)
    marker = state_dir / f"{parsed['session_id']}_last_checkpoint_turns"
    legacy = state_dir / f"{parsed['session_id']}_last_save_turns"
    try:
        last = int((marker if marker.exists() else legacy).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        last = 0

    if turns - last < SAVE_INTERVAL or turns <= 0:
        print(json.dumps({}))
        return

    try:
        marker.write_text(str(turns), encoding="utf-8")
    except OSError:
        pass
    hooks_cli._maybe_auto_ingest()  # noqa: SLF001 — сохраняем поведение обёртки

    reason = hooks_cli.STOP_BLOCK_REASON.rstrip() + OBSIDIAN_ADDENDUM
    if lab_base_configured(harness, str(data.get("cwd") or "")):
        reason = reason.rstrip() + "\n" + LAB_ADDENDUM
    print(json.dumps({"decision": "block", "reason": reason}))


if __name__ == "__main__":
    main()
