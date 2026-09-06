"""Короткая долговременная память по узлу + raw-архив попыток.

Идея: graph.json остаётся памятью решений, git branch idea/<node> — canonical
accepted state, а rejected attempts уходят в sidecar. В prompt попадает только
CARD.md, raw derive/verify лежат по ссылке для аудита и ручного rescue.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from .config import ARConfig
from . import graph_hygiene


_MAX_CARD = 1800
_KEEP_RAW_REJECTED = 2


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s or "").strip())
    return s.strip("._-") or "node"


def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def node_dir(cfg: ARConfig, node_id: str) -> str:
    return os.path.join(cfg.workdir, "memory", "nodes", _slug(node_id))


def card_path(cfg: ARConfig, node_id: str) -> str:
    return os.path.join(node_dir(cfg, node_id), "CARD.md")


def attempts_path(cfg: ARConfig, node_id: str) -> str:
    return os.path.join(node_dir(cfg, node_id), "attempts.jsonl")


def tree_ref(lineage_ids: Iterable[str]) -> str:
    parts = [_slug(x) for x in lineage_ids if x]
    if not parts:
        parts = ["root"]
    # Leaf branches live under _node so parent and child nodes never conflict
    # as Git refs (refs cannot be both a file and a directory).
    return "idea-tree/" + "/".join(parts) + "/_node"


def branch_meta(node: Dict[str, Any], branch: str, lineage_ids: List[str],
                round_no: int, status: str = "candidate") -> Dict[str, Any]:
    return {
        "schema": 1,
        "kind": "autoresearch-node-branch",
        "node_id": node.get("id"),
        "node_short": node.get("short", ""),
        "parent": node.get("parent"),
        "lineage_ids": lineage_ids,
        "canonical_branch": branch,
        "tree_branch": tree_ref(lineage_ids),
        "round": round_no,
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def _classify(feedback: str) -> str:
    low = (feedback or "").lower()
    if "local_git_check" in low or "git push failed" in low or "tree branch push failed" in low:
        return "local_git_failure"
    # experiment — ДО code_error: фидбек «exit 0 … нужен реальный LLM-эксперимент» классифицировался
    # как code_error из-за жадной подстроки "exit" (искажал таксономию провалов).
    if "experiment" in low or "реальный llm" in low:
        return "needs_real_experiment"
    if "syntaxerror" in low or "не запускается" in low or "exit code" in low \
            or "exit 1" in low or "traceback" in low:
        return "code_error"
    if "доказ" in low or "proof" in low or "обры" in low:
        return "proof_gap"
    if "duplicate" in low or "дубл" in low:
        return "duplicate"
    if "overclaim" in low or "шире" in low:
        return "overclaim"
    return "gate_reject"


def _summary(text: str, limit: int = 500) -> str:
    text = " ".join((text or "").split())
    return text[:limit]


def _load_attempts(cfg: ARConfig, node_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    p = attempts_path(cfg, node_id)
    if not os.path.exists(p):
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return rows[-limit:]


def _append_attempt(cfg: ARConfig, node_id: str, rec: Dict[str, Any]) -> None:
    os.makedirs(node_dir(cfg, node_id), exist_ok=True)
    with open(attempts_path(cfg, node_id), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def _prune_raw(cfg: ARConfig, node_id: str) -> None:
    rdir = os.path.join(node_dir(cfg, node_id), "rejected")
    if not os.path.isdir(rdir):
        return
    dirs = [os.path.join(rdir, d) for d in os.listdir(rdir)
            if os.path.isdir(os.path.join(rdir, d))]
    dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    for p in dirs[_KEEP_RAW_REJECTED:]:
        shutil.rmtree(p, ignore_errors=True)


def _write_card(cfg: ARConfig, node: Dict[str, Any], lineage_ids: List[str],
                branch: str) -> None:
    nid = node["id"]
    attempts = _load_attempts(cfg, nid, limit=6)
    last = attempts[-1] if attempts else {}
    reject_rows = [a for a in attempts if a.get("status") == "rejected"][-3:]
    lines = [
        f"# {nid}",
        "",
        f"Short: {node.get('short', '')}",
        f"Status: {node.get('status', '')}",
        f"Parent: {node.get('parent', '')}",
        "Lineage: " + (" -> ".join(lineage_ids) if lineage_ids else nid),
        f"Canonical branch: {branch}",
        f"Tree branch: {tree_ref(lineage_ids)}",
        "",
    ]
    if last:
        lines += [
            "## Last Attempt",
            f"- status: {last.get('status')}",
            f"- round: {last.get('round')}",
            f"- failure_class: {last.get('failure_class', '')}",
            f"- gate: {last.get('gate_summary', '')}",
            f"- raw: {last.get('raw_dir', '') or last.get('commit', '')}",
            "",
        ]
    if reject_rows:
        lines.append("## Do Not Repeat")
        seen = set()
        for r in reject_rows:
            cls = r.get("failure_class") or "gate_reject"
            gate = r.get("gate_summary", "")
            key = (cls, gate[:140])
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {cls}: {gate}")
        lines.append("")
    if node.get("last_gate"):
        lines += ["## Current Gate Feedback", _summary(node.get("last_gate", ""), 650), ""]
    exp_confirmed = graph_hygiene.typed_exp_confirmed(node)
    if exp_confirmed:
        lines += ["## Real Experiment", _summary(exp_confirmed, 650), ""]
    lines += [
        "## Prompt Rule",
        "Use this card as compact memory. Do not load raw rejected artifacts unless explicitly needed.",
        "",
    ]
    txt = "\n".join(lines)
    if len(txt) > _MAX_CARD:
        txt = txt[:_MAX_CARD].rstrip() + "\n"
    os.makedirs(node_dir(cfg, nid), exist_ok=True)
    with open(card_path(cfg, nid), "w", encoding="utf-8") as f:
        f.write(txt)


def prompt_block(cfg: ARConfig, node: Dict[str, Any], lineage_ids: List[str],
                 branch: str) -> str:
    """Return small skills-like memory block for prompt, creating CARD.md if needed."""
    _write_card(cfg, node, lineage_ids, branch)
    try:
        txt = open(card_path(cfg, node["id"]), encoding="utf-8").read()
    except OSError:
        return ""
    return "NODE MEMORY CARD (compact; raw attempts stay out of context):\n" + txt[:_MAX_CARD]


def record_rejected(cfg: ARConfig, node: Dict[str, Any], lineage_ids: List[str],
                    branch: str, round_no: int, files: Dict[str, str],
                    feedback: str, push_output: str) -> Dict[str, Any]:
    nid = node["id"]
    aid = f"{_now_id()}-r{round_no}"
    raw = os.path.join(node_dir(cfg, nid), "rejected", aid)
    os.makedirs(raw, exist_ok=True)
    for name, content in files.items():
        if content:
            with open(os.path.join(raw, name), "w", encoding="utf-8") as f:
                f.write(content)
    with open(os.path.join(raw, "gate.txt"), "w", encoding="utf-8") as f:
        f.write(push_output or feedback or "")
    rec = {
        "attempt_id": aid,
        "status": "rejected",
        "round": round_no,
        "branch": branch,
        "tree_branch": tree_ref(lineage_ids),
        "failure_class": _classify(feedback or push_output),
        "gate_summary": _summary(feedback or push_output, 700),
        "raw_dir": raw,
        "ts": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(raw, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    _append_attempt(cfg, nid, rec)
    _prune_raw(cfg, nid)
    _write_card(cfg, node, lineage_ids, branch)
    return rec


def record_pass(cfg: ARConfig, node: Dict[str, Any], lineage_ids: List[str],
                branch: str, round_no: int, commit: str, feedback: str) -> None:
    rec = {
        "attempt_id": f"{_now_id()}-r{round_no}",
        "status": "passed",
        "round": round_no,
        "branch": branch,
        "tree_branch": tree_ref(lineage_ids),
        "commit": commit,
        "gate_summary": _summary(feedback, 700),
        "ts": datetime.now().isoformat(timespec="seconds"),
    }
    _append_attempt(cfg, node["id"], rec)
    _write_card(cfg, node, lineage_ids, branch)
