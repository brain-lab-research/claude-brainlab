"""Graph hygiene helpers for empirical evidence.

Typed EXP_SPEC results are the only empirical results that may promote a node to
needs_writeup/confirmed_emp.  Older untyped llama/D-Muon payloads are kept for
audit, but they must not enter prompts or report headline selection as evidence.
"""
from __future__ import annotations

import re
from typing import Any, Dict


QUARANTINE_REASON = (
    "legacy untyped empirical result quarantined; rerun this claim through typed "
    "EXP_SPEC before using it as real-network evidence"
)

LEGACY_CHECKPOINT_TEXT_RE = re.compile(
    r"(?:EleutherAI/)?pythia-\d+(?:\.\d+)?[a-z]?|\bPythia\b|\b(?:160m|410m)\b",
    re.IGNORECASE,
)


def has_typed_exp(node: Dict[str, Any]) -> bool:
    return bool(node.get("exp_spec"))


def is_policy_blocked(node: Dict[str, Any]) -> bool:
    return bool(node.get("policy_blocked_exp") or node.get("mechanical_invalid_exp"))


def report_eligible(node: Dict[str, Any]) -> bool:
    # report_excluded: узел, который A*-критик многократно называл weak/диллютящим ТЕКУЩИЙ headline
    # (но он proven/conditional → prune не трогает статус). Убираем из МАТЕРИАЛА отчёта, статус цел.
    if node.get("report_excluded"):
        return False
    return not is_policy_blocked(node)


def mentions_legacy_checkpoint(node: Dict[str, Any]) -> bool:
    text = "\n".join(str(node.get(k) or "") for k in (
        "id", "short", "idea", "open", "plan", "verdict", "last_gate",
        "exp_confirmed", "exp_refuted", "exp_confirmed_summary",
        "exp_refuted_summary",
    ))
    return bool(LEGACY_CHECKPOINT_TEXT_RE.search(text))


def project_policy_eligible(node: Dict[str, Any], cfg: Any) -> bool:
    if str(getattr(cfg, "exp_kind", "") or "").strip().lower() == "llama":
        return not mentions_legacy_checkpoint(node)
    return True


def scrub_project_policy_text(text: str, cfg: Any) -> str:
    if str(getattr(cfg, "exp_kind", "") or "").strip().lower() != "llama":
        return text or ""
    kept = []
    dropped = 0
    for line in (text or "").splitlines():
        if LEGACY_CHECKPOINT_TEXT_RE.search(line):
            dropped += 1
            continue
        kept.append(line)
    if dropped:
        kept.insert(0, (
            "[policy-filter] legacy checkpoint-size/model directives were omitted "
            "under exp_kind=llama; use llama-style evidence only."
        ))
    return "\n".join(kept).strip()


def typed_exp_confirmed(node: Dict[str, Any]) -> str:
    if is_policy_blocked(node):
        return ""
    if has_typed_exp(node):
        return str(node.get("exp_confirmed") or "")
    return ""


def has_untyped_exp_evidence(node: Dict[str, Any]) -> bool:
    return bool(node.get("exp_confirmed")) and not has_typed_exp(node)


def is_quarantined_untyped(node: Dict[str, Any]) -> bool:
    return bool(node.get("empirical_quarantine")) and not has_typed_exp(node)


def is_typed_writeup(node: Dict[str, Any]) -> bool:
    return (
        node.get("status") == "needs_writeup"
        and has_typed_exp(node)
        and not is_policy_blocked(node)
        and bool(node.get("exp_confirmed"))
    )


def writeup_attempts(node: Dict[str, Any]) -> int:
    return int(node.get("writeup_attempts", 0) or 0)


def needs_writeup_priority(node: Dict[str, Any]) -> bool:
    return is_typed_writeup(node) and writeup_attempts(node) < 2


def writeup_exhausted(node: Dict[str, Any]) -> bool:
    return (
        node.get("status") in ("needs_writeup", "conditional")
        and has_typed_exp(node)
        and bool(node.get("exp_confirmed"))
        and writeup_attempts(node) >= 2
    )


def is_headline_win(node: Dict[str, Any]) -> bool:
    if is_policy_blocked(node):
        return False
    status = node.get("status")
    if status == "proven":
        return True
    if status != "confirmed_emp":
        return False
    return has_typed_exp(node) and not has_untyped_exp_evidence(node)


def safe_verdict(node: Dict[str, Any]) -> str:
    verdict = str(node.get("verdict") or "")
    if not node.get("empirical_quarantine"):
        return verdict
    markers = (
        "реальный llm",
        "llm-эксперимент",
        "эксп подтверд",
        "эксперимент подтверд",
        "реальными эксп",
        "confirmed_emp",
        "ci∌0",
    )
    parts = []
    for part in re.split(r"\s+\|\s+", verdict):
        low = part.lower()
        if any(marker in low for marker in markers):
            continue
        parts.append(part)
    kept = " | ".join(p for p in parts if p.strip()).strip()
    note = "legacy untyped empirical claims quarantined; ignore old real-network evidence"
    return f"{kept} | {note}" if kept else note


def scrub_legacy_empirical(data: Dict[str, Any]) -> int:
    """Move untyped empirical payloads out of active evidence fields.

    Returns the number of node mutations.  The old text is preserved in
    legacy_exp_confirmed for audit/rescue, but exp_confirmed is removed so
    prompts/reports cannot treat it as current evidence.
    """
    changed = 0
    for node in data.get("nodes", []):
        status = node.get("status")
        legacy = has_untyped_exp_evidence(node)
        untyped_confirmed = status == "confirmed_emp" and not has_typed_exp(node)
        invalid_writeup = status == "needs_writeup" and not has_typed_exp(node)
        if not legacy and not untyped_confirmed and not invalid_writeup:
            continue

        old_status = status
        if legacy:
            node["legacy_exp_confirmed"] = node.pop("exp_confirmed")
            changed += 1

        node["empirical_quarantine"] = {
            "kind": "legacy_untyped_empirical",
            "reason": QUARANTINE_REASON,
            "old_status": old_status,
        }

        if old_status in ("confirmed_emp", "needs_writeup", "experiment_running"):
            node["status"] = "conditional"
            changed += 1

        last_gate = str(node.get("last_gate") or "")
        low = last_gate.lower()
        if "реальный эксперимент" in low or "llm-эксперимент" in low or "llm-experiment" in low:
            node["last_gate"] = QUARANTINE_REASON
            changed += 1
    return changed
