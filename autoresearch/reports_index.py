"""Индекс отчётов по узлам (тег = линии/узлы, которые отчёт покрыл).

Зачем: когда идейный агент берёт узел из линии, по которой УЖЕ был отчёт, ему в память
подмешивается КОМПАКТНЫЙ указатель на тот отчёт (балл A* + ключевые претензии + какие узлы
критик признал слабыми). Так агент сверяется с прошлым результатом, а не повторяет уже
отвергнутое. Токен-бюджет: только указатель (~250 токенов), НЕ весь отчёт.

Хранится в <reports_dir>/report_index.json (durable, дописывается на каждом report-cycle).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .config import ARConfig
from . import graph_hygiene, graph_io as gio
from .reporter import _reports_dir


def _path(cfg: ARConfig) -> str:
    return os.path.join(_reports_dir(cfg), "report_index.json")


def _load(cfg: ARConfig) -> List[Dict[str, Any]]:
    try:
        idx = json.load(open(_path(cfg), encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(idx, list):
        return []
    return [_scrub_entry(cfg, e) for e in idx if isinstance(e, dict)]


def _scrub_entry(cfg: ARConfig, entry: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(entry)
    if out.get("summary"):
        out["summary"] = graph_hygiene.scrub_project_policy_text(str(out.get("summary") or ""), cfg)
    tags = []
    for tag in out.get("tags", []) or []:
        clean = graph_hygiene.scrub_project_policy_text(str(tag), cfg)
        if clean and not clean.startswith("[policy-filter]"):
            tags.append(clean)
    out["tags"] = tags
    return out


def add(cfg: ARConfig, entry: Dict[str, Any]) -> None:
    """Дописать запись об отчёте (тег + покрытые узлы + балл + слабые + краткая суть)."""
    idx = _load(cfg)
    idx.append(_scrub_entry(cfg, entry))
    os.makedirs(os.path.dirname(_path(cfg)), exist_ok=True)
    tmp = _path(cfg) + ".tmp"
    json.dump(idx[-50:], open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, _path(cfg))


def relevant(cfg: ARConfig, nid: str, by: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Самый свежий отчёт, чей тег покрывает этот узел (узел сам или его предок-ветка)."""
    anc = {a["id"] for a in gio.ancestors(nid, by)} | {nid}
    best = None
    for e in _load(cfg):
        if anc & set(e.get("node_ids", [])):
            best = e  # перебор по порядку → последняя совпавшая = самая свежая
    return best


def pointer(entry: Dict[str, Any], nid: str) -> str:
    """Компактный блок для подмешивания в память агента (бюджет ~250 токенов)."""
    weak = entry.get("weak", [])
    flagged = " ⚠️ ЭТОТ УЗЕЛ критик пометил СЛАБЫМ — переосмысли или закрой претензию." \
        if nid in weak else ""
    tags = ", ".join(entry.get("tags", [])[:4]) or "—"
    return (f"СВЕРЬСЯ С ПРОШЛЫМ ОТЧЁТОМ по этой линии (тег: {tags}; A*={entry.get('score','?')}/10; "
            f"{entry.get('ts','')}). Не повторяй уже отвергнутое, закрывай претензии:\n"
            f"{(entry.get('summary') or '')[:360]}{flagged}")
