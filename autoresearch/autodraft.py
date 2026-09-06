"""#8 Авто-драфт статьи из выживших узлов (идея из ResearchOS / AI Scientist-v2).

Берёт узлы proven/confirmed_emp/conditional, группирует по ветке-сюжету и генерит
LaTeX-стабы секций (claim + вердикт + открытое + TODO). Скелет для ручной доводки,
не финальный текст. Чистое шаблонирование — тестируемо без LLM.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from .config import ARConfig
from . import graph_hygiene, graph_io as gio

logger = logging.getLogger(__name__)

_STATUS_TAG = {"proven": "доказано", "confirmed_emp": "подтверждено на реальной сети",
               "conditional": "условно"}


def _esc(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}")):
        s = s.replace(a, b)
    return s


def survived_nodes(data: Dict[str, Any], cfg: ARConfig) -> List[Dict[str, Any]]:
    return [n for n in data["nodes"]
            if n.get("status") in cfg.survived_status and n.get("col") not in (0, None)
            and graph_hygiene.report_eligible(n)
            and graph_hygiene.project_policy_eligible(n, cfg)
            and (n.get("status") != "confirmed_emp" or graph_hygiene.is_headline_win(n))]


def build_draft(data: Dict[str, Any], cfg: ARConfig) -> str:
    by = gio.index(data)
    surv = survived_nodes(data, cfg)
    branches: Dict[str, List[Dict[str, Any]]] = {}
    for n in surv:
        chain = gio.ancestors(n["id"], by)
        branch = next((a for a in chain if a.get("col") == 1), None)
        key = branch["id"] if branch else (n["id"] if n.get("col") == 1 else "misc")
        branches.setdefault(key, []).append(n)

    L = [r"% AUTO-DRAFT — стабы секций из выживших узлов графа идей (autoresearch #8).",
         r"% Скелет для ручной доводки, не финальный текст. Источник: IdeaGraph/graph.json.",
         f"% Проект: {cfg.project_name}",
         r"\section{Результаты (авто-сборка)}",
         "",
         _esc(data.get("overall_verdict", "")), ""]
    for bkey, nodes in branches.items():
        L.append(r"\subsection{%s}" % _esc(by.get(bkey, {}).get("short", bkey)))
        for n in nodes:
            tag = _STATUS_TAG.get(n.get("status"), n.get("status", ""))
            L.append(r"\paragraph{%s (%s).}" % (_esc(n.get("short", n["id"])), tag))
            if n.get("idea"):
                L.append(_esc(n["idea"]))
            if n.get("verdict"):
                L.append(r"\emph{Вердикт:} " + _esc(graph_hygiene.safe_verdict(n)))
            if n.get("open"):
                L.append(r"\emph{Открыто:} " + _esc(n["open"]))
            L.append(r"% TODO: развернуть в полную формулировку из IdeaGraph/" + n["id"] + ".md")
            L.append("")
    return "\n".join(L)


def write_draft(data: Dict[str, Any], ts: str, cfg: ARConfig) -> str:
    os.makedirs(cfg.paper, exist_ok=True)
    path = os.path.join(cfg.paper, f"auto-draft-{ts}.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_draft(data, cfg))
    logger.info("auto-draft -> %s", path)
    return path
