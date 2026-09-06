"""Единый НАРРАТИВ проекта — центральная история, в которую складываются кирпичи.

Зачем: чтобы идейные агенты работали в ОДНОМ русле (а не плодили разрозненные леммы),
а критик гейта судил, вписывается ли узел в общую историю и не дубль ли он. Это «дом»,
который собирается из кирпичей-узлов.

Хранится в <project>/IdeaGraph/narrative.md (durable, синкается). Если нет — генерится
детерминированно из графа (root_idea + overall_verdict + доказанные результаты).
Обновляется на report-cycle (A*-критик отчёта видит, как складывается история).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

from .config import ARConfig
from . import graph_hygiene, graph_io as gio


def _path(cfg: ARConfig) -> str:
    return os.path.join(cfg.gdir, "narrative.md")


def get(cfg: ARConfig) -> str:
    try:
        return open(_path(cfg), encoding="utf-8").read().strip()
    except OSError:
        return ""


def write(cfg: ARConfig, text: str) -> str:
    os.makedirs(cfg.gdir, exist_ok=True)
    open(_path(cfg), "w", encoding="utf-8").write(text.rstrip() + "\n")
    return _path(cfg)


def ensure(cfg: ARConfig, data: Dict[str, Any]) -> str:
    """Вернуть нарратив; если файла нет — собрать стартовый из графа."""
    cur = get(cfg)
    if cur:
        return cur
    return write(cfg, _from_graph(data))


def _from_graph(data: Dict[str, Any]) -> str:
    proven = [n for n in data["nodes"] if graph_hygiene.is_headline_win(n)]
    branches = [n for n in data["nodes"] if n.get("col") == 1]
    L = ["# Нарратив проекта (единая история — собираем кирпичи в неё)", "",
         "## Центральный тезис", (data.get("root_idea") or "").strip(), "",
         (data.get("overall_verdict") or "").strip(), "",
         "## Сюжетные линии (ветки)"]
    for b in branches:
        L.append(f"- {b.get('short','')}: {(b.get('idea') or '')[:140]}")
    L += ["", "## Уже доказанные кирпичи (опора)"]
    for n in proven[:15]:
        L.append(f"- {n.get('short','')}: {(graph_hygiene.safe_verdict(n) or n.get('idea',''))[:140]}")
    L += ["", "## Куда движемся",
          "Каждый новый узел должен УСИЛИВАТЬ эту историю: либо закрывать открытый вопрос "
          "линии, либо связывать линии, либо укреплять центральный тезис. Не разрозненные "
          "леммы, а кирпичи одного дома."]
    return "\n".join(x for x in L if x is not None)


def boss(cfg: ARConfig) -> str:
    """Жёсткие требования научного руководителя — всегда учитываются критиком/составителем."""
    try:
        return open(os.path.join(cfg.gdir, "boss_critique.md"), encoding="utf-8").read().strip()
    except OSError:
        return ""


def node_map(data: Dict[str, Any], limit: int = 120) -> str:
    """Компактная карта существующих узлов (для проверки дубля критиком)."""
    out = []
    for n in data["nodes"]:
        if n.get("col") in (0, None):
            continue
        out.append(f"[{n['id']}|{n.get('status','')}] {n.get('short','')}")
        if len(out) >= limit:
            break
    return "\n".join(out)
