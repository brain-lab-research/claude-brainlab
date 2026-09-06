"""Шина сообщений между агентами-членами команды (механизм очередей ResearchOS).

Агент не «спавнит-и-умирает», а кладёт запрос в inbox пира (нужна лемма / ревью /
мнение); целевой член команды разбирает, когда дойдёт. Файлы: <workdir>/.run/<agent>-inbox.json
(JSON-список сообщений). Декаплинг — агенты переживают отдельные задачи.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from .config import ARConfig


def _dir(cfg: ARConfig) -> str:
    d = os.path.join(cfg.workdir, ".run")
    os.makedirs(d, exist_ok=True)
    return d


def _path(cfg: ARConfig, agent: str) -> str:
    return os.path.join(_dir(cfg), f"{agent}-inbox.json")


def post(cfg: ARConfig, to_agent: str, msg: Dict[str, Any]) -> None:
    """Положить сообщение в inbox пира. msg: {from, type, node, text, ts?}."""
    p = _path(cfg, to_agent)
    try:
        q = json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        q = []
    q.append(msg)
    json.dump(q, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def pull(cfg: ARConfig, agent: str) -> List[Dict[str, Any]]:
    """Забрать и очистить inbox агента."""
    p = _path(cfg, agent)
    try:
        q = json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        return []
    json.dump([], open(p, "w", encoding="utf-8"))
    return q


def peek(cfg: ARConfig, agent: str) -> List[Dict[str, Any]]:
    try:
        return json.load(open(_path(cfg, agent), encoding="utf-8"))
    except (OSError, ValueError):
        return []


def list_pending(cfg: ARConfig) -> Dict[str, int]:
    d = _dir(cfg)
    out: Dict[str, int] = {}
    for fn in os.listdir(d):
        if fn.endswith("-inbox.json"):
            out[fn[:-len("-inbox.json")]] = len(peek(cfg, fn[:-len("-inbox.json")]))
    return out
