"""Единый слой чтения/записи graph.json + active.json и перерисовки канваса.

Все модули autoresearch ходят в граф ТОЛЬКО через этот модуль. cfg обязателен
(никаких глобалов/хардкода путей) — его собирает config.load(). Новые поля узла
добавляются лениво; старый граф читается без миграции.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

from .config import ARConfig

logger = logging.getLogger(__name__)

NODE_DEFAULTS: Dict[str, Any] = {
    "elo": None,            # None -> трактуется как cfg.elo.start
    "attempts": 0,
    "writeup_attempts": 0,
    "scoop": None,          # {"verdict", "refs", "ts"}
    "failure_class": None,
}


def load_graph(cfg: ARConfig) -> Dict[str, Any]:
    if getattr(cfg, "store", "graph") == "koi":
        from . import koi_store
        return koi_store.load_graph_like(cfg.koi_dir, cfg)
    with open(cfg.graph, encoding="utf-8") as f:
        return json.load(f)


def save_graph(data: Dict[str, Any], cfg: ARConfig, backup: bool = True) -> None:
    if getattr(cfg, "store", "graph") == "koi":
        from . import koi_store
        koi_store.save_graph_like(data, cfg.koi_dir, cfg)
        logger.info("koi store saved: %d nodes -> %s", len(data.get("nodes", [])), cfg.koi_dir)
        return
    path = cfg.graph
    if backup and os.path.exists(path):
        shutil.copy2(path, path + ".bak")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    logger.info("graph saved: %d nodes -> %s", len(data.get("nodes", [])), path)


def load_active(cfg: ARConfig) -> List[str]:
    try:
        with open(cfg.active, encoding="utf-8") as f:
            payload = json.load(f)
            if isinstance(payload, dict):
                return list(payload.get("running", []))
            if isinstance(payload, list):
                return list(payload)
            return []
    except (OSError, ValueError):
        return []


def save_active(ids: List[str], cfg: ARConfig, detail: str = "",
                last_verdict: str = "") -> None:
    payload: Dict[str, Any] = {"running": list(ids)}
    if detail:
        payload["detail"] = detail
    if last_verdict:
        payload["last_verdict"] = last_verdict
    with open(cfg.active, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)


def index(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {n["id"]: n for n in data["nodes"]}


def node_text(node: Dict[str, Any]) -> str:
    return " ".join(str(node.get(k, "")) for k in ("short", "idea", "open", "plan")).strip()


def get_elo(node: Dict[str, Any], cfg: ARConfig) -> float:
    v = node.get("elo")
    return cfg.elo.start if v is None else float(v)


def ancestors(node_id: str, by: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    chain, p = [], by.get(node_id, {}).get("parent")
    seen = {node_id}
    while p and p in by and p not in seen:
        seen.add(p)
        chain.append(by[p])
        p = by[p].get("parent")
    return chain


def ensure_fields(data: Dict[str, Any]) -> int:
    changed = 0
    for n in data["nodes"]:
        for k, v in NODE_DEFAULTS.items():
            if k not in n:
                n[k] = v
                changed += 1
    try:
        from . import graph_hygiene
        changed += graph_hygiene.scrub_legacy_empirical(data)
    except Exception as e:
        logger.warning("graph hygiene skipped: %s", e)
    return changed


_LAST_CANVAS_REGEN = [0.0]


def regen_canvas(cfg: ARConfig, force: bool = False) -> bool:
    # ТРОТТЛ: canvas — косметика дашборда, но зовётся на КАЖДУЮ правку графа (десятки раз/мин от
    # воркеров) под общим glock → contention, главный поток (report) голодает. Не чаще раз в 15с.
    # + timeout: subprocess без него мог ПОВИСНУТЬ навсегда (iCloud-I/O на canvas-файле) и заморозить
    # держателя glock. force=True обходит троттл (для явных ключевых моментов).
    now = time.time()
    if not force and now - _LAST_CANVAS_REGEN[0] < 15.0:
        return True
    script = cfg.canvas_script
    if not os.path.exists(script):
        logger.warning("canvas script not found: %s", script)
        return False
    try:
        r = subprocess.run(["python3", script], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        logger.warning("canvas regen timed out (>60s) — пропуск, не вешаю поток")
        _LAST_CANVAS_REGEN[0] = now
        return False
    _LAST_CANVAS_REGEN[0] = now
    if r.returncode != 0:
        logger.error("canvas regen failed: %s", r.stderr.strip())
        return False
    logger.info("canvas regenerated: %s", r.stdout.strip())
    return True


def add_node(data: Dict[str, Any], node: Dict[str, Any]) -> None:
    data["nodes"].append({**NODE_DEFAULTS, **node})


def add_xref(data: Dict[str, Any], src: str, dst: str, rel: str) -> None:
    data.setdefault("xref", []).append({"from": src, "to": dst, "type": rel})
