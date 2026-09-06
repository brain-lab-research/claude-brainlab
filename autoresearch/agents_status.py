"""Живой статус агентов для UI-дашборда (аватары за столом).

Каждый агент пишет СВОЙ файл <workdir>/.run/agents/<agent>.json — нет гонки при
параллельных воркерах (общий мутируемый JSON её вызывал → агенты пропадали).
Сборщик collect() объединяет все файлы в один agents-status.json для дашборда.
Время — GMT+3 (MSK). Без зависимостей.
"""
from __future__ import annotations

import glob
import json
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from .config import ARConfig

MSK = timezone(timedelta(hours=3))
_SNAP_LOCK = threading.Lock()  # параллельные воркеры/poll пишут общий снапшот → без лока JSON бьётся
STAGES = ("idle", "thinking", "writing", "pushing", "gate", "passed", "rejected",
          "experiment", "waiting_gpu", "reviewing", "scored", "done", "needswork", "failed")


def now_msk() -> str:
    return datetime.now(MSK).strftime("%H:%M:%S MSK")


def _dir(cfg: ARConfig) -> str:
    d = os.path.join(cfg.workdir, ".run", "agents")
    os.makedirs(d, exist_ok=True)
    return d


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def update(cfg: ARConfig, agent: str, *, node: str = "", short: str = "",
           stage: str = "thinking", detail: str = "", engine: str = "",
           role: str = "idea-агент", astar: str = "", ts: str = "") -> None:
    """Атомарно записать СВОЙ файл агента (без чтения общего → без гонки)."""
    detail_limit = 1400 if stage in ("experiment", "waiting_gpu") or "экспериментатор" in role.lower() else 280
    rec = {"agent": agent, "node": node, "short": short, "stage": stage,
           "detail": detail[:detail_limit], "engine": engine, "role": role, "astar": astar,
           "ts": ts or now_msk()}
    p = os.path.join(_dir(cfg), _safe(agent) + ".json")
    tmp = p + ".tmp"
    json.dump(rec, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
    os.replace(tmp, p)


def set_round(cfg: ARConfig, it: int, targets: List[str], ts: str = "") -> None:
    p = os.path.join(_dir(cfg), "_round.json")
    json.dump({"iter": it, "targets": targets, "ts": ts or now_msk()},
              open(p, "w", encoding="utf-8"), ensure_ascii=False)


def report_flag(cfg: ARConfig, active: bool, stage: str = "") -> None:
    """Отметить, что идёт длинный report-cycle (составитель/критик opus, минуты) — для баннера."""
    p = os.path.join(_dir(cfg), "_report.json")
    json.dump({"active": active, "stage": stage, "ts": now_msk()},
              open(p, "w", encoding="utf-8"), ensure_ascii=False)
    try:
        write_snapshot(cfg)
    except Exception:
        pass


def _report(cfg: ARConfig) -> Dict[str, Any]:
    try:
        return json.load(open(os.path.join(_dir(cfg), "_report.json"), encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def record_score(cfg: ARConfig, score: int) -> None:
    """Дописать (время, балл A*-отчёта) в историю — для графика роста на дашборде."""
    p = os.path.join(_dir(cfg), "_scores.json")
    try:
        hist = json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        hist = []
    hist.append({"ts": now_msk(), "score": int(score)})
    json.dump(hist[-100:], open(p, "w", encoding="utf-8"), ensure_ascii=False)


def _scores(cfg: ARConfig) -> List[Dict[str, Any]]:
    try:
        return json.load(open(os.path.join(_dir(cfg), "_scores.json"), encoding="utf-8"))
    except (OSError, ValueError):
        return []


def _graph_stats(cfg: ARConfig) -> Dict[str, int]:
    """Статистика узлов проекта: сколько прошло/не прошло/в работе."""
    try:
        from . import graph_io as gio
        data = gio.load_graph(cfg)
        c: Dict[str, int] = {}
        for n in data["nodes"]:
            if n.get("col") in (0, None):
                continue
            c[n.get("status", "?")] = c.get(n.get("status", "?"), 0) + 1
        from . import graph_hygiene
        confirmed = sum(1 for n in data["nodes"]
                        if n.get("status") == "confirmed_emp"
                        and graph_hygiene.is_headline_win(n))
        passed = c.get("proven", 0) + confirmed + c.get("conditional", 0)
        failed = c.get("rejected", 0) + c.get("refuted_breadth", 0) + c.get("deferred", 0)
        return {"passed": passed, "failed": failed, "open": c.get("open", 0),
                "weak": c.get("weak", 0), "proven": c.get("proven", 0),
                "confirmed": confirmed, "total": sum(c.values())}
    except Exception:
        return {}


def log_event(cfg: ARConfig, who: str, msg: str) -> None:
    """Дописать строку в общую ленту событий (append атомарен → безопасно для воркеров)."""
    line = f"{now_msk()} · {who}: {msg}\n"
    try:
        with open(os.path.join(_dir(cfg), "_log.txt"), "a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass


def _read_log(cfg: ARConfig, n: int = 60) -> List[str]:
    try:
        lines = open(os.path.join(_dir(cfg), "_log.txt"), encoding="utf-8").read().splitlines()
        return lines[-n:]
    except OSError:
        return []


def collect(cfg: ARConfig) -> Dict[str, Any]:
    """Собрать все per-agent файлы в один снапшот для дашборда."""
    d = _dir(cfg)
    agents: List[Dict[str, Any]] = []
    rnd: Dict[str, Any] = {}
    # ВАЖНО: пропускаем ТОЛЬКО реальные служебные файлы. Имена агентов вроде
    # "СОСТАВИТЕЛЬ-отчёта" после _safe() становятся "_____…" (кириллица→_) и раньше
    # ошибочно отсекались как служебные → составитель не показывался.
    service = {"_scores.json", "_report.json"}
    for fp in glob.glob(os.path.join(d, "*.json")):
        base = os.path.basename(fp)
        if base in service:
            continue
        try:
            rec = json.load(open(fp, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(rec, dict):
            continue
        if base == "_round.json":
            rnd = rec
        else:
            rec["_mtime"] = os.path.getmtime(fp)
            agents.append(rec)
    agents.sort(key=lambda a: a.get("_mtime", 0), reverse=True)
    try:
        from . import graph_io as gio
        active_nodes = set(gio.load_active(cfg))
    except Exception:
        active_nodes = set()
    live_active = set(active_nodes)  # СЫРОЙ активный набор (до фолбэка на устаревший _round.json)
    if not active_nodes and isinstance(rnd.get("targets"), list):
        active_nodes.update(str(x) for x in rnd.get("targets", []))
    # Дашборд помечает идейного агента активным ⟺ его node ∈ round.targets. В --continuous режиме
    # set_round НЕ зовётся (он только в run_round) → _round.json застревает на старом раунде и ЖИВЫЕ
    # идейные агенты скрывались (баг «нет агентов»). Берём актуальный active.json (его continuous-
    # воркеры пишут на каждый claim/finish) как истинные targets, iter/ts сохраняем.
    if live_active:
        rnd = dict(rnd)
        rnd["targets"] = sorted(live_active)
        rnd["ts"] = now_msk()
    round_mtime = 0.0
    try:
        round_mtime = os.path.getmtime(os.path.join(d, "_round.json"))
    except OSError:
        pass
    transient = {"thinking", "writing", "pushing", "gate", "reviewing"}
    cleaned: List[Dict[str, Any]] = []
    for a in agents:
        agent = str(a.get("agent") or "")
        node = str(a.get("node") or "")
        stage = str(a.get("stage") or "")
        if agent == "GPU-lane":
            continue
        if agent.startswith("agent-") and node and node not in active_nodes and stage in transient:
            continue
        if agent.startswith("idea-rest-") and round_mtime and a.get("_mtime", 0) < round_mtime:
            continue
        cleaned.append(a)
    agents = cleaned
    # ПРУНИНГ: дашборд зарастал сотней старых карточек (failed/passed) → живой эксперимент тонул.
    # Показываем только СВЕЖИЕ (≤30мин) + ЖИВЫЕ эксперименты/ожидание GPU всегда, и капим до 40.
    now = time.time()
    FRESH_S = 1800
    live_stages = ("experiment", "waiting_gpu")
    agents = [a for a in agents
              if (now - a.get("_mtime", 0)) < FRESH_S or a.get("stage") in live_stages][:40]
    scores = _scores(cfg)
    return {"agents": agents, "round": rnd, "log": list(reversed(_read_log(cfg))),
            "scores": scores, "last_astar": (scores[-1]["score"] if scores else None),
            "stats": _graph_stats(cfg), "report": _report(cfg),
            "experiments": _exp_counter(cfg), "updated": now_msk()}


def _exp_counter(cfg: ARConfig) -> Dict[str, int]:
    """Счётчик проведённых GPU-экспериментов (пишет empirical) — для плитки дашборда."""
    try:
        return json.load(open(os.path.join(cfg.workdir, ".run", "exp_counter.json"), encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_snapshot(cfg: ARConfig) -> str:
    """Записать собранный снапшот в <workdir>/.run/agents-status.json (его синкаем/отдаём)."""
    snap = collect(cfg)
    p = os.path.join(cfg.workdir, ".run", "agents-status.json")
    # уникальный tmp на поток + лок → нет гонки/склейки JSON при параллельных писателях
    tmp = p + f".{os.getpid()}.{threading.get_ident()}.tmp"
    with _SNAP_LOCK:
        json.dump(snap, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        os.replace(tmp, p)
    return p
