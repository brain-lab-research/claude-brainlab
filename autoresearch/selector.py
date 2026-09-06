"""#2 Авто-выбор узла — Experiment-Manager (идея из AIDE / AI Scientist-v2).

Политика скорит узлы фронтира и берёт top-k для следующего раунда (полу-UCT):
эксплуатация (Elo + перспективный предок) против исследования (богатство открытых
вопросов) и штрафа за число потраченных попыток. Чистая функция — тестируема без LLM.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

from .config import ARConfig
from . import graph_hygiene, graph_io as gio

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Score:
    node_id: str
    short: str
    total: float
    elo_term: float
    open_term: float
    promise_term: float
    attempts_term: float
    pressure_term: float = 0.0

    def explain(self) -> str:
        return (f"[{self.node_id}] {self.short[:48]} -> {self.total:+.2f} "
                f"(elo {self.elo_term:+.2f}, open {self.open_term:+.2f}, "
                f"promise {self.promise_term:+.2f}, attempts {self.attempts_term:+.2f}"
                + (f", pressure {self.pressure_term:+.2f}" if self.pressure_term else "")
                + ")")


def _open_richness(node: Dict[str, Any], cfg: ARConfig) -> float:
    toks = len((node.get("open") or "").split())
    return min(toks / cfg.selector.open_token_norm, 1.0)


def _promise(node: Dict[str, Any], by: Dict[str, Dict[str, Any]], cfg: ARConfig) -> float:
    chain = gio.ancestors(node["id"], by)
    if not chain:
        return cfg.selector.status_promise.get(node.get("status", "open"), 0.5)
    return max(cfg.selector.status_promise.get(a.get("status", "open"), 0.0) for a in chain)


def _latest_pressure_text(cfg: ARConfig) -> str:
    fb = os.path.join(cfg.workdir, "astar_feedback.md")
    try:
        text = open(fb, encoding="utf-8", errors="replace").read()
        if text.strip():
            return graph_hygiene.scrub_project_policy_text(text, cfg)
    except OSError:
        pass
    try:
        rd = os.path.join(cfg.project_root, "Reports")
        paths = [os.path.join(rd, f) for f in os.listdir(rd) if f.endswith(".review.md")]
        if not paths:
            return ""
        path = max(paths, key=os.path.getmtime)
        return graph_hygiene.scrub_project_policy_text(
            open(path, encoding="utf-8", errors="replace").read(), cfg)
    except OSError:
        return ""


def pressure_weak_nodes(cfg: ARConfig) -> List[str]:
    """Weak/extra nodes named by the latest no-cornerstone sub-A* report.

    These are report-critic labels, not necessarily work targets.  In
    particular, A* feedback often names nodes that should be pruned from the
    core story.  Use pressure_action_targets() when selecting new work.
    """
    text = _latest_pressure_text(cfg)
    if not text:
        return []
    score = None
    for m in re.finditer(r"(?im)^\s*SCORE\s*:?\s*(\d+)", text):
        try:
            score = int(m.group(1))
        except ValueError:
            pass
    if score is None or score >= cfg.report_min_score:
        return []  # порог pressure = порог публикации (min_score) → нет «мёртвой зоны» между ними
    cornerstones = [m.group(1).strip() for m in re.finditer(
        r"(?im)^\s*CORNERSTONE\s*:?\s*(.+?)\s*$", text)]
    if cornerstones:
        latest = cornerstones[-1].lower()
        if latest and not latest.startswith(("нет", "none", "no", "n/a", "-")):
            return []
    weak = ""
    for m in re.finditer(r"(?im)^\s*WEAK_NODES\s*:?\s*(.+?)\s*$", text):
        weak = m.group(1)
    if not weak:
        return []
    ids = []
    for part in re.split(r"[,;\s]+", weak):
        nid = part.strip().strip("`'\".()[]{}")
        if re.match(r"^[A-Za-z0-9_.-]+$", nid):
            ids.append(nid)
    return list(dict.fromkeys(ids))


def pressure_prune_nodes(cfg: ARConfig) -> List[str]:
    """Weak nodes the latest A* report explicitly says to drop from the story."""
    text = _latest_pressure_text(cfg)
    if not text:
        return []
    weak = pressure_weak_nodes(cfg)
    if not weak:
        return []
    prune_markers = [
        r"выкинуть\s*/\s*не\s+тащить",
        r"не\s+тащить\s+в\s+core\s+story",
        r"слабые\s*/\s*лишние\s+узлы[^\n]{0,80}выкинуть",
        r"лишние\s+узлы[^\n]{0,80}выкинуть",
        r"выкинуть\s+неподдерживающие\s+ветки",
        r"\bвыкинуть\b",
        r"убрать\s+loose\s+bridges",
        r"remove\s+loose\s+bridges",
        r"drop\s+loose\s+bridges",
        r"\bscope[- ]reduc",
        r"\bprun(?:e|ing)\b",
    ]
    if any(re.search(p, text, re.IGNORECASE) for p in prune_markers):
        return weak
    return []


def pressure_action_targets(cfg: ARConfig) -> List[str]:
    """Concrete pressure targets for workers to act on.

    WEAK_NODES can mean "remove these from the core story".  Routing workers to
    pressure_<weak> in that case makes the loop do the opposite of the review.
    Collapse prune-only feedback into one story/scope pivot target instead.
    """
    weak = pressure_weak_nodes(cfg)
    if not weak:
        return []
    if pressure_prune_nodes(cfg):
        return ["scope_pivot"]
    return weak


def _pressure_target(node: Dict[str, Any]) -> str:
    nid = str(node.get("id") or "")
    if not nid.startswith("pressure_"):
        return ""
    return str(node.get("pressure_target") or nid[len("pressure_"):]).strip()


def _latest_report_mtime(cfg: ARConfig) -> float:
    try:
        rd = os.path.join(cfg.project_root, "Reports")
        paths = [os.path.join(rd, f) for f in os.listdir(rd) if f.endswith(".review.md")]
        return max((os.path.getmtime(p) for p in paths), default=0.0)
    except OSError:
        return 0.0


def _pressure_latest_pass_mtime(node: Dict[str, Any], cfg: ARConfig) -> float:
    nid = str(node.get("id") or "")
    if not nid:
        return 0.0
    best = 0.0
    ts = str(node.get("last_pass_ts") or "")
    if ts:
        try:
            best = max(best, datetime.fromisoformat(ts).timestamp())
        except ValueError:
            pass
    path = os.path.join(cfg.workdir, "memory", "nodes", nid, "attempts.jsonl")
    try:
        for line in open(path, encoding="utf-8", errors="replace"):
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("status") != "passed":
                continue
            ts = str(rec.get("ts") or "")
            try:
                best = max(best, datetime.fromisoformat(ts).timestamp())
            except ValueError:
                pass
    except OSError:
        pass
    card = os.path.join(cfg.workdir, ".run", "agents", f"agent-{nid}.json")
    try:
        rec = json.load(open(card, encoding="utf-8", errors="replace"))
        if rec.get("stage") == "passed":
            best = max(best, os.path.getmtime(card))
    except (OSError, ValueError, TypeError):
        pass
    return best


def pressure_node_satisfied(node: Dict[str, Any], weak_nodes: set[str],
                            cfg: ARConfig | None = None) -> bool:
    target = _pressure_target(node)
    if not target or target not in weak_nodes:
        return False
    if node.get("status") not in ("conditional", "proven"):
        return False
    if node.get("exp_spec") or node.get("exp_refuted"):
        return False
    verdict = str(node.get("verdict") or "")
    last_pass = str(node.get("last_pass") or "")
    if "agent push PASS" not in verdict and "PASS" not in last_pass:
        return False
    if cfg is not None:
        latest_report = _latest_report_mtime(cfg)
        if latest_report and _pressure_latest_pass_mtime(node, cfg) <= latest_report:
            return False
    return True


def pressure_unresolved_targets(data: Dict[str, Any], cfg: ARConfig) -> List[str]:
    targets = set(pressure_action_targets(cfg))
    if not targets:
        return []
    terminal = set(cfg.selector.terminal)
    done = set()
    alive_for_target: Dict[str, bool] = {}  # target -> есть ли хоть один ЧИНИБЕЛЬНЫЙ pressure-узел
    for n in data.get("nodes", []):
        t = _pressure_target(n)
        if not t:
            continue
        if pressure_node_satisfied(n, targets, cfg):
            done.add(t)
        # pressure-узел с ПОДТВЕРЖДЁННЫМ typed-экспериментом сделал свою работу → target закрыт.
        # (pressure_node_satisfied жёстко отсекает любой exp_spec-узел → без этого confirmed-путь
        #  вечно держал бы target «unresolved» и воркеры зависали в rest — supervisor livelock.)
        if n.get("exp_confirmed") and not n.get("exp_refuted"):
            done.add(t)
        # МЁРТВЫЙ pressure-узел (эксперимент опровергнут ИЛИ терминальный статус) починить нечем.
        alive = not (n.get("exp_refuted") or n.get("status") in terminal)
        alive_for_target[t] = alive_for_target.get(t, False) or alive
    result = []
    for nid in pressure_action_targets(cfg):
        if nid in done:
            continue
        # pressure-узел(ы) для target существуют, но ВСЕ мертвы (refuted/terminal) → резолвить нечем;
        # иначе воркеры уходят в вечный rest «жду unresolved target» и луп встаёт (livelock).
        if nid in alive_for_target and not alive_for_target[nid]:
            continue
        result.append(nid)
    return result


def score_nodes(data: Dict[str, Any], cfg: ARConfig,
                exclude_running: bool = True) -> List[Score]:
    by = gio.index(data)
    running = set(gio.load_active(cfg)) if exclude_running else set()
    s = cfg.selector
    pressure_nodes = set(pressure_action_targets(cfg))
    unresolved_pressure = set(pressure_unresolved_targets(data, cfg))
    out: List[Score] = []
    for n in data["nodes"]:
        if n.get("status") in s.terminal or n["id"] in running:
            continue
        if graph_hygiene.is_quarantined_untyped(n):
            continue
        if not graph_hygiene.report_eligible(n):
            continue
        if not graph_hygiene.project_policy_eligible(n, cfg):
            continue
        if graph_hygiene.writeup_exhausted(n):
            continue
        if n.get("exp_refuted"):
            continue
        if n.get("policy_blocked_exp"):
            continue  # эмп-линия запаркована (скрипт не строится автогеном) → не грызём слоты повторно
        if n.get("col") in (None, 0):
            continue
        elo_t = s.w_elo * (gio.get_elo(n, cfg) - cfg.elo.start) / 400.0
        open_t = s.w_open * _open_richness(n, cfg)
        prom_t = s.w_promise * _promise(n, by, cfg)
        att_t = -s.w_attempts * float(n.get("attempts", 0) or 0)
        # Only typed EXP_SPEC confirmations get the writeup priority lane.
        wu_t = 5.0 if graph_hygiene.needs_writeup_priority(n) else 0.0
        nid = str(n["id"])
        ptarget = _pressure_target(n)
        if pressure_nodes and pressure_node_satisfied(n, pressure_nodes, cfg):
            continue
        # анти-застой (проект-агностично): узел, перемолотый >= max_attempts раз, выбывает из
        # РАБОЧЕГО фронтира → освобождает место под свежие узлы/эволюцию (supervisor при тонком
        # фронтире сам зовёт evolve). Из графа/отчёта узел НЕ удаляется. Исключения: приоритет
        # writeup (эксп подтвердил — надо вписать) и активная unresolved-pressure цель (ещё чинибельна).
        if (int(n.get("attempts", 0) or 0) >= s.max_attempts
                and not graph_hygiene.needs_writeup_priority(n)
                and not (unresolved_pressure and ptarget in unresolved_pressure)):
            continue
        # фильтр применяем ТОЛЬКО к pressure-узлам: у обычного узла ptarget="" и "" not in unresolved
        # → раньше это выкидывало ВЕСЬ обычный фронтир при любом unresolved pressure (team-wide голодание).
        # pressure-узлы и так наверху из-за pressure_t=+8.0; обычную работу душить не нужно.
        if unresolved_pressure and nid.startswith("pressure_") and ptarget not in unresolved_pressure:
            continue
        pressure_t = 8.0 if (
            nid in pressure_nodes
            or (nid.startswith("pressure_") and nid[len("pressure_"):] in pressure_nodes)
        ) else 0.0
        out.append(Score(n["id"], n.get("short", ""),
                         elo_t + open_t + prom_t + att_t + wu_t + pressure_t,
                         elo_t, open_t, prom_t, att_t, pressure_t))
    out.sort(key=lambda x: x.total, reverse=True)
    return out


def select_next(data: Dict[str, Any], cfg: ARConfig, k: int = 0) -> List[str]:
    k = k or cfg.selector.pick_k
    return [s.node_id for s in score_nodes(data, cfg)[:k]]
