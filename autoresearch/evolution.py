"""#4 Evolution-агент: скрещивание/упрощение топ-узлов (идея из Google AI co-scientist).

Берёт два сильнейших по Elo узла из РАЗНЫХ веток и просит агента синтезировать
дочерний узел-гибрид. Новый узел добавляется с xref на родителей. Домен — из cfg.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import ARConfig
from .engines import engine_call
from . import graph_io as gio

logger = logging.getLogger(__name__)


def _branch_of(node_id: str, by: Dict[str, Dict[str, Any]]) -> Optional[str]:
    for a in gio.ancestors(node_id, by):
        if a.get("col") == 1:
            return a["id"]
    n = by.get(node_id, {})
    return node_id if n.get("col") == 1 else None


def top_cross_branch_pairs(data: Dict[str, Any], cfg: ARConfig,
                           limit: int = 3) -> List[Tuple[Dict, Dict]]:
    by = gio.index(data)
    alive = [n for n in data["nodes"]
             if n.get("status") not in ("rejected", "refuted_breadth", "root", "background")
             and n.get("col") not in (0, None)]
    alive.sort(key=lambda n: gio.get_elo(n, cfg), reverse=True)
    pairs: List[Tuple[Dict, Dict]] = []
    for i in range(len(alive)):
        for j in range(i + 1, len(alive)):
            if _branch_of(alive[i]["id"], by) != _branch_of(alive[j]["id"], by):
                pairs.append((alive[i], alive[j]))
                break
        if len(pairs) >= limit:
            break
    return pairs


_CROSS_PROMPT = """\
Ты EVOLUTION-АГЕНТ проекта: {domain}
Даны ДВА перспективных результата из разных веток. Придумай ОДНУ новую
гипотезу-теорему, которая ОСМЫСЛЕННО комбинирует/мостит их (не механически склеивает).
Формальную, доказуемую из первых принципов, проверяемую кодом/экспериментом. Честно
укажи, как может провалиться. Если осмысленного моста нет — NO_BRIDGE.

УЗЕЛ A [{aid}]: {aidea}  | вердикт: {averd}
УЗЕЛ B [{bid}]: {bidea}  | вердикт: {bverd}

Верни СТРОГО JSON одной строкой и больше ничего:
{{"id":"<краткий_id_латиницей>","short":"<заголовок <=60 симв>","idea":"<1-2 предложения>","open":"<что проверить>","plan":"<план>","bridge":true}}
Если моста нет: {{"bridge":false}}"""


def _parse_json_line(text: str) -> Optional[Dict[str, Any]]:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    i, j = text.find("{"), text.rfind("}")
    if 0 <= i < j:
        try:
            return json.loads(text[i:j + 1])
        except ValueError:
            return None
    return None


def propose_crossover(a: Dict[str, Any], b: Dict[str, Any], cfg: ARConfig,
                      call: Optional[Callable[[str], str]] = None) -> Optional[Dict[str, Any]]:
    call = call or engine_call(cfg.engines.evolve, cfg)
    out = call(_CROSS_PROMPT.format(
        domain=cfg.domain,
        aid=a["id"], aidea=a.get("idea", ""), averd=a.get("verdict", "")[:200],
        bid=b["id"], bidea=b.get("idea", ""), bverd=b.get("verdict", "")[:200]))
    obj = _parse_json_line(out)
    if not obj or not obj.get("bridge"):
        return None
    return obj


def evolve_into_graph(data: Dict[str, Any], cfg: ARConfig, limit: int = 3,
                      call: Optional[Callable[[str], str]] = None,
                      guard_check: Optional[Callable[[str], bool]] = None) -> List[str]:
    """Гибриды для топ-пар -> новые узлы col=2. guard_check(text)->True если свежо.
    Граф НЕ сохраняется здесь (это делает вызывающий)."""
    by = gio.index(data)
    added: List[str] = []
    for a, b in top_cross_branch_pairs(data, cfg, limit):
        obj = propose_crossover(a, b, cfg, call)
        if not obj:
            continue
        nid = obj.get("id", "").strip()
        if not nid or nid in by:
            nid = f"evo_{a['id']}_{b['id']}"[:40]
        if nid in by:
            # фолбэк-id детерминирован от пары и обрезан до 40 симв → одна и та же топ-пара
            # давала ОДИН и тот же id и пришивала дубль (было 40 копий одного узла). Пропускаем.
            logger.info("evolution: гибрид %s уже есть — пропуск (без дубля)", nid)
            continue
        text = f"{obj.get('short','')} {obj.get('idea','')} {obj.get('open','')}"
        if guard_check is not None and not guard_check(text):
            logger.info("evolution: гибрид %s отброшен дедупом", nid)
            continue
        branch = _branch_of(a["id"], by) or a.get("parent") or "root"
        node = {"id": nid, "short": obj.get("short", nid)[:60], "parent": branch, "col": 2,
                "status": "open", "idea": obj.get("idea", ""),
                "verdict": "", "open": obj.get("open", ""), "plan": obj.get("plan", "")}
        gio.add_node(data, node)
        gio.add_xref(data, nid, a["id"], "комбинирует")
        gio.add_xref(data, nid, b["id"], "комбинирует")
        by[nid] = node
        added.append(nid)
        logger.info("evolution: новый гибрид %s (от %s + %s)", nid, a["id"], b["id"])
    return added
