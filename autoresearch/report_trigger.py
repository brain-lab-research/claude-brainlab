"""Лёгкий codex-агентик «планировщик отчёта»: решает, пора ли делать отчёт.

Вместо тупого порога (каждые N узлов) — смотрит на НАКОПЛЕННЫЕ новые прошедшие
результаты + общую картину и решает: (1) достаточно ли нового связного материала,
чтобы обновлять отчёт; (2) тянет ли текущая теория на A* (main-track). Дёшево (codex).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .config import ARConfig
from .engines import engine_call, grep_tail
from . import graph_hygiene, graph_io as gio

logger = logging.getLogger(__name__)

_PROMPT = """\
Ты ПЛАНИРОВЩИК отчёта в автономном research-лупе. Проект: {domain}
Идейные агенты только что провели через жёсткий тройной гейт НОВЫЕ результаты:
{new_passed}

Общая картина проекта:
- доказано/подтверждено всего: {n_proven}
- условно (выжило): {n_cond}
- последний отчёт: {last}

РЕШИ две вещи коротко и трезво (honesty>hype):
1. Накопилось ли достаточно НОВОГО СВЯЗНОГО материала, чтобы СЕЙЧАС обновить отчёт-статью?
   (одиночный мелкий результат — рано; несколько связанных или один сильный — пора.)
2. Тянет ли текущая совокупность на A* (main-track NeurIPS/ICLR)?

Выведи РОВНО три строки:
REASON: <одно предложение>
ASTAR_READY: yes|no
DECISION: REPORT|WAIT"""


def should_report(cfg: ARConfig, data: Dict[str, Any], new_passed: List[str],
                  last_score: Optional[int] = None,
                  call=None) -> Tuple[bool, str, str]:
    """Вернуть (делать_отчёт, astar_ready, reason). Решает codex-агентик."""
    if not new_passed:
        return False, "no", "нет новых прошедших gate/typed-writeup результатов; отчёт был бы перерендером старого материала"
    by = gio.index(data)
    lines = []
    for nid in new_passed:
        n = by.get(nid, {})
        if not graph_hygiene.report_eligible(n):
            continue
        if not graph_hygiene.project_policy_eligible(n, cfg):
            continue
        exp = graph_hygiene.typed_exp_confirmed(n)
        signal = (f"RAW typed EXP_SPEC confirmed; {exp[:220]}"
                  if exp and n.get("status") == "needs_writeup"
                  else (graph_hygiene.safe_verdict(n) or n.get("idea", "")))
        lines.append(f"- [{nid}] {n.get('short','')}: "
                     f"{signal[:260]}")
    if not lines:
        return False, "no", "новые результаты есть только в policy-blocked/mechanical-invalid узлах; отчёт не обновляю"
    n_proven = sum(1 for n in data["nodes"]
                   if graph_hygiene.is_headline_win(n) and graph_hygiene.report_eligible(n)
                   and graph_hygiene.project_policy_eligible(n, cfg))
    n_cond = sum(1 for n in data["nodes"]
                 if n.get("status") == "conditional" and graph_hygiene.report_eligible(n)
                 and graph_hygiene.project_policy_eligible(n, cfg))
    last = f"A*={last_score}/10" if last_score is not None else "ещё не делался"
    call = call or engine_call("codex", cfg)
    out = call(_PROMPT.format(domain=cfg.domain, new_passed="\n".join(lines) or "(нет)",
                              n_proven=n_proven, n_cond=n_cond, last=last))
    decision = grep_tail(out, "DECISION:").upper()
    astar = grep_tail(out, "ASTAR_READY:")
    reason = grep_tail(out, "REASON:")
    return decision.startswith("REPORT"), astar, reason
