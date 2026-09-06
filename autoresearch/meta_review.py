"""#5 Meta-review: самоулучшение FRAMEWORK из накопленной критики (co-scientist).

Агент читает таксономию провалов + хвосты критики и выводит 3–6 повторяющихся классов
ошибок -> короткие hard-rules в framework_overlay.md. Этот overlay вклеивается в
FRAMEWORK цикла, так что промпт чинит сам себя между раундами. Домен — из cfg.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional

from .config import ARConfig
from .engines import engine_call
from . import failure_taxonomy as ftax

logger = logging.getLogger(__name__)

_HEADER = "=== AUTO-OVERLAY (meta-review): свежие hard-rules из накопленной критики ==="


def _collect_critic(cfg: ARConfig, limit: int = 40) -> str:
    rl = cfg.workdir
    if not os.path.isdir(rl):
        return ""
    chunks: List[str] = []
    # reverse=True: имена r5_<ts>_critic.md сортируются по таймстампу → берём СВЕЖАЙШИЕ 40, а не
    # старейшие (иначе hard-rules выводились из устаревшей критики и свежие уроки не попадали никогда).
    for fn in sorted(os.listdir(rl), reverse=True):
        if fn.startswith("r5_") and fn.endswith("_critic.md"):
            try:
                txt = open(os.path.join(rl, fn), encoding="utf-8").read()
            except OSError:
                continue
            tail = "\n".join(txt.strip().splitlines()[-12:])
            chunks.append(f"# {fn}\n{tail}")
        if len(chunks) >= limit:
            break
    return "\n\n".join(chunks)


_META_PROMPT = """\
Ты META-РЕЦЕНЗЕНТ автономного research-лупа. Проект: {domain}
Дана таксономия провалов и хвосты вердиктов критика. Найди 3–6 ПОВТОРЯЮЩИХСЯ классов
ошибок и сформулируй для каждого ОДНО короткое императивное правило (hard-rule),
которое не даст будущим агентам повторить промах. Правила конкретные для проекта, не
общие банальности.

ТАКСОНОМИЯ ПРОВАЛОВ:
{tax}

ХВОСТЫ КРИТИКИ:
{critic}

Верни ТОЛЬКО маркированный список правил (по одному на строку, начинай с '- ')."""


def generate_rules(data: Dict[str, Any], cfg: ARConfig,
                   call: Optional[Callable[[str], str]] = None) -> str:
    call = call or engine_call(cfg.engines.meta, cfg)
    tax = ftax.memory_block(data) or "(нет мёртвых узлов)"
    critic = _collect_critic(cfg) or "(файлов критика нет)"
    out = call(_META_PROMPT.format(domain=cfg.domain, tax=tax, critic=critic[:6000]))
    rules = [ln.rstrip() for ln in out.splitlines() if ln.strip().startswith("- ")]
    return "\n".join(rules)


def write_overlay(rules: str, ts: str, cfg: ARConfig) -> str:
    os.makedirs(os.path.dirname(cfg.framework_overlay), exist_ok=True)
    with open(cfg.framework_overlay, "w", encoding="utf-8") as f:
        f.write(f"{_HEADER}\n(обновлено {ts})\n{rules}\n")
    logger.info("framework overlay -> %s", cfg.framework_overlay)
    return cfg.framework_overlay


def read_overlay(cfg: ARConfig) -> str:
    try:
        return open(cfg.framework_overlay, encoding="utf-8").read().strip()
    except OSError:
        return ""
