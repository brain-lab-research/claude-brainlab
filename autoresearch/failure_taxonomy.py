"""#6 Память о КЛАССАХ провалов (урок из «Why LLMs Aren't Scientists Yet»).

Извлекаем ПОВТОРЯЮЩИЙСЯ класс ошибки из вердиктов мёртвых узлов, чтобы агент учился
типу промаха, а не конкретному узлу. Классификация — по ключевым словам (офлайн).
Паттерны достаточно общие для любого theory-проекта; при желании расширяй PATTERNS.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from .config import ARConfig
from . import graph_io as gio

logger = logging.getLogger(__name__)

_DEAD = {"rejected", "refuted_breadth"}

PATTERNS: List[Tuple[str, Tuple[str, ...]]] = [
    ("оракул / leak ответа", ("оракул", "oracle", "teacher", "ground truth", "leak", "утечк")),
    ("результат зашит выбором метода", ("forced", "зашит", "by-g", "порядок задан", "tautolog", "тавтолог")),
    ("циркулярность / самоподтверждение", ("циркуляр", "circular", "самоподтвержд", "self-confirm")),
    ("overclaim ('=' вместо 'пропорц.')", ("overclaim", "пропорц", "не равно", "не является", "не lmo")),
    ("хэндвейв / 'схема' доказательства", ("схема", "руками", "легко видеть", "hand-wav", "хэндвейв", "пропуск шаг", "gaps")),
    ("нечестное сравнение / baseline", ("baseline", "same-class", "нечестн", "unfair")),
    ("слабый / шумный сигнал", ("слаб", "шум", "noisy", "weak", "не значим", "ci включает ноль")),
    ("конфаунд", ("конфаунд", "confound", "медленнее", "scale-fair")),
    ("опровергнуто кодом/экспериментом", ("код опроверг", "fail", "ломается", "не держится", "refuted")),
]


def classify(text: str) -> str:
    t = (text or "").lower()
    for label, kws in PATTERNS:
        if any(kw.lower() in t for kw in kws):
            return label
    return "прочее"


def annotate(data: Dict[str, Any]) -> int:
    n = 0
    for node in data["nodes"]:
        if node.get("status") in _DEAD and not node.get("failure_class"):
            node["failure_class"] = classify(node.get("verdict", ""))
            n += 1
    return n


def taxonomy(data: Dict[str, Any]) -> List[Tuple[str, int, List[str]]]:
    buckets: Dict[str, List[str]] = {}
    for node in data["nodes"]:
        if node.get("status") in _DEAD:
            cls = node.get("failure_class") or classify(node.get("verdict", ""))
            buckets.setdefault(cls, []).append(node["id"])
    ranked = sorted(buckets.items(), key=lambda kv: len(kv[1]), reverse=True)
    return [(cls, len(ids), ids) for cls, ids in ranked]


def memory_block(data: Dict[str, Any]) -> str:
    tax = taxonomy(data)
    if not tax:
        return ""
    L = ["=== ⚠️ ТАКСОНОМИЯ ПРОВАЛОВ — КЛАССЫ ошибок, НЕ повторять паттерн ==="]
    for cls, cnt, ids in tax:
        L.append(f"- [{cnt}×] {cls} (напр.: {', '.join(ids[:4])})")
    return "\n".join(L)
