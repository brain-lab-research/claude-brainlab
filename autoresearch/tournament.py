"""#1 Elo-турнир отбора кандидатов (идея из Google AI co-scientist ranking agent).

Попарные дебаты: судья-агент решает, какой кандидат перспективнее по
(ригор × новизна × проверяемость). Победы двигают Elo. Чистая Elo-математика
тестируется отдельно от LLM. Домен проекта инжектится из cfg.domain.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Callable, List, Tuple

from .config import ARConfig

logger = logging.getLogger(__name__)

JudgeFn = Callable[[str, str], str]   # (a_text, b_text) -> 'A' | 'B'


@dataclass
class Candidate:
    text: str
    cid: str = ""
    elo: float = 1500.0
    wins: int = 0
    losses: int = 0


def expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def update_elo(ra: float, rb: float, a_won: bool, k: float) -> Tuple[float, float]:
    ea = expected(ra, rb)
    sa = 1.0 if a_won else 0.0
    return ra + k * (sa - ea), rb + k * ((1.0 - sa) - (1.0 - ea))


def _pairings(n: int, rounds: int, rng: random.Random) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    target = max(1, (n * rounds) // 2)
    idx = list(range(n))
    while len(pairs) < target:
        rng.shuffle(idx)
        for a, b in zip(idx[::2], idx[1::2]):
            if a != b:
                pairs.append((a, b))
            if len(pairs) >= target:
                break
    return pairs


def run_tournament(candidates: List[Candidate], judge: JudgeFn,
                   cfg: ARConfig, seed: int = 0) -> List[Candidate]:
    if len(candidates) < 2:
        return list(candidates)
    rng = random.Random(seed)
    for i, c in enumerate(candidates):
        if not c.cid:
            c.cid = f"c{i}"
    for a, b in _pairings(len(candidates), cfg.elo.rounds, rng):
        ca, cb = candidates[a], candidates[b]
        try:
            verdict = judge(ca.text, cb.text).strip().upper()
        except Exception as e:
            logger.warning("judge failed (%s) -> draw", e)
            continue
        a_won = verdict.startswith("A")
        ca.elo, cb.elo = update_elo(ca.elo, cb.elo, a_won, cfg.elo.k_factor)
        if a_won:
            ca.wins += 1; cb.losses += 1
        else:
            cb.wins += 1; ca.losses += 1
    return sorted(candidates, key=lambda c: c.elo, reverse=True)


def keep_top(ranked: List[Candidate], cfg: ARConfig) -> List[Candidate]:
    return ranked[: cfg.elo.keep_top]


_JUDGE_PROMPT = """\
Ты ОТБОРЩИК идей для проекта: {domain}
Сравни ДВА кандидата и реши, какой ПЕРСПЕКТИВНЕЕ строго по трём осям:
  (1) РИГОР — доказуемость из первых принципов (не хэндвейв);
  (2) НОВИЗНА — не переоткрытие известного/уже сделанного;
  (3) ПРОВЕРЯЕМОСТЬ — фальсифицируемо в коде/эксперименте.
Честность важнее хайпа: узкий доказуемый результат лучше громкого недоказуемого.

КАНДИДАТ A:
{a}

КАНДИДАТ B:
{b}

Ответь РОВНО одной строкой: 'WINNER: A' или 'WINNER: B'."""


def make_llm_judge(call: Callable[[str], str], cfg: ARConfig) -> JudgeFn:
    def judge(a: str, b: str) -> str:
        out = call(_JUDGE_PROMPT.format(domain=cfg.domain, a=a[:1500], b=b[:1500]))
        for line in reversed(out.splitlines()):
            up = line.strip().upper()
            if "WINNER:" in up:
                return "A" if up.split("WINNER:", 1)[1].strip().startswith("A") else "B"
        # нет строки WINNER → НЕ дефолтить в "A" (систематически завышало Elo позиции A на мусорных
        # ответах судьи); исключение = ничья, run_tournament её уже обрабатывает как пропуск партии.
        raise ValueError("судья не выдал WINNER — партия пропущена (draw)")
    return judge
