"""#3 Doom-loop / детектор дубль-углов (идея из ml-intern).

Семантика — офлайн, без ключей. ДВА бэкенда за одним интерфейсом:
  * sklearn  — TF-IDF по символьным n-граммам + косинус (если есть numpy+sklearn);
  * stdlib   — TF-косинус по символьным n-граммам на чистом Python (фолбэк для
               серверов без numpy/sklearn).
Оба ловят точные/почти-дубли; дубль отвергнутого ловим строже (порог − penalty).
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import ARConfig
from . import graph_io as gio

logger = logging.getLogger(__name__)

_DEAD = {"rejected", "refuted_breadth"}

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import linear_kernel  # type: ignore
    _HAVE_SKLEARN = True
except Exception:  # numpy/sklearn отсутствуют -> stdlib backend
    _HAVE_SKLEARN = False


@dataclass(frozen=True)
class DupReport:
    is_dup: bool
    warn: bool
    score: float
    nearest_id: Optional[str]
    nearest_short: str
    nearest_status: str
    nearest_verdict: str

    def message(self) -> str:
        if self.nearest_id is None:
            return "DEDUP: граф пуст — дублей нет."
        tag = "БЛОК (дубль)" if self.is_dup else ("ВНИМАНИЕ (близко)" if self.warn else "OK (ново)")
        m = (f"DEDUP {tag}: cos={self.score:.2f} к [{self.nearest_id}] "
             f"«{self.nearest_short}» (status={self.nearest_status}).")
        if self.is_dup and self.nearest_status in _DEAD:
            m += f" Этот угол УЖЕ отвергнут: {self.nearest_verdict[:160]} — возьми НОВУЮ линзу."
        elif self.is_dup:
            m += " Угол уже в графе — углубляй существующий узел, а не дублируй."
        return m


def _char_ngrams(text: str, lo: int, hi: int) -> Counter:
    """char_wb-подобные n-граммы: слова в пробелах, скан внутри слова."""
    text = text.lower()
    grams: Counter = Counter()
    for w in re.findall(r"\w+", text, flags=re.UNICODE):
        ww = " " + w + " "
        for n in range(lo, hi + 1):
            for i in range(len(ww) - n + 1):
                grams[ww[i:i + n]] += 1
    return grams


def _cos(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[g] * b[g] for g in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


class DedupGuard:
    """Индекс по графу; .check(text) для нового кандидата. Бэкенд авто-выбирается."""

    def __init__(self, data: Dict[str, Any], cfg: ARConfig):
        self.cfg = cfg
        self.nodes: List[Dict[str, Any]] = data["nodes"]
        self._texts = [gio.node_text(n) for n in self.nodes]
        self.backend = "sklearn" if _HAVE_SKLEARN else "stdlib"
        lo, hi = cfg.dedup.char_ngram
        if _HAVE_SKLEARN and any(t.strip() for t in self._texts):
            self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(lo, hi), min_df=1)
            self._mat = self._vec.fit_transform(self._texts)
        else:
            self._vec = None
            self._grams = [_char_ngrams(t, lo, hi) for t in self._texts]

    def check(self, text: str, exclude_id: Optional[str] = None) -> DupReport:
        if not text.strip() or not self.nodes:
            return DupReport(False, False, 0.0, None, "", "", "")
        if self._vec is not None:
            q = self._vec.transform([text])
            sims = list(linear_kernel(q, self._mat).ravel())
        else:
            lo, hi = self.cfg.dedup.char_ngram
            qg = _char_ngrams(text, lo, hi)
            sims = [_cos(qg, g) for g in self._grams]
        order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        for idx in order:
            n = self.nodes[idx]
            if exclude_id is not None and n["id"] == exclude_id:
                continue
            score = float(sims[idx])
            d = self.cfg.dedup
            thr = d.block_threshold - (d.rejected_penalty if n.get("status") in _DEAD else 0.0)
            return DupReport(
                is_dup=score >= thr, warn=score >= d.warn_threshold, score=score,
                nearest_id=n["id"], nearest_short=n.get("short", ""),
                nearest_status=n.get("status", ""), nearest_verdict=n.get("verdict", ""))
        return DupReport(False, False, 0.0, None, "", "", "")

    def filter_candidates(self, candidates: List[str]) -> List[Dict[str, Any]]:
        out = []
        for c in candidates:
            r = self.check(c)
            out.append({"text": c, "fresh": not r.is_dup, "report": r})
        return out
