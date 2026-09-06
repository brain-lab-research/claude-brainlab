"""autoresearch — переиспользуемый апгрейд автономного research-лупа (проект-агностичный).

Восемь механизмов из лучших autonomous-research систем (AIDE, AI Scientist-v2,
Google AI co-scientist, ml-intern, claude-autoresearch, ResearchOS), адаптированных
под theory-трек с тройными воротами (derive+proof+code+critic):

  #3 dedup_guard      — детектор дубль-углов (sklearn ИЛИ stdlib-фолбэк)
  #1 tournament       — Elo-турнир отбора кандидатов
  #2 selector         — авто-выбор узла (experiment-manager)
  #4 evolution        — скрещивание топ-узлов
  #5 meta_review      — самоулучшение FRAMEWORK из критики
  #6 failure_taxonomy — память о КЛАССАХ провалов
  #7 scoop_gate       — prior-art проверка до derive
  #8 autodraft        — выжившие узлы -> LaTeX-стабы

Не привязан к проекту: корень ищется по IdeaGraph/graph.json, домен и пути —
из per-project IdeaGraph/autoresearch.json. Stdlib-only (sklearn опционален).
CLI: python3 -m autoresearch.orchestrator <cmd> [--project PATH].
"""
from __future__ import annotations

__version__ = "0.2.0"

__all__ = [
    "config", "graph_io", "engines", "dedup_guard", "tournament", "selector",
    "evolution", "meta_review", "failure_taxonomy", "scoop_gate", "autodraft",
    "reporter", "paper", "review_gate", "inbox", "agent_worker", "supervisor",
    "agents_status", "report_trigger", "narrative", "node_memory", "exp_spec",
    "graph_hygiene",
    "koi_store", "koi_compat", "koi_ui", "orchestrator",
]
