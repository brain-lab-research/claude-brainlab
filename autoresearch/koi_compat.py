"""Слой совместимости с движком ResearchOS (koi): импорт его md_io/models.

Единый стек: проект живёт как koi-structure (project.md + research.json), а наши
поля движка (elo/attempts/scoop/failure_class/status-нюанс/idea/open/plan/xref) —
в koi-structure/engine.json тех же узлов. Один store, без мостов.

koi-движок требует pydantic + pyyaml. На сервере без них поставить:
  apt-get install -y python3-pydantic python3-yaml   (или pip, если есть)
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Any, Tuple

DEFAULT_ENGINE_PATH = os.path.expanduser("~/research_os/ReseachOS")


def engine_path(cfg: Any = None) -> str:
    p = getattr(cfg, "koi_engine_path", None) if cfg is not None else None
    return os.path.abspath(os.path.expanduser(p or DEFAULT_ENGINE_PATH))


@lru_cache(maxsize=4)
def _load(engine: str):
    if engine not in sys.path:
        sys.path.insert(0, engine)
    from koi.core import md_io, models  # type: ignore
    return md_io, models


def koi_modules(cfg: Any = None) -> Tuple[Any, Any]:
    """Вернуть (md_io, models) движка koi. Бросит ImportError, если движок/deps нет."""
    return _load(engine_path(cfg))


def available(cfg: Any = None) -> bool:
    try:
        koi_modules(cfg)
        return True
    except Exception:
        return False
