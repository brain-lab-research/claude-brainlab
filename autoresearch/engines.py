"""Обёртки над движками генерации (codex exec / claude headless).

Popen + hard timeout + killpg всей группы. engine_call() делает graceful-fallback
codex->claude, если codex недоступен (на части серверов его нет). cfg обязателен.
"""
from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import tempfile
from typing import Callable, Optional

from .config import ARConfig

logger = logging.getLogger(__name__)


def _run(cmd, prompt: str, timeout: int, cwd: Optional[str]) -> str:
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True, cwd=cwd)
    try:
        out, err = proc.communicate(prompt, timeout=timeout)
        return (out or "") + "\n" + (err or "")
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            pass
        try:
            out, err = proc.communicate(timeout=15)
        except Exception:
            out, err = "", ""
        return (out or "") + "\n" + (err or "") + \
            "\nTIMEOUT_STAGE (engine hung > %ds, killed)" % timeout


def codex_available() -> bool:
    return shutil.which("codex") is not None


def claude_available() -> bool:
    return shutil.which("claude") is not None


def claude_disabled() -> bool:
    return os.environ.get("AUTORESEARCH_DISABLE_CLAUDE", "").lower() in ("1", "true", "yes")


def _cwd(cfg: ARConfig) -> Optional[str]:
    return cfg.workdir if os.path.isdir(cfg.workdir) else cfg.project_root


def codex_call(prompt: str, cfg: ARConfig, write: bool = True, allow_web: bool = False) -> str:
    mode = "workspace-write" if write else "read-only"
    cwd = _cwd(cfg)
    # codex exec пишет в stdout СВОИ баннеры (`session id: ...`, `tokens used N`) вперемешку с ответом
    # модели → они протекали в derive.md/verify.py и ломали код-гейт (SyntaxError «посторонняя строка
    # session id»). --output-last-message пишет в файл ТОЛЬКО финальное сообщение агента — берём его.
    fd, msgf = tempfile.mkstemp(suffix=".txt", prefix="codexmsg_")
    os.close(fd)
    cmd = ["codex", "exec", "--skip-git-repo-check", "-s", mode, "-C", cwd,
           "--output-last-message", msgf]
    eff = getattr(cfg.engines, "codex_reasoning_effort", "") or ""
    if eff:
        cmd += ["-c", f'model_reasoning_effort="{eff}"']  # codex «на максимум» = high
    mdl = getattr(cfg.engines, "codex_model", "") or ""
    if mdl:
        cmd += ["-m", mdl]
    if allow_web:
        cmd += ["-c", "tools.web_search=true"]  # live web search в codex exec — для обзора литературы
    cmd += ["-"]
    raw = _run(cmd, prompt, cfg.engines.codex_timeout, cwd)
    try:
        msg = open(msgf, encoding="utf-8", errors="replace").read().strip()
    except OSError:
        msg = ""
    finally:
        try:
            os.unlink(msgf)
        except OSError:
            pass
    # файл пуст (старый codex / таймаут / ошибка) → отдаём raw stdout как раньше (в т.ч. TIMEOUT_STAGE)
    return msg or raw


def codex_write_artifact(prompt: str, cfg: ARConfig, artifact_name: str = "codex_artifact.txt") -> str:
    """codex-АГЕНТ пишет ПОЛНЫЙ длинный артефакт (код/пруф/отчёт) ПРЯМО В ФАЙЛ своими file-инструментами,
    а не в ответное сообщение — так длинный вывод НЕ обрезается лимитом финального сообщения.
    Имя файла УНИКАЛЬНО на вызов (mkstemp) → параллельные агенты не перетирают друг друга.
    Возвращает содержимое файла (или фолбэк на last-message, если файл не создан)."""
    cwd = _cwd(cfg)
    _root, _ext = os.path.splitext(os.path.basename(artifact_name))
    # уникальное имя в cwd → нет гонки между параллельными идейными агентами/судьями
    fd_a, target = tempfile.mkstemp(suffix=(_ext or ".txt"), prefix=f"_wip_{_root}_", dir=cwd)
    os.close(fd_a)
    rel_name = os.path.basename(target)
    fd, msgf = tempfile.mkstemp(suffix=".txt", prefix="codexmsg_")
    os.close(fd)
    full = (prompt + f"\n\n=== КАК ВЫДАТЬ РЕЗУЛЬТАТ ===\nЗАПИШИ полный результат ЦЕЛИКОМ в файл "
            f"`{rel_name}` (в текущей рабочей папке) своими file-инструментами. Пиши в ФАЙЛ, "
            "а НЕ в ответное сообщение — длинный артефакт в сообщении обрезается, в файле нет. "
            "Можешь дописывать файл по частям (несколько правок), пока он не будет ПОЛНЫМ и "
            "завершённым. В ответном сообщении верни только слово DONE.")
    cmd = ["codex", "exec", "--skip-git-repo-check", "-s", "workspace-write", "-C", cwd,
           "--output-last-message", msgf]
    eff = getattr(cfg.engines, "codex_reasoning_effort", "") or ""
    if eff:
        cmd += ["-c", f'model_reasoning_effort="{eff}"']
    mdl = getattr(cfg.engines, "codex_model", "") or ""
    if mdl:
        cmd += ["-m", mdl]
    cmd += ["-"]
    raw = _run(cmd, full, cfg.engines.codex_timeout, cwd)
    try:
        content = open(target, encoding="utf-8", errors="replace").read()
    except OSError:
        content = ""
    finally:
        try:
            os.remove(target)
        except OSError:
            pass
    if content.strip():
        return content
    # фолбэк: файл не создан → берём last-message (как обычный codex_call)
    try:
        msg = open(msgf, encoding="utf-8", errors="replace").read().strip()
    except OSError:
        msg = ""
    finally:
        try:
            os.unlink(msgf)
        except OSError:
            pass
    return msg or raw


def claude_call(prompt: str, cfg: ARConfig, model: str = "", allow_web: bool = False) -> str:
    if claude_disabled():
        return "CLAUDE_DISABLED: AUTORESEARCH_DISABLE_CLAUDE=1"
    cmd = ["claude", "-p", "--model", model or cfg.engines.claude_model]
    if allow_web:
        cmd += ["--allowedTools", "WebSearch"]  # веб-поиск в headless claude (подтверждено на сервере)
    return _run(cmd, prompt, cfg.engines.claude_timeout, _cwd(cfg))


def engine_call(name: str, cfg: ARConfig) -> Callable[[str], str]:
    """name: 'codex' | 'codex_web' | 'sonnet' | 'opus' | 'haiku_web' | 'claude'. codex->claude fallback."""
    if name == "codex_web":  # codex + live web search (обзор литературы), не зависит от claude
        if codex_available():
            return lambda p: codex_call(p, cfg, allow_web=True)
        name = "haiku_web"  # фолбэк на web-capable claude, если codex нет
    if name == "haiku_web":  # дешёвый web-capable claude (WebSearch)
        if claude_disabled():
            return lambda p: "CLAUDE_DISABLED: AUTORESEARCH_DISABLE_CLAUDE=1"
        return lambda p: claude_call(p, cfg, model="claude-haiku-4-5-20251001", allow_web=True)
    if name == "codex" and not codex_available():
        if claude_disabled():
            logger.error("codex недоступен, но Claude отключён")
            return lambda p: "CODEX_UNAVAILABLE: codex missing and Claude disabled"
        logger.warning("codex недоступен -> fallback на claude(%s)", cfg.engines.claude_model)
        name = "claude"
    if name in ("sonnet", "opus", "claude") and claude_disabled():
        logger.error("запрошен %s, но Claude отключён", name)
        return lambda p: "CLAUDE_DISABLED: AUTORESEARCH_DISABLE_CLAUDE=1"
    if name in ("sonnet", "opus"):
        return lambda p: claude_call(p, cfg, model=name)
    if name == "claude":
        return lambda p: claude_call(p, cfg)
    # ВСЕ codex-роли (writer/derive/verify/reporter/gate_code/gate_critic/meta/evolve/scoop/…) пишут
    # результат ПРЯМО В ФАЙЛ (codex_write_artifact) → НИГДЕ обрывов на длинном выводе. Уникальное имя
    # на вызов (mkstemp) — параллельные агенты не коллизят. codex_web (web-поиск) обработан выше отдельно.
    return lambda p: codex_write_artifact(p, cfg)


def grep_tail(text: str, key: str) -> str:
    for line in reversed(text.splitlines()):
        # sonnet оборачивает вывод в markdown (**GATE_VERDICT: FAIL**, ## A_STAR, > FEEDBACK):
        # снять ведущий декор перед матчем и хвостовые `*`, иначе строка не стартует с ключа → "?".
        s = line.strip().lstrip("*#>-• \t").strip()
        if s.startswith(key):
            return s.split(":", 1)[1].strip().rstrip("*").strip()
    return "?"
