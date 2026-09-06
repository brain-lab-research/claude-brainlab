"""Реальная LLM-валидация узла на GPU (второй уровень после numpy-toy-гейта) — АСИНХРОННО.

Для узла с эмпирическим claim (forgetting / Muon-vs-SGD / real-LLM transfer) codex пишет
самодостаточный exp_<node>.py поверх готовых хелперов сервера (llm_exp_kit + transfer_
protocol). Скрипт гоняется на A100 (пул серверов: brain_lab + <gpu-host>) в tmux.

АСИНХРОННО (важно): launch() НЕ блокирует воркер — ставит прогон в tmux и возвращается, так
идейный слот сразу освобождается (всегда 2 идейных агента). supervisor.poll() на каждой
итерации проверяет, не завершились ли прогоны, и читает вердикт по bootstrap-CI (CI∌0 =>
confirmed_emp). Нет свободной карты ни на одном сервере => статус 'waiting' (виден в дашборде),
poll() повторит размещение позже.
"""
from __future__ import annotations

import glob
import hashlib
import json
import logging
import math
import os
import re
import shlex
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

STALE_LOG_S = 300  # лог не растёт дольше 5 мин ИЛИ виден ERROR → зовём агента-наблюдателя (он решает)
# Детерминированная страховка от вечного experiment_running (когда _watch-LLM всегда отвечает RUNNING):
WATCHDOG_STALE_S = 14400      # лог не растёт 4ч → прогон завис → kill + invalid
                              # (было 1800: тяжёлые multi-seed прогоны молчат в логе >30мин между
                              #  per-seed-флашами → ложный stale-kill → orphan-double → doom-loop
                              #  жёг invalid_attempts до permanent-guard; см. watchdog-server-log волна 78)
WATCHDOG_WALL_S = 12 * 3600   # абсолютный потолок wall-clock (пато-случай) → kill + invalid
# ФИКС ВОЛНЫ 213: сколько запись может ждать НЕСУЩЕСТВУЮЩУЮ в пуле карту, продолжая держать слот
# GPU-полосы. Ожидание освобождающейся карты — законный слот; ожидание длиной в часы означает, что
# карты такого размера в пуле нет вовсе, и слот запирает ВСЮ фазу подтверждения (см. _holds_gpu_slot).
NOFIT_SLOT_RELEASE_S = 5400   # 1,5 ч
_NOFIT_LOGGED: Dict[str, float] = {}   # nid → когда в последний раз писали о снятии слота

from .config import ARConfig
from .engines import engine_call, grep_tail
from . import agents_status, exp_spec, engines

logger = logging.getLogger(__name__)

_EXP_PROMPT = """\
Ты ЭКСПЕРИМЕНТАТОР на реальной LLM. Проект: {domain}
Узел [{nid}]: {short}
идея: {idea}
теория (derive, кратко): {derive}

Напиши самодостаточный Python-скрипт exp_{nid}.py для РЕАЛЬНОЙ проверки на GPU-сервере.
Доступно (импортируй из HOME сервера):
- llm_exp_kit: project-allowed model helpers, finetune, _newton_schulz (Muon), eval_loss,
  get_text_batches, matrix_param_names.
- transfer_protocol: preflight (гейты валидности), partial_spearman, bootstrap (CI),
  full_report → вердикт ABORT_INVALID/CONFOUND_SIGNFLIP/SURVIVES_CONTROL/NO_SIGNAL.
ТРЕБОВАНИЯ: matched-step (не matched-loss), >=3 сида, bootstrap-CI; модель из sys.argv[2]
первой строкой print({{"model":...}}). Верить результату ТОЛЬКО если
CI исключает ноль.
ГИПЕРЫ ИЗ ENV (чтобы агент-экспериментатор мог их перебирать без правки логики): читай через
os.environ с дефолтами — LR (float, дефолт по оптимизатору), BATCH (int), STEPS (int),
SEEDS (csv int). Печатай их значения в начале. Логируй ПРОГРЕСС построчно (per-seed/opt числа),
чтобы наблюдатель видел, что прогон жив. Избегай eigendecomp каждый шаг (виснет).
В КОНЦЕ напечатай ровно:
RESULT_JSON: {{...ключевые числа+CI...}}
EMP_VERDICT: confirmed|no_signal|invalid
EXP_DONE
Выведи ТОЛЬКО код скрипта (один блок ```python)."""


# ---------------- серверы и выбор GPU ----------------

def _servers(cfg: ARConfig) -> List[str]:
    return list(cfg.gpu_servers) or [cfg.gpu_server]


_KIT_OK_CACHE: Dict[Tuple[str, str], bool] = {}


def _server_has_kit(srv: str, kit: str = "llm") -> bool:
    """Есть ли kit-хелпер на сервере. ФИКС ВОЛНЫ 185: одиночный ssh с timeout=15 через прокси
    регулярно давал ЛОЖНОЕ «missing» (файл на месте: `ls -la ~/llama_exp_kit.py` → 11763 байта,
    а в логе `skip GPU server <gpu-host>: ~/llama_exp_kit.py missing`), и сервер выпадал из пула
    размещения на весь цикл: записи оси E оставались `waiting` при свободных картах. Лечим двумя
    дешёвыми средствами — ретрай и КЭШ ТОЛЬКО ПОЛОЖИТЕЛЬНОГО ответа (отрицательный не кэшируем:
    kit может появиться, а ложное «нет» не должно застревать в памяти процесса)."""
    marker = "~/llama_exp_kit.py" if kit == "llama" else "~/llm_exp_kit.py"
    key = (srv, kit)
    if _KIT_OK_CACHE.get(key):
        return True
    for _try in range(2):
        try:
            r = subprocess.run(["ssh", srv, f"test -f {marker} && echo OK || echo NO"],
                               capture_output=True, text=True, timeout=25)
            if "OK" in r.stdout:
                _KIT_OK_CACHE[key] = True
                return True
            if "NO" in r.stdout:
                return False  # сервер ответил внятно: файла действительно нет
        except (subprocess.TimeoutExpired, OSError):
            pass
    return False


_STACK_OK_CACHE: Dict[Tuple[str, str, str], bool] = {}


def _server_has_working_stack(cfg: ARConfig, srv: str, kit: str = "llm") -> bool:
    py = cfg.gpu_python_map.get(srv, cfg.gpu_python)
    key = (srv, py, kit)
    if key in _STACK_OK_CACHE:
        return _STACK_OK_CACHE[key]
    if kit == "llama":
        probe = (
            "test -f ~/llama_exp_kit.py || exit 42; "
            "test -d ~/llm-baselines || exit 43; "
            "export PYTHONPATH=$HOME:$HOME/llm-baselines:$HOME/llm-baselines/src:${PYTHONPATH:-}; "
            f"{_q_path(py)} - <<'PY'\n"
            "import tiktoken\n"
            "from llama_exp_kit import build_llama, eval_loss\n"
            "from optim.muon import Muon\n"
            "PY"
        )
    else:
        probe = (
            "test -f ~/llm_exp_kit.py || exit 42; "
            "export PYTHONPATH=$HOME:${PYTHONPATH:-}; "
            f"{_q_path(py)} - <<'PY'\n"
            "import datasets\n"
            "from llm_exp_kit import _newton_schulz, eval_loss, load_model\n"
            "PY"
        )
    try:
        r = subprocess.run(["ssh", srv, probe], capture_output=True, text=True, timeout=60)
        ok = r.returncode == 0
        if not ok:
            msg = (r.stderr or r.stdout or "").strip().splitlines()
            logger.info("skip GPU server %s: python stack probe failed: %s",
                        srv, (msg[-1] if msg else f"rc={r.returncode}")[:180])
        _STACK_OK_CACHE[key] = ok
        return ok
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.info("skip GPU server %s: python stack probe failed: %s", srv, e)
        _STACK_OK_CACHE[key] = False
        return False


_SCALE_FREE_MB: Dict[str, int] = {
    "124M": 11000, "350M": 26000, "760M": 44000, "1B": 60000, "1.4B": 60000,
}
# Дефолт списка масштабов ВНУТРИ скрипта: `TAGS=os.environ.get("SCALE_TAGS","124M,350M")`.
# Ищем ИМЕННО фактический список, а НЕ таблицу геометрии (`TABLE={"124M":…,"760M":…}` есть почти
# всегда, и по ней бюджет уехал бы в 44 ГБ ⇒ kind стал бы неразмещаем навсегда).
_SCALE_TAG_ENV_RE = re.compile(
    r"""os\.environ\.get\(\s*["'](?:SCALE_TAGS|SCALE_LIST|TAGS|SCALES)["']\s*,\s*["']([^"']*)["']""")


def _runtime_scale_tags(cfg: ARConfig, rec: Dict[str, Any], kind: str) -> List[str]:
    """Теги масштабов, которые прогон РЕАЛЬНО построит (фикс волны 211).

    Источник истины — не спека, а то, что исполнится: env записи, если он задаёт список масштабов,
    иначе ДЕФОЛТ из текста одобренного скрипта. По контракту в.183 геометрия именованных масштабов
    берётся из таблицы скрипта и env её НЕ переопределяет, поэтому оценивать требуемую VRAM по
    claim/env спеки — значит бюджетировать не тот прогон (см. комментарий в `_place`).
    """
    env: Dict[str, Any] = dict(rec.get("env") or {})
    for _k, _v in (((rec.get("exp_spec") or {}).get("env") or {}).items()):
        env.setdefault(_k, _v)
    for _k in ("SCALE_TAGS", "SCALE_LIST", "TAGS", "SCALES"):
        raw = str(env.get(_k) or "").strip()
        if raw:
            tags = [t.strip() for t in raw.split(",") if t.strip() in _SCALE_FREE_MB]
            if tags:
                return tags
    from . import exp_spec as _es
    rel = str((rec.get("exp_spec") or {}).get("script") or _es.KIND_SCRIPTS.get(kind, "") or "")
    if not rel:
        return []
    src = _project_script_src(cfg, rel) or ""
    m = _SCALE_TAG_ENV_RE.search(src)
    if not m:
        return []
    return [t.strip() for t in m.group(1).split(",") if t.strip() in _SCALE_FREE_MB]


def _pick_gpu_multi(cfg: ARConfig, require_kit: Any = False,
                    min_free_mb: int = 11000) -> Tuple[Optional[str], Optional[int]]:
    """ГЛОБАЛЬНО самая свободная карта по всему пулу серверов (спред для параллельных экспов).

    Исключает (server,gpu), уже занятые НАШИМИ бегущими экспами (из pending-записей) — иначе два
    быстрых параллельных запуска заняли бы одну карту до того, как nvidia-smi её покажет busy.
    min_free_mb: минимум СВОБОДНОЙ VRAM (масштаб-зависимо: крупная модель требует больше — иначе OOM
    на карте с малым остатком). Карта с достаточной free-памятью подходит ДАЖЕ при высоком util
    (чужой compute делится по времени); util лишь деприоритизирует (сорт (util,-free)), НЕ исключает —
    иначе карты brain_lab с 56GB free и util 100% (чужой прогон) ошибочно пропускались и крупные
    модели вообще не находили карту.
    """
    kit = "llm" if require_kit is True else str(require_kit or "")
    in_use = set()
    for r in _records(cfg):
        # ФИКС ВОЛНЫ 241 (ПРИЗРАЧНАЯ БРОНЬ КАРТЫ). Карту занимает ТОЛЬКО реально идущий прогон.
        # Раньше сюда попадала ЛЮБАЯ запись с непустыми (server,gpu) — в том числе `waiting` с
        # ОСТАТОЧНЫМИ полями размещения, которые по в.186 намеренно НЕ вычищаются при downgrade
        # (`_save_rec` подмешивает их с диска, чтобы можно было адоптировать осиротевший прогон).
        # Итог: запись `directive_abf750cc_emp_760M` (waiting, без tmux-сессии и без процесса на
        # сервере) вечно держала бронь на <gpu-host> cuda:3 — 54 ГБ свободно, util 0 %, а в логе
        # каждую итерацию «ждёт карту >=44000 МБ … в пуле такой карты нет». Бронь была
        # САМОБЛОКИРУЮЩЕЙ: сама запись тоже не могла разместиться на своей же карте, поэтому
        # четыре 760M-узла простояли 11-26 ч при физически свободной карте, а ось E (одна из двух,
        # что держат SCORE<=6) не получала материала. Вахтёры в.208-240 списывали это на «дефицит
        # физический», проверяя nvidia-smi руками и видя чужой util — но util здесь НЕ исключает
        # (см. докстринг), так что причина всё это время была софтовая.
        # Безопасность: настоящий осиротевший прогон не теряется — `_place` перед размещением
        # зовёт `_adopt_live_run` (в.186) и вернёт запись в `running` вместе с бронью, а дублю на
        # том же имени сессии мешает проверка `has-session … && echo BUSY` (в.171).
        if str(r.get("status") or "") != "running":
            continue
        s, g = r.get("server"), r.get("gpu")
        if s is not None and g is not None:
            try:
                in_use.add((str(s), int(g)))
            except (TypeError, ValueError):
                pass
    best = None  # (free, srv, idx)
    for srv in _servers(cfg):
        if kit and not _server_has_kit(srv, kit):
            logger.info("skip GPU server %s: ~/%s_exp_kit.py missing", srv, kit)
            continue
        if kit and not _server_has_working_stack(cfg, srv, kit):
            continue
        try:
            r = subprocess.run(
                ["ssh", srv,
                 "nvidia-smi --query-gpu=index,memory.free,utilization.gpu "
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, errors="replace", timeout=30)
        except (subprocess.TimeoutExpired, OSError):
            continue
        for line in r.stdout.splitlines():
            try:
                idx, free, util = [int(x.strip()) for x in line.split(",")]
            except ValueError:
                continue
            if (srv, idx) in in_use:
                continue  # уже занята нашим бегущим экспом → не занимать повторно
            # ВЫБОР: карта годна, если СВОБОДНОЙ памяти хватает под масштаб модели (min_free_mb).
            # Ключ сортировки (util, -free): среди годных сначала минимальный util (реально свободные),
            # при равном — больше free. util НЕ исключает: карта с 56GB free и util 100% (чужой прогон)
            # всё равно годна для крупной модели (compute делится по времени) — раньше util<92 её
            # ошибочно отсекал и 350M/760M не находили карту → «нет масштаба».
            if free > min_free_mb:
                key = (util, -free)
                if best is None or key < best[0]:
                    best = (key, srv, idx)
    if best is not None:
        return best[1], best[2]
    return None, None


def _extract_py(out: str) -> str:
    m = (re.search(r"```python\s*\n(.*?)```", out, re.DOTALL)
         or re.search(r"```python\s*\n(.*)$", out, re.DOTALL))
    return m.group(1).strip() if m else out.strip()


# ---------------- хранилище записей об экспериментах ----------------

def _exp_dir(cfg: ARConfig) -> str:
    d = os.path.join(cfg.workdir, ".run", "experiments")
    os.makedirs(d, exist_ok=True)
    return d


def _safe(nid: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", nid)


def _q_path(path: str) -> str:
    """Quote a shell path, leaving trusted ~/ paths expandable on the remote host."""
    return path if path.startswith("~/") else shlex.quote(path)


def _env_assign(env: Dict[str, Any]) -> str:
    return " ".join(f"{k}={shlex.quote(str(v))}" for k, v in (env or {}).items())


def _project_script(cfg: ARConfig, script: str) -> str:
    root = os.path.abspath(cfg.project_root)
    path = os.path.abspath(os.path.join(root, script))
    if not (path == root or path.startswith(root + os.sep)):
        raise ValueError(f"EXP_SPEC.script escapes project root: {script}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"EXP_SPEC.script not found: {script}")
    return path


def _shadow_script_path(cfg: ARConfig, rel: str) -> str:
    """Теневая копия ОДОБРЕННОГО ревью скрипта ВНЕ вольта (фикс волны 194).

    Канонический скрипт лежит в `Results/scripts/` — а это (а) папка Syncthing-вольта, общая с
    Маком, и (б) рабочее пространство, которое отдаётся codex-детям (артефакт `_exp_wip_<kind>.py`
    пишется туда же). Поэтому одобренную версию может затереть кто угодно БЕЗ СЛЕДА: `.prev` не
    трогается, строки «прошёл код-ревью» в логе нет. Улика волны 194: версия 27851 симв с
    `full_gn`×12/`"row"`×5/`per_arm_time_s`/`parameter_check` (единственная одобренная, коммит
    04:57) к 08:17 стала БАЙТ-В-БАЙТ прежней 19960-байтной без НИ ОДНОГО из этих токенов, и десять
    волн ушло на вопрос «почему в скрипте 0 вхождений контракта». Тень держим вне вольта и вне
    workspace codex'а."""
    name = re.sub(r"[^a-zA-Z0-9_.-]", "_", os.path.basename(os.path.abspath(cfg.project_root)))
    d = os.path.expanduser(os.path.join("~/.autoresearch_script_vault", name))
    return os.path.join(d, os.path.basename(rel))


# ОБРАЗЦЫ-КАРКАСЫ: скрипты, написанные руками и НЕ пересеваемые автогеном (фикс волны 171). Одна
# константа на все три точки использования — волна 193 показала, что пара «строка запуска ↔ строка
# поиска» в разных местах файла расходится молча; здесь тот же класс (пересев/усыновление тени).
_REGEN_SKIP = ("exp_llama_checkpoint_pool_selector.py", "exp_llama_gnmuon_audit.py")


def _shadow_save(cfg: ARConfig, rel: str, code: str) -> None:
    try:
        sp = _shadow_script_path(cfg, rel)
        os.makedirs(os.path.dirname(sp), exist_ok=True)
        with open(sp + ".tmp", "w", encoding="utf-8") as f:
            f.write(code)
        os.replace(sp + ".tmp", sp)
    except OSError as e:
        logger.warning("[exp] тень скрипта %s не сохранена: %s", rel, e)


def _shadow_guard(cfg: ARConfig, kind: str, rel: str) -> bool:
    """Сверить канонический скрипт с тенью и ВОССТАНОВИТЬ его, если подменён извне.

    Возвращает True, если восстановление состоялось. mtime восстановленного файла = mtime тени
    (время одобрения), чтобы логика «уроки свежее скрипта» продолжала работать как раньше и
    улучшение шло дальше от ОДОБРЕННОЙ базы, а не от подменённой."""
    try:
        sp = _shadow_script_path(cfg, rel)
        path = os.path.abspath(os.path.join(os.path.abspath(cfg.project_root), rel))
        if not os.path.exists(sp):
            # ФИКС ВОЛНЫ 197: образцы-каркасы (`_regen_skip`) НИКОГДА не проходят accept-путь, значит
            # `_shadow_save` для них не звался и тени у них нет — то есть единственная РАБОТАЮЩАЯ
            # реализация 4 рук (`exp_llama_gnmuon_audit.py`, живой рычаг оси E) не защищена от
            # внешней перезаписи из вольта, ровно от которой спасал фикс волны 194. Усыновляем её
            # содержимое как эталон при первом же использовании.
            if os.path.basename(path) in _REGEN_SKIP and os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        _seed = f.read()
                    if _seed.strip():
                        _shadow_save(cfg, rel, _seed.decode("utf-8", "replace"))
                        logger.info("[exp] образец-каркас %s усыновлён в тень (%d б) — теперь его "
                                    "подмена извне будет замечена", rel, len(_seed))
                except OSError:
                    pass
            return False
        shadow = open(sp, "rb").read()
        try:
            cur = open(path, "rb").read()
        except OSError:
            cur = b""
        if cur == shadow:
            return False
        import hashlib
        _h = lambda b: hashlib.sha256(b).hexdigest()[:12]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".ext", "wb") as f:  # что лежало на диске — на всякий случай рядом
            f.write(cur)
        with open(path + ".new", "wb") as f:
            f.write(shadow)
        os.replace(path + ".new", path)
        st = os.stat(sp)
        os.utime(path, (st.st_mtime, st.st_mtime))
        logger.warning("[exp] ⚠️ скрипт kind=%s ПОДМЕНЁН ИЗВНЕ (на диске %d б sha %s, одобренный "
                       "%d б sha %s) — ВОССТАНОВЛЕН из тени %s; подменённая версия рядом в .ext",
                       kind, len(cur), _h(cur), len(shadow), _h(shadow), sp)
        return True
    except OSError as e:
        logger.warning("[exp] сверка скрипта kind=%s с тенью не удалась: %s", kind, e)
        return False


def _copy_project_script(cfg: ARConfig, srv: str, nid: str, kind: str, script: str) -> str:
    _shadow_guard(cfg, kind, script)  # фикс в.194: на GPU уезжает ОДОБРЕННАЯ версия, а не подменённая
    local = _project_script(cfg, script)
    remote_dir = "~/autoresearch_exp_scripts"
    remote = f"{remote_dir}/exp_{_safe(nid + '_' + kind)}_{os.path.basename(script)}"
    mk = subprocess.run(["ssh", srv, f"mkdir -p {remote_dir}"],
                        capture_output=True, text=True, timeout=30)
    if mk.returncode != 0:
        raise RuntimeError(f"mkdir failed on {srv}: {(mk.stderr or mk.stdout)[:200]}")
    cp = subprocess.run(["scp", "-q", local, f"{srv}:{remote}"],
                        capture_output=True, text=True, timeout=60)
    if cp.returncode != 0:
        raise RuntimeError(f"scp failed to {srv}: {(cp.stderr or cp.stdout)[:200]}")
    return remote


_EXP_SCRIPT_PROMPT = """\
Ты ЭКСПЕРИМЕНТАТОР. Напиши САМОДОСТАТОЧНЫЙ Python-скрипт llama-эксперимента для typed EXP_SPEC.
Домен: {domain}
EXP_SPEC:
  kind: {kind}
  claim: {claim}
  success_metric: {success_metric}
  confirm_rule: {confirm_rule}
ЖЁСТКИЕ ТРЕБОВАНИЯ:
- Модель — llama-like FROM SCRATCH через хелперы `llama_exp_kit` (НИКАКОЙ pythia, НИКАКИХ HF-загрузок).
  РЕАЛЬНЫЙ API llama_exp_kit (импортируй ТОЛЬКО существующее): build_llama, eval_loss,
  make_optimizer(model, opt_name, lr), train, run_forgetting, _bootstrap_diff.
  Оптимизаторы бери ТОЛЬКО через make_optimizer (opt_name: "adamw"/"muon"/...); Muon(=D-Muon, с
  Newton-Schulz внутри) уже там. НЕ импортируй `_newton_schulz` из llama_exp_kit — его НЕТ (будет
  invalid). Если нужен сырой NS/Muon — он в `optim.muon` (llm-baselines, доступен по PYTHONPATH).
  НЕ пиши свой оптимизатор. Перед импортом не выдумывай символы — используй только перечисленные.
- **ЛОГИТЫ МОДЕЛИ (фикс волны 197, детерминированный креш):** сигнатура llama-модели —
  `forward(idx, targets=None, get_logits=False)`, и внутри стоит `logits = logits if get_logits
  else None`. Поэтому `m(x)["logits"]` — это **None**, а `m(x)["logits"].float()` падает
  `AttributeError: 'NoneType' object has no attribute 'float'` (наблюдалось: прогон 124M честно
  отучился 600 шагов, напечатал parameter_check и УМЕР на первой же оценке → `EMP_VERDICT: invalid`,
  и так на КАЖДОМ прогоне этого kind'а). Нужны логиты/лог-вероятности → ВСЕГДА
  `m(x, get_logits=True)["logits"]`. Нужен только лосс → `m(x, targets=y)["loss"]` или `eval_loss`
  кита. Любое обращение к `["logits"]` без `get_logits=True` — гарантированный invalid.
- ДАННЫЕ — ТОЛЬКО через хелперы кита: `_tokens_wikitext()` и `_tokens_shakespeare()` (оба возвращают
  np.ndarray токенов, с HF-фолбэком и кешем внутри), split — через `_split(toks, frac)`, батч — `_batch`.
  Доступны РОВНО ДВА домена: wikitext и shakespeare. НЕ используй gsm8k/c4/openwebtext или любые
  другие датасеты (их НЕТ → рантайм-креш). НЕ пиши свою функцию скачивания, НЕ хардкодь URL/mirror
  (github-mirror'ы мертвы → HTTP 404 → invalid). Для переноса домен A→B бери оба: wikitext и shakespeare.
- argv: sys.argv[1]=device (cuda:N); sys.argv[2]=model-тег (строка); опц. `--out <file>` (финальный JSON туда).
- Размер модели/бюджет/сиды бери ИЗ ENV (os.environ) с разумными дефолтами: N_EMBD, N_LAYER, N_HEAD,
  SEQ, BATCH, PRE_STEPS, FT_STEPS, SEEDS (csv, напр. "0,1,2,3,4"). НЕ хардкодь мелкие значения —
  проект задаёт ~60M через env. Печатай использованные значения первой строкой.
- Несколько сидов (>=3, из SEEDS); печатай РЕАЛЬНЫЕ числа; bootstrap-CI где уместно; assert только на гарантируемом.
- СИДЫ: валидируй УНИКАЛЬНОСТЬ, не только длину — `assert len(set(seeds)) >= 3` (SEEDS=0,0,0 = три копии
  одного запуска, ревью отклонит). Целые пиши БЕЗ ведущих нулей (0350 — SyntaxError в Python3).
- МАСШТАБ МОДЕЛИ: если claim заявляет масштаб (124M/350M/760M), посчитай и напечатай ЧИСЛО ПАРАМЕТРОВ
  (`sum(p.numel() for p in model.parameters())`) и `assert`, что оно в ±30% заявленного масштаба —
  иначе ревью отклонит «N_EMBD=512 не является сетью масштаба 124M».
- В САМОМ КОНЦЕ напечатай РОВНО три строки:
    RESULT_JSON: {{<ключевые числа + CI под confirm_rule>}}
    EMP_VERDICT: confirmed|no_signal|invalid
    EXP_DONE
  (confirmed ТОЛЬКО если confirm_rule реально выполнен на числах).
- МЕТОДОЛОГИЯ (иначе код-ревью отклонит): метрику реализуй ИМЕННО как в claim — если claim про
  spectral/C_energy, считай по СИНГУЛЯРНЫМ ЧИСЛАМ/spectral norm (torch.linalg.svdvals), НЕ Frobenius/L2
  суррогат. Любую калибровку/clock делай ЧЕСТНО held-out: подгоняй на ОДНИХ checkpoints/данных, оценивай
  ошибку на ДРУГИХ (никакого leakage — не фитить и не мерить на тех же строках). Bootstrap-CI строй по
  >=3 валидным сидам. NaN/невалидное значение метрики → EMP_VERDICT invalid (НЕ молча пропускать в rejected).
  Модель строится с нуля через build_llama (реального LLaMA-чекпоинта нет — это правильно).
- КОНСТРУКЦИЯ ДЛЯ IMPOSSIBILITY / INDISTINGUISHABILITY (если claim про «target-only не сертифицирует»,
  «две конфигурации неразличимы», impossibility): построй РОВНО ДВЕ инстанции, у которых ВСЁ НАБЛЮДАЕМОЕ
  target-only совпадает (ОДНА общая target-finetune траектория / идентичный target-loss+градиенты+спектр),
  а РАЗЛИЧАЮТСЯ ТОЛЬКО source-компонентой (source-градиент/направление), дающей ПРОТИВОПОЛОЖНЫЙ знак
  source-retention Delta. НЕ pretrain'ь на разных корпусах (это уже различает target-наблюдаемое → invalid).
  Проверь численно: |target-observable_A − target-observable_B| < tol (одинаково) И sign(retention_A) !=
  sign(retention_B). Отсюда target-only rule имеет err=1/2. Bootstrap-CI по >=3 seeds.
  РЕЦЕПТ sign-paired source-пары (как в теореме, БЕЗ фейка): один общий target-finetune даёт W_0→W_T и
  направление D = W_T − W_0. Миры «+» и «−» обязаны быть CRN-ПАРНЫМИ: ОБЩИЙ пул source-примеров и
  ОБЩИЙ RNG/сиды для обоих (НЕ разные непересекающиеся токены и НЕ разные сиды — это конфаунд,
  ревью отклонит «paired объявлен постфактум»). Различие ТОЛЬКО в знаковой компоненте: например,
  метки/таргеты со сдвигом +beta vs −beta вдоль D на ОДНИХ И ТЕХ ЖЕ примерах (CE-конструкция
  теоремы), либо парная перестановка знака вклада каждого примера. При этом величины обоих миров
  считаются НЕЗАВИСИМЫМИ честными форвардами (не отрицанием массива соседа).
  Затем НЕЗАВИСИМО вычисли retention_plus = L_Splus(W_0)−L_Splus(W_T) и retention_minus по S_minus —
  ОБА честными форвардами по СВОИМ данным. ЗАПРЕЩЕНО задавать minus-величины как отрицание/копию plus
  (`ret_minus = -ret_plus`, `probe_minus = -probe_plus` и т.п.) — код-ревью это отклоняет как фейк:
  двух миров при этом реально НЕТ. Если после честного построения знаки НЕ противоположны — печатай
  реальные числа и EMP_VERDICT=no_signal, НЕ подгоняй.
- ПРОТОКОЛ-ПОЛНОТА (частые причины отклонения, соблюдай СРАЗУ): (1) если claim называет НЕСКОЛЬКО
  масштабов (напр. 124M И 760M) — ОДИН скрипт обязан прогнать ВСЕ (цикл по тегам/env-пресетам),
  confirmed только если правило выполнено на КАЖДОМ; (2) каждый control реализуй ЧЕСТНО: full-GN
  контроль = настоящее GN/KFAC-взвешенное направление (список/матрица), а не один скаляр-суррогат
  вида d^T G d; (3) ВСЕ статистики/bootstrap-CI считай по ВСЕМ заявленным сидам (len(SEEDS)) — если
  часть сидов ушла на selection-split, объяви это в RESULT_JSON и считай certification-CI по своему
  сплиту, но НЕ заявляй 12 сидов при CI на 6; (4) ЧЕСТНЫЙ НЕЙМИНГ контролей: если контроль — прокси
  (K-FAC низкого ранга, диагональный Фишер и т.п.), называй его kfac_proxy/gn_proxy, а НЕ full_gn —
  ревью отклоняет overclaim в имени; target-only baseline-правила МОГУТ быть прокси, это валидно,
  главное — точное имя и описание в RESULT_JSON; (5) NON-INFERIORITY/разности измеряй с РЕАЛЬНОЙ
  вариацией: eval_loss кита ДЕТЕРМИНИРОВАН (seed=0) — два вызова на одном чекпоинте дают идентичные
  числа и НИЧЕГО не измеряют; сравнивай на РАЗНЫХ held-out блоках данных (разные срезы токенов на
  блок) или разных eval-сидах, чтобы разность имела дисперсию и CI был содержательным.
- КОНСТРУКЦИЯ ДЛЯ ACHIEVABILITY / SOURCE-PROBE (если claim про «source-probe восстанавливает
  сертифицируемость»): на РЕАЛЬНОЙ finetune-траектории посчитай дешёвый source-probe (напр.
  sign(v^T H_S u) или source-gradient-alignment) и покажи, что он ПРЕДСКАЗЫВАЕТ знак retention с
  bootstrap-CI выше chance (0.5), held-out по чекпоинтам. Это проще impossibility — просто корреляция
  probe↔retention на реальной сети.
- OOM-САМОАДАПТАЦИЯ (обязательно): оберни обучение/финетюн в try/except torch.cuda.OutOfMemoryError
  (и generic RuntimeError с 'out of memory'): при OOM вызови torch.cuda.empty_cache(), УМЕНЬШИ batch
  вдвое (BATCH из env как СТАРТ, дальше половинь до min 1) и ПОВТОРИ; логируй 'OOM: batch->N retry'.
  Скрипт САМ подстраивает batch под память карты (модель 124M может не влезть на полном batch) —
  НЕ падай с CUDA OOM в invalid, а адаптируйся и доведи прогон до RESULT_JSON.
- Скрипт обязан запускаться (exit 0) и проходить ast.parse. Только код, без prose.
- ДЛИНА: держи скрипт КОРОТКИМ (<= ~180 строк). ВСЮ тяжёлую логику (finetune, eval, bootstrap,
  оптимизаторы, токены) бери ГОТОВОЙ из llama_exp_kit/transfer_protocol — НЕ переписывай её с нуля.
  Короткий скрипт = ты гарантированно ДОПИШЕШЬ его до конца (RESULT_JSON+EMP_VERDICT+EXP_DONE) без
  обрыва. НИКОГДА не обрывай вывод на полуслове: лучше меньше вспомогательного кода, но полный скрипт
  с финальными тремя строками. Если не помещаешься — сократи, но ЗАВЕРШИ маркерами.
Выведи РОВНО один ```python блок."""


_EXP_REVIEW_PROMPT = """\
Ты СТРОГИЙ КОД-РЕВЬЮЕР экспериментов. Проверь: РЕАЛЬНО ли скрипт измеряет заявленное И валиден ли он.
Красота НЕ важна — важна ЧЕСТНОСТЬ и соответствие claim.
EXP_SPEC:
  claim: {claim}
  success_metric: {success_metric}
  confirm_rule: {confirm_rule}
  env ЭТОГО прогона (ФАКТИЧЕСКИЕ значения, с которыми скрипт будет запущен): {env}
СКРИПТ:
```python
{code}
```
Проверь по пунктам:
1) Скрипт РЕАЛЬНО делает то, что в claim (те оптимизаторы/метрика/сравнение/фаза, а не заглушка/другое).
2) Реальное обучение/вычисление (llama_exp_kit, несколько сидов), а НЕ хардкод чисел.
3) EMP_VERDICT выводится ИЗ ПОСЧИТАННЫХ чисел по confirm_rule (НЕ захардкожен "confirmed", НЕ всегда один и тот же).
4) Печатает RESULT_JSON + EMP_VERDICT + EXP_DONE; синтаксически исполним; берёт размер/сиды из env.
5) Нет читерства: не «подтверждает» тривиально, метрика соответствует confirm_rule.
ПРИНЯТАЯ УСТАНОВКА (НЕ штрафуй за это): модель — llama-like FROM SCRATCH через `build_llama` кита;
реального pretrained-LLaMA-чекпоинта в окружении НЕТ и не будет. НЕ требуй загрузки настоящего
LLaMA-чекпоинта и НЕ отклоняй за «строится с нуля» — это правильный дизайн. Строгость МЕТОДОЛОГИИ
сохраняй полностью: метрика должна соответствовать claim (если claim про spectral — по сингулярным
числам/spectral norm, а НЕ Frobenius-суррогат), held-out без leakage (калибровка и оценка на РАЗНЫХ
checkpoints/данных), bootstrap по >=3 валидным сидам, NaN/невалидные метрики не должны молча проходить.
ПРИНЯТАЯ УСТАНОВКА №2 — ГДЕ ЖИВЁТ ГЕОМЕТРИЯ И ЧТО МОЖНО СУЖАТЬ (фикс волны 190; прогон ограничен
12 ч на ОДНОЙ карте, поэтому без права сужать бюджет гейт физически недостижим):
 - Геометрия каждого масштаба берётся ИСКЛЮЧИТЕЛЬНО из таблицы SCALES внутри скрипта (формат
   tag:n_embd,n_layer,n_head). Общие env-ключи N_EMBD/N_LAYER/N_HEAD геометрию тега НЕ переопределяют —
   так уже ломался тег 124M, получавший геометрию 350M. НЕ требуй, чтобы скрипт брал размер из этих
   env-ключей; env вправе задавать только бюджеты (SEEDS/PRE_STEPS/FT_STEPS/BATCH/SEQ) и СПИСОК тегов.
   Обязательно отклоняй, если тег не сверен с фактическим n_params_total до тяжёлого цикла.
 - СУЖАТЬ МОЖНО (и нужно, чтобы влезть в 12 ч): число шагов, точки LR-сетки (минимум 2 на руку), число
   обучающих сидов (минимум 3 НЕЗАВИСИМЫХ), число eval/cov-батчей. Легально ТОЛЬКО когда фактическое
   сужение против спеки объявлено ключом budget_reduction в RESULT_JSON; молчаливое сужение = подлог, FAIL.
 - НЕПРИКОСНОВЕННО (сужать нельзя, отсутствие = FAIL): число scale-тегов, которое требует claim (если
   claim говорит про два и более scale tags — прогон обязан отработать их все, урезание до одного тега
   недопустимо даже с объявленным budget_reduction); все руки/методы контраста из env (METHODS/ARMS/
   BASELINES) целиком; НЕЗАВИСИМЫЕ обучающие сиды (сиды обязаны менять инициализацию/порядок ОБУЧАЮЩИХ
   данных, а не только выбор eval-батчей); отдельный tuning-сплит для выбора LR; раздельные
   calibration- и certification-данные; bootstrap по кластерам сидов; пометка вырожденных полос.
ПРИНЯТАЯ УСТАНОВКА №3 — ГДЕ И КАК СКРИПТ ИСПОЛНЯЕТСЯ (фикс волны 196; НЕ штрафуй за это):
 - Скрипт исполняется НЕ на этой машине, а на УДАЛЁННОМ GPU-сервере, в его `$HOME` (torch+CUDA,
   A100). Файловая система машины, где ты сейчас читаешь код, к делу НЕ относится: отсутствие
   `/root/llm-baselines`, `/root/llama_exp_kit.py` и т.п. ЗДЕСЬ — не дефект скрипта. НЕ отклоняй
   за «в текущем окружении нет такого пути/пакета».
 - На целевом сервере ГАРАНТИРОВАННО есть: `~/llama_exp_kit.py` (build_llama, eval_loss,
   make_optimizer, train, run_forgetting, _bootstrap_diff, _tokens_wikitext, _tokens_shakespeare)
   и `~/llm-baselines/` с `src/models/llama.py`, `src/optim/muon.py`. Запуск экспортирует
   PYTHONPATH=$HOME:$HOME/llm-baselines:$HOME/llm-baselines/src, поэтому `from llama_exp_kit import …`,
   `from optim.muon import Muon`, `from models.llama import …` — ЗАКОННЫЕ импорты, как и
   дублирующий их `sys.path.insert(0, os.path.join(os.path.expanduser("~"), "llm-baselines", "src"))`.
 - **ОБЯЗАН ОТКЛОНИТЬ (фикс волны 197):** любое обращение к `["logits"]` результата модели без
   `get_logits=True` в ЭТОМ ЖЕ вызове. Сигнатура llama — `forward(idx, targets=None,
   get_logits=False)` и `logits = logits if get_logits else None`, поэтому `m(x)["logits"]` = None и
   скрипт падает `AttributeError` на первой оценке (наблюдалось: 600 шагов обучения впустую,
   вердикт invalid). Правильно: `m(x, get_logits=True)["logits"]`, либо `m(x, targets=y)["loss"]`,
   либо `eval_loss` кита. Это требование зеркально прописано писателю — гейт достижим.
ПРИНЯТАЯ УСТАНОВКА №4 — ДОСТИЖИМОСТЬ СЕРТИФИКАЦИИ ПРИ ФАКТИЧЕСКОМ БЮДЖЕТЕ СИДОВ (фикс волны 199):
 - **ОБЯЗАН ОТКЛОНИТЬ:** скрипт, у которого при ФАКТИЧЕСКИХ env встроенный power/feasibility-check
   даёт `achievable=False` и `run()` выходит `invalid`/`abstain` ДО обучения и до сравнения рук.
   Такой прогон не измеряет claim ни при каком качестве кода (наблюдалось: `SEEDS=0..7`, tail
   0.05/36, exact Clopper–Pearson UCB для 0 false accepts при пороге .2 требует 30 сидов ⇒ 12 ч GPU
   впустую). Требуй, чтобы скрипт считал минимально необходимое n ДО тяжёлого цикла и приводил
   конфигурацию к достижимой.
 - **НЕ штрафуй** за ОБЪЯВЛЕННУЮ деградацию критерия сертификации, если скрипт печатает ключи
   `certification_feasibility` (n_required, n_available, выбранная опция) и `budget_reduction`, а сама
   деградация идёт по этому порядку: (1) сузить simultaneous family до сравнений, реально входящих в
   headline; (2) поднять число certification-сидов ВНУТРИ скрипта до вычисленного минимума, если
   влезает в 12 ч (сиды дешевле невалидного прогона); (3) заменить exact 0-false-accept UCB на
   объявленный сертификат по НАБЛЮДЁННОЙ доле (Wilson/Clopper–Pearson) и понизить headline до
   «sufficient, not validated». Молчаливая деградация без этих ключей = подлог, FAIL.
 - **СТУПЕНЬ (3) ПРОВЕРЯЕТСЯ ПО РЕАЛИЗАЦИИ, НЕ ПО ДЕКЛАРАЦИИ (фикс волны 206).** Твоя же
   доминирующая претензия к трём последним kind'ам была верной и повторялась: «код заявляет
   option=3, но деградацию №3 не реализует — сертификата наблюдённой доли нет, certify()
   по-прежнему percentile seed-bootstrap с прежними порогами, а family_size_headline просто
   обнуляется». Теперь это ловится машинно ДО тебя: при `option=3` обязательны токены
   `wilson`/`clopper` и ДОСЛОВНАЯ строка «sufficient, not validated», а писателю выдана дословная
   референс-реализация `wilson_bounds`/`certify_option3` (stdlib `NormalDist`, alpha_adj =
   alpha/family_size_headline, двусторонняя Wilson-полоса, флаг `degenerate`). ⇒ (а) увидев ИМЕННО
   эту форму — НЕ отклоняй её как «нестандартную статистику» и не требуй exact Clopper–Pearson
   вместо Wilson: обе допустимы, выбор объявлен; (б) по-прежнему ОБЯЗАН отклонять, если
   `certify_option3` (или её аналог) лежит неиспользованным рядом с прежним percentile-путём —
   решение гейта должно приходить из `passed` сертификата наблюдённой доли; (в) ОБЯЗАН отклонять
   обнуление `family_size_headline` — headline-family остаётся >= 1 (сама проверяемая claim),
   демотировать ВСЕ гипотезы = ослабить критерий до пустоты.
 - **СТУПЕНЬ (4): ЗАМЕР БЕЗ СЕРТИФИКАЦИИ — ЗАКОННЫЙ ИСХОД (фикс волны 206).** Ступени (1)-(3)
   КОНЕЧНЫ, и это арифметика, а не мнение: для гейта «false-accept <= .2» идеальный исход 0 из n при
   family=1 даёт Wilson-верх .390 (n=6), .324 (n=8), .242 (n=12) и лишь .176 (n=18). Значит при
   6-8 сертификационных сидах порог .2 недостижим НИКАКИМ сертификатом, и твоя же прежняя претензия
   «confirmed математически недостижим при любых результатах» остаётся верной даже у безупречно
   реализованной ступени (3). Поэтому: если скрипт ДО тяжёлого цикла считает best-case границу,
   объявляет `"option": 4` + `"reason": "certification out of budget"` + `n_required_for_threshold`,
   ВЫПОЛНЯЕТ прогон целиком, печатает решающие величины как descriptive с фактической Wilson-полосой
   и полем `"certification": "out of budget"`, НЕ ставит `confirmed` и выходит
   `EMP_VERDICT: no_signal` — это ПРАВИЛЬНОЕ поведение, НЕ ОТКЛОНЯЙ его за «отсутствие
   сертификации». Отклоняй только: выход `invalid`/`abstain` ДО обучения; молчаливое занижение
   порога; `confirmed` по недостижимому гейту; отсутствие расчёта best-case границы.
 - **«SIMULTANEOUS FAMILY» — ЭТО НЕ «КОНТРАСТ РУК» (фикс волны 204).** Сужение headline-family
   (числа ОДНОВРЕМЕННО сертифицируемых гипотез) — РАЗРЕШЁННАЯ и ТРЕБУЕМАЯ ступень (1) лестницы
   выше; НЕ отклоняй за неё, если объявлена в `certification_feasibility` и все руки/методы
   контраста из env всё равно обучены, а вынесенные из family числа напечатаны как descriptive
   (без права открывать гейт). Запрет «сужать контраст» — про РУКИ, а не про family.
 - Писателю ДОСЛОВНО зеркально предписан пункт 6 машинного инфра-контракта: при наличии в коде
   любого из `feasib`/`n_required`/`achievable` ключ `certification_feasibility` обязателен и
   проверяется автоматически. Поэтому претензию «нет обоснования достижимости» формулируй через
   этот ключ, а не требуй недостижимого порога: если при ФАКТИЧЕСКОМ бюджете сидов нижняя граница
   сертификата не дотягивается до порога ни при каких данных (наблюдалось: family=68 при 3-6
   сидах, n_required=46), то ОТКЛОНЯТЬ надо отсутствие лестницы деградации, а НЕ её применение.

ПРИНЯТАЯ УСТАНОВКА №5 — ПРОВЕНАНС ОСИ R ПЕЧАТАЕТ САМ СКРИПТ (фикс волны 208; писателю сказано
ДОСЛОВНО зеркально — это пункт 5 его машинного инфра-контракта):
 - Директива №3 A*-рецензии требует per-task rows, bootstrap indices, model/data config и per-arm
   wall-clock. Ни луп, ни рецензент восстановить их задним числом НЕ могут («старые metadata не
   реконструировать»), поэтому их обязан напечатать прогон: ключи `per_block_rows` (устойчивый
   `block_id`, строка на блок×руку), `bootstrap_indices` (`bootstrap_seed`, `cluster_key`, сами
   индексы или их sha256; индексы ОБЩИЕ для рук — CRN) и `run_provenance` (sha256 конфига/данных/
   токенайзера, `model_init_seed`, `replicates`, `rng_seeds`, `n_params_actual`).
 - **ОБЯЗАН ОТКЛОНИТЬ** скрипт без этих ключей (проверяется и машинно, ДО тебя) и скрипт, где
   `block_id` — просто порядковый индекс перечисления (реанализ по нему невозможен) или где руки
   ресэмплятся РАЗНЫМИ индексами (тогда familywise CI не пересчитывается).
 - **НЕ требуй** от скрипта команду запуска, sha256 самого файла, training commit и отпечаток
   окружения: их снимает САМ луп (в.167/в.205), дублирование в RESULT_JSON — не дефект и не
   достоинство. И НЕ штрафуй за `"indices": {{"sha256": …}}` вместо полного массива при больших N.
 - НЕПРИКОСНОВЕННО и здесь: раздельность calibration-/certification-данных, seed-кластерный
   bootstrap, полный контраст рук (METHODS/ARMS/BASELINES), число scale-тегов из claim.
 - ОТКЛОНЯЙ по окружению ТОЛЬКО: захардкоженные абсолютные пути чужих машин (`/root/...`,
   `/home/<чужой-юзер>/...`) вместо `$HOME`/`os.path.expanduser`; импорт символов, которых в ките
   нет (напр. `_newton_schulz`); скачивание датасетов мимо хелперов кита.
Ответь РОВНО:
ISSUES: <кратко конкретные проблемы, или "нет">
REVIEW_VERDICT: PASS|FAIL
(PASS только если скрипт действительно проверяет заявленное и честен.)"""


def _review_exp_script(cfg: ARConfig, kind: str, spec: Dict[str, Any], code: str) -> Tuple[bool, str]:
    """Независимый код-эксперт (gate_code) проверяет, что самописный скрипт РЕАЛЬНО тестирует claim
    и не подделывает вердикт. Возвращает (ok, issues)."""
    # ФИКС ВОЛНЫ 190: ревьюер видел ТОЛЬКО claim и потому требовал от скрипта цикл по всем масштабам
    # claim'а, тогда как геометрию задаёт env ЭТОГО прогона (одна шкала за запуск, остальные — в
    # родственных узлах _emp_<scale>). Отсюда отказ «RUN_SCALES=REQUESTED_SCALES[:1] запускает только
    # одну шкалу» на версии, которая ровно ВЫПОЛНЯЛА контракт сужения бюджета волны 189: писателю
    # велено сузить, ревьюеру велено (неявно) отклонять сужение. Даём ревьюеру фактический env.
    out = engine_call(cfg.engines.gate_code, cfg)(_EXP_REVIEW_PROMPT.format(
        claim=str(spec.get("claim", ""))[:800],
        success_metric=str(spec.get("success_metric", ""))[:400],
        confirm_rule=str(spec.get("confirm_rule", ""))[:400],
        env=json.dumps(spec.get("env") or {}, ensure_ascii=False)[:800],
        code=code[:48000]))  # полный скрипт 60M-эксп (>12k симв) — иначе ревьюер видит обрезок без финального RESULT_JSON/EMP_VERDICT и всегда FAIL
    verdict = grep_tail(out, "REVIEW_VERDICT:").upper()
    issues = grep_tail(out, "ISSUES:")
    ok = verdict.startswith("PASS")
    return ok, (issues if issues != "?" else out[-400:])


_MECH_CONTRACT = (
    ('"row"', 'нет прогресс-строк print(json.dumps({"row": {...}}), flush=True) после каждой '
              'тройки (масштаб, сид, метод) — многочасовой прогон неотличим от зависшего'),
    ("per_arm_time_s", 'нет ключа "per_arm_time_s" (wall-clock секунды КАЖДОЙ руки/метода) — '
                       'рецензент требует стоимость на руку, а не только суммарный wall_clock_s'),
    ("parameter_check", 'нет печати {"parameter_check": {"tag":…, "n_params_total":…, "ok":…}} '
                        'ДО тяжёлого цикла — тег масштаба ничем не сверен с фактической геометрией'),
    # ФИКС ВОЛНЫ 190: сужение бюджета разрешено (иначе сетка не влезает в 12 ч), но ТОЛЬКО объявленное.
    # Ревьюеру сказано «сужение легально, если объявлено ключом budget_reduction» — значит наличие
    # ключа обязано проверяться машинно, иначе мягкий судья пропустит молчаливое сужение (урок в.185).
    ("budget_reduction", 'нет ключа "budget_reduction" в RESULT_JSON (что именно сужено против спеки: '
                         'шаги/точки LR/число сидов/масштаб; пустой dict, если не сужено ничего) — '
                         'молчаливое сужение сетки неотличимо от подлога'),
    # ФИКС ВОЛНЫ 208 — ПРОВЕНАНС ОСИ R. Директива №3 A*-рецензии («сохранить per-task rows, bootstrap
    # indices, exact command/hash/commit/environment/model-data config и per-arm wall-clock») требует
    # ровно тех величин, которые НИКТО, кроме самого скрипта, дать не может. Волна 207 написала это
    # контрактом в уроки, но (а) только для kind'а `llama_source_bit_frontier_audit`, а ВСЕ узлы
    # директив критика — kind `custom_llama`, и (б) без машинной проверки: две одобренные 01-08
    # версии (`exp_custom_eab06e246e.py`, `exp_custom_85239ab438.py`) содержат все ТРИ прежних
    # мех-токена и НОЛЬ этих — мягкий судья пропустил ровно то, что не проверялось (класс в.185).
    ("per_block_rows", 'нет ключа "per_block_rows" (строка на КАЖДЫЙ блок/задачу с устойчивым '
                       '"block_id") — рецензент требует per-task rows, реанализ чужими руками '
                       'невозможен, ось R остаётся partial'),
    ("bootstrap_indices", 'нет ключа "bootstrap_indices" (общие для всех рук индексы ресэмпла + '
                          '"bootstrap_seed"/"cluster_key") — CI невоспроизводим, familywise правило '
                          'нельзя пересчитать'),
    ("run_provenance", 'нет ключа "run_provenance" (sha256 конфига/данных/токенайзера, '
                       '"model_init_seed", "replicates", "rng_seeds", "n_params_actual") — прогон '
                       'auditable, но НЕ rerunnable, а это дословная претензия W4'),
)


def _mech_contract_violations(code: str) -> List[str]:
    """Детерминированная проверка инфра-контракта A*-рецензии (фикс волны 185).

    Мотивация: код-ревью — МЯГКИЙ судья (codex), он пропускал версии без единого требуемого токена
    (свежий `exp_llama_source_bit_frontier_audit.py`, одобренный 09:50 UTC 31-07: `{"row"` 0,
    `per_arm_time_s` 0, `parameter_check` 0 — при 11 блоках-КОНТРАКТАХ в уроках). Машинная проверка
    даёт писателю ТОЧНУЮ, невыторговываемую обратную связь. Гейт НЕ применяется на последней попытке
    (см. вызов) — иначе kind без одобренного скрипта заморозился бы навсегда."""
    out = [why for token, why in _MECH_CONTRACT if token not in code]
    # УСЛОВНОЕ ПРАВИЛО (фикс волны 197): `m(x)["logits"]` — это None (llama: `forward(idx,
    # targets=None, get_logits=False)`, `logits = logits if get_logits else None`), поэтому
    # `.float()` на нём падает AttributeError на ПЕРВОЙ оценке. Прогон 124M успел отучить 600
    # шагов, напечатать parameter_check — и умер с `EMP_VERDICT: invalid`; kind терял каждую
    # попытку по одной и той же причине, а мягкий код-ревьюер её не видел.
    # УСЛОВНОЕ ПРАВИЛО (фикс волны 204): ревьюер (ПРИНЯТАЯ УСТАНОВКА №4) ОБЯЗАН отклонять скрипт, чья
    # power/feasibility-проверка либо захардкожена, либо уводит прогон в `invalid` ДО обучения, и НЕ
    # штрафовать объявленную деградацию с ключом `certification_feasibility`. Писатель про этот ключ
    # до в.204 не знал вовсе ⇒ гейт был недостижим (тот же класс, что в.190, но в обратную сторону):
    # 26 отказов подряд (llama_source_bit_frontier_audit 21 + два custom_llama-скрипта 3 и 2).
    # Правило УСЛОВНОЕ: kind'ы без сертификации вообще его не касаются — иначе новый вечный дедлок.
    if any(t in code for t in ("n_required", "feasib", "achievable")) and "certification_feasibility" not in code:
        out.append('есть power/feasibility-проверка, но нет ключа "certification_feasibility" '
                   '(n_available, n_required, family_size_headline, family_size_demoted, option, '
                   'reason) в RESULT_JSON: ревьюер не может отличить объявленную деградацию '
                   'критерия от подлога и обязан отклонить. Достижимость приводить лестницей '
                   '(1) сузить simultaneous family до headline → (2) поднять число certification-'
                   'сидов до вычисленного минимума → (3) Wilson/Clopper-Pearson по наблюдённой доле '
                   'и headline «sufficient, not validated». Выход invalid/abstain ДО обучения '
                   'запрещён')
    # УСЛОВНОЕ ПРАВИЛО (фикс волны 206): ступень (3) лестницы ОБЪЯВЛЯЛАСЬ, но НЕ РЕАЛИЗОВЫВАЛАСЬ.
    # Доминирующая причина отказов у ВСЕХ трёх живых kind'ов (дословно из `.run/exp_reseed_state`):
    # «Код заявляет option=3, но разрешённую деградацию №3 не реализует: сертификата наблюдённой доли
    # с Wilson/Clopper–Pearson нет, certify() по-прежнему использует прежний percentile seed-bootstrap
    # и прежние пороги, а family_size_headline просто обнуляется». То есть `certification_feasibility`
    # писатель понял как ДЕКЛАРАЦИЮ-отписку. Гейт в.204 наличие ключа проверял, содержание — нет,
    # поэтому каждая попытка доходила до мягкого судьи и сгорала. Требование машинно-проверяемо и
    # ДОСТИЖИМО: дословная референс-реализация выдана писателю в промпте (класс в.191 — не требовать
    # того, чем нельзя воспользоваться), и ровно эти же токены названы ревьюеру.
    if re.search(r'option["\']?\s*[:=]\s*3', code):
        _low = code.lower()
        if ("wilson" not in _low) and ("clopper" not in _low):
            out.append('объявлена ступень (3) лестницы достижимости (option=3), но в коде нет '
                       'сертификата по НАБЛЮДЁННОЙ доле: ни `wilson`, ни `clopper`. Ступень (3) — '
                       'это ЗАМЕНА статистики, а не строка в RESULT_JSON: exact 0-false-accept UCB / '
                       'percentile seed-bootstrap обязаны быть заменены на Wilson/Clopper-Pearson по '
                       'фактическим k из n (Wilson при 6/6 даёт невырожденную полосу, percentile — '
                       'CI=[1,1], т.е. НЕ измерение). Возьми ДОСЛОВНО референс-реализацию '
                       '`wilson_bounds`/`certify_option3` из пункта 5 задания')
        if "sufficient, not validated" not in code:
            out.append('объявлена ступень (3), но headline не понижен: в коде нет ДОСЛОВНОЙ строки '
                       '"sufficient, not validated". Ступень (3) законна только вместе с явным '
                       'понижением статуса headline — иначе это молчаливое ослабление критерия '
                       '(подлог). И НЕ обнуляй family_size_headline: демотировать ВСЕ гипотезы '
                       'нельзя, headline-family обязана остаться >= 1 (сама проверяемая claim)')
    # УСЛОВНОЕ ПРАВИЛО (фикс волны 233): СМЕШАННЫЕ ЧАСЫ В per-arm wall-clock. Дословный отказ
    # код-ревью custom_llama (попытка 3, 01-08 23:49): «`setup_s=time.perf_counter()-T0` смешивает
    # часы разных эпох (`T0` получен через `time.time()`), поэтому у всех recipient-рук
    # `per_arm_time_s` будет огромным отрицательным числом. Это делает обязательный per-arm
    # wall-clock provenance ложным; проверка `badnum` дефект не ловит, поскольку число конечное».
    # Класс: контракт в.185 требует ключ `per_arm_time_s`, но его ПРАВДИВОСТЬ не проверялась ничем —
    # мягкий ревьюер ловит это через раз, а каждая поимка стоит целого codex-раунда (15-25 мин) и
    # пишет килобайты в уроки (давление вытеснения курированных блоков, класс в.189/190). Проверка
    # детерминированная и без ложных срабатываний: имя должно быть присвоено РОВНО одной из двух
    # функций (переприсвоенные обеими — пропускаем).
    _t_names = set(re.findall(r"(\w+)\s*=\s*time\.time\(\)", code))
    _p_names = set(re.findall(r"(\w+)\s*=\s*time\.perf_counter\(\)", code))
    _mixed = sorted(
        {n for n in re.findall(r"time\.perf_counter\(\)\s*-\s*(\w+)", code)
         if n in _t_names and n not in _p_names}
        | {n for n in re.findall(r"time\.time\(\)\s*-\s*(\w+)", code)
           if n in _p_names and n not in _t_names}
    )
    if _mixed:
        out.append('СМЕШАННЫЕ ЧАСЫ в замере времени: разность берётся между time.perf_counter() и '
                   'переменной из time.time() (или наоборот) для ' + ", ".join(_mixed[:6]) + '. Это '
                   'РАЗНЫЕ эпохи: результат — конечное, но бессмысленное (обычно огромное '
                   'отрицательное) число, поэтому обязательный per-arm wall-clock provenance '
                   'оказывается ложным, а проверка на nan/inf дефект не видит. Засекай и вычитай '
                   'ОДНОЙ функцией: t0 = time.perf_counter(); …; per_arm_time_s = '
                   'time.perf_counter() - t0')
    if '["logits"]' in code and "get_logits=True" not in code:
        out.append('обращение к ["logits"] без get_logits=True: llama-модель возвращает '
                   'logits=None по умолчанию → AttributeError на первой же оценке (гарантированный '
                   'invalid). Нужно m(x, get_logits=True)["logits"], либо m(x, targets=y)["loss"], '
                   'либо eval_loss кита')
    return out


# КОНТРАСТНЫЕ ENV-КЛЮЧИ (фикс волны 188): ключи, которые несут САМ НАУЧНЫЙ КОНТРАСТ спеки — набор
# рук/методов, сетки LR для тюнинга каждой руки, лестницу масштабов, базлайны. Если такой ключ
# объявлен в `exp_spec.env`, а скрипт его НЕ читает и даже не упоминает его значения, то прогон
# физически измеряет НЕ ТО, что заявлено в claim/success_metric.
_CONTRAST_ENV_RE = re.compile(
    r"^(METHODS|ARMS|OPTS?|OPTIMIZERS?|LR_GRID(_[A-Z0-9_]+)?|BASELINES?|CONTROLS?|SCALES)$")
_IDENT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _contrast_gap(spec: Dict[str, Any], code: str, strict: bool = False) -> List[str]:
    """Контрастные ENV-ключи спеки, которых код не реализует НИКАК (фикс волны 188).

    Мотивация (волна 188, диагноз): у `llama_source_bit_frontier_audit` спека объявляла
    `METHODS=gnmuon,muon,adamw,full_gn` + четыре `LR_GRID_*`, а одобренный скрипт не содержал ни
    одного вхождения `muon`/`full_gn`/`LR_GRID` — он честно мерил abstention-rules на AdamW-траектории.
    Последствия каскадом: (1) код-ревью пересева ОБЯЗАНО отклонять любую версию, минимально
    отличную от такой БАЗЫ («Claim не проверяется: Muon и full-GN отсутствуют, нет tuned grid»), то
    есть требование волны 187 «минимальный дифф» прямо противоречило требованию ревьюера →
    незакрываемый цикл отказов; (2) многочасовой прогон мог напечатать `EMP_VERDICT: confirmed` по
    своей внутренней метрике и узел получил бы ПОДЛОГ вердикта — «confirmed» для claim'а, чей
    контраст в коде отсутствует.

    Ключ пропущен, если в коде нет ни его имени, ни ВСЕХ его ИМЕНОВАННЫХ значений. Требуется именно
    ВСЕ (не «хотя бы одно»): у `METHODS=gnmuon,muon,adamw,full_gn` слово `AdamW` встречается в любом
    finetune-скрипте, поэтому проверка «хотя бы одно» пропускала главный ключ (проверено на живом
    скрипте волны 188 — гейт молчал именно там, где контраст отсутствовал).

    `strict=True` (используется анти-подлогом в `_finalize`) оставляет ТОЛЬКО ключи с ИМЕНОВАННЫМИ
    значениями, отсутствие которых — прямое доказательство, что рук/масштабов в коде нет. Ключи с
    числовыми значениями (`LR_GRID_MUON=0.01,0.02`) в strict не попадают: скрипт мог держать сетку
    внутри кода, и понижать по такому признаку живой вердикт было бы ложным срабатыванием.
    """
    env = (spec or {}).get("env") or {}
    if not isinstance(env, dict):
        return []
    low = code.lower()
    out: List[str] = []
    for k in sorted(env):
        key = str(k)
        if not _CONTRAST_ENV_RE.match(key) or key in code:
            continue
        vals = [v.strip() for v in str(env[k]).replace(";", ",").replace(":", ",").split(",")
                if v.strip()]
        names = [v for v in vals if _IDENT_RE.match(v)]
        if names:
            missing = [n for n in names if n.lower() not in low]
            if not missing:
                continue
            out.append(f"{key} (нет: {', '.join(missing)})")
        elif not strict:
            out.append(key)
    return out


def _missing_arm_names(spec: Dict[str, Any], code: str) -> List[str]:
    """ИМЕНОВАННЫЕ руки/методы контраста спеки, которых в коде нет (подмножество `_contrast_gap`).

    Отличие от `_contrast_gap`: возвращает не «какой ENV-ключ не реализован», а КОНКРЕТНЫЕ имена
    рук (`gnmuon`, `full_gn`, `muon`) — по ним ищется уже работающая реализация в проекте.
    Числа (`LR_GRID_*=0.01,0.02`) и теги масштаба (`124M`) сюда не попадают: `_IDENT_RE` требует,
    чтобы имя начиналось с буквы.
    """
    env = (spec or {}).get("env") or {}
    if not isinstance(env, dict):
        return []
    low = code.lower()
    out: List[str] = []
    for k in sorted(env):
        if not _CONTRAST_ENV_RE.match(str(k)):
            continue
        for v in str(env[k]).replace(";", ",").replace(":", ",").split(","):
            v = v.strip()
            if v and _IDENT_RE.match(v) and v.lower() not in low and v not in out:
                out.append(v)
    return out


_ARM_REF_MAX_FILES = 60


def _arm_impl_excerpt(root: str, missing: List[str], exclude_rel: str,
                      budget: int = 14000) -> Tuple[str, str]:
    """ФИКС ВОЛНЫ 191 — КОРЕНЬ ДЕВЯТИ ОТКАЗОВ ПОДРЯД ПО ОСИ E.

    Когда БАЗА kind'а не реализует контраст спеки, писателю (фикс в.188) выдаётся требование
    «дострой ВСЕ руки из ENV» — но `for _ref_rel in () if _base_injected else (...)` выключало
    выдачу образцов ИМЕННО в этом случае. То есть от писателя требовали `gnmuon`/`full_gn` и при
    этом прятали от него единственную рабочую реализацию этих рук в проекте
    (`Results/scripts/exp_llama_gnmuon_audit.py`: `run_optimizer_phase`, KFAC-факторы full_gn,
    LR-сетка на руку — скрипт прошёл код-ревью и бегал на GPU). Наблюдаемый итог: попытка №1
    честно написала «gnmuon/full_gn не поддержаны → EMP_VERDICT invalid» (ревью справедливо
    отклонило: «не выполняет заявленный эксперимент»), попытка №2 подделала контраст (`target_plus`
    и `target_minus` двумя ОДИНАКОВЫМИ вызовами eval_loss) — и так девять попыток за четыре волны.

    Возвращает (rel-путь образца, вырезка ТОЛЬКО релевантных top-level определений). Вырезка, а не
    файл целиком: промпт уже несёт БАЗУ (до 30 КБ) + курированные КОНТРАКТЫ (до 40 КБ), и полный
    образец выдавил бы их — ровно тот класс регрессии, что чинили в.189/в.190.
    """
    if not missing:
        return "", ""
    import ast as _ast
    d = os.path.join(root, "Results", "scripts")
    try:
        names = sorted(n for n in os.listdir(d) if n.startswith("exp_") and n.endswith(".py"))
    except OSError:
        return "", ""
    best_score, best_rel, best_src = 0, "", ""
    for n in names[:_ARM_REF_MAX_FILES]:
        rel = f"Results/scripts/{n}"
        if rel == exclude_rel or "_exp_wip" in n:
            continue
        try:
            src = open(os.path.join(d, n), encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        if len(src) > 200000:
            continue
        low = src.lower()
        score = sum(1 for m in missing if m.lower() in low)
        if score > best_score:  # детерминированно: имена уже отсортированы, строгое `>`
            best_score, best_rel, best_src = score, rel, src
    if not best_rel or (best_score < len(missing) and best_score < 2):
        return "", ""   # одна случайно совпавшая рука — это не образец реализации
    try:
        tree = _ast.parse(best_src)
    except SyntaxError:
        return best_rel, best_src[:budget]
    chunks: List[str] = []
    for node in tree.body:
        if not isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.Assign,
                                 _ast.AnnAssign, _ast.ClassDef)):
            continue
        seg = _ast.get_source_segment(best_src, node) or ""
        if seg and any(m.lower() in seg.lower() for m in missing):
            chunks.append(seg)
    out, total = [], 0
    for c in chunks:
        if total + len(c) > budget:
            break
        out.append(c)
        total += len(c)
    return best_rel, ("\n\n".join(out) if out else best_src[:budget])


_LEADZERO_RE = re.compile(r"(?<![\w.])0+(\d)")


def _fix_leading_zero_ints(code: str, max_passes: int = 8) -> str:
    """codex-экспериментатор регулярно эмитит целые с ведущим нулём (0350, 007) → в Python3 это
    `SyntaxError: leading zeros in decimal integer literals`, которая съедает все ретраи, и реальный
    скрипт не доходит до код-ревью (ось E стоит). Детерминированно нормализуем ТОЛЬКО ту строку, на
    которую ругается парсер (строковые литералы/комменты на других строках не трогаем)."""
    import ast
    for _ in range(max_passes):
        try:
            ast.parse(code)
            return code
        except SyntaxError as e:
            if "leading zeros" not in (e.msg or "") or not e.lineno:
                return code
            lines = code.splitlines(keepends=True)
            i = e.lineno - 1
            if not (0 <= i < len(lines)):
                return code
            fixed = _LEADZERO_RE.sub(r"\1", lines[i])
            if fixed == lines[i]:
                return code  # не удалось нормализовать — отдаём как есть, пусть решает ast.parse
            lines[i] = fixed
            code = "".join(lines)
    return code


class ReseedPending(Exception):
    """Пересев скрипта kind'а идёт в ФОНОВОМ потоке — запуск ЭТОГО прогона отложен на следующую
    итерацию, но главный цикл НЕ стоит (фикс волны 173)."""


# ПЕРЕСЕВ: один поток НА KIND (фикс волны 179). Раньше single-flight был ГЛОБАЛЬНЫМ («только один
# пересев за раз»), потому что `codex_write_artifact` писал в ОДИН общий `Results/scripts/_exp_wip.py`
# и параллельные пересевы затирали бы друг друга. Побочный эффект оказался дороже защиты: пока
# kind, чей пересев ЦИКЛИЧЕСКИ отклоняется код-ревью (до 3 раундов × 15-25 мин), держал единственный
# слот, размещение ВСЕХ ОСТАЛЬНЫХ kind'ов отклонялось каждой итерацией («пересев kind=custom_llama
# уже идёт в фоне — запуск отложен») — в т.ч. `llama_source_bit_frontier_audit`, единственный рычаг
# оси E. Теперь артефакт пересева ПЕР-KIND (`_exp_wip_<kind>.py` → затирания нет по построению),
# лок держится на kind, а общий потолок `_RESEED_MAX_PARALLEL` не даёт пересевам съесть codex.
_reseed_threads: Dict[str, Any] = {}
_reseed_lock = threading.Lock()
_RESEED_MAX_PARALLEL = 2
# ФИКС ВОЛНЫ 202 (ИНВЕРСИЯ ПРИОРИТЕТА В ОЧЕРЕДИ ПЕРЕСЕВА). Оба слота держали (а) УЛУЧШЕНИЕ скрипта
# kind'а, у которого одобренная версия УЖЕ лежит на диске и прогон стартует и без пересева
# (`llama_source_bit_frontier_audit`, 18+ отказов ревью подряд), и (б) один первый сев. При этом в
# очереди стояли 5 узлов, у которых скрипта НЕТ ВООБЩЕ, то есть без сева они не могут запуститься
# ФИЗИЧЕСКИ — и все они directive-узлы A*-критика, единственный рычаг оси E. Улучшение (неблокирующее)
# вытесняло первый сев (блокирующий) неограниченно долго. Потолок НЕ поднимаем (1 vCPU, RAM 961 МБ —
# OOM дороже задержки, класс в.193): улучшение просто УСТУПАЕТ слот, пока есть живой спрос на первый
# сев. Улучшение при этом ничего не теряет: его прогон и так идёт на прежней одобренной версии.
_RESEED_YIELD_S = 1800.0
_reseed_last_blocking = 0.0
# ФИКС ВОЛНЫ 207 (ТА ЖЕ ИНВЕРСИЯ, НО УСТУПКА в.202 ФИЗИЧЕСКИ НЕ МОГЛА СРАБОТАТЬ). Уступка выше
# проверяется ТОЛЬКО в момент, когда улучшение САМО пытается стартовать, то есть когда слот СВОБОДЕН.
# А инверсия возникает ПОСЛЕ старта: улучшение уже держит слот 15-75 мин (до 3 раундов codex), и
# пришедший позже первый сев обязан ждать. Замер в в.207: строк `уступает слот узлам без скрипта` в
# логе за всё время — 0, строк «жду слот», где слот держит УЛУЧШЕНИЕ `llama_source_bit_frontier_audit`
# (скрипт одобрен, прогоны идут и без пересева) — 208 из 273. Поэтому уступка «по спросу» заменяется
# СТАТИЧЕСКОЙ РЕЗЕРВАЦИЕЙ: улучшения занимают максимум `_RESEED_MAX_PARALLEL - 1` слотов, один слот
# всегда свободен под первый сев, и порядок прихода больше ничего не решает.
_RESEED_IMPROVE_CAP = max(1, _RESEED_MAX_PARALLEL - 1)
_reseed_is_blocking: Dict[str, bool] = {}
# Вторая половина фикса в.207. Резервации «улучшения занимают <= MAX-1 слотов» НЕ ХВАТИЛО: при двух и
# более узлах без скрипта улучшение всё равно держит один из двух слотов, и второй ждущий первый сев
# стоит (наблюдалось сразу после рестарта волны 207). Спрос теперь определяется НЕ таймштампом
# `_reseed_last_blocking` (он обнуляется рестартом, и гонка «кто первый после старта» всегда
# доставалась улучшению — отсюда 0 срабатываний уступки в.202 за всё время), а ДЕТЕРМИНИРОВАННЫМ
# опросом записей прогонов: есть ли запись, чей скрипт ещё не написан. Голодание улучшения ограничено
# клапаном `_RESEED_IMPROVE_STARVE_S`.
_RESEED_IMPROVE_STARVE_S = 2700.0
_reseed_yield_since: Dict[str, float] = {}


def _rec_unplaceable(rec: Dict[str, Any]) -> bool:
    """в.226: запись, которой в пуле НЕТ подходящей карты дольше `NOFIT_SLOT_RELEASE_S`.

    Тот же критерий, по которому в.213 снимает с неё слот GPU-полосы: если карта ≥`_nofit_mb` не
    нашлась за 1,5 ч, её появление зависит от ЮЗЕРА (ёмкость), а не от лупа."""
    _n = float(rec.get("_nofit_since") or 0)
    return bool(_n and (time.time() - _n) >= NOFIT_SLOT_RELEASE_S)


def _min_free_hint(cfg: ARConfig, rel: str) -> int:
    """Сколько МБ ждёт запись с этим скриптом (для текста лога; 0 = неизвестно)."""
    try:
        for rec in _records(cfg):
            if str((rec.get("exp_spec") or {}).get("script") or "") == rel:
                return int(rec.get("_nofit_mb") or 0)
    except Exception as e:  # noqa: BLE001
        logger.info("[exp] _min_free_hint(%s) не удался: %s", rel, e)
    return 0


def _first_sow_split(cfg: ARConfig) -> Tuple[List[str], List[str], List[str]]:
    """(размещаемые nid, неразмещаемые nid, rel-пути неразмещаемых) среди спроса на первый сев."""
    ok: List[str] = []
    nofit: List[str] = []
    nofit_rel: List[str] = []
    try:
        root = os.path.abspath(cfg.project_root)
        for rec in _records(cfg):
            if rec.get("status") in ("running", "done", "failed", "blocked"):
                continue
            rel = str((rec.get("exp_spec") or {}).get("script") or "")
            if not rel:
                continue
            p = os.path.join(root, rel)
            if os.path.exists(p) and os.path.getsize(p) > 0:
                continue
            _t = _reseed_threads.get(_reseed_key(str(rec.get("kind") or ""), rel))
            if _t is not None and _t.is_alive():
                continue           # в.210: спрос, который уже обслуживается, — не спрос
            if _rec_unplaceable(rec):
                nofit.append(str(rec.get("nid") or "?")); nofit_rel.append(rel)
            else:
                ok.append(str(rec.get("nid") or "?"))
    except Exception as e:  # noqa: BLE001 — правило в.183: причину логируем, не глотаем
        logger.info("[exp] опрос спроса на первый сев не удался (%s) — считаю спрос нулевым", e)
    return ok, nofit, nofit_rel


def _pending_first_sow(cfg: ARConfig) -> List[str]:
    """в.207: РЕАЛЬНЫЙ спрос на первый сев — записи прогонов, чей скрипт ещё не написан.
    Такой узел не запустится ФИЗИЧЕСКИ, тогда как улучшение ничего не теряет: его прогон идёт на
    прежней одобренной ревью версии. Ошибку не глотаем молча (правило в.183) — логируем причину.

    В.226: спрос считается ПО РАЗМЕЩАЕМЫМ записям; неразмещаемые (ждут карту, которой в пуле нет
    >1,5 ч) учитываются только когда размещаемого спроса нет вовсе — иначе улучшение уступало бы
    слот узлу, чей прогон всё равно не стартует до появления ЖЕЛЕЗА (решение юзера, не лупа).
    Комментарий в.210 ниже сохранён: обслуживаемый живым потоком спрос спросом не считается."""
    _ok, _nofit, _ = _first_sow_split(cfg)
    return _ok or _nofit


def _pending_first_sow_legacy(cfg: ARConfig) -> List[str]:
    """Прежняя (до в.226) реализация — оставлена как эталон для пробы инвентаря."""
    out: List[str] = []
    try:
        root = os.path.abspath(cfg.project_root)
        for rec in _records(cfg):
            if rec.get("status") in ("running", "done", "failed", "blocked"):
                continue
            rel = str((rec.get("exp_spec") or {}).get("script") or "")
            if not rel:
                continue
            p = os.path.join(root, rel)
            if not (os.path.exists(p) and os.path.getsize(p) > 0):
                # ФИКС ВОЛНЫ 210: спрос, который УЖЕ ОБСЛУЖИВАЕТСЯ живым потоком первого сева, — НЕ
                # спрос. Иначе резервация в.207 вырождается в ПРОСТОЙ слота: при единственном узле без
                # скрипта его же первый сев занимает слот 1, а слот 2 никто не берёт — улучшения
                # уступают «спросу», которого фактически нет (наблюдение волны 210: 1 живой поток из 2
                # слотов, три ключа улучшения уступают каждую итерацию по 45 мин до клапана голодания,
                # в т.ч. рычаг оси E `llama_source_bit_frontier_audit`, чей прогон стоит в КАРАНТИНЕ по
                # трейсбэку и ждёт ровно нового пересева). Приоритет НЕ инвертируется: НЕобслуживаемый
                # первый сев по-прежнему выталкивает улучшение, а потолок `_RESEED_IMPROVE_CAP` не даёт
                # улучшениям занять больше MAX-1 слотов.
                _t = _reseed_threads.get(_reseed_key(str(rec.get("kind") or ""), rel))
                if _t is not None and _t.is_alive():
                    continue
                out.append(str(rec.get("nid") or "?"))
    except Exception as e:  # noqa: BLE001
        logger.info("[exp] опрос спроса на первый сев не удался (%s) — считаю спрос нулевым", e)
    return out


def _reseed_state_path(cfg: ARConfig, kind: str) -> str:
    return os.path.join(cfg.workdir, ".run", "exp_reseed_state", _safe(kind)[:60] + ".json")


def _reseed_key(kind: str, rel: str) -> str:
    """Ключ single-flight'а и состояния пересева (ФИКС ВОЛНЫ 198).

    Весь механизм пересева (лок потока, артефакт `_exp_wip_*`, накопленный фидбек ревью в.192)
    был ключован по KIND в предположении «один kind — один скрипт». Для `custom_llama` это
    предположение ЛОЖНО: у kind'а НЕТ постоянного скрипта (`KIND_SCRIPTS['custom_llama'] == ''`),
    путь берётся из спеки и уникален для КАЖДОГО узла (`Results/scripts/exp_custom_<hash>.py`).
    Последствия, измеренные в волне 198: шесть ждущих узлов (все директивы A*-критика!) делили
    ОДИН слот, а `_load_reseed_feedback` подставлял писателю НОВОГО узла текст отказа по ЧУЖОМУ
    скрипту («исправь ИМЕННО ЭТО: `LRS[method]` без сетки…») — файла, которого у этого узла нет
    вообще (первый сев с нуля). Итог: 141 отказ ревью, 76 сдач «не прошёл за N попыток» и НИ
    ОДНОГО одобренного custom-скрипта с 24 июля ⇒ директивы критика физически не могли получить
    эмпирику, ось E стояла. Теперь per-node kind'ы ключуются по ИМЕНИ ФАЙЛА: фидбек накапливается
    по своему скрипту, разные узлы сеются параллельно (в пределах `_RESEED_MAX_PARALLEL`).
    Для kind'ов с постоянным скриптом ключ остаётся прежним (== kind), чтобы не потерять уже
    накопленное состояние (`llama_source_bit_frontier_audit`, 11 попыток).
    """
    if exp_spec.KIND_SCRIPTS.get(kind):
        return kind
    base = os.path.basename(rel or "")
    if base.endswith(".py"):
        base = base[:-3]
    return f"{kind}__{base}" if base else kind


def _load_reseed_feedback(cfg: ARConfig, kind: str) -> Tuple[str, int]:
    """Накопленный фидбек код-ревью ЭТОГО kind'а, переживший смерть процесса (фикс волны 192).

    Пересев живёт в daemon-потоке и стоит до 3 codex-раундов × 15-25 мин, а orch перезапускается
    (вахтёром после каждой правки кода, cron'ом после OOM) заметно чаще: daemon-поток умирает
    ВМЕСТЕ с процессом, поэтому попытка начиналась с чистого листа КАЖДЫЙ раз. Измерено в волне
    192: 19 запусков фонового пересева `llama_source_bit_frontier_audit`, из них лишь 2 дошли до
    вердикта ревью — остальные убиты рестартом до конца первой попытки. Теперь причина последнего
    отказа лежит на диске и подмешивается в промпт СЛЕДУЮЩЕЙ жизни процесса: прогресс пересева
    больше не обнуляется рестартом.
    """
    try:
        st = json.load(open(_reseed_state_path(cfg, kind), encoding="utf-8"))
    except Exception:  # noqa: BLE001 — нет файла/битый json = просто нет накопленного фидбека
        return "", 0
    fb = str(st.get("feedback") or "")
    if not fb:
        return "", 0
    try:  # устаревший фидбек хуже отсутствия: контракт мог поменяться
        if time.time() - float(st.get("ts") or 0) > 6 * 3600:
            return "", 0
    except (TypeError, ValueError):
        return "", 0
    return fb, int(st.get("attempts") or 0)


_RESEED_HISTORY_MAX = 6
_RESEED_HISTORY_TRIM = 700


def _load_reseed_history(cfg: ARConfig, kind: str) -> List[str]:
    """ВСЕ различные причины отказов код-ревью этого же скрипта (фикс волны 200).

    Память фидбека была ДЛИНЫ ОДИН (`feedback` — только ПОСЛЕДНИЙ отказ), поэтому писатель
    правил свежую претензию и РЕ-ЛОМАЛ более раннюю: у `llama_source_bit_frontier_audit`
    15 попыток подряд отклонены, и причины РОТИРУЮТ (по логу ~20 различных текстов, темы
    возвращаются в новой формулировке: «нет tuned Muon/AdamW/full-GN controls» ×2, «нарушено
    ОДИН общий target-finetune» ×2, «невалидны simultaneous 95% CI» ×2, недостижимая
    сертификация ×3). Накопленные отказы лежали только в уроках — НЕкурированным автодампом,
    который проигрывает бюджет курированным блокам (класс волн 189/190), т.е. были мёртвой
    буквой. Теперь список различных причин живёт в состоянии пересева и подмешивается ЦЕЛИКОМ.
    """
    try:
        st = json.load(open(_reseed_state_path(cfg, kind), encoding="utf-8"))
    except Exception:  # noqa: BLE001 — нет файла/битый json = нет истории
        return []
    try:
        if time.time() - float(st.get("ts") or 0) > 6 * 3600:
            return []  # тот же TTL, что у `_load_reseed_feedback`: контракт мог поменяться
    except (TypeError, ValueError):
        return []
    hist = st.get("history")
    if not isinstance(hist, list):
        return []
    return [str(h) for h in hist if str(h).strip()][:_RESEED_HISTORY_MAX]


def _reason_key(text: str) -> str:
    """Ключ дедупа причины отказа: первые 120 значимых символов без пробелов/регистра."""
    return "".join(str(text).lower().split())[:120]


def _save_reseed_feedback(cfg: ARConfig, kind: str, feedback: str, attempts: int) -> None:
    try:
        p = _reseed_state_path(cfg, kind)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        # фикс волны 200: копим РАЗЛИЧНЫЕ причины, а не только последнюю (см. _load_reseed_history)
        hist = _load_reseed_history(cfg, kind)
        new = feedback.strip()[:_RESEED_HISTORY_TRIM]
        if new:
            seen = {_reason_key(new)}
            merged = [new]
            for h in hist:
                if _reason_key(h) in seen:
                    continue
                seen.add(_reason_key(h))
                merged.append(h[:_RESEED_HISTORY_TRIM])
            hist = merged[:_RESEED_HISTORY_MAX]
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"kind": kind, "feedback": feedback[:4000], "attempts": attempts,
                       "history": hist, "ts": time.time()}, f, ensure_ascii=False)
        os.replace(tmp, p)
    except OSError as e:
        logger.warning("[exp] не смог сохранить состояние пересева kind=%s: %s", kind, e)


def _clear_reseed_state(cfg: ARConfig, kind: str) -> None:
    try:
        os.remove(_reseed_state_path(cfg, kind))
    except OSError:
        pass


def _ensure_exp_script(cfg: ARConfig, kind: str, spec: Dict[str, Any], script_rel: str) -> str:
    """Неблокирующая обёртка над севом/пересевом скрипта (фикс волны 173).

    Раньше `_ensure_exp_script` звался ПРЯМО из главного цикла supervisor'а, а внутри сидели до
    3 codex-раундов «напиши скрипт → код-ревью» по 15-25 минут каждый. На всё это время главный
    цикл замирал ЦЕЛИКОМ: `empirical.poll()` не вызывался, поэтому завершившиеся на GPU прогоны
    часами не финализировались (волна 173: `em_5` отдал `EMP_VERDICT: confirmed` в 04:19 MSK, а
    запись висела `running` — запись не трогалась 2 ч 17 мин), вердикт не доезжал до графа, и
    ни один новый эксперимент не мог стартовать. Теперь тяжёлая часть уходит в daemon-поток,
    а вызывающая сторона получает `ReseedPending` → запись просто `waiting` до следующей итерации.
    """
    rel = script_rel or exp_spec.KIND_SCRIPTS.get(kind, "")
    if not rel:
        return script_rel
    root = os.path.abspath(cfg.project_root)
    path = os.path.abspath(os.path.join(root, rel))
    lessons_path = os.path.join(cfg.workdir, ".run", "exp_review_lessons", f"{kind}.md")
    need = not os.path.exists(path)
    if not need and os.path.basename(path) not in _REGEN_SKIP:
        try:
            need = os.path.getmtime(lessons_path) > os.path.getmtime(path)
        except OSError:
            need = False
    if not need:
        return rel  # быстрый путь: скрипт есть и свежее уроков — запуск идёт как раньше
    # ПЕРЕСЕВ НЕ ИМЕЕТ ПРАВА ОСТАНАВЛИВАТЬ ПРОГОНЫ (фикс волны 180). На диске лежит ПОСЛЕДНЯЯ
    # версия скрипта, ПРОШЕДШАЯ код-ревью (кандидаты, отклонённые ревью, до `path` не доезжают —
    # см. конец `_ensure_exp_script_sync`), поэтому ждать пересева, чтобы запустить прогон, незачем:
    # улучшение скрипта — фоновая работа, а не предусловие. Раньше `ReseedPending` поднимался всегда,
    # и каждое обновление уроков (в т.ч. ОБЯЗАТЕЛЬНАЯ по правилу в.172 дописка КОНТРАКТа вахтёром)
    # запирало ВСЕ прогоны kind'а на 3 раунда × 15-25 мин: в волне 180 три записи оси E простояли
    # `waiting` 40+ мин при свободных GPU, и цикл был самоподдерживающимся (отказ ревью → новый
    # урок → новый пересев → снова стоп). Теперь `ReseedPending` только когда скрипта НЕТ ВООБЩЕ.
    _approved = os.path.exists(path) and os.path.getsize(path) > 0
    _key = _reseed_key(kind, rel)  # фикс в.198: per-node kind'ы (custom_llama) — ключ по файлу
    global _reseed_last_blocking
    with _reseed_lock:
        if not _approved:
            # Живой спрос на ПЕРВЫЙ сев: без скрипта этот узел не запустится вообще (фикс в.202).
            _reseed_last_blocking = time.time()
        mine = _reseed_threads.get(_key)
        if mine is not None and mine.is_alive():
            if _approved:
                return rel
            raise ReseedPending(f"пересев {_key} уже идёт в фоне")
        live = [k for k, t in _reseed_threads.items() if t is not None and t.is_alive()]
        # ФИКС ВОЛНЫ 207: один слот ЗАРЕЗЕРВИРОВАН под первый сев. Улучшение не занимает последний
        # слот НИКОГДА — независимо от того, стоит ли прямо сейчас кто-то в очереди (проверка «есть ли
        # спрос» в.202 достижима только при свободном слоте и потому мертва, см. комментарий выше).
        if _approved:
            _demand = _pending_first_sow(cfg)
            _since = _reseed_yield_since.get(_key)
            if _demand and (_since is None or time.time() - _since < _RESEED_IMPROVE_STARVE_S):
                _reseed_yield_since.setdefault(_key, time.time())
                logger.info("[exp] пересев %s (УЛУЧШЕНИЕ) уступает слот: РЕАЛЬНЫЙ спрос на первый сев "
                            "— %d узлов без скрипта (%s); прогон идёт на прежней одобренной версии",
                            _key, len(_demand), ", ".join(_demand[:4]))
                return rel
            if _demand:
                logger.warning("[exp] пересев %s (УЛУЧШЕНИЕ) стартует ВОПРЕКИ спросу на первый сев "
                               "(%d узлов): клапан голодания %.0f мин исчерпан",
                               _key, len(_demand), _RESEED_IMPROVE_STARVE_S / 60)
            _reseed_yield_since.pop(_key, None)
            live_improve = [k for k in live if not _reseed_is_blocking.get(k, False)]
            if len(live_improve) >= _RESEED_IMPROVE_CAP:
                logger.info("[exp] пересев %s (УЛУЧШЕНИЕ) не берёт последний слот: %d/%d слотов уже у "
                            "улучшений (%s), слот зарезервирован под первый сев; прогон идёт на "
                            "прежней одобренной версии", _key, len(live_improve),
                            _RESEED_MAX_PARALLEL, ", ".join(live_improve))
                return rel
        # ФИКС ВОЛНЫ 226: ПЕРВЫЙ СЕВ ДЛЯ НЕРАЗМЕЩАЕМОГО ПРОГОНА УСТУПАЕТ СЛОТ РАЗМЕЩАЕМОМУ.
        # Измерено в в.226: `directive_07d8006a_emp_760M` (нужна карта ≥44 ГБ, максимум в пуле
        # 33,9 ГБ; `_nofit` уже снял с неё слот GPU-полосы по в.213) числилась спросом на первый сев
        # и забирала полный codex-цикл «напиши скрипт → код-ревью» (до 3 раундов × 15-25 мин) из
        # ДВУХ слотов, пока размещаемые `directive_07d8006a_emp_350M` и `gnmuon_custom_llama_*`
        # стояли в очереди. Дефицит ЖЕЛЕЗА конвертировался в потерю пропускной способности ПИСАТЕЛЯ
        # скриптов — а она и есть узкое место оси E. Приоритет, не запрет: если размещаемого спроса
        # нет, неразмещаемый сев идёт как прежде (иначе такой узел не получил бы скрипт НИКОГДА).
        if not _approved:
            _ok_demand, _nofit_demand, _nofit_rels = _first_sow_split(cfg)
            if rel in _nofit_rels and _ok_demand:
                raise ReseedPending(
                    f"первый сев отложен: прогон неразмещаем (карты ≥{_min_free_hint(cfg, rel)} МБ в "
                    f"пуле нет >1,5 ч), слот отдан размещаемым узлам ({', '.join(_ok_demand[:3])})")
        if len(live) >= _RESEED_MAX_PARALLEL:
            if _approved:
                return rel
            raise ReseedPending(f"параллельных пересевов уже {len(live)} ({', '.join(live)}) — жду слот")
        # ФИКС ВОЛНЫ 202: УЛУЧШЕНИЕ уступает слот ПЕРВОМУ СЕВУ. Прогон этого kind'а всё равно идёт на
        # прежней одобренной ревью версии, а узлы без скрипта без сева не запускаются в принципе.
        if _approved and live and (time.time() - _reseed_last_blocking) < _RESEED_YIELD_S:
            logger.info("[exp] пересев %s (УЛУЧШЕНИЕ, скрипт уже одобрен) уступает слот узлам без "
                        "скрипта — есть живой спрос на первый сев; прогон идёт на прежней версии",
                        _key)
            return rel

        def _run() -> None:
            try:
                _ensure_exp_script_sync(cfg, kind, spec, script_rel)
            except Exception as e:  # noqa: BLE001 — фон не имеет права уронить процесс
                logger.warning("[exp] фоновый пересев kind=%s упал: %s", kind, e)

        th = threading.Thread(target=_run, daemon=True, name=f"reseed-{_key}")
        _reseed_threads[_key] = th
        _reseed_is_blocking[_key] = not _approved  # в.207: чей это слот — первого сева или улучшения
        th.start()
    logger.info("[exp] сев/пересев %s (kind=%s) ушёл в ФОНОВЫЙ поток — главный цикл продолжает poll()",
                _key, kind)
    if _approved:
        logger.info("[exp] kind=%s: прогон стартует на ПРЕЖНЕМ одобренном ревью скрипте, пересев "
                    "улучшает его в фоне (не блокируем ось E)", kind)
        return rel
    raise ReseedPending(f"пересев {_key} запущен в фоне")


def _ensure_exp_script_sync(cfg: ARConfig, kind: str, spec: Dict[str, Any], script_rel: str) -> str:
    """Автономия экспериментатора: если типизированного скрипта kind НЕТ в проекте — агент (codex)
    ПИШЕТ его сам (llama-контракт, без pythia), сохраняет в Results/scripts/ и переиспользует далее.
    Невалидный/пустой код → ValueError (в _place ловится → _blocked, безопасный фолбэк)."""
    rel = script_rel or exp_spec.KIND_SCRIPTS.get(kind, "")
    if not rel:
        return script_rel  # нет маппинга (напр. legacy_llama_forgetting) — оставляем как есть
    root = os.path.abspath(cfg.project_root)
    path = os.path.abspath(os.path.join(root, rel))
    if not (path == root or path.startswith(root + os.sep)):
        raise ValueError(f"script escapes project root: {rel}")
    lessons_path = os.path.join(cfg.workdir, ".run", "exp_review_lessons", f"{kind}.md")
    # ФИКС ВОЛНЫ 194: прежде чем решать «пересевать или нет», убедись, что на диске лежит ИМЕННО
    # одобренная ревью версия. Иначе пересев улучшает ПОДМЕНЁННУЮ базу (и каждый прогон уезжает на
    # GPU с ней же) — ровно так были потеряны `full_gn`/`"row"`/`per_arm_time_s`/`parameter_check`.
    _shadow_guard(cfg, kind, rel)
    # ПЕРЕСЕВ ПО СВЕЖИМ УРОКАМ (фикс волны 171). Раньше было `if os.path.exists(path): return rel` —
    # скрипт kind'а писался ОДИН раз и жил вечно, поэтому все контракты, дописанные в уроки ПОЗЖЕ
    # (прогресс-строки `{"row":...}`, OOM→invalid, bootstrap по seed-кластерам, per_arm_time_s),
    # были МЁРТВОЙ БУКВОЙ для уже существующих kind'ов: агент их читал только при первом севе.
    # Теперь скрипт пересевается, если файл уроков НОВЕЕ скрипта. Образцы-каркасы не трогаем.
    _existing = os.path.exists(path)
    if _existing:
        _stale = False
        if os.path.basename(path) not in _REGEN_SKIP:
            try:
                _stale = os.path.getmtime(lessons_path) > os.path.getmtime(path)
            except OSError:
                _stale = False
        if not _stale:
            return rel
        logger.info("[exp] уроки kind=%s СВЕЖЕЕ скрипта %s → пересев по обновлённому контракту", kind, rel)
    else:
        logger.info("[exp] скрипт %s отсутствует → экспериментатор пишет его сам (kind=%s)", rel, kind)
    import ast
    base_prompt = _EXP_SCRIPT_PROMPT.format(
        domain=str(cfg.domain)[:600], kind=kind,
        claim=str(spec.get("claim", ""))[:800],
        success_metric=str(spec.get("success_metric", ""))[:400],
        confirm_rule=str(spec.get("confirm_rule", ""))[:400])
    # МАШИННО ПРОВЕРЯЕМЫЙ ИНФРА-КОНТРАКТ (фикс волны 185): три требования A*-рецензии проверяются
    # детерминированно ПОСЛЕ генерации (см. `_mech_contract_violations`), поэтому писатель обязан
    # знать точные токены — иначе он «выполняет» контракт своими именами и проверка его валит.
    base_prompt += (
        "\n\n=== МАШИННО ПРОВЕРЯЕМЫЙ ИНФРА-КОНТРАКТ (проверяется автоматически, БЕЗ ревьюера; "
        "нарушение = отклонённая попытка) ===\n"
        "1) ПРОГРЕСС: после КАЖДОЙ тройки (масштаб, сид, метод/рука) печатай одну строку\n"
        "   print(json.dumps({\"row\": {...ключевые числа этой тройки...}}), flush=True)\n"
        "   — без неё наблюдатель не отличает живой многочасовой прогон от зависшего.\n"
        "2) ВРЕМЯ: в результат КАЖДОЙ руки/метода клади ключ \"per_arm_time_s\" (float, wall-clock "
        "секунды этой руки) — рецензент требует стоимость на руку, а не только суммарный wall.\n"
        "2b) ОДИН ЧАСОВОЙ ИСТОЧНИК НА ОДНУ РАЗНОСТЬ (фикс волны 233, проверяется автоматически): "
        "`time.time()` и `time.perf_counter()` — РАЗНЫЕ эпохи, их разность = мусор конечной величины, "
        "который не ловит ни `badnum`, ни мягкий ревьюер. Засекай и вычитай ОДНОЙ функцией: "
        "`t0 = time.perf_counter(); …; per_arm_time_s = time.perf_counter() - t0`. Вычитание "
        "perf_counter из переменной, полученной через time.time() (и наоборот), = отклонённая попытка.\n"
        "3) СВЕРКА МАСШТАБА: ДО тяжёлого цикла печатай\n"
        "   print(json.dumps({\"parameter_check\": {\"tag\": tag, \"n_params_total\": N, "
        "\"n_params_non_embed\": M, \"ok\": bool}}), flush=True)\n"
        "   и падай с EMP_VERDICT: invalid, если фактическое число параметров не соответствует тегу.\n"
        "4) ОБЪЯВЛЕННОЕ СУЖЕНИЕ: в RESULT_JSON всегда клади ключ \"budget_reduction\" — dict «что именно "
        "сужено против спеки» (шаги, точки LR, число обучающих сидов, масштаб) и ПОЧЕМУ; пустой dict, "
        "если не сужено ничего. Сужать бюджет МОЖНО и НУЖНО, чтобы влезть в 12 ч на одной карте, но "
        "молчаливое сужение = подлог.\n"
        "5) ПРОВЕНАНС ОСИ R — ЭТО ОБЯЗАН НАПЕЧАТАТЬ САМ СКРИПТ (фикс волны 208; дословная директива "
        "№3 A*-рецензии, ревьюеру сказано ЗЕРКАЛЬНО — его ПРИНЯТАЯ УСТАНОВКА №5). В RESULT_JSON:\n"
        "   \"per_block_rows\": [{\"block_id\": <устойчивый идентификатор блока/задачи, НЕ индекс "
        "перечисления>, \"arm\": …, \"metric\": …, …}, …] — строка на КАЖДЫЙ блок и КАЖДУЮ руку;\n"
        "   \"bootstrap_indices\": {\"bootstrap_seed\": int, \"cluster_key\": \"<по чему кластеры: "
        "seed/block>\", \"indices\": [[…], …] ИЛИ \"sha256\": <хэш массива, если он огромен>} — "
        "индексы ресэмпла ОБЩИЕ для всех рук (CRN), иначе CI не пересчитывается;\n"
        "   \"run_provenance\": {\"config_sha256\": …, \"data_sha256\": …, \"tokenizer_sha256\": …, "
        "\"model_init_seed\": int, \"replicates\": int, \"rng_seeds\": [...], \"n_params_actual\": int}.\n"
        "   Команду запуска, sha256 самого файла скрипта, training commit и отпечаток окружения луп "
        "снимает САМ — их дублировать НЕ надо.\n"
        "Токены \"row\", \"per_arm_time_s\", \"parameter_check\", \"budget_reduction\", "
        "\"per_block_rows\", \"bootstrap_indices\", \"run_provenance\" должны "
        "присутствовать в коде ДОСЛОВНО.\n"
        "6) ДОСТИЖИМОСТЬ СЕРТИФИКАЦИИ (фикс волны 204; ревьюеру сказано ДОСЛОВНО то же — это его "
        "ПРИНЯТАЯ УСТАНОВКА №4, по которой он тебя и валит). Если у скрипта есть power/feasibility/"
        "achievability-проверка (любое из `feasib`, `n_required`, `achievable`), он ОБЯЗАН класть в "
        "RESULT_JSON ключ \"certification_feasibility\" = {\"n_available\": …, \"n_required\": …, "
        "\"family_size_headline\": …, \"family_size_demoted\": …, \"option\": 1|2|3, \"reason\": …}.\n"
        "   ЗАПРЕЩЕНО (гарантированное отклонение, наблюдалось 26 раз подряд у kind'ов custom_llama и "
        "llama_source_bit_frontier_audit): (а) захардкоженные `feasible=True`/`n_required=3` вместо "
        "ВЫЧИСЛЕНИЯ из фактического числа сидов, alpha/family_size и типа CI; (б) выход "
        "`invalid`/`abstain` ДО обучения, потому что сертификация недостижима — такой прогон не "
        "измеряет claim ни при каком качестве кода; (в) молчаливое продолжение с заведомо "
        "недостижимым порогом (например simultaneous family=68 при 3-6 сидах: нижняя Wilson-граница "
        "физически не дотягивается до порога ни при каких данных) — это сожжённые 12 ч GPU.\n"
        "   ОБЯЗАН вместо этого привести конфигурацию к ДОСТИЖИМОЙ строго в таком порядке и объявить "
        "выбранную ступень в \"certification_feasibility\": (1) СУЗИТЬ simultaneous family до "
        "сравнений, реально входящих в headline claim, остальные напечатать как descriptive "
        "(без членства в familywise-коррекции и без права открывать гейт); (2) поднять число "
        "CERTIFICATION-сидов внутри скрипта до вычисленного минимума, если укладывается в 12 ч "
        "(сиды дешевле невалидного прогона; компенсируй шагами и объяви в budget_reduction); "
        "(3) заменить exact 0-false-accept UCB на сертификат по НАБЛЮДЁННОЙ доле (Wilson/"
        "Clopper–Pearson) и понизить headline до «sufficient, not validated».\n"
        "   ВАЖНО, ЭТО НЕ ОДНО И ТО ЖЕ: «simultaneous family» (число одновременно сертифицируемых "
        "гипотез) — ЭТО НЕ «контраст рук». Сужать family МОЖНО и НУЖНО; при этом ВСЕ руки/методы "
        "контраста из env всё равно обучаются и их числа печатаются descriptive. Запрет «сужать "
        "контраст» ниже про РУКИ, а не про family.\n"
        "   Вырожденные bootstrap-полосы (CI=[1,1] при 6/6) помечай `degenerate` И ЗАПРЕЩАЙ им "
        "открывать гейт — вырожденный сертификат не считается пройденным.\n"
        "   СТУПЕНЬ (3) — ЭТО ЗАМЕНА СТАТИСТИКИ, А НЕ СТРОКА В RESULT_JSON (фикс волны 206; "
        "доминирующая причина отказов у всех трёх kind'ов была дословно «код заявляет option=3, но "
        "деградацию №3 не реализует: сертификата наблюдённой доли нет, certify() по-прежнему "
        "использует прежний percentile seed-bootstrap и прежние пороги»). Если объявил option=3, "
        "МАШИННО проверяется наличие `wilson`/`clopper` И ДОСЛОВНОЙ строки "
        "\"sufficient, not validated\". Бери эту реализацию ДОСЛОВНО (stdlib, без scipy):\n"
        "```python\n"
        "from statistics import NormalDist\n"
        "def wilson_bounds(k, n, alpha):\n"
        "    \"\"\"Двусторонний Wilson-CI для НАБЛЮДЁННОЙ доли k/n; alpha УЖЕ Bonferroni-скорректирована\n"
        "    (alpha_adj = alpha / family_size_headline). При k=n полоса НЕвырожденная — в отличие от\n"
        "    percentile-bootstrap, который при 6/6 даёт CI=[1,1], т.е. не измерение.\"\"\"\n"
        "    if n <= 0:\n"
        "        return (0.0, 1.0)\n"
        "    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)\n"
        "    p = k / n\n"
        "    d = 1.0 + z * z / n\n"
        "    c = (p + z * z / (2 * n)) / d\n"
        "    h = (z / d) * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n))\n"
        "    return (max(0.0, c - h), min(1.0, c + h))\n"
        "\n"
        "def certify_option3(k, n, alpha_adj, thr_lo=None, thr_hi=None):\n"
        "    \"\"\"Ступень (3): сертификат по наблюдённой доле вместо exact 0-false-accept UCB.\n"
        "    thr_lo — порог для гейтов вида «доля >= X» (например P_delta >= .67);\n"
        "    thr_hi — для гейтов вида «доля <= X» (например false-accept <= .2).\"\"\"\n"
        "    lo, hi = wilson_bounds(k, n, alpha_adj)\n"
        "    degenerate = (n < 2) or (hi - lo < 1e-12)\n"
        "    passed = ((not degenerate)\n"
        "              and (thr_lo is None or lo >= thr_lo)\n"
        "              and (thr_hi is None or hi <= thr_hi))\n"
        "    return {\"k\": k, \"n\": n, \"alpha_adj\": alpha_adj, \"wilson_lo\": lo, \"wilson_hi\": hi,\n"
        "            \"degenerate\": degenerate, \"passed\": bool(passed),\n"
        "            \"headline_status\": \"sufficient, not validated\"}\n"
        "```\n"
        "   `certify_option3` обязан ЗАМЕНИТЬ прежний путь сертификации решающих гейтов, а не лежать "
        "рядом неиспользованным; его `passed` и есть решение гейта. И НЕ обнуляй "
        "`family_size_headline`: демотировать ВСЕ гипотезы нельзя (наблюдалось: 18 из 18) — "
        "headline-family остаётся >= 1 (сама проверяемая claim), остальное уходит в descriptive.\n"
        "   СТУПЕНЬ (4) — ЗАМЕР БЕЗ СЕРТИФИКАЦИИ, ЕСЛИ ЛЕСТНИЦА ИСЧЕРПАНА (фикс волны 206). Ступени "
        "(1)-(3) конечны, и это ПРОВЕРЕНО арифметикой: для порога «false-accept <= .2» даже идеальный "
        "исход 0 из n при family=1 даёт Wilson-верх .390 при n=6, .324 при n=8, .242 при n=12 и "
        "только .176 при n=18 — то есть при 6-8 сертификационных сидах порог .2 НЕ достигается ни "
        "Wilson, ни Clopper-Pearson, ни каким иным сертификатом. Поэтому ДО тяжёлого цикла посчитай "
        "BEST-CASE границу (`wilson_bounds(0, n, alpha_adj)` для гейтов «<= thr», "
        "`wilson_bounds(n, n, alpha_adj)` для «>= thr») и, если она не дотягивается до порога даже в "
        "идеале, объяви `\"option\": 4` с `\"reason\": \"certification out of budget\"` и: (а) прогон "
        "ВСЁ РАВНО выполняется полностью и печатает все числа/`row`/`per_arm_time_s`; (б) решающие "
        "величины печатаются как descriptive с фактической Wilson-полосой и полем "
        "`\"certification\": \"out of budget\"`; (в) `confirmed` НЕ ставится, вердикт "
        "`EMP_VERDICT: no_signal` (а НЕ `invalid` — измерение состоялось, недостаёт только "
        "сертификата); (г) в `certification_feasibility` кладёшь `n_required_for_threshold` — сколько "
        "сидов нужно, чтобы порог стал достижим. Это ЧЕСТНЫЙ и ОЖИДАЕМЫЙ исход, а не провал: он даёт "
        "и числа, и точную цену следующей попытки. Чего делать НЕЛЬЗЯ: (i) выходить `invalid` до "
        "обучения; (ii) занижать порог молча; (iii) объявлять `confirmed` по недостижимому гейту.\n"
        "=== ЧТО СУЖАТЬ МОЖНО, А ЧТО НЕЛЬЗЯ (фикс волны 190; ревьюеру сказано ДОСЛОВНО то же) ===\n"
        "Сужать МОЖНО: число шагов, точки LR-сетки (минимум 2 на руку), число обучающих сидов (минимум 3 "
        "НЕЗАВИСИМЫХ), число eval/cov-батчей — и ОБЯЗАН объявить это в budget_reduction.\n"
        "Сужать НЕЛЬЗЯ (отклонение даже при объявленном budget_reduction): число scale-тегов, которое "
        "требует claim (claim про два и более scale tags ⇒ отрабатывай ВСЕ, `SCALES[:1]` = отказ; это "
        "ровно то, за что отклонена попытка волны 190); руки/методы контраста из env (METHODS/ARMS/"
        "BASELINES) целиком; независимость ОБУЧАЮЩИХ сидов; tuning-сплит; calibration/certification "
        "split; seed-кластерный bootstrap; пометку вырожденных полос.\n"
        "Геометрия тега — ТОЛЬКО из таблицы SCALES внутри скрипта; общие env N_EMBD/N_LAYER/N_HEAD "
        "геометрию тега НЕ переопределяют (так тег 124M однажды получил геометрию 350M и упал по "
        "parameter_check). Если 12 ч не хватает даже после разрешённого сужения — честный "
        "EMP_VERDICT: invalid с error: budget_infeasible, а НЕ тихий срез масштабов.")
    # ФИКС ВОЛНЫ 187: ПЕРЕСЕВ ПИСАЛСЯ С НУЛЯ, а не как ПРАВКА уже одобренной версии ЭТОГО kind'а.
    # Писателю давали лишь ЧУЖОЙ образец-каркас, поэтому каждая попытка заново изобретала весь
    # эксперимент: уцелевшие после ревью части терялись, а вместе с ними и уже выполненные пункты
    # контракта. Отсюда бесконечный цикл отказов (волны 186-187: `full_gn` то без embeddings, то
    # rank-8 KFAC; tuning то есть, то «no tuning claim»; в custom_llama свежий `fac, cur, handles =
    # {}, {"on": False}` — распаковка 2 в 3) И невозможность приземлить инфра-токены: они уезжали
    # вместе с отклонённой научной переписью. Даём СВОЮ действующую версию как БАЗУ и требуем
    # МИНИМАЛЬНОГО диффа — правим только то, чего требуют уроки, остальное не трогаем.
    _base_injected = False
    if _existing and os.path.basename(path) not in _REGEN_SKIP:
        try:
            _cur_src = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            _cur_src = ""
        if _cur_src and len(_cur_src) <= 30000:
            base_prompt += (
                f"\n\n=== ТЕКУЩАЯ ДЕЙСТВУЮЩАЯ ВЕРСИЯ ЭТОГО ЖЕ kind={kind} ({rel}) — она УЖЕ прошла "
                "код-ревью и РЕАЛЬНО бегала на GPU. Это твоя БАЗА, а не образец для подражания.\n"
                "ТРЕБОВАНИЕ: выдай МИНИМАЛЬНО ИЗМЕНЁННУЮ версию этого файла. Правь ТОЛЬКО то, что "
                "требуют уроки/контракты ниже; всё остальное (импорты, разбор env, лестницу "
                "масштабов, сиды, bootstrap, печать RESULT_JSON/EMP_VERDICT/EXP_DONE) сохрани "
                "ДОСЛОВНО. НЕ переписывай эксперимент с нуля: переписанная с нуля версия теряет уже "
                "принятые ревью части и отклоняется снова. Выведи ПОЛНЫЙ файл целиком. ===\n"
                + _cur_src)
            _base_injected = True
            # ФИКС ВОЛНЫ 188: «минимальный дифф» ПРОТИВОРЕЧИЛ ревьюеру, когда БАЗА не реализует
            # контраст спеки. У `llama_source_bit_frontier_audit` база не содержала ни `muon`, ни
            # `full_gn`, ни `LR_GRID` — а ревьюер (справедливо) требовал именно их, поэтому писатель
            # был зажат между «ничего не меняй» и «добавь четыре руки с тюнингом»: 5+ отказов за три
            # волны. Разрешаем противоречие ДЕТЕРМИНИРОВАННО: перечисляем пропущенные ключи и явно
            # снимаем требование минимального диффа ДЛЯ ЭТОЙ ЧАСТИ (инфраструктура — по-прежнему
            # дословно).
            _gap = _contrast_gap(spec, _cur_src)
            if _gap:
                _env_decl = json.dumps((spec or {}).get("env") or {}, ensure_ascii=False)[:600]
                base_prompt += (
                    "\n\n=== ⚠️ БАЗА НЕ РЕАЛИЗУЕТ КОНТРАСТ, ЗАЯВЛЕННЫЙ СПЕКОЙ ===\n"
                    f"Спека объявляет ENV: {_env_decl}\n"
                    f"В БАЗЕ полностью отсутствуют (ни чтения ключа, ни его значений): {', '.join(_gap)}.\n"
                    "Это значит: claim/success_metric спеки базой НЕ проверяется, и код-ревью ОБЯЗАН "
                    "отклонять версию, которая их не реализует (именно эта формулировка уже "
                    "отклоняла предыдущие попытки: «Claim не проверяется: … отсутствуют, нет tuned "
                    "grid»).\n"
                    "ПОЭТОМУ требование «минимальный дифф» на ЭТУ часть НЕ распространяется: ты "
                    "ОБЯЗАН достроить недостающий контраст (все руки/методы из ENV, тюнинг LR по "
                    "СВОЕЙ сетке для КАЖДОЙ руки на ОТДЕЛЬНОМ сплите, честная compute-matched "
                    "постановка) — при этом инфраструктуру БАЗЫ (импорты, llama_exp_kit, разбор env, "
                    "лестницу масштабов, сиды, bootstrap, печать RESULT_JSON/EMP_VERDICT/EXP_DONE) "
                    "сохрани дословно, а объём вычислений держи в рамках (руки × сиды × масштабы "
                    "должны укладываться в бюджет: при нехватке сокращай шаги, НЕ выбрасывай руки).\n"
                    "Имена ключей из ENV читай ДОСЛОВНО через os.environ — это проверяется машинно.")
                # ФИКС ВОЛНЫ 191: требовать руки и ОДНОВРЕМЕННО прятать их рабочую реализацию —
                # недостижимый гейт (см. `_arm_impl_excerpt`). Образцы выдавались только при
                # ОТСУТСТВИИ базы (`() if _base_injected else (...)`), то есть никогда там, где
                # писателю как раз и велено «дострой контраст».
                _miss = _missing_arm_names(spec, _cur_src)
                _ref_rel, _ref_src = _arm_impl_excerpt(root, _miss, rel)
                if _ref_src:
                    base_prompt += (
                        f"\n\n=== ✅ ЭТИ ЖЕ РУКИ УЖЕ РЕАЛИЗОВАНЫ В ЭТОМ ПРОЕКТЕ ({_ref_rel} — прошёл "
                        f"код-ревью и бегал на GPU). Ниже ВЫРЕЗКА релевантных определений для "
                        f"{', '.join(_miss)}.\n"
                        "ПРАВИЛО: механику построения рук (оптимизатор, факторы, шаг, LR-сетку на "
                        "руку, compute-matched бюджет) БЕРИ ОТСЮДА — переноси код, а не изобретай. "
                        "МЕТРИКУ/дизайн своего claim'а НЕ меняй: сюда переносится ТОЛЬКО механика "
                        "рук, всё остальное остаётся из твоей БАЗЫ.\n"
                        "ЗАПРЕЩЕНО объявлять эти руки «неподдержанными» и печатать по этому поводу "
                        "EMP_VERDICT: invalid — реализация перед тобой, значит рука реализуема; "
                        "именно на такой попытке уже сгорел цикл («при неподдержанных gnmuon/full_gn "
                        "обучение не запускается» → отказ ревью). Так же запрещено имитировать "
                        "контраст одинаковыми вычислениями для разных рук/миров (вторая сгоревшая "
                        "попытка: target_plus и target_minus считались двумя ОДИНАКОВЫМИ вызовами "
                        "eval_loss и не зависели от плеча). ===\n" + _ref_src)
                else:
                    logger.warning("[exp] kind=%s: реализация рук %s в проекте НЕ найдена — писатель "
                                   "достраивает контраст без образца", kind, ", ".join(_miss) or "-")
    for _ref_rel in () if _base_injected else (
                    "Results/scripts/exp_llama_checkpoint_pool_selector.py",
                     "Results/scripts/exp_llama_gnmuon_audit.py"):
        _ref = os.path.join(root, _ref_rel)
        if os.path.exists(_ref):
            try:
                base_prompt += (
                    f"\n\n=== ОБРАЗЕЦ РАБОЧЕГО СКРИПТА ПРОЕКТА ({_ref_rel}; прошёл код-ревью и "
                    "бегал на GPU). Используй его КАРКАС: импорты/хелперы llama_exp_kit, разбор env, "
                    "сиды, bootstrap-CI, печать RESULT_JSON/EMP_VERDICT/EXP_DONE. Логику измерения "
                    "АДАПТИРУЙ под свой claim — не копируй чужую метрику слепо. ===\n"
                    + open(_ref, encoding="utf-8", errors="replace").read()[:18000])
            except OSError:
                pass
            break
    # САМОНАКОПЛЕНИЕ УРОКОВ РЕВЬЮ (per-kind, durable): раньше история отказов жила только ВНУТРИ
    # одного сева (3 попытки) — каждый новый сев начинал с нуля и повторял класс ошибок, уроки
    # приходилось вручную зашивать в промпт. Теперь каждая причина отказа пишется в
    # .run/exp_review_lessons/<kind>.md и подмешивается во все будущие автогены этого kind.
    try:
        _lessons = open(lessons_path, encoding="utf-8").read().strip()
    except OSError:
        _lessons = ""
    try:  # снимок «уроки, которые этот пересев РЕАЛЬНО прочитал» (см. фикс волны 174 ниже)
        _self_lessons_mtime = os.path.getmtime(lessons_path)
    except OSError:
        _self_lessons_mtime = 0.0
    # ФИКС ВОЛНЫ 176: mtime-учёт внешней правки сбивался ПОРЯДКОМ событий. Если вахтёр дописал
    # КОНТРАКТ во время пересева, а ПОСЛЕ него хоть одна попытка получила отказ, внутренняя
    # запись причины отказа обновляла `_self_lessons_mtime` ПОВЕРХ внешней правки ⇒ `_external`
    # снова False ⇒ слепой utime(script) ⇒ контракт не прочитан НИКОГДА (тот же дедлок в.172/174,
    # третья его инкарнация). Поэтому ведём БАЙТОВЫЙ учёт: размер файла на снимке + сумма того,
    # что дописал САМ этот пересев. Любое расхождение (внешний append, усечение, перезапись) =
    # внешняя правка ⇒ не штампуем, следующий цикл пересеет и прочитает её.
    try:
        _self_lessons_size = os.path.getsize(lessons_path)
    except OSError:
        _self_lessons_size = 0
    _self_written_bytes = 0
    if _lessons:
        # ПРИОРИТЕТ КОНТРАКТОВ (фикс волны 172). Раньше подмешивался слепой хвост `_lessons[-6000:]`:
        # файл уроков растёт на ~600 символов с КАЖДЫМ отказом, поэтому явные блоки-КОНТРАКТЫ
        # (прогресс-строки `{"row":…}`, OOM→invalid, seed-cluster bootstrap, per_arm_time_s —
        # прямые директивы A*-рецензии) молча выдавливались из окна и снова становились мёртвой
        # буквой. Теперь строки-контракты подмешиваются ВСЕГДА и целиком, а хвост обычных отказов
        # добирается в оставшийся бюджет.
        #
        # ФИКС ВОЛНЫ 185 — фильтр был ПОСТРОЧНЫМ, и это обесценивало весь механизм. Контракт вахтёра
        # записан блоком: заголовок `### КОНТРАКТ (…): …` + 5-40 строк ТЕЛА с самим требованием.
        # `"КОНТРАКТ" in l` истинно только для ЗАГОЛОВКА, поэтому в промпт всегда попадали одни
        # заголовки, а тела падали в `_rest` и выдавливались хвостовым бюджетом (файл уроков вырос
        # до 86 КБ). Наблюдаемый ущерб: 11 блоков-контрактов в уроках kind'а
        # `llama_source_bit_frontier_audit` — и НИ ОДНОГО из требуемых токенов в свежем скрипте,
        # который в 09:50 UTC прошёл код-ревью (`{"row"` 0, `per_arm_time_s` 0, `parameter_check` 0),
        # то есть ось E физически не может уйти с `partial`. Теперь блок берётся ЦЕЛИКОМ (до
        # следующего `###`), новейшие блоки — дословно в бюджет `_CONTRACT_BUDGET`, более старые —
        # заголовками (чтобы писатель знал об их существовании), хвост обычных отказов — в остаток.
        _lines = _lessons.split("\n")
        _blocks: List[Tuple[int, str]] = []   # (индекс начала, текст блока)
        _in_block = set()
        _i = 0
        while _i < len(_lines):
            if "КОНТРАКТ" in _lines[_i]:
                _body = [_lines[_i]]; _in_block.add(_i)
                _j = _i + 1
                while _j < len(_lines) and not _lines[_j].startswith("###") and len(_body) < 60:
                    _body.append(_lines[_j]); _in_block.add(_j); _j += 1
                _blocks.append((_i, "\n".join(_body).strip()))
                _i = _j
            else:
                _i += 1
        # ПРИОРИТЕТ КУРИРОВАННЫХ КОНТРАКТОВ (фикс волны 189). Отбор «новейшие вперёд, что не влезло —
        # только заголовком» на практике выбрасывал ТЕЛА коротких РУЧНЫХ контрактов вахтёра, потому что
        # бюджет 16 000 съедали 8 новейших блоков (15 640 симв), а автоблоки-дампы отказов бывают по
        # 6-30 КБ. Измерено в волне 189: `### КОНТРАКТ … НЕЗАВИСИМЫЕ TRAINING SEEDS` (851 симв!) уехал
        # в заголовки — и СВЕЖИЙ отказ код-ревью дословно звучал «Независимых обучающих сидов нет».
        # Контур самоподдерживающийся: вытесненное требование нарушается → отказ дописывает в файл
        # ещё килобайты → давление вытеснения растёт. Теперь: (1) ручные блоки (`вахтёр` в заголовке)
        # получают бюджет ПЕРВЫМИ, автодампы — из остатка; (2) блок длиннее `_MAX_BLOCK` не
        # выбрасывается целиком, а ОБРЕЗАЕТСЯ (требование обычно в первых строках).
        _CONTRACT_BUDGET, _TAIL_BUDGET, _MAX_BLOCK = 20000, 4000, 4000

        def _curated(_b: str) -> bool:
            # ручные блоки вахтёра помечены «вахтёр»/«волна N» в заголовке; автодампы отказов —
            # «[КОНТРАКТ РАНТАЙМА/ABSTAIN/СТРУКТУРЫ …]» без этих слов.
            _h = _b.split("\n", 1)[0]
            return ("вахтёр" in _h) or ("волна" in _h)

        def _clip(_b: str) -> str:
            return _b if len(_b) <= _MAX_BLOCK else (
                _b[:_MAX_BLOCK] + "\n…[тело контракта обрезано бюджетом; требование — выше]")

        # ФИКС ВОЛНЫ 190: приоритет курированных блоков (в.189) ВСЁ РАВНО их вытеснял, потому что
        # бюджет был ОБЩИМ: курированные контракты сами доросли до 19 663 симв из 20 000, и первый же
        # новый блок вахтёра (1,6 КБ) выбил в заголовки `КОНТРАКТ (волна 182): НЕЗАВИСИМЫЕ TRAINING
        # SEEDS` — ровно тот блок, который в.189 и спасала. Класс регрессии самовоспроизводящийся:
        # каждая волна дописывает контракт и молча выдавливает более старый, а обнаруживается это
        # только ручной репликацией отбора. Теперь курированные блоки НЕДРОПАБЕЛЬНЫ (каждый обрезан
        # до `_MAX_BLOCK`, суммарно до `_CURATED_CAP`), автодампы конкурируют за ОСТАТОК, а факт
        # вытеснения любого курированного блока ЛОГИРУЕТСЯ — чтобы следующая волна видела это в логе.
        # ВОЛНА 197: у `llama_source_bit_frontier_audit` курированные блоки заняли 37 795 из 40 000 —
        # следующий же контракт вахтёра начал бы вытеснять более старый (самовоспроизводящаяся
        # регрессия в.190). Поднимаю потолок; когда и он подойдёт к концу, правильнее удалять
        # УСТАРЕВШИЕ блоки (в.187/189 частично отменены поздними), а не растить бесконечно.
        # ВОЛНА 204: у `llama_source_bit_frontier_audit` курированные блоки заняли 54 903 из
        # 56 000 — следующий контракт вахтёра снова начал бы вытеснять более старый. Поднимаю
        # до 64 000; дальше правильнее ПРОРЕЖИВАТЬ устаревшие блоки, а не растить cap.
        _CURATED_CAP = 64000
        _full, _acc = [], 0
        _curated_acc, _lost_curated = 0, []
        for _idx, _b in reversed(_blocks):           # курированные: новейшие важнее, дроп только по cap
            if not _curated(_b):
                continue
            _c = _clip(_b)
            if _curated_acc + len(_c) > _CURATED_CAP:
                _lost_curated.append(_b.split("\n", 1)[0][:80]); continue
            _full.append((_idx, _c)); _curated_acc += len(_c)
        if _lost_curated:
            logger.warning("[exp] kind=%s: курированные КОНТРАКТЫ вытеснены cap'ом (%d): %s — "
                           "подними _CURATED_CAP или удали устаревшие блоки из уроков",
                           kind, len(_lost_curated), "; ".join(_lost_curated))
        _acc = _curated_acc
        for _idx, _b in reversed(_blocks):           # автодампы отказов — из остатка общего бюджета
            if _curated(_b):
                continue
            _c = _clip(_b)
            if _acc + len(_c) > max(_CONTRACT_BUDGET, _curated_acc + 4000):
                continue
            _full.append((_idx, _c)); _acc += len(_c)
        _full_idx = {i for i, _ in _full}
        _heads = [b.split("\n", 1)[0] for i, b in _blocks if i not in _full_idx]
        _rest = [l.strip() for k, l in enumerate(_lines) if l.strip() and k not in _in_block]
        _keep, _acc2 = [], 0
        for l in reversed(_rest):
            if _acc2 + len(l) > _TAIL_BUDGET:
                break
            _keep.append(l); _acc2 += len(l)
        _keep.reverse()
        _text = "\n".join(
            (["(ещё контракты, тела вытеснены бюджетом — соблюдай и их):"] + _heads if _heads else [])
            + [b for _, b in sorted(_full)] + ["", "--- прочие уроки отказов ---"] + _keep)
        base_prompt += ("\n\n=== УРОКИ ПРОШЛЫХ ОТКАЗОВ КОД-РЕВЬЮ ПО ЭТОМУ KIND (НЕ повторяй ни одну "
                        "из этих ошибок — каждая уже стоила сгоревшей попытки; блоки со словом "
                        "КОНТРАКТ обязательны к ДОСЛОВНОМУ исполнению) ===\n" + _text)
    # пишем → синтакс-парс → НЕЗАВИСИМЫЙ КОД-РЕВЬЮ (реально ли тестирует claim, не подделан ли вердикт);
    # фейл → 1 ретрай с фидбеком ревьюера; иначе ValueError → _place → _blocked (безопасный фолбэк).
    feedback, code = "", ""
    exp_eng = cfg.engines.exp_writer
    # ПРОГРЕСС ПЕРЕСЕВА ПЕРЕЖИВАЕТ СМЕРТЬ ПРОЦЕССА (фикс волны 192): фидбек последнего отказа
    # ревью читается с диска, поэтому убитая рестартом попытка не обнуляет накопленное знание.
    # ФИКС ВОЛНЫ 198: ключ состояния — НЕ kind, а конкретный скрипт для per-node kind'ов
    # (`custom_llama` даёт каждому узлу свой `exp_custom_<hash>.py`), иначе писателю нового узла
    # подставлялся отказ по ЧУЖОМУ файлу («исправь ИМЕННО ЭТО» про код, которого у него нет).
    _rs_key = _reseed_key(kind, rel)
    _prev_fb, _prev_att = _load_reseed_feedback(cfg, _rs_key)
    _att_total = _prev_att
    if _prev_fb:
        feedback = _prev_fb
        logger.info("[exp] пересев %s продолжает с накопленного фидбека ревью (попыток до "
                    "рестартов: %d): %s", _rs_key, _prev_att, _prev_fb[:160])
    # ФИКС ВОЛНЫ 200: писателю идут ВСЕ различные прежние отказы, а не только последний —
    # иначе он правит свежую претензию и ре-ломает более раннюю (у source_bit 15 отказов
    # с ротирующими причинами). Требование «не регрессируй» ревьюер и так проверяет.
    _prev_hist = _load_reseed_history(cfg, _rs_key)
    if _prev_hist:
        logger.info("[exp] пересев %s: в промпт идут %d различных прежних отказов ревью",
                    _rs_key, len(_prev_hist))
    for attempt in range(3):
        _att_total += 1
        prompt = base_prompt if not feedback else (
            base_prompt + "\n\nПРЕДЫДУЩАЯ ВЕРСИЯ ОТКЛОНЕНА КОД-РЕВЬЮ — ИСПРАВЬ ИМЕННО ЭТО:\n" + feedback)
        _older = [h for h in _prev_hist if _reason_key(h) != _reason_key(feedback or "")]
        if _older:
            prompt += ("\n\nПРЕЖНИЕ ОТКАЗЫ КОД-РЕВЬЮ ЭТОГО ЖЕ СКРИПТА — НИ ОДИН НЕ ВОСКРЕШАЙ "
                       "(правка свежей претензии ценой возврата старой = снова отказ; ревьюер "
                       "проверяет ВСЕ требования сразу, а не только последнее):\n"
                       + "\n".join(f"{i}. {h}" for i, h in enumerate(_older, 1)))
        # codex-агент ПИШЕТ скрипт прямо в файл (не в обрезаемое сообщение) → длинный код не рвётся.
        if exp_eng == "codex" and engines.codex_available():
            # артефакт ПЕР-KIND (фикс в.179): общий `_exp_wip.py` был единственной причиной
            # глобального single-flight, который блокировал размещение чужих kind'ов.
            code = _extract_py(engines.codex_write_artifact(
                prompt, cfg, f"Results/scripts/_exp_wip_{_safe(_rs_key)[:40]}.py"))
        else:
            code = _extract_py(engine_call(exp_eng, cfg)(prompt))
        if not code.strip():
            feedback = "пустой вывод — выдай полный самодостаточный python-скрипт"; continue
        code = _fix_leading_zero_ints(code)  # детерминированно чиним 0350/007, чтобы не жечь ретраи
        try:
            ast.parse(code)
        except SyntaxError as e:
            feedback = f"скрипт не парсится: {e}"; continue
        # МАШИННАЯ проверка инфра-контракта ДО codex-ревью (фикс волны 185): дешевле (0 токенов) и
        # жёстче мягкого судьи. На ПОСЛЕДНЕЙ попытке не валим — иначе kind без одобренного скрипта
        # (первый сев) заморозился бы навсегда; там нарушение только логируется и идёт в уроки.
        _viol = _mech_contract_violations(code)
        # ФИКС ВОЛНЫ 188: к инфра-контракту добавлен КОНТРАСТ спеки — кандидат, который не читает
        # объявленные `METHODS`/`LR_GRID_*`/`SCALES`, измеряет не заявленный claim, и его всё равно
        # отклонит codex-ревью (только на 15-25 мин дороже). Проверяем машинно и бесплатно.
        _viol += [f'скрипт не реализует ENV-ключ "{k}" из спеки (ни чтения, ни его значений) — '
                  f'заявленный контраст claim\'а физически не измеряется' for k in _contrast_gap(spec, code)]
        if _viol and attempt < 2:
            issues = "МАШИННАЯ ПРОВЕРКА ИНФРА-КОНТРАКТА: " + "; ".join(_viol)
            logger.info("[exp] мех-контракт отклонил kind=%s (попытка %d): %s", kind, attempt + 1,
                        issues[:200])
            feedback = issues
            _save_reseed_feedback(cfg, _rs_key, feedback, _att_total)  # фикс в.192
            try:
                os.makedirs(os.path.dirname(lessons_path), exist_ok=True)
                _line = f"- {issues[:600]}\n"
                with open(lessons_path, "a", encoding="utf-8") as _lf:
                    _lf.write(_line)
                _self_lessons_mtime = os.path.getmtime(lessons_path)
                _self_written_bytes += len(_line.encode("utf-8"))
            except OSError:
                pass
            continue
        if _viol:
            logger.warning("[exp] kind=%s: инфра-контракт НЕ выполнен даже на последней попытке "
                           "(%s) — принимаю, чтобы не заморозить kind", kind, "; ".join(_viol)[:200])
        ok, issues = _review_exp_script(cfg, kind, spec, code)
        if ok:
            break
        logger.info("[exp] код-ревью отклонил kind=%s (попытка %d): %s", kind, attempt + 1, issues[:160])
        feedback = issues
        _save_reseed_feedback(cfg, _rs_key, feedback, _att_total)  # фикс в.192
        try:  # durable-урок: причина отказа переживает сев и попадает в будущие автогены этого kind
            os.makedirs(os.path.dirname(lessons_path), exist_ok=True)
            _line = f"- {issues[:600]}\n"
            with open(lessons_path, "a", encoding="utf-8") as _lf:
                _lf.write(_line)
            _self_lessons_mtime = os.path.getmtime(lessons_path)
            _self_written_bytes += len(_line.encode("utf-8"))
        except OSError:
            pass
    else:
        if _existing:
            # ПЕРЕСЕВ не удался, но РАБОЧИЙ скрипт есть — не блокируем узел (это стоило бы прогона).
            # Помечаем скрипт свежим (utime), чтобы не жечь codex на каждом запуске: следующий
            # пересев случится, когда уроки снова обновятся.
            logger.warning("[exp] пересев kind=%s отклонён ревью (%s) — оставляю прежний скрипт",
                           kind, feedback[:160])
            # ...НО (фикс волны 174) только если уроки не трогал НИКТО СНАРУЖИ, пока шёл пересев.
            # `_lessons` читается ОДИН раз до цикла попыток, поэтому КОНТРАКТ, дописанный вахтёром
            # в середине пересева, ни одной попыткой не прочитан; слепой utime(now) делал скрипт
            # свежее этого контракта ⇒ он не попадал в код НИКОГДА (тот же дедлок, что в в.172,
            # только с гонкой по времени). Внешняя правка ⇒ НЕ штампуем: следующий цикл пересеет
            # заново и прочитает её. Свои же дописанные причины отказов (их видели попытки 2-3
            # как feedback) штамп по-прежнему гасит — иначе codex жёгся бы на каждом опросе.
            try:  # байтовый учёт (фикс в.176) — устойчив к ПОРЯДКУ внешней и внутренних записей
                _external = os.path.getsize(lessons_path) != _self_lessons_size + _self_written_bytes
            except OSError:
                _external = False
            if _external:
                logger.info("[exp] уроки kind=%s правлены ИЗВНЕ во время пересева — не штампую "
                            "скрипт, следующий цикл пересеет по новому контракту", kind)
            else:
                try:
                    os.utime(path, None)
                except OSError:
                    pass
            return rel
        raise ValueError(f"скрипт kind={kind} не прошёл код-ревью за 2 попытки: {feedback[:200]}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if _existing:  # прежняя версия остаётся рядом (провенанс + возможность сравнить контракты)
        try:
            open(path + ".prev", "w", encoding="utf-8").write(
                open(path, encoding="utf-8", errors="replace").read())
        except OSError:
            pass
    # АТОМАРНАЯ подмена (фикс волны 180): с тех пор как прогон стартует на прежнем одобренном
    # скрипте ПАРАЛЛЕЛЬНО пересеву, `_copy_project_script` может читать этот файл ровно в момент
    # записи новой версии → на GPU уехал бы обрезанный скрипт (SyntaxError на первой секунде и
    # сожжённая попытка узла). `os.replace` делает подмену неделимой: читатель видит либо старую,
    # либо новую версию целиком.
    _tmp = path + ".new"
    with open(_tmp, "w", encoding="utf-8") as _f:
        _f.write(code)
    os.replace(_tmp, path)
    _shadow_save(cfg, rel, code)  # фикс в.194: одобренная версия дублируется ВНЕ вольта/workspace codex'а
    _clear_reseed_state(cfg, _rs_key)  # фикс в.192: версия принята — накопленный фидбек больше не нужен
    logger.info("[exp] самописный скрипт %s прошёл код-ревью (%d симв, попыток с учётом "
                "убитых рестартами: %d)", rel, len(code), _att_total)
    return rel


def _blocked(cfg: ARConfig, rec: Dict[str, Any], error: str) -> Dict[str, Any]:
    rec["status"] = "blocked_no_spec"
    rec["error"] = error
    _card(cfg, rec.get("nid", "?"), rec.get("short", ""), "failed",
          f"typed EXP_SPEC router отказал: {error}", log=True)
    return rec


def _plan(cfg: ARConfig, node: Dict[str, Any], derive: str,
          spec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return exp_spec.canonicalize(
        spec or node.get("exp_spec"),
        context=f"node={node.get('id')} short={node.get('short','')} "
                f"idea={node.get('idea','')} derive={derive[:1000]}",
        exp_kind=getattr(cfg, "exp_kind", ""),
    )


def _rec_path(cfg: ARConfig, nid: str) -> str:
    return os.path.join(_exp_dir(cfg), _safe(nid) + ".json")


def _save_rec(cfg: ARConfig, rec: Dict[str, Any]) -> None:
    p = _rec_path(cfg, rec["nid"])
    # ФИКС ВОЛНЫ 186 (КОРЕНЬ осиротевших прогонов): запись пишется словарём ЦЕЛИКОМ, а копий этой
    # записи в памяти несколько — `poll()` в главном цикле и `_drain_exp` из потока отчёта грузят
    # свои снимки с диска. Снимок, сделанный ДО размещения (`waiting`, без полей `server/tmux/log`),
    # позже затирал уже размещённый `running` → живой прогон на GPU становился «ничьим»: poll о нём
    # не знает, вердикт не доедет до графа, узел навсегда в `experiment_running`. Downgrade
    # running→waiting/new с ПОТЕРЕЙ полей размещения = всегда потерянное обновление, а не решение:
    # подмешиваем поля с диска обратно. Терминальные статусы (finalize/blocked) не трогаем — они
    # снимают запись через `_del_rec`/осознанно.
    if str(rec.get("status")) in ("waiting", "new") and not rec.get("tmux"):
        try:
            disk = json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError):
            disk = None
        if disk and str(disk.get("status")) == "running" and disk.get("tmux"):
            for k in ("status", "server", "gpu", "tmux", "log", "done", "started_at",
                      "_log_size", "_grow_ts", "_cpu_time", "_cpu_ts", "out"):
                if k in disk:
                    rec[k] = disk[k]
            logger.warning("[%s] потерянное обновление записи прогона: снимок без полей размещения "
                           "пытался затереть running (%s) — поля восстановлены с диска",
                           rec.get("nid"), disk.get("tmux"))
    json.dump(rec, open(p + ".tmp", "w", encoding="utf-8"), ensure_ascii=False)
    os.replace(p + ".tmp", p)


def _del_rec(cfg: ARConfig, nid: str) -> None:
    try:
        os.remove(_rec_path(cfg, nid))
    except OSError:
        pass
    # Syncthing-клоны того же nid («rec 2.json», «rec.sync-conflict-...json») иначе остаются
    # навсегда: _del_rec чистит только каноническое имя, а _records их снова подхватывает.
    stem = _safe(nid)
    for fp in glob.glob(os.path.join(_exp_dir(cfg), stem + " *.json")) + \
              glob.glob(os.path.join(_exp_dir(cfg), stem + ".sync-conflict*")):
        try:
            os.remove(fp)
        except OSError:
            pass


def _records(cfg: ARConfig) -> List[Dict[str, Any]]:
    out = []
    for fp in glob.glob(os.path.join(_exp_dir(cfg), "*.json")):
        try:
            rec = json.load(open(fp, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        # Файл, имя которого НЕ каноническое для своего nid (Syncthing-дубль «... 2.json»,
        # sync-conflict) — зомби: _del_rec его не удалит, poll будет вечно «завершать» его
        # заново (спам графа/лога) И он навсегда занимает слот pending_count < max_concurrent_exp,
        # блокируя запуск НОВЫХ экспериментов. Не грузим такие записи; уводим в experiments_stale/.
        if os.path.basename(fp) != _safe(rec.get("nid") or "") + ".json":
            try:
                stale = os.path.join(_exp_dir(cfg), "..", "experiments_stale")
                os.makedirs(stale, exist_ok=True)
                os.replace(fp, os.path.join(stale, os.path.basename(fp)))
                logger.warning("не-канонический exp-record (Syncthing-дубль?) → experiments_stale/: %s",
                               os.path.basename(fp))
            except OSError:
                pass
            continue
        out.append(rec)
    return out


def _holds_gpu_slot(cfg: ARConfig, rec: Dict[str, Any]) -> bool:
    """Занимает ли запись слот GPU-полосы (`pending_count < max_concurrent_exp`).

    ФИКС ВОЛНЫ 184. `waiting`-запись, у которой скрипта kind'а НЕТ НА ДИСКЕ ВООБЩЕ (первый сев
    циклически отклоняется код-ревью — у `custom_llama` три отказа по СУЩЕСТВУ науки), GPU не
    занимает и занять НЕ МОЖЕТ, но слот держала вечно: два таких узла съедали 2 из 4 слотов, из-за
    чего `_pick/_seed_confirm_candidate` не вызывался (`pending_count < MAX_EXP` ложно) → новые дети
    оси E не сеялись, а значит не звался и `_ensure_exp_script` → фоновый ПЕРЕСЕВ её скрипта под
    контракты рецензента не запускался вовсе. Плюс `pending_count` никогда не обнулялся → отчётный
    гейт вечно считал «прогоны в полёте» (quiescent только по 2×cooldown-клапану).
    `waiting` из-за занятых карт/несостоявшегося tmux — ЭТО слот (реальная попытка запуска).

    ФИКС ВОЛНЫ 213 (тот же класс, что в.184 и в.210: «счётчик обязан считать только то, что МОЖЕТ
    получить ресурс»). Ожидание карты — законный слот, ПОКА оно правдоподобно кончится. Но запись,
    которой карта нужного размера не находится в пуле уже НЕСКОЛЬКО ЧАСОВ (три записи 760M требуют
    ≥44 ГБ free, а в пуле один сервер, где столько не освобождается сутками), GPU не занимает и
    занять не может — а слот держит. Замер волны 213: `pending_count = 4 == max_concurrent_exp`, из
    них 3 такие записи ⇒ ВЕСЬ блок фазы подтверждения (`supervisor:1774` — и выбор ГОТОВОГО
    кандидата, и посев нового) пропускался КАЖДЫЙ тик, то есть директива критика №4 («новый
    prospective audit») физически не могла получить узел. Снятие слота НЕ запускает ничего лишнего:
    размещение по-прежнему требует реально свободной карты (`_pick_gpu_multi`), поэтому физический
    параллелизм прогонов не растёт; растёт только число ЗАПИСЕЙ, и оно ограничено прежним клапаном
    `len(recs) >= 2*mx` в `pending_count`. Запись остаётся `waiting` и будет размещена, как только
    карта появится."""
    if str(rec.get("status")) != "waiting":
        return True
    try:
        _nofit = float(rec.get("_nofit_since") or 0)
    except (TypeError, ValueError):
        _nofit = 0.0
    if _nofit and (time.time() - _nofit) >= NOFIT_SLOT_RELEASE_S:
        # лог — не чаще раза в час на запись: `_holds_gpu_slot` зовётся из `pending_count` по
        # несколько раз за тик, и безусловная строка залила бы лог (наблюдалось в волне 213).
        _nid = str(rec.get("nid"))
        if time.time() - _NOFIT_LOGGED.get(_nid, 0.0) > 3600:
            _NOFIT_LOGGED[_nid] = time.time()
            logger.info("[%s] ждёт карту ≥%s МБ уже %.1f ч — слот GPU-полосы снят (в.213): в пуле "
                        "такой карты нет, запись ресурса не занимает; размещение произойдёт, когда "
                        "карта освободится", _nid, rec.get("_nofit_mb"), (time.time() - _nofit) / 3600.0)
        return False
    return _script_on_disk(cfg, rec)


def _script_on_disk(cfg: ARConfig, rec: Dict[str, Any]) -> bool:
    """Лежит ли на диске непустой скрипт этой записи (пер-claim'овый или kind'овый).

    Вынесено из `_holds_gpu_slot` (фикс волны 225), чтобы клапан `pending_count` мог отличать
    «запись ждёт СВОЙ скрипт от codex» (растущая очередь — её и надо ограничивать) от «запись ждёт
    карту, которой в пуле нет» (число таких записей фиксировано, ограничивать их бессмысленно).
    Консервативно True при неизвестном пути/ошибке ФС: неизвестность не должна снимать слот."""
    kind = str(rec.get("kind") or "")
    rel = str((rec.get("exp_spec") or {}).get("script") or exp_spec.KIND_SCRIPTS.get(kind, "") or "")
    if not rel:
        return True
    try:
        path = os.path.join(os.path.abspath(cfg.project_root), rel)
        return os.path.exists(path) and os.path.getsize(path) > 0
    except OSError:
        return True


def pending_count(cfg: ARConfig) -> int:
    recs = _records(cfg)
    held = sum(1 for r in recs if _holds_gpu_slot(cfg, r))
    mx = int(getattr(cfg, "max_concurrent_exp", 0) or 0) or 4
    # ФИКС ВОЛНЫ 225 (тот же класс, что в.184/в.210/в.213: «счётчик обязан считать только то, что
    # МОЖЕТ получить ресурс»). Клапан стоял на `len(recs)`, то есть на ВСЕХ записях — включая те,
    # у которых слот уже СНЯТ фиксом в.213 (ждут карту ≥44 ГБ, которой нет ни на одном сервере
    # пула; их число не убывает СУТКАМИ и зависит от решения юзера о ёмкости). Замер волны 225:
    # 8 записей при max=4 ⇒ клапан вернул 4 == MAX_EXP при ОДНОМ реально живом прогоне ⇒ весь блок
    # фазы подтверждения (`supervisor:1853`) пропускался каждый тик НАВСЕГДА: ни выбор готового
    # кандидата, ни посев нового узла (а значит и директива критика про новый prospective прогон —
    # единственный рычаг оси E) физически не могли случиться. То есть клапан гасил ровно тот фикс,
    # который снимал слот. Считаем клапаном ТОЛЬКО очередь на скрипт: она и есть растущая величина
    # (каждый посев добавляет запись, ждущую своего codex-скрипта), и именно её ограничивал
    # исходный замысел («kind без скрипта копит waiting-записи»).
    unscripted = sum(1 for r in recs
                     if not _holds_gpu_slot(cfg, r) and not _script_on_disk(cfg, r))
    if unscripted >= 2 * mx:
        # страховка: записи без слота не имеют права размножаться без границ (иначе confirm-lane
        # сеял бы новый узел каждый тик, пока kind без скрипта копит waiting-записи).
        return max(held, mx)
    return held


def record_nids(cfg: ARConfig) -> set:
    """nid'ы всех живых записей прогонов (нужно гигиене графа: узел в experiment_running без записи
    залипает навсегда — см. `supervisor._unwedge_phantom_experiment_nodes`, фикс волны 179)."""
    return {str(r.get("nid")) for r in _records(cfg) if r.get("nid")}


def _counter_path(cfg: ARConfig) -> str:
    return os.path.join(cfg.workdir, ".run", "exp_counter.json")


def _bump(cfg: ARConfig, key: str) -> None:
    """Счётчик проведённых GPU-экспериментов для дашборда (ran/confirmed/no_signal)."""
    p = _counter_path(cfg)
    try:
        d = json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        d = {"ran": 0, "confirmed": 0, "no_signal": 0, "invalid": 0}
    d[key] = int(d.get(key, 0)) + 1
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    json.dump(d, open(tmp, "w", encoding="utf-8"))
    os.replace(tmp, p)


# ---------------- карточка эксперимента в дашборде ----------------

def _card(cfg: ARConfig, nid: str, short: str, stage: str, detail: str, log: bool = False) -> None:
    # одна когерентная карточка на узел: та же agent-<nid>, что вёл идейный агент → не плодим
    try:
        agents_status.update(cfg, f"agent-{nid}", node=nid, short=short, stage=stage,
                             detail=detail, engine="codex→GPU", role="экспериментатор (GPU)")
        if log:
            agents_status.log_event(cfg, f"agent-{nid}", f"{stage} · {detail[:90]}")
        agents_status.write_snapshot(cfg)
    except Exception:
        pass


# ---------------- размещение прогона на свободной карте ----------------

def _adopt_live_run(cfg: ARConfig, rec: Dict[str, Any], sess: str,
                    log: str, done: str) -> Optional[Dict[str, Any]]:
    """ФИКС ВОЛНЫ 186. Вернуть запись под управление poll'а, если прогон этого nid РЕАЛЬНО жив.

    Запись прогона может потерять поля размещения (`server/gpu/tmux/log/done`) при затирании
    свежего `running` устаревшим снимком той же записи (`_save_rec` пишет словарь целиком, а
    `poll()`/`_drain_exp` держат СВОИ копии, загруженные с диска до размещения). Последствия те же,
    что у фантомных узлов волны 179, только наоборот: процесс на GPU идёт часами, а лупу о нём
    ничего не известно — вердикт не доедет, узел вечно `experiment_running`, карта занята «ничьим»
    процессом, а сама запись каждый цикл падает в `waiting` (карту-то занял её же прогон).
    Пробуем адоптировать: ищем ЖИВУЮ tmux-сессию с каноническим именем этого nid по серверам пула.
    GPU-индекс восстанавливаем из cmdline процесса (`cuda:N`), он нужен только для дашборда.
    Возврат: обновлённая запись (адопция удалась) либо None (живого прогона нет — размещаем как обычно).
    """
    nid = str(rec.get("nid") or "")
    for srv in _servers(cfg):
        try:
            r = subprocess.run(["ssh", srv,
                                f"tmux has-session -t {sess} 2>/dev/null && echo ALIVE || echo NO"],
                               capture_output=True, text=True, timeout=20)
        except (subprocess.TimeoutExpired, OSError):
            continue                                  # сервер не ответил — не наше дело, идём дальше
        if "ALIVE" not in r.stdout:
            continue
        gpu = -1
        try:
            ps = subprocess.run(["ssh", srv, f"ps -eo cmd | grep '[e]xp_{_safe(nid)}' | head -3"],
                                capture_output=True, text=True, timeout=20).stdout
            m = re.search(r"cuda:(\d+)", ps)
            if m:
                gpu = int(m.group(1))
        except (subprocess.TimeoutExpired, OSError, ValueError):
            pass
        rec.update({"status": "running", "server": srv, "gpu": gpu,
                    "tmux": sess, "log": log, "done": done})
        rec.setdefault("started_at", time.time())
        rec["_grow_ts"] = time.time()                 # окно watchdog'а считаем с момента адопции
        rec["_counted"] = True                        # запуск уже был посчитан
        rec.pop("_nofit_since", None)                 # в.213: прогон РАЗМЕЩЁН, метка неразмещаемости снимается
        rec.pop("_nofit_mb", None)
        _save_rec(cfg, rec)
        logger.warning("[%s] АДОПЦИЯ: живая tmux-сессия %s на %s (cuda:%s) — запись потеряла поля "
                       "размещения, возвращаю прогон под управление poll()", nid, sess, srv, gpu)
        _card(cfg, nid, rec.get("short", ""), "experiment",
              f"адоптирован живой прогон на {srv} cuda:{gpu} (запись была осиротевшей)", log=True)
        return rec
    return None


def _remote_launch_bundle(srv: str, py: str, remote_script: str, gpu: int,
                          rec: Dict[str, Any]) -> Dict[str, Any]:
    """IMMUTABLE LAUNCH BUNDLE прогона: sha256 РЕАЛЬНО исполняемого файла + отпечаток GPU-хоста.

    ОСЬ R, фикс волны 205. Волна 189 сохранила `launch_command`, и рецензент это зачёл — но W4
    свежей рецензии (04:24 MSK, SCORE 6) держит `R=partial` СЛЕДУЮЩИМ пунктом: «load-bearing
    reanalysis bundles ... не восстанавливают исходный GPU training run из полного immutable launch
    bundle», и в требованиях к проспективному аудиту перечислено дословно: exact remote command,
    **exact executed-script hash**, training commit, model/data/tokenizer/config hashes,
    **environment fingerprint**, per-block rows, shared bootstrap indices, per-arm wall-clock.
    Из этого списка луп снимал ТОЛЬКО команду и commit ⇒ даже успешный краеугольный прогон закрыл бы
    E, но НЕ R, а по детерминированной таблице критика два partial'а = потолок 6 («не 7: E, N и R
    остаются partial»). То есть отсутствие этой пробы было потолком балла, а не мелочью провенанса.

    ХЭШ СНИМАЕТСЯ НА GPU-ХОСТЕ, НЕ ЛОКАЛЬНО — это принципиально (класс волны 194: канонический
    артефакт лежит в Syncthing-вольте, куда пишет ещё и Мак, и уже наблюдалась байт-в-байт подмена
    одобренной версии). Хэш локальной копии доказывал бы целостность НЕ ТОГО файла — ровно тот класс
    («печать поставлена не на тот экземпляр»), за который в.200 получала SCORE≤4.

    Проба — ОДИН plain ssh сразу после подтверждённого старта, без subshell-`$()` (§5 CLAUDE.md).
    Любой сбой пробы НЕ ломает запуск: bundle просто не появится. Реконструировать поля задним
    числом ЗАПРЕЩЕНО рецензией («старые metadata не реконструировать»), поэтому лучше отсутствие
    поля, чем догадка.
    """
    if not remote_script:
        return {}
    probe = (f"sha256sum {_q_path(remote_script)}; "
             f"nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i {int(gpu)}; "
             f"{_q_path(py)} -c 'import sys,torch;print(\"PYENV \"+sys.version.split()[0]+\" \"+"
             f"torch.__version__+\" \"+str(torch.version.cuda))'")
    try:
        p = subprocess.run(["ssh", srv, probe], capture_output=True, text=True, timeout=90)
    except Exception as e:                      # проба диагностическая — запуск уже идёт
        logger.info("[R] проба провенанса на %s не выполнена: %s", srv, e)
        return {}
    sha = gpu_name = driver = pyver = torchver = cudaver = ""
    for ln in (p.stdout or "").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if re.fullmatch(r"[0-9a-f]{64}\s+\S.*", ln):
            sha = ln.split()[0]
        elif ln.startswith("PYENV "):
            parts = ln.split()
            pyver, torchver, cudaver = (parts + ["", "", ""])[1:4]
        elif "," in ln and not gpu_name:        # строка nvidia-smi: "<name>, <driver>"
            cols = [c.strip() for c in ln.split(",")]
            gpu_name = cols[0]
            driver = cols[1] if len(cols) > 1 else ""
    if not sha:
        logger.info("[R] проба провенанса на %s не дала sha256 (stderr: %s)",
                    srv, (p.stderr or "")[:200])
        return {}
    env = rec.get("env") or {}
    bundle = {
        "executed_script_path": remote_script,
        "executed_script_sha256": sha,
        "executed_script_sha256_command": f"ssh {srv} sha256sum {remote_script}",
        "host": srv,
        "gpu_index": int(gpu),
        "gpu_name": gpu_name,
        "driver_version": driver,
        "python_version": pyver,
        "torch_version": torchver,
        "cuda_version": cudaver,
        "model": str((rec.get("exp_spec") or {}).get("model") or ""),
        "model_config_hash": "sha256:" + hashlib.sha256(json.dumps(
            {k: env[k] for k in sorted(env) if k in (
                "N_EMBD", "N_LAYER", "N_HEAD", "SEQ", "BATCH", "PRE_STEPS", "FT_STEPS")},
            sort_keys=True).encode()).hexdigest()[:16],
        "run_env_hash": "sha256:" + hashlib.sha256(
            json.dumps({str(k): str(v) for k, v in sorted(env.items())},
                       sort_keys=True).encode()).hexdigest()[:16],
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "containerized": False,
    }
    bundle["environment_fingerprint"] = "sha256:" + hashlib.sha256(json.dumps(
        {k: bundle[k] for k in ("host", "gpu_name", "driver_version", "python_version",
                                "torch_version", "cuda_version")},
        sort_keys=True).encode()).hexdigest()[:16]
    logger.info("[R] launch bundle снят на %s: script sha256=%s… env=%s",
                srv, sha[:12], bundle["environment_fingerprint"])
    return bundle


def _place(cfg: ARConfig, rec: Dict[str, Any]) -> Dict[str, Any]:
    """Попытаться поставить прогон на свободную GPU. Нет карты → статус waiting."""
    nid, short = rec["nid"], rec.get("short", "")
    kind = rec.get("kind") or (rec.get("exp_spec") or {}).get("kind") or "legacy_llama_forgetting"
    llama_kinds = {
        "legacy_llama_forgetting",
        "llama_gnmuon_audit",
        "llama_checkpoint_pool_selector",
        "llama_firewall_calibration_audit",
        "llama_alignment_bridge_audit",
        "llama_proxy_falsification_grid",
        "llama_fixed_baseline_audit",
        "llama_source_bit_frontier_audit",  # llama-kind: скрипты этого kind'а строят сеть через
        # llama_exp_kit, но kind отсутствовал в списке → required_kit ошибочно = "llm", и сервер
        # отбраковывался по llm-пробе (numpy ABI), хотя llama-стек на нём рабочий.
        "custom_llama",
    }
    required_kit = "llama" if kind in llama_kinds else "llm"
    # масштаб-зависимый минимум свободной VRAM: крупная модель (N_EMBD) требует больше, иначе OOM
    # (350M/760M падали на картах с 10-21GB free). 768→11GB, 1024→26GB, 1536→44GB.
    try:
        _nembd = int((rec.get("env") or {}).get("N_EMBD")
                     or ((rec.get("exp_spec") or {}).get("env") or {}).get("N_EMBD") or 768)
    except (TypeError, ValueError):
        _nembd = 768
    _min_free = (11000 if _nembd <= 768 else
                 (26000 if _nembd <= 1024 else (44000 if _nembd <= 1536 else 60000)))
    # МУЛЬТИ-МАСШТАБНЫЕ прогоны (frontier-audit гоняет 124M→350M→760M ВНУТРИ одного скрипта) не
    # видны по N_EMBD: он описывает только ПЕРВЫЙ слайс (768) → карта на 11GB бралась под прогон,
    # которому нужен 760M, и крупные слайсы детерминированно падали в OOM (рецензия: «350M/760M
    # frontier slices: OOM, не результат» → ось E не может стать pass). Берём максимум по тегам
    # масштаба, упомянутым в claim/confirm_rule/env спеки.
    try:
        _spec_txt = json.dumps({"k": kind, "s": rec.get("exp_spec") or {}, "e": rec.get("env") or {}},
                               ensure_ascii=False)
    except (TypeError, ValueError):
        _spec_txt = str(kind)
    for _tag, _need in (("1.4B", 60000), ("1B", 60000), ("760M", 44000), ("350M", 26000)):
        if _tag in _spec_txt:
            _min_free = max(_min_free, _need)
            break
    # ФИКС ВОЛНЫ 211 — БЮДЖЕТ ПАМЯТИ СЧИТАЛСЯ ПО СПЕКЕ, А ГЕОМЕТРИЮ ЗАДАЁТ СКРИПТ.
    # Спека `gnfirewall_custom_llama` говорит «на LLaMA РАЗНЫХ МАСШТАБОВ» словами и не содержит ни
    # одного тега, env — только SEEDS ⇒ `_min_free` = 11000, и карта с 14,7 ГБ free прошла отбор.
    # А в самом скрипте `SCALE_TAGS` по умолчанию = "124M,350M" (и контракт в.183 прямо говорит, что
    # геометрия именованных масштабов берётся ТОЛЬКО из таблицы скрипта, env её не переопределяет —
    # прогон это и напечатал: `generic_geometry_env_ignored_for_named_scales`). Итог: скрипт, впервые
    # прошедший код-ревью с 8-й попытки, умер OOM за 47 секунд, сжёг попытку узла и получил ЛОЖНЫЙ
    # урок про память. Тот же класс, что в.188 («спека и скрипт мерят разное»), только в РЕСУРСЕ.
    # Берём теги, которые прогон РЕАЛЬНО построит: env, если задан, иначе дефолт из текста скрипта.
    # Важно: сканировать НЕЛЬЗЯ по таблице геометрии (`TABLE={... "760M": ...}` есть почти всегда) —
    # только по фактическому значению SCALE_TAGS, иначе `_min_free` = 44000 и kind неразмещаем.
    try:
        for _t in _runtime_scale_tags(cfg, rec, kind):
            _need = _SCALE_FREE_MB.get(_t, 0)
            if _need > _min_free:
                logger.info("[%s] бюджет VRAM поднят по СКРИПТУ: масштаб %s требует %d МБ "
                            "(спека тега не содержит) — иначе детерминированный OOM (в.211)",
                            nid, _t, _need)
                _min_free = _need
    except Exception as _se:                          # бюджет не должен ронять размещение
        logger.info("[%s] теги масштабов из скрипта не определены (%s) — бюджет по спеке", nid, _se)
    # ИМЯ ПРОГОНА ОБЯЗАНО БЫТЬ УНИКАЛЬНЫМ (баг волны 171): раньше было `[:80]` — слепая обрезка
    # срезала РАЗЛИЧАЮЩИЙ хвост nid (`..._em_3` и `..._em_5` давали ОДНО имя) → два прогона на одном
    # сервере писали в ОДИН лог, делили ОДНУ tmux-сессию (второй `new-session` молча падал, но
    # `has-session` отвечал OK → запись помечалась running при мёртвом прогоне) и ОДИН .done-маркер
    # → poll финализировал ОБА узла результатом ОДНОГО прогона (подлог вердикта). Теперь при обрезке
    # добавляется хэш полного имени: длина та же, коллизии невозможны.
    # ВОЛНА 186: блок поднят ВЫШЕ GPU-пика — имена нужны для АДОПЦИИ живого прогона (см. ниже).
    _full = _safe(f"{nid}_{kind}")
    safe = _full if len(_full) <= 80 else (
        _full[:71] + "_" + hashlib.sha1(_full.encode("utf-8")).hexdigest()[:8])
    log, done = f"~/exp_{safe}.log", f"~/exp_{safe}.done"
    sess = f"exp_{safe}"  # ВАЖНО: tmux НЕ принимает точки в имени (был баг)
    # ФИКС ВОЛНЫ 186 — АДОПЦИЯ ОСИРОТЕВШЕГО ПРОГОНА. Если запись потеряла поля размещения
    # (`server/tmux/log/done`), а tmux-сессия ЭТОГО ЖЕ nid на сервере ЖИВА, то прогон реально идёт,
    # просто poll о нём больше не знает: вердикт никогда не доедет до графа, узел навсегда застрянет
    # в `experiment_running`, а карта будет занята «ничьим» процессом. Раньше такая запись каждый
    # цикл уходила в `waiting` (нет карты нужного размера — её же занял наш собственный прогон) и
    # висела так вечно. Теперь поля восстанавливаются из живой сессии и poll снова ведёт прогон.
    if str(rec.get("status")) != "running" or not rec.get("tmux"):
        _ad = _adopt_live_run(cfg, rec, sess, log, done)
        if _ad is not None:
            return _ad
    srv, gpu = _pick_gpu_multi(cfg, require_kit=required_kit, min_free_mb=_min_free)
    if srv is None:
        # ФИКС ВОЛНЫ 185: пересев скрипта стоял ЗА GPU-пиком (`_ensure_exp_script` ниже по телу),
        # поэтому пока в пуле нет карты нужного размера, УЛУЧШЕНИЕ скрипта не запускалось вовсе:
        # свежедописанный КОНТРАКТ ждал не своего повода (кода), а чужого (освободившейся карты) —
        # у мультимасштабного audit-kind'а это 10-12 часов до конца текущего прогона. Тот же класс,
        # что в.180, только зеркальный. Пересев — ФОНОВАЯ работа и от карты не зависит: кикаем его
        # прямо здесь (обёртка неблокирующая, тяжёлая часть в daemon-потоке).
        try:
            _sp = rec.get("exp_spec") or {}
            if _sp.get("script"):
                # ФИКС ВОЛНЫ 188: несоответствие «спека объявляет руки — скрипт их не знает» должно
                # попадать в уроки НЕ ДОЖИДАЯСЬ свободной карты (иначе контракт ждёт 10-12 ч чужого
                # прогона, а пересев всё это время улучшает скрипт по НЕПОЛНОМУ списку требований).
                _g = _contrast_gap(_sp, _project_script_src(cfg, str(_sp.get("script", ""))))
                if _g:
                    _note_contrast_gap_lesson(cfg, kind, _g, _sp)
                _ensure_exp_script(cfg, kind, _sp, str(_sp.get("script", "")))
        except ReseedPending:
            pass
        except Exception as _e:                      # не роняем ожидание карты из-за пересева
            logger.info("[%s] фоновый пересев при ожидании карты не стартовал: %s", nid, _e)
        rec["status"] = "waiting"
        # ФИКС ВОЛНЫ 213: метка «карты такого размера в пуле не нашлось» ставится ОДИН раз (первым
        # отказом) и живёт до успешного размещения — по ней `_holds_gpu_slot` через несколько часов
        # снимает слот GPU-полосы, чтобы неразмещаемая запись не запирала фазу подтверждения целиком.
        if not rec.get("_nofit_since"):
            rec["_nofit_since"] = time.time()
        rec["_nofit_mb"] = _min_free
        _save_rec(cfg, rec)
        _card(cfg, nid, short, "waiting_gpu",
              f"ждёт свободную карту ≥{_min_free} МБ в пуле ({', '.join(_servers(cfg))}) — поставлю как освободится")
        return rec
    # Страховка второго уровня: если сессия с таким именем УЖЕ жива (наш же прогон этого nid ещё
    # не умер), НЕ запускаем второй — иначе снова два процесса на один лог/маркер. Ждём и повторим.
    # ФИКС ВОЛНЫ 208: этот ssh был БЕЗ обработки таймаута, а `poll()` зовёт `_place` тоже без
    # обработки → одна медленная проба через прокси роняла ВЕСЬ цикл опроса (`poll экспериментов
    # упал: … tmux has-session … timed out after 20 seconds`, 3 события): остальные записи в этой
    # итерации не опрашивались вовсе — ни размещение, ни финализация вердикта, ни обновление
    # `_grow_ts` (а по нему страж в.168 решает, жив ли молчащий прогон). Асимметрия ущерба: не
    # ответивший ssh НЕ означает «сессии нет», поэтому неизвестность трактуем как ЗАНЯТО (хуже
    # всего — запустить дубль на тот же лог/маркер, класс в.171).
    try:
        _pre = subprocess.run(["ssh", srv, f"tmux has-session -t {sess} 2>/dev/null && echo BUSY || echo FREE"],
                              capture_output=True, text=True, timeout=20)
    except (subprocess.TimeoutExpired, OSError) as _pe:
        rec["status"] = "waiting"
        _save_rec(cfg, rec)
        _card(cfg, nid, short, "waiting_gpu",
              f"проба tmux-сессии на {srv} не ответила — повтор на следующей итерации")
        logger.warning("[%s] проба tmux-сессии %s на %s не ответила (%s) — запуск отложен, "
                       "дубль не рискуем плодить", nid, sess, srv, _pe)
        return rec
    if "BUSY" in _pre.stdout:
        rec["status"] = "waiting"
        _save_rec(cfg, rec)
        _card(cfg, nid, short, "waiting_gpu", f"tmux-сессия {sess} уже занята на {srv} — повтор позже")
        logger.warning("[%s] tmux-сессия %s УЖЕ существует на %s — не запускаю дубль", nid, sess, srv)
        return rec
    py = cfg.gpu_python_map.get(srv, cfg.gpu_python)  # путь к python разный на серверах
    # env-правки typed EXP_SPEC (перебор гиперов без правки логики/claim).
    envs = _env_assign(rec.get("env") or {})
    spec = rec.get("exp_spec") or {}
    remote_script = ""   # ветка legacy_llama_forgetting его не задаёт (см. _remote_launch_bundle)
    try:
        if kind == "legacy_llama_forgetting":
            script = "~/llama_exp_kit.py"
            cmd = f"{_q_path(py)} -u {script} cuda:{gpu} 512 8 8 300 150"
        else:
            script_rel = _ensure_exp_script(cfg, kind, spec, spec.get("script", ""))
            # Версия скрипта, уже разбившаяся трейсбэком на GPU, запускается только впустую
            # (фикс волны 181): попытка узла сгорит за 3 секунды, чисел не будет, а на invalid
            # ещё и авто-сеется новый ребёнок. Ждём пересева (он уже идёт в фоне: рантайм-урок
            # сделал файл уроков новее скрипта). TTL карантина — 6 ч, запереть ось E нельзя.
            _quar = _script_quarantined(cfg, kind, script_rel)
            if _quar:
                rec["status"] = "waiting"
                _save_rec(cfg, rec)
                _card(cfg, nid, short, "waiting_gpu", _quar)
                logger.warning("[%s] запуск отложен: %s", nid, _quar)
                return rec
            # Предусловия САМОГО скрипта авторитетнее claim-текста (фикс волны 178): поднимаем
            # SEEDS до требуемого скриптом минимума — только ВВЕРХ, гейт не ослабляется. Иначе
            # пересеянный по КОНТРАКТУ скрипт (assert >=12 сидов) детерминированно падает invalid
            # на env с 6 сидами и сжигает попытку узла (см. _script_min_seeds).
            _need = _script_min_seeds(_project_script(cfg, script_rel))
            _have = {s for s in str((rec.get("env") or {}).get("SEEDS", "")).replace(" ", "").split(",") if s}
            if _need and len(_have) < _need:
                rec.setdefault("env", {})["SEEDS"] = ",".join(str(i) for i in range(_need))
                envs = _env_assign(rec["env"])
                logger.warning("[%s] скрипт kind=%s требует >=%d уникальных сидов, в env было %d "
                               "→ SEEDS=0..%d", nid, kind, _need, len(_have), _need - 1)
            # ФИКС ВОЛНЫ 188 (АНТИ-ПОДЛОГ): снимаем «покрытие контраста» ИМЕННО той версией скрипта,
            # которая реально уезжает на GPU. Пересев может подменить файл позже, поэтому решение о
            # доверии вердикту принимается по снимку на момент запуска (в `_finalize`), а не по
            # текущему содержимому диска.
            try:
                _src_now = _project_script_src(cfg, script_rel)
                _gap_now = _contrast_gap(spec, _src_now, strict=True)   # для анти-подлога — только
                _gap_soft = _contrast_gap(spec, _src_now)               # сильные признаки; уроку — все
            except Exception as _ge:                            # диагностика не должна ронять запуск
                logger.info("[%s] проверка контраста не выполнена: %s", nid, _ge)
                _gap_now, _gap_soft = [], []
            if _gap_now:
                rec["_contrast_gap"] = _gap_now
                logger.warning("[%s] скрипт kind=%s НЕ реализует контраст спеки (%s): вердикт "
                               "confirmed будет понижен до invalid, пересев обязателен",
                               nid, kind, ", ".join(_gap_now))
            else:
                rec.pop("_contrast_gap", None)
            if _gap_soft:  # урок пишем и по «мягким» признакам (незачитанные LR_GRID_* и т.п.)
                _note_contrast_gap_lesson(cfg, kind, _gap_soft, spec)
            # ФИКС ВОЛНЫ 197 (класс волны 191 «предрешённо-невалидный прогон»): если одобренная
            # версия скрипта содержит ДЕТЕРМИНИРОВАННЫЙ креш `m(x)["logits"]` (llama возвращает
            # logits=None без get_logits=True), то прогон гарантированно умрёт после претрейна с
            # `invalid` — 124M так сжёг 600 шагов и карту. Такой запуск — чистая потеря GPU и слота,
            # поэтому не размещаем, а кикаем пересев (он фоновый и никого не блокирует).
            try:
                with open(_project_script(cfg, script_rel), encoding="utf-8") as _sf:
                    _src_now = _sf.read()
            except OSError:
                _src_now = ""
            if '["logits"]' in _src_now and "get_logits=True" not in _src_now:
                _lp = os.path.join(cfg.workdir, ".run", "exp_review_lessons", f"{kind}.md")
                _lb = ("\n\n### КОНТРАКТ (вахтёр, волна 197): ЛОГИТЫ ТОЛЬКО ЧЕРЕЗ get_logits=True\n"
                       "Машинная проверка ПЕРЕД запуском нашла в одобренном скрипте `[\"logits\"]` "
                       "без `get_logits=True`. Сигнатура llama — `forward(idx, targets=None, "
                       "get_logits=False)`, внутри `logits = logits if get_logits else None`, "
                       "поэтому `m(x)[\"logits\"]` = None и прогон падает AttributeError на первой "
                       "же оценке (наблюдалось: 600 шагов претрейна и карта — впустую). "
                       "Правильно: `m(x, get_logits=True)[\"logits\"]`, либо "
                       "`m(x, targets=y)[\"loss\"]`, либо `eval_loss` кита. Запуск этого kind'а "
                       "ОТМЕНЯЕТСЯ, пока скрипт не исправлен.\n")
                try:
                    if _lb.split("\n")[2][:60] not in open(
                            _lp, encoding="utf-8", errors="replace").read():
                        with open(_lp, "a", encoding="utf-8") as _fh:
                            _fh.write(_lb)
                except OSError:
                    try:
                        os.makedirs(os.path.dirname(_lp), exist_ok=True)
                        with open(_lp, "a", encoding="utf-8") as _fh:
                            _fh.write(_lb)
                    except OSError:
                        pass
                logger.warning("[%s] запуск ОТМЕНЁН: kind=%s содержит детерминированный креш "
                               '`["logits"]` без get_logits=True — прогон был бы предрешённо '
                               "invalid; ждём пересева", nid, kind)
                rec["status"] = "waiting"
                rec["note"] = "скрипт падает на ['logits'] без get_logits=True — ждём пересева"
                return rec
            remote_script = _copy_project_script(cfg, srv, nid, kind, script_rel)
            model = shlex.quote(str(spec.get("model") or exp_spec.DEFAULT_MODEL))
            cmd = f"{_q_path(py)} -u {_q_path(remote_script)} cuda:{gpu} {model}"
            if kind in ("c1_retention", "llama_gnmuon_audit",
                        "llama_checkpoint_pool_selector", "llama_firewall_calibration_audit",
                        "llama_alignment_bridge_audit",
                        "llama_proxy_falsification_grid",
                        "llama_fixed_baseline_audit", "custom_llama",
                        "gnmuon_wallclock", "gnmuon_matched_target", "rho_predictor"):
                rec["out"] = f"~/exp_{safe}.json"
                cmd += f" --out {rec['out']}"
            if kind == "rho_predictor":
                env = rec.get("env") or {}
                if env.get("COV_BATCHES"):
                    cmd += f" --cov-batches {shlex.quote(str(env['COV_BATCHES']))}"
                if env.get("GRAD_BATCHES"):
                    cmd += f" --grad-batches {shlex.quote(str(env['GRAD_BATCHES']))}"
                if env.get("SEQ"):
                    cmd += f" --seq {shlex.quote(str(env['SEQ']))}"
                if env.get("BATCH"):
                    cmd += f" --batch-size {shlex.quote(str(env['BATCH']))}"
    except ReseedPending as e:
        # Скрипт этого kind'а прямо сейчас пересевается по свежим урокам (в фоне). Узел НЕ блокируем
        # и попытку НЕ жжём — просто ждём следующей итерации, когда скрипт будет готов.
        rec["status"] = "waiting"
        _save_rec(cfg, rec)
        _card(cfg, nid, short, "waiting_gpu",
              f"скрипт kind={kind} пересевается по свежим урокам — запуск отложен")
        logger.info("[%s] %s — запуск отложен до готовности скрипта", nid, e)
        return rec
    except Exception as e:
        logger.warning("[%s] EXP_SPEC router refused: %s", nid, e)
        return _blocked(cfg, rec, str(e))
    # OMP/MKL лимит → эксперимент не грузит CPU сервера (вежливость к общим машинам)
    # SCRATCH НЕ В /tmp (фикс волны 181). Автоген-скрипты держат чекпоинты руки в
    # `tempfile.TemporaryDirectory()`, а он идёт в /tmp — на brain_lab это корневой раздел (28 ГБ,
    # 90 % занято ЧУЖИМ, 2,9 ГБ свободно), тогда как чекпоинт 124M-модели ~3 ГБ. Результат: прогон
    # честно доходит до 800 шагов претрейна и умирает в `torch.save`
    # (`RuntimeError: [enforce fail at inline_container.cc:668] unexpected pos ...`) → вердикт
    # invalid, попытка узла сожжена, ось E без чисел. $HOME лежит на 12 ТБ-томе (1,3 ТБ свободно).
    # ФИКС ВОЛНЫ 196: писателю прямым текстом обещано «сырой NS/Muon — в `optim.muon` (llm-baselines,
    # доступен по PYTHONPATH)», а запускались прогоны с PYTHONPATH=$HOME БЕЗ llm-baselines/src ⇒
    # `from optim.muon import Muon` = ImportError на первой строке, вердикт invalid, попытка узла
    # сожжена. Проба кита (выше в этом же файле) давно ставит оба пути — приводим запуск к ней.
    # ⚠️ ФИКС ВОЛНЫ 211, КОРНЕВАЯ ПРИЧИНА СЕРИИ OOM: `nvidia-smi` И `torch` НУМЕРУЮТ КАРТЫ ПО-РАЗНОМУ.
    # Отбор карты (`_pick_gpu_multi`) идёт по индексу nvidia-smi (порядок PCI), а запускается
    # `python … cuda:<тот же индекс>`, где индекс интерпретирует CUDA — и по умолчанию у CUDA порядок
    # `FASTEST_FIRST`, а не PCI. На РАЗНОРОДНОМ <gpu-host> (2×A100-SXM4-80GB, 4×A100-PCIE-40GB,
    # 2×2080Ti) это разные перестановки — замерено в волне 211 по UUID:
    #     nvidia-smi 1(2080Ti)→torch 5 · 2→torch 1 · 3(A100-80GB)→torch 2 · 4→torch 3 · 5→torch 4.
    # Совпадают только индексы 0 и 6. То есть луп МЕРИЛ свободную память на одной карте, а СЧИТАЛ на
    # другой. Прямая улика: прогон `gnfirewall_custom_llama` выбран на nvidia-smi 4 (свободной памяти
    # хватало), запущен `cuda:4` → физически попал на nvidia-smi 5, где было 3,5 ГБ free, и умер OOM
    # за 47 с даже после падения микробатча до 1. Вторая улика: у ЖИВОГО прогона с `gpu=2` вся память
    # (15 496 МБ) висит на nvidia-smi 3. Этим же объясняется класс «сосед занял карту ПОСЛЕ
    # размещения» (в.203): соседа на измеренной карте не было — мы просто считали не на ней.
    # `CUDA_DEVICE_ORDER=PCI_BUS_ID` приводит нумерацию CUDA к нумерации nvidia-smi, после чего
    # `cuda:N` означает ровно ту карту, которую отобрал `_pick_gpu_multi`.
    launch = (f"cd ~ && rm -f {done}; "
              f"export PYTHONPATH=$HOME:$HOME/llm-baselines:$HOME/llm-baselines/src:${{PYTHONPATH:-}}; "
              f"export CUDA_DEVICE_ORDER=PCI_BUS_ID; "
              f"mkdir -p $HOME/tmp; export TMPDIR=$HOME/tmp TMP=$HOME/tmp TEMP=$HOME/tmp; "
              f"OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 {envs} "
              f"nohup {cmd} > {log} 2>&1 < /dev/null; touch {done}")
    # env-значения с ; (напр. SCALES) shlex.quote оборачивает в одинарные кавычки — внутри
    # внешних одинарных кавычек tmux-команды это ломало запуск (tmux-сессия молча не создавалась).
    # POSIX-экранируем кавычки внутри launch.
    _launch_q = launch.replace("'", "'\\''")
    subprocess.run(["ssh", srv, f"tmux new-session -d -s {sess} '{_launch_q}'"],
                   capture_output=True, text=True, timeout=30)
    # проверяем, что сессия РЕАЛЬНО создалась (иначе эксп молча не запустится → узел зависнет)
    chk = subprocess.run(["ssh", srv, f"tmux has-session -t {sess} 2>/dev/null && echo OK || echo NO"],
                         capture_output=True, text=True, timeout=20)
    if "OK" not in chk.stdout:
        rec["status"] = "waiting"
        _save_rec(cfg, rec)
        _card(cfg, nid, short, "waiting_gpu", f"tmux не стартовал на {srv} — повтор позже")
        logger.warning("[%s] tmux-сессия %s не создалась на %s — повтор", nid, sess, srv)
        return rec
    rec.update({"status": "running", "server": srv, "gpu": gpu, "tmux": sess, "log": log, "done": done})
    # ФИКС ВОЛНЫ 213: карта нашлась — метка неразмещаемости снимается (иначе следующее короткое
    # ожидание унаследовало бы многочасовой возраст и слот снялся бы сразу).
    rec.pop("_nofit_since", None)
    rec.pop("_nofit_mb", None)
    # ОСЬ R (фикс волны 189): ТОЧНАЯ команда запуска сохраняется в записи прогона. Рецензент держит
    # R=partial в т.ч. строкой «Exact training launch command отсутствует и честно указан как
    # unreconstructed»: команда собиралась здесь и терялась после ssh, поэтому registry вечно писал
    # training_command=null. Теперь строка (env + интерпретатор + argv, без tmux-обёртки) живёт в
    # записи → попадает в контекст составителя и в submission registry как ФАКТ, а не реконструкция.
    # CUDA_DEVICE_ORDER входит в команду и здесь: для оси R это ФАКТ запуска, а без него `cuda:N`
    # воспроизводит ДРУГУЮ карту (фикс волны 211) — то есть команда была бы невоспроизводимой.
    rec["launch_command"] = (f"ssh {srv} \"cd ~ && CUDA_DEVICE_ORDER=PCI_BUS_ID "
                             f"OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 {envs} {cmd}\"")
    if isinstance(rec.get("exp_spec"), dict):
        # в exp_spec — чтобы команда доехала до графа через payload вердикта и попала в
        # `paper._exp_protocol_summary` (колонка `command` protocol-таблицы, требование W4 рецензии).
        rec["exp_spec"]["launch_command"] = rec["launch_command"]
    # ОСЬ R (фикс волны 205): команды запуска ОДНОЙ НЕ ХВАТАЕТ. W4 свежей рецензии держит R=partial
    # дословно за «не восстанавливают исходный GPU training run из полного immutable launch bundle»
    # и перечисляет, что обязано быть у ПРОСПЕКТИВНОГО аудита: exact executed-script hash,
    # environment fingerprint, model/config hashes. Ни одного из них луп не снимал — то есть даже
    # успешный краеугольный прогон оставил бы R=partial (потолок балла 6 при закрытом E).
    _bundle = _remote_launch_bundle(srv, py, remote_script, gpu, rec)
    if _bundle:
        rec["launch_bundle"] = _bundle
        if isinstance(rec.get("exp_spec"), dict):
            rec["exp_spec"]["launch_bundle"] = _bundle
    rec.setdefault("started_at", time.time())  # для тайминга на дашборде (переживает рестарты оркестратора)
    _save_rec(cfg, rec)
    if not rec.get("_counted"):  # считаем запуск один раз (не на каждый retry размещения)
        _bump(cfg, "ran"); rec["_counted"] = True; _save_rec(cfg, rec)
    logger.info("[%s] typed EXP_SPEC %s запущен на %s cuda:%d", nid, kind, srv, gpu)
    detail = _exp_dashboard_detail(rec)
    _card(cfg, nid, short, "experiment", detail, log=True)
    try:
        agents_status.update(
            cfg, "agent-experimenter", node="EXP_SPEC", short="typed experiment lane",
            stage="experiment",
            detail=detail,
            engine="codex→GPU",
            role="экспериментатор (typed EXP_SPEC)",
        )
        agents_status.write_snapshot(cfg)
    except Exception:
        pass
    return rec


_SEED_REQ_RE = re.compile(r">=?\s*(\d+)\s*(?:[^\s,;.]+\s+){0,2}seeds?\b", re.I)


def _claim_min_seeds(*texts: str) -> int:
    """Минимальное число УНИКАЛЬНЫХ сидов, ЯВНО требуемое claim'ом (напр. '>=20 seeds').

    Автоген пишет скрипт ПОД claim: если claim обещает '>=N независимых seeds', скрипт
    несёт `assert len(set(seeds))>=N`. Но env SEEDS (авторитетный — перебивает argv в kit)
    ставится дефолтом '0,1,2,3,4' (5) → детерминированный AssertionError→invalid НАВСЕГДА
    (до permanent-guard), краеугольный эксп никогда не landed-valid, ось E замерзает.
    Решение (см. lessons custom_llama): конфиг НЕСЁТ >=N сидов, НЕ ослабляем gate. Возвращает
    0, если требование не заявлено; кап 24 (мощность vs GPU-стоимость)."""
    best = 0
    blob = " ".join(t for t in texts if t)
    for m in _SEED_REQ_RE.finditer(blob):
        try:
            best = max(best, int(m.group(1)))
        except ValueError:
            pass
    return min(best, 24)


_INFRA_ERR_RE = re.compile(
    r"no space left|disk quota|inline_container|basic_ios::clear|iostream error|"
    r"out of memory|CUDA error|NCCL|Connection reset|Broken pipe|Read-only file system",
    re.I)
_QUAR_RETRY_S = 6 * 3600      # через сколько карантин версии скрипта снимается «на одну попытку»
_QUAR_KEEP_S = 7 * 24 * 3600  # хранение записей карантина (гигиена файла)


def _quar_path(cfg: ARConfig) -> str:
    return os.path.join(cfg.workdir, ".run", "exp_script_quarantine.json")


def _quar_load(cfg: ARConfig) -> Dict[str, Any]:
    try:
        with open(_quar_path(cfg), "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _quar_save(cfg: ARConfig, data: Dict[str, Any]) -> None:
    path = _quar_path(cfg)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except OSError as e:
        logger.warning("не смог записать карантин скриптов: %s", e)


def _script_sha(cfg: ARConfig, kind: str, script_rel: str = "") -> str:
    """sha1 ИМЕННО ТОЙ версии скрипта kind'а, что лежит сейчас в проекте (её и увозит scp)."""
    rel = script_rel or exp_spec.KIND_SCRIPTS.get(kind, "")
    if not rel:
        return ""
    try:
        with open(os.path.join(os.path.abspath(cfg.project_root), rel), "rb") as fh:
            return hashlib.sha1(fh.read()).hexdigest()
    except OSError:
        return ""


def _quarantine_script(cfg: ARConfig, kind: str, nid: str, err: str, script_rel: str = "") -> None:
    """КАРАНТИН ВЕРСИИ СКРИПТА, ДЕТЕРМИНИРОВАННО ПАДАЮЩЕЙ НА GPU (фикс волны 181).

    После фикса волны 180 («пересев не предусловие запуска») размещение перестало ждать пересева и
    стало запускать прогон на ПОСЛЕДНЕЙ одобренной ревью версии скрипта. Для здорового скрипта это
    правильно, но для скрипта, который падает трейсбэком на первых секундах, получился спин-луп:
    прогон стартует → traceback через ~3 с → вердикт invalid → попытка узла сожжена, урок дописан,
    узел вернулся в conditional → следующая итерация ставит НОВЫЙ прогон на ТОЙ ЖЕ версии скрипта.
    Волна 181: 8 прогонов оси E (`em_7`…`em_14`) за 32 минуты, ноль чисел, восемь сожжённых попыток,
    плюс auto-seed новых детей на каждый invalid (граф пух впустую).

    Решение — карантин не kind'а, а КОНКРЕТНОЙ версии (sha1 файла): пока на диске лежит именно та
    версия, что уже разбилась трейсбэком, размещение уходит в `waiting` (пересев идёт в фоне и
    поднимет sha). Гейты не ослаблены, наука не тронута; TTL `_QUAR_RETRY_S` даёт одну повторную
    попытку раз в 6 ч, чтобы карантин не мог запереть ось E навсегда при вечно отклоняемом пересеве.
    """
    sha = _script_sha(cfg, kind, script_rel)
    if not sha:
        return
    # ИНФРА-СБОЙ НЕ ВИНА СКРИПТА: нет места на диске сервера, CUDA OOM, оборванная сеть — при
    # следующем запуске может пройти. Карантин только для ДЕТЕРМИНИРОВАННЫХ дефектов самого кода.
    if _INFRA_ERR_RE.search(err or ""):
        logger.warning("[%s] рантайм-сбой kind=%s похож на ИНФРА-проблему (%s) — версию скрипта "
                       "в карантин НЕ ставлю", nid, kind, (err or "")[:80])
        return
    data = _quar_load(cfg)
    now = time.time()
    for k, v in list(data.items()):
        try:
            if now - float((v or {}).get("ts") or 0) > _QUAR_KEEP_S:
                data.pop(k, None)
        except (TypeError, ValueError):
            data.pop(k, None)
    data[sha] = {"kind": kind, "nid": nid, "ts": now, "err": (err or "")[:300]}
    _quar_save(cfg, data)
    logger.warning("[%s] версия скрипта kind=%s (sha %s) в КАРАНТИНЕ до пересева: %s",
                   nid, kind, sha[:8], (err or "")[:120])


def _script_quarantined(cfg: ARConfig, kind: str, script_rel: str = "") -> str:
    """Причина, по которой ЭТУ версию скрипта запускать нельзя ('' = можно)."""
    sha = _script_sha(cfg, kind, script_rel)
    if not sha:
        return ""
    data = _quar_load(cfg)
    ent = data.get(sha)
    if not isinstance(ent, dict):
        return ""
    try:
        age = time.time() - float(ent.get("ts") or 0)
    except (TypeError, ValueError):
        age = _QUAR_RETRY_S + 1
    if age > _QUAR_RETRY_S:  # аварийный выпуск: пересев за 6 ч ничего не дал — пробуем ещё раз
        data.pop(sha, None)
        _quar_save(cfg, data)
        logger.warning("[exp] карантин версии скрипта kind=%s (sha %s) снят по TTL — одна повторная "
                       "попытка", kind, sha[:8])
        return ""
    return (f"версия скрипта kind={kind} (sha {sha[:8]}) уже падала трейсбэком "
            f"({str(ent.get('err') or '')[:100]}) — жду пересева")


def _lesson_from_runtime_failure(cfg: ARConfig, kind: str, nid: str, rj: str,
                                 script_rel: str = "") -> None:
    """РАНТАЙМ-СБОЙ ТОЖЕ УРОК (фикс волны 178).

    Самообучение экспериментатора было ОДНОСТОРОННИМ: в `.run/exp_review_lessons/<kind>.md`
    попадали только причины отказов КОД-РЕВЬЮ, а traceback реального прогона — никуда. Скрипт,
    прошедший ревью и упавший на GPU (волна 178: `m(x)["logits"]` = None, потому что kit отдаёт
    логиты только при переданных targets), пересевался бы с ТЕМ ЖЕ багом бесконечно: генератор
    физически не видел, обо что прогон разбился. Теперь любой `invalid` с traceback дописывает
    блок-КОНТРАКТ в уроки kind'а — файл становится новее скрипта, и следующее размещение
    пересевает скрипт по этому уроку (байтовый детектор в.176 считает это внешней правкой,
    поэтому штамп-заморозки не будет)."""
    if not kind:
        return
    try:
        data = json.loads(rj or "{}")
    except (TypeError, ValueError):
        data = {}
    err = str((data or {}).get("error") or "").strip()
    tb = str((data or {}).get("traceback") or "").strip()
    if not err and not tb:
        return
    # ЧУЖОЙ ПРОЦЕСС СЪЕЛ КАРТУ — ЭТО НЕ ДЕФЕКТ СКРИПТА (фикс волны 203). Оба 350M-прогона оси E
    # умерли `OutOfMemoryError`, где ПО ТЕКСТУ САМОГО СООБЩЕНИЯ память держал ПОСТОРОННИЙ процесс
    # (`Process 3517315 has 36.03 GiB in use` при `this process has 3.43 GiB`): карта выбиралась по
    # free-mem, а сосед занял её ПОСЛЕ размещения — brain_lab/<gpu-host> общие. Прежний код всё
    # равно дописывал урок с текстом «сбой воспроизводится ДЕТЕРМИНИРОВАННО» и советом проверять
    # контракт API: файл уроков становился новее скрипта ⇒ пересев ЕДИНСТВЕННОГО рычага оси E по
    # ЛОЖНОЙ причине, а писатель жёг попытки код-ревью, «чиня» бездефектный код. Урок обязан быть
    # честным: внешний OOM — повод сузить след по памяти, но НЕ повод переписывать логику.
    _ext_oom = False
    if re.search(r"out of memory", err, re.I):
        _mine = re.search(r"this process has ([\d.]+) GiB", err)
        _hog = re.search(r"Process \d+ has ([\d.]+) GiB", err)
        try:
            _ext_oom = bool(_mine and _hog and float(_hog.group(1)) > 2 * float(_mine.group(1)))
        except (TypeError, ValueError):
            _ext_oom = False
    if _ext_oom:
        logger.warning("[%s] OOM kind=%s вызван ЧУЖИМ процессом на карте (сосед %s, мы %s) — "
                       "урок-КОНТРАКТ НЕ пишу (пересев по ложной причине), нужен ре-плейсмент",
                       nid, kind, _hog.group(1) if _hog else "?", _mine.group(1) if _mine else "?")
        return
    # ФИКС ВОЛНЫ 211 — УРОК ТРЕБОВАЛ ТОГО, ЧТО СКРИПТ УЖЕ СДЕЛАЛ. Блок в.209 про БЮДЖЕТ ПАМЯТИ
    # предписывает «уменьшай микробатч с градиентным накоплением». Прогон `gnfirewall_custom_llama`
    # ровно это и выполнил — `OOM: batch->4/2/1 retry`, в RESULT_JSON `budget_reduction.micro_batch
    # {from: 8, to: 1}` — и всё равно умер: карта была на 14,7 ГБ free под масштаб 350M. Значит рычаг
    # снижения памяти ИСЧЕРПАН, и урок физически невыполним: он лишь делает файл уроков новее скрипта
    # ⇒ пересев только что одобренного (8 попыток!) кода и сожжённые раунды ревью на бездефектном
    # скрипте — ровно тот ущерб, который в.203 и в.209 уже оплатили дважды. Причина здесь ЁМКОСТЬ
    # (её чинит бюджет размещения в `_place`, тот же фикс волны), поэтому урока нет — как и при
    # внешнем OOM выше. Гейт узкий: срабатывает ТОЛЬКО при доказанном микробатче ≤1.
    if re.search(r"out of memory|OutOfMemoryError", err, re.I):
        _floor_to = None
        try:
            _mb = ((data.get("budget_reduction") or {}).get("micro_batch") or {})
            _floor_to = int(_mb.get("to")) if _mb.get("to") is not None else None
        except (TypeError, ValueError, AttributeError):
            _floor_to = None
        if _floor_to is not None and _floor_to <= 1:
            logger.warning("[%s] OOM kind=%s ПРИ МИКРОБАТЧЕ %s (скрипт сам понизил %s→%s с "
                           "накоплением) — рычаг памяти исчерпан, это ЁМКОСТЬ КАРТЫ, а не дефект "
                           "кода: урок-КОНТРАКТ НЕ пишу, нужен ре-плейсмент под объявленные "
                           "масштабы (в.211)", nid, kind, _floor_to,
                           (data.get("budget_reduction") or {}).get("micro_batch", {}).get("from"),
                           _floor_to)
            return
    tail = "\n".join(tb.splitlines()[-6:])[:900]
    path = os.path.join(cfg.workdir, ".run", "exp_review_lessons", f"{kind}.md")
    # ПОЛОВИНА ЗАЩИТЫ В.203 НЕ РАБОТАЛА: ГЕЙТ ТРЕБОВАЛ УЛИК, КОТОРЫЕ СКРИПТ ВЫБРАСЫВАЕТ (фикс в.209).
    # Краеугольный `method_llama_firewall_calibration_audit` умер `OutOfMemoryError` — и урок «сбой
    # воспроизводится ДЕТЕРМИНИРОВАННО, проверь контракт API» был всё равно записан, потому что
    # атрибуция в.203 ищет в тексте ошибки `Process N has X GiB` / `this process has Y GiB`, а в
    # RESULT_JSON приехало РОВНО `"error": "OutOfMemoryError"` с ПУСТЫМ traceback: свой же except
    # положил `type(e).__name__` вместо `str(e)`. Итог тот же, что в.203 уже оплатила: пересев рычага
    # по ложной причине + сожжённые раунды код-ревью на бездефектном коде. При ПУСТОМ traceback у нас
    # нет НИ ОДНОЙ улики дефекта кода — «детерминированность» тогда чистая спекуляция. Поэтому OOM без
    # атрибуции получает ЧЕСТНЫЙ блок про память (он же требует сохранять полный текст ошибки, чтобы
    # атрибуция в.203 в следующий раз сработала), а не «перепиши логику».
    if re.search(r"out of memory|OutOfMemoryError", err, re.I) and not tail:
        logger.warning("[%s] OOM kind=%s БЕЗ атрибуции и без traceback (error=%r) — пишу урок про "
                       "БЮДЖЕТ ПАМЯТИ, а не про дефект логики (в.209)", nid, kind, err[:80])
        # Заголовок НЕ содержит «вахтёр»/«волна» СПЕЦИАЛЬНО: это автодамп, он обязан конкурировать за
        # остаток бюджета, а не занимать недропабельный _CURATED_CAP (иначе повторные OOM вытеснят
        # настоящие ручные контракты — класс в.190).
        block = (f"\n\n[КОНТРАКТ РАНТАЙМА — прогон {nid} упал по ПАМЯТИ (атрибуция неизвестна), "
                 f"вердикт invalid]\n"
                 f"Ошибка: {err[:300]} (traceback пуст, виновник НЕ атрибутирован)\n"
                 "Это НЕ повод переписывать логику или «проверять контракт API»: карты GPU общие, и "
                 "сосед может занять память ПОСЛЕ размещения. Требования к новой версии РОВНО два, оба "
                 "машинно проверяемые:\n"
                 "1) ПОЛНЫЙ ТЕКСТ ОШИБКИ: в обработчике исключений печатай `str(e)` целиком (для CUDA "
                 "OOM там есть строки `Process N has X GiB in use` / `this process has Y GiB`) и "
                 "`traceback.format_exc()` — оба в RESULT_JSON (`error`, `traceback`). `type(e).__name__` "
                 "БЕЗ текста запрещён: без этих строк невозможно отличить чужой процесс от своего "
                 "перерасхода, и следующая версия правится наугад.\n"
                 "2) БЮДЖЕТ ПАМЯТИ: печатай `torch.cuda.max_memory_allocated()/1e9` после КАЖДОЙ фазы "
                 "(pretrain, tuning, каждая рука) в прогресс-строке `{\"row\": …}`; освобождай граф и "
                 "оптимизатор предыдущей фазы (`del`, `torch.cuda.empty_cache()`) ПЕРЕД следующей; "
                 "уважай `BATCH`/`SEQ` из env и, если пик > 0.8 от `torch.cuda.mem_get_info()` на "
                 "старте, уменьшай микробатч с градиентным накоплением, СОХРАНЯЯ эффективный батч "
                 "(менять число сидов/рук/scale-тегов и контраст — НЕЛЬЗЯ, см. установку про бюджет).\n"
                 "Научную часть, руки и статистику НЕ меняй — прежняя версия прошла код-ревью.\n")
    else:
        block = (f"\n\n[КОНТРАКТ РАНТАЙМА — прогон {nid} упал на GPU, вердикт invalid]\n"
                 f"Ошибка: {err[:300]}\n"
                 f"Хвост traceback:\n```\n{tail}\n```\n"
                 "Этот сбой воспроизводится детерминированно: скрипт прошёл код-ревью и всё равно не дал "
                 "ни одного числа, то есть попытка узла и часы GPU потрачены впустую. Новый скрипт ОБЯЗАН "
                 "не повторять эту ошибку: сначала проверь контракт API, которым пользуешься (какие поля "
                 "модель/кит реально возвращает при ТВОИХ аргументах), сделай дешёвую smoke-проверку этого "
                 "вызова на одном микробатче ДО тяжёлого цикла и падай с понятным сообщением, если контракт "
                 "не выполняется. Инфраструктурные падения на первых секундах прогона недопустимы.\n")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(block)
        logger.warning("[%s] рантайм-сбой записан в уроки kind=%s (%d симв) → скрипт пересеется",
                       nid, kind, len(block))
    except OSError as e:
        logger.warning("не смог записать рантайм-урок kind=%s: %s", kind, e)
    # Трейсбэк = скрипт разбился детерминированно (не «не повезло с GPU»): пока на диске ЭТА версия,
    # новые прогоны kind'а только сожгут попытки узлов (см. `_quarantine_script`, фикс волны 181).
    if tb:
        why = err or (tail.splitlines()[-1] if tail else "")
        _quarantine_script(cfg, kind, nid, why, script_rel)


def _project_script_src(cfg: ARConfig, script_rel: str) -> str:
    """Исходник скрипта проекта (для диагностики покрытия контраста; фикс волны 188)."""
    if not script_rel:
        return ""
    try:
        return open(_project_script(cfg, script_rel), encoding="utf-8", errors="replace").read()
    except (OSError, ValueError, FileNotFoundError):
        return ""


def _note_contrast_gap_lesson(cfg: ARConfig, kind: str, gap: List[str],
                              spec: Dict[str, Any]) -> None:
    """Дописать блок-КОНТРАКТ про НЕреализованный контраст спеки (фикс волны 188).

    Без этого несоответствие «спека объявляет руки — скрипт их не знает» жило только в логе: пересев
    получал от ревьюера ту же претензию в свободной форме («Claim не проверяется»), а точного списка
    ключей не видел никогда. Файл уроков становится новее скрипта ⇒ следующее размещение кикает
    пересев (запуск при этом НЕ блокируется, в.180). Дедуп по списку ключей — иначе блок дописывался
    бы каждый цикл и раздувал уроки."""
    if not kind or not gap:
        return
    path = os.path.join(cfg.workdir, ".run", "exp_review_lessons", f"{kind}.md")
    marker = f"[КОНТРАКТ КОНТРАСТА: {', '.join(gap)}]"
    try:
        if marker in open(path, encoding="utf-8", errors="replace").read():
            return
    except OSError:
        pass
    env_decl = json.dumps((spec or {}).get("env") or {}, ensure_ascii=False)[:600]
    block = (f"\n\n### КОНТРАКТ (несоответствие спеке) {marker}\n"
             f"Спека этого kind объявляет ENV `{env_decl}`, а действующий скрипт НЕ содержит ни "
             f"чтения ключей {', '.join(gap)}, ни их значений. Значит прогон измеряет НЕ заявленный "
             "claim: сравнения объявленных рук/методов в коде нет вообще, tuned-grid нет, "
             "compute-matched постановки нет. Именно за это код-ревью отклоняет попытки формулировкой "
             "«Claim не проверяется». ТРЕБОВАНИЕ: читать перечисленные ключи ДОСЛОВНО через "
             "os.environ, реализовать КАЖДУЮ руку из METHODS, тюнить LR каждой руки по её "
             "LR_GRID_<METHOD> на ОТДЕЛЬНОМ сплите (печатать {\"lr_select\": ...}), и выводить "
             "метрику сравнения рук в RESULT_JSON. Если бюджет не позволяет — сокращай ШАГИ/сиды, но "
             "НЕ выбрасывай руки: вердикт confirmed при отсутствующем контрасте автоматически "
             "понижается до invalid (анти-подлог), то есть прогон без рук бессмысленен.\n")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(block)
        logger.warning("[exp] kind=%s: несоответствие контраста (%s) записано в уроки → пересев",
                       kind, ", ".join(gap))
    except OSError as e:
        logger.warning("не смог записать урок о контрасте kind=%s: %s", kind, e)


_SCRIPT_SEED_REQ_RE = re.compile(
    r"len\(\s*set\(\s*([A-Za-z_]*SEEDS?[A-Za-z_]*)\s*\)\s*\)\s*>=\s*(\d+)"
    r"|len\(\s*([A-Za-z_]*SEEDS?[A-Za-z_]*)\s*\)\s*>=\s*(\d+)", re.I)


def _script_min_seeds(path: str) -> int:
    """Минимум УНИКАЛЬНЫХ сидов, которого требует САМ скрипт (`assert len(set(SEEDS))>=N`).

    Волна 178: КОНТРАКТ рецензента («>=12 независимых source blocks на масштаб») доехал до
    пересеянного скрипта в виде жёсткого assert, а env EXP_SPEC нёс 6 сидов → прогон умирал
    AssertionError через 3 с, EMP_VERDICT invalid, попытка узла сожжена, ось E мертва.
    `_enforce_claim_seeds` это не ловит: он читает claim/derive и ищет формулу «>=N seeds»,
    а контракт был сформулирован как «source blocks». Скрипт — авторитетный артефакт, поэтому
    требование берём из него. Кап 24 (мощность vs цена GPU)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            code = fh.read()
    except OSError:
        return 0
    best = 0
    for m in _SCRIPT_SEED_REQ_RE.finditer(code):
        for g in (m.group(2), m.group(4)):
            if g:
                try:
                    best = max(best, int(g))
                except ValueError:
                    pass
    return min(best, 24)


def _enforce_claim_seeds(env: Dict[str, Any], *texts: str) -> None:
    """Если claim требует >=N уникальных сидов, а env несёт меньше — поднимаем SEEDS до N
    (только ВВЕРХ). Иначе автоген-скрипт с `assert len(set(seeds))>=N` падает invalid вечно."""
    need = _claim_min_seeds(*texts)
    if not need:
        return
    have = len({s for s in str(env.get("SEEDS", "")).replace(" ", "").split(",") if s})
    if have < need:
        env["SEEDS"] = ",".join(str(i) for i in range(need))


def launch(node: Dict[str, Any], derive: str, cfg: ARConfig,
           model: str = "llama", call=None, spec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """НЕ блокирует: ставит typed EXP_SPEC на свободную карту пула и возвращается.

    The old behavior was hard-coded llama_exp_kit for every empirical claim.  Now
    a node must carry a normalized EXP_SPEC; missing specs are inferred only into
    an allowlisted kind, never into an arbitrary script.
    """
    nid, short = node["id"], node.get("short", "")
    try:
        plan = _plan(cfg, node, derive, spec)
    except ValueError as e:
        return _blocked(cfg, {"nid": nid, "short": short, "model": model}, str(e))
    env = dict(plan.get("env") or {})
    # проектные env-оверрайды (размер модели/шаги/LR-сетка) из autoresearch.json → cfg.exp_env.
    # setdefault: EXP_SPEC-env (plan) выигрывает у проекта, проект выигрывает у хардкод-дефолтов ниже.
    for _k, _v in (getattr(cfg, "exp_env", None) or {}).items():
        env.setdefault(str(_k), str(_v))
    # анти-вырождение сидов: одиночный сид (напр. агент написал SEEDS=5 = сид №5) даёт zero-width CI,
    # который строгий A*-критик не зачтёт как confirm. Нормализуем к >=3 сидам (проект-агностично).
    _sd = [s for s in str(env.get("SEEDS", "")).replace(" ", "").split(",") if s]
    if 0 < len(_sd) < 3:
        env["SEEDS"] = "0,1,2,3,4"
    # claim мог ЯВНО потребовать >=N сидов (скрипт-автоген несёт assert) → поднимаем SEEDS вверх.
    _enforce_claim_seeds(env, plan.get("claim", ""), node.get("idea", ""), derive or "")
    if plan["kind"] == "legacy_llama_forgetting":
        env.setdefault("PRE_STEPS", "300")
        env.setdefault("FT_STEPS", "150")
        env.setdefault("SEEDS", "0,1,2")
    elif plan["kind"] in {"llama_gnmuon_audit", "llama_proxy_falsification_grid",
                          "llama_fixed_baseline_audit"}:
        env.setdefault("PRE_STEPS", "240")
        env.setdefault("FT_STEPS", "120")
        env.setdefault("SEEDS", "0,1,2")
        if plan["kind"] == "llama_proxy_falsification_grid":
            env.setdefault("AUDIT_KIND", "llama_proxy_falsification_grid")
        if plan["kind"] == "llama_fixed_baseline_audit":
            env.setdefault("AUDIT_KIND", "llama_fixed_baseline_audit")
        env.setdefault("METHODS", "gnmuon,muon,adamw,full_gn")
        env.setdefault("LR_GRID_GNMUON", "0.01,0.02")
        env.setdefault("LR_GRID_MUON", "0.01,0.02")
        env.setdefault("LR_GRID_ADAMW", "0.0001,0.0003")
        env.setdefault("LR_GRID_FULL_GN", "0.005,0.01")
        env.setdefault("N_LAYER", "4")
        env.setdefault("N_HEAD", "4")
        env.setdefault("N_EMBD", "256")
        env.setdefault("SEQ", "192")
        env.setdefault("BATCH", "8")
        env.setdefault("COV_BATCHES", "8")
        env.setdefault("EVAL_BATCHES", "4")
    elif plan["kind"] in {"llama_checkpoint_pool_selector", "llama_firewall_calibration_audit",
                          "llama_alignment_bridge_audit"}:
        env.setdefault("PRE_STEPS", "240")
        env.setdefault("POOL_STEPS", "60,120,180,240")
        env.setdefault("PROBE_FT_STEPS", "24")
        env.setdefault("FT_STEPS", "120")
        env.setdefault("SEEDS", "0,1,2")
        env.setdefault("SELECTORS", "c1,loss,time,gn_proxy")
        env.setdefault("N_LAYER", "4")
        env.setdefault("N_HEAD", "4")
        env.setdefault("N_EMBD", "256")
        env.setdefault("SEQ", "192")
        env.setdefault("BATCH", "8")
        env.setdefault("EVAL_BATCHES", "4")
        if plan["kind"] == "llama_firewall_calibration_audit":
            env.setdefault("AUDIT_KIND", "llama_firewall_calibration_audit")
        if plan["kind"] == "llama_alignment_bridge_audit":
            env.setdefault("AUDIT_KIND", "llama_alignment_bridge_audit")
    rec = {"nid": nid, "short": short, "model": plan["model"], "status": "new",
           "idea": node.get("idea", ""), "env": env, "fixes": 0,
           "kind": plan["kind"], "exp_spec": plan, "prev_status": node.get("status")}
    return _place(cfg, rec)


def launch_cornerstone(node: Dict[str, Any], spec: str, cfg: ARConfig,
                       pre_steps: int = 1500, ft_steps: int = 600,
                       seeds: str = "0,1,2,3,4") -> Dict[str, Any]:
    """КРАЕУГОЛЬНЫЙ эксперимент по требованию A*-критика: тот же llama_exp_kit, но ДОЛГИЙ и с
    бОльшим числом сидов (env PRE_STEPS/FT_STEPS/SEEDS перебивают argv в kit). Один авторитетный
    прогон, вокруг которого команда координируется (мыслители отдыхают, confirm-фаза молчит)."""
    nid, short = node["id"], node.get("short", "")
    try:
        plan = _plan(cfg, node, spec, node.get("exp_spec"))
    except ValueError:
        try:
            plan = exp_spec.canonicalize(
                {"kind": exp_spec.infer_kind(spec), "claim": spec},
                context=f"node={nid} {node.get('idea','')} cornerstone={spec}",
                exp_kind=getattr(cfg, "exp_kind", ""))
        except ValueError as e:
            return _blocked(cfg, {"nid": nid, "short": short}, str(e))
    env = dict(plan.get("env") or {})
    # claim мог ЯВНО потребовать >=N сидов (скрипт-автоген несёт assert) → поднимаем SEEDS вверх
    # ДО ветвей (их setdefault("SEEDS",...) станет no-op). Иначе краеугольный вечно invalid.
    _enforce_claim_seeds(env, plan.get("claim", ""), spec or "", node.get("idea", ""))
    if plan["kind"] == "legacy_llama_forgetting":
        env.update({"PRE_STEPS": str(pre_steps), "FT_STEPS": str(ft_steps), "SEEDS": seeds})
    elif plan["kind"] in {"llama_gnmuon_audit", "llama_proxy_falsification_grid",
                          "llama_fixed_baseline_audit"}:
        env.setdefault("PRE_STEPS", str(min(pre_steps, 600)))
        env.setdefault("FT_STEPS", str(min(ft_steps, 240)))
        env.setdefault("SEEDS", seeds)
        if plan["kind"] == "llama_proxy_falsification_grid":
            env.setdefault("AUDIT_KIND", "llama_proxy_falsification_grid")
        if plan["kind"] == "llama_fixed_baseline_audit":
            env.setdefault("AUDIT_KIND", "llama_fixed_baseline_audit")
        env.setdefault("METHODS", "gnmuon,muon,adamw,full_gn")
        env.setdefault("LR_GRID_GNMUON", "0.01,0.02")
        env.setdefault("LR_GRID_MUON", "0.01,0.02")
        env.setdefault("LR_GRID_ADAMW", "0.0001,0.0003")
        env.setdefault("LR_GRID_FULL_GN", "0.005,0.01")
        env.setdefault("N_LAYER", "4")
        env.setdefault("N_HEAD", "4")
        env.setdefault("N_EMBD", "256")
        env.setdefault("SEQ", "192")
        env.setdefault("BATCH", "8")
        env.setdefault("COV_BATCHES", "8")
        env.setdefault("EVAL_BATCHES", "4")
    elif plan["kind"] in {"llama_checkpoint_pool_selector", "llama_firewall_calibration_audit",
                          "llama_alignment_bridge_audit"}:
        env.setdefault("PRE_STEPS", str(min(pre_steps, 600)))
        env.setdefault("POOL_STEPS", "120,240,360,600" if pre_steps >= 600 else "60,120,180,240")
        env.setdefault("PROBE_FT_STEPS", "32")
        env.setdefault("FT_STEPS", str(min(ft_steps, 240)))
        env.setdefault("SEEDS", seeds)
        env.setdefault("SELECTORS", "c1,loss,time,gn_proxy")
        env.setdefault("N_LAYER", "4")
        env.setdefault("N_HEAD", "4")
        env.setdefault("N_EMBD", "256")
        env.setdefault("SEQ", "192")
        env.setdefault("BATCH", "8")
        env.setdefault("EVAL_BATCHES", "4")
        if plan["kind"] == "llama_firewall_calibration_audit":
            env.setdefault("AUDIT_KIND", "llama_firewall_calibration_audit")
        if plan["kind"] == "llama_alignment_bridge_audit":
            env.setdefault("AUDIT_KIND", "llama_alignment_bridge_audit")
    else:
        env.setdefault("STEPS", str(ft_steps))
        env.setdefault("SEEDS", seeds)
        env.setdefault("FORGET_STEPS", str(ft_steps))
        env.setdefault("FORGET_SEEDS", seeds)
        env.setdefault("GATE_STEPS", str(max(24, ft_steps // 10)))
        env.setdefault("GATE_SEEDS", seeds)
        env.setdefault("MECH_STEPS", str(max(24, ft_steps // 10)))
        env.setdefault("MECH_SEEDS", seeds)
    rec = {"nid": nid, "short": short, "model": plan["model"], "status": "new",
           "idea": node.get("idea", ""), "fixes": 0, "cornerstone": True, "spec": spec[:300],
           "env": env, "kind": plan["kind"], "exp_spec": plan, "prev_status": node.get("status")}
    rec = _place(cfg, rec)
    if rec.get("status") in ("running", "waiting"):
        _card(cfg, nid, short, rec.get("status") if rec["status"] == "waiting" else "experiment",
              f"КРАЕУГОЛЬНЫЙ эксп (по требованию A*-критика, {pre_steps}+{ft_steps} шагов, {seeds.count(',')+1} сидов): "
              f"{spec[:120]}", log=True)
    return rec


# ---------------- агент-экспериментатор: мониторит свой прогон с тиком, понимает идею ----------------

_WATCH_PROMPT = """\
Ты АГЕНТ-ЭКСПЕРИМЕНТАТОР, ведёшь СВОЙ реальный GPU-эксперимент по узлу [{nid}]: {short}
Идея/что проверяем (ГИПОТЕЗА): {idea}
Typed EXP_SPEC: {spec}
Текущие гиперы (env-override): {env}
ХВОСТ ЛОГА прогона. Стандартный контракт: RESULT_JSON + EMP_VERDICT + EXP_DONE в конце.
Промежуточные строки зависят от EXP_SPEC.kind: это могут быть {{"trace":...}}, {{"row":...}},
или другие JSON-события конкретного harness-а.
{tail}

Ты НЕ болванчик: ты ПОНИМАЕШЬ гипотезу и НАУЧНО судишь, есть ли смысл жечь GPU дальше.
Реши состояние:
- идёт нормально, числа осмысленны, есть прогресс → RUNNING;
- МЕЛКАЯ чинимая проблема, НЕ меняющая суть: расходится/NaN → меньше LR; OOM → меньше BATCH;
  явно мало объёма для сигнала → больше STEPS/SEEDS. Тогда → NEEDS_FIX и дай env-пары
  (только гиперы {{LR,BATCH,STEPS,SEEDS,PRE_STEPS,FT_STEPS,FORGET_STEPS,FORGET_SEEDS,
  FORGET_BS,FORGET_SEQ,GATE_STEPS,GATE_SEEDS,GATE_SEQ,MECH_STEPS,MECH_SEEDS,MECH_SEQ}},
  НЕ логику/гипотезу);
  ВНИМАНИЕ: NEEDS_FIX = ПЕРЕЗАПУСК С НУЛЯ (прогон убивается, часы GPU теряются). Давай его
  ТОЛЬКО если требуемые значения ОТЛИЧАЮТСЯ от «Текущих гиперов» выше. Строка
  {{"oom_retry": {{"tag":..., "new_BATCH":..., ...}}}} — это скрипт САМ погасил OOM и идёт дальше
  на меньшем батче: это ПРОГРЕСС, а не сбой ⇒ RUNNING (не проси BATCH, который уже стоит);
- завис/сломан непоправимо (повторяющийся ERROR MKL/SSYEVD/CUDA, минутами ничего) → DEAD;
- прогон фактически завершён по логу → DONE;
- ГИПОТЕЗА ТОЧНО НЕ ВЫПОЛНИТСЯ или модель НЕ УЧИТСЯ → ABORT_HOPELESS (ранний стоп, экономим GPU):
    * по trace видно, что loss претрейна НЕ падает за много точек / взрывается / NaN с начала
      (модель не учится — чинить гиперы бессмысленно, это не мелкая правка), ИЛИ
    * по уже готовым {{"row"}} разрыв оптимизаторов стабильно ~0 или ОБРАТНОГО гипотезе знака,
      и оставшихся сидов заведомо не хватит, чтобы CI исключил ноль.
  ВАЖНО: ABORT_HOPELESS только когда ОЧЕВИДНО (не из-за шума пары точек). Если сомневаешься — RUNNING.
Выведи РОВНО три строки:
NOTE: <что видишь + ПОЧЕМУ такой вердикт, одна строка для дашборда>
FIX: <KEY=VAL через пробел из {{LR,BATCH,STEPS,SEEDS}}, или ->
EXP_STATE: RUNNING|NEEDS_FIX|DEAD|DONE|ABORT_HOPELESS"""


def _watch(cfg: ARConfig, rec: Dict[str, Any], tail: str) -> Tuple[str, Dict[str, str], str]:
    """codex-экспериментатор смотрит хвост лога СВОЕГО прогона и решает, что делать."""
    out = engine_call("codex", cfg)(_WATCH_PROMPT.format(
        nid=rec["nid"], short=rec.get("short", ""), idea=(rec.get("idea", "") or "")[:800],
        spec=exp_spec.dump(rec.get("exp_spec") or {"kind": rec.get("kind", "?")})[:1000],
        env=rec.get("env") or {}, tail=(tail or "(лог пуст)")[-3000:]))
    state = grep_tail(out, "EXP_STATE:").upper()
    if state not in ("RUNNING", "NEEDS_FIX", "DEAD", "DONE", "ABORT_HOPELESS"):
        state = "RUNNING"  # codex отзеркалил шаблон/мусор → дефолт RUNNING (консервативно)
    note = grep_tail(out, "NOTE:")
    # САНИТАЙЗ: codex иногда возвращает строку-плейсхолдер из промпта (<что видишь...>) → чистим
    if note.lstrip().startswith("<") or "для дашборда" in note or "<что видишь" in note:
        note = ""
    fix: Dict[str, str] = {}
    fs = grep_tail(out, "FIX:")
    if fs and fs.strip() not in ("-", ""):
        for tok in fs.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                kk = k.strip().upper()
                if (kk.startswith("LR_GRID_")
                        or kk in ("LR", "BATCH", "STEPS", "SEEDS", "METHODS", "PRE_STEPS", "FT_STEPS",
                          "FORGET_STEPS", "FORGET_SEEDS", "FORGET_BS", "FORGET_SEQ",
                          "N_LAYER", "N_HEAD", "N_EMBD", "SEQ", "COV_BATCHES", "EVAL_BATCHES",
                          "GATE_STEPS", "GATE_SEEDS", "GATE_SEQ",
                          "MECH_STEPS", "MECH_SEEDS", "MECH_SEQ")):
                    cleaned = exp_spec.clean_env({kk: v.strip()})
                    fix.update(cleaned)
    # ФИКС ВОЛНЫ 187: НО-ОП «ФИКС» УБИВАЛ ЖИВОЙ МНОГОЧАСОВОЙ ПРОГОН.
    # Автоген-скрипты САМИ гасят CUDA-OOM (печатают {"oom_retry": {"new_BATCH": 8, ...}} и
    # продолжают на меньшем батче). Watcher-codex видит эту строку как «OOM → меньше BATCH» и
    # отдаёт NEEDS_FIX с BATCH=8 — РОВНО тем значением, которое уже стоит в env. Ветка NEEDS_FIX
    # безусловно делает _kill_tmux + перезапуск, поэтому каждый такой «фикс» = минус часы GPU и
    # +1 к rec['fixes'] (у method_..._emp_124M так сгорели ДВЕ попытки: 12:29 и 14:25 MSK, оба раза
    # BATCH 8→8), а после kill'а прогон ещё и не может вернуться на карту, если свободной нет.
    # Правило: фикс имеет смысл ТОЛЬКО если хотя бы один ключ РЕАЛЬНО меняет значение.
    if state == "NEEDS_FIX" and fix:
        _cur = {str(k).upper(): str(v) for k, v in (rec.get("env") or {}).items()}
        _eff = {k: v for k, v in fix.items() if str(v) != _cur.get(str(k).upper())}
        if not _eff:
            logger.warning("[%s] NEEDS_FIX с НО-ОП фиксом %s (значения уже стоят в env) — "
                           "прогон НЕ убиваю, считаю RUNNING", rec["nid"], fix)
            return "RUNNING", {}, note
        fix = _eff
    return state, fix, note


def _cpu_seconds(rec: Dict[str, Any]) -> Optional[float]:
    """Суммарное CPU-время процессов прогона на хосте, сек (None = узнать не удалось).

    Нужно watchdog'у: лог-тишина НЕ равна зависанию — автоген-скрипты печатают только
    config и финальный RESULT_JSON (per-seed флашей нет), поэтому тяжёлый multi-seed
    прогон молчит часами, будучи живым и считая на 99 % CPU. Растущее CPU-время —
    объективное доказательство, что убивать нечего. Матчим по уникальному basename
    лога узла; bracket-трюк [x]... — чтобы grep не поймал сам себя (как в _kill_tmux).
    """
    srv = rec.get("server", "")
    base = os.path.basename(rec.get("log", "") or "")
    if base.endswith(".log"):
        base = base[:-4]
    if not srv or len(base) <= 4:
        return None
    pat = "[" + base[0] + "]" + base[1:]
    cmd = f"ps -eo times,cmd | grep '{pat}' | awk '{{s+=$1}} END {{print s+0}}'"
    try:
        out = subprocess.run(["ssh", srv, cmd],
                             capture_output=True, text=True, timeout=20).stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        return None
    try:
        return float(out)
    except ValueError:
        return None


def _kill_tmux(rec: Dict[str, Any]) -> None:
    srv = rec.get("server", "")
    try:
        subprocess.run(["ssh", srv, f"tmux kill-session -t {rec.get('tmux','')} 2>/dev/null"],
                       capture_output=True, text=True, timeout=20)
    except (subprocess.TimeoutExpired, OSError):
        pass
    # nohup-python детачится (PPID=1) и ПЕРЕЖИВАЕТ kill-session → осиротевший писатель корраптит
    # общий лог при следующем запуске (double-write → порча JSON → invalid). Добиваем его по
    # уникальному basename скрипта/лога узла. bracket-трюк [x]... чтобы pkill НЕ убил собственный
    # ssh-шелл (его cmdline содержит [x]..., а regex [x]xx ищет xxx подряд → self-match исключён).
    log = rec.get("log", "") or ""
    base = os.path.basename(log)
    if base.endswith(".log"):
        base = base[:-4]
    if srv and len(base) > 4:
        pat = "[" + base[0] + "]" + base[1:]
        try:
            subprocess.run(["ssh", srv, f"pkill -9 -f '{pat}'"],
                           capture_output=True, text=True, timeout=20)
        except (subprocess.TimeoutExpired, OSError):
            pass


# ---------------- поллинг завершения (вызывается из supervisor каждую итерацию) ----------------

def _read_verdict(cfg: ARConfig, rec: Dict[str, Any]) -> Tuple[str, str]:
    tail = subprocess.run(["ssh", rec["server"], f"tail -25 {rec['log']}"],
                          capture_output=True, text=True, errors="replace", timeout=20).stdout
    if rec.get("kind") == "c1_retention":
        for line in reversed(tail.splitlines()):
            if '"FINAL"' not in line:
                continue
            try:
                payload = json.loads(line)
                c1 = payload.get("FINAL", {}).get("C1", {})
                mean = float(c1.get("mean", 0.0))
                ok = bool(c1.get("excl0")) and mean > 0
                return ("confirmed" if ok else "no_signal",
                        json.dumps(payload, ensure_ascii=False, sort_keys=True))
            except (TypeError, ValueError):
                continue
        return "invalid", json.dumps({"error": "c1_retention finished without FINAL payload"},
                                      ensure_ascii=False)
    if rec.get("kind") == "rho_predictor":
        payload = None
        if rec.get("out"):
            try:
                txt = subprocess.run(["ssh", rec["server"], f"cat {rec['out']} 2>/dev/null"],
                                     capture_output=True, text=True, errors="replace",
                                     timeout=20).stdout
                if txt.strip():
                    payload = json.loads(txt)
            except (subprocess.TimeoutExpired, OSError, ValueError):
                payload = None
        if payload is None:
            for line in reversed(tail.splitlines()):
                if '"FINAL"' not in line:
                    continue
                try:
                    payload = json.loads(line)
                    break
                except ValueError:
                    continue
        if not payload or "FINAL" not in payload:
            return "invalid", json.dumps({"error": "rho_predictor finished without FINAL payload"},
                                         ensure_ascii=False)
        final = payload.get("FINAL") or {}
        per = final.get("per_layer") or []
        vals = [float(r.get("rho")) for r in per
                if isinstance(r, dict) and r.get("rho") is not None]
        rho_range = (max(vals) - min(vals)) if vals else 0.0
        rho_w = float(final.get("rho_grad_weighted", float("nan")))
        dim_ok = int(final.get("n_dim_ok", 0)) == int(final.get("n_matrices", -1))
        enough = int(final.get("n_layers", 0)) >= 8
        finite = math.isfinite(rho_w)
        nondeg = rho_range >= 0.02
        result = {
            "FINAL": final,
            "rho_range": round(rho_range, 6),
            "confirm_rule": (
                "confirmed iff dimensions are valid, n_layers>=8, "
                "rho_grad_weighted is finite, and rho_range>=0.02"
            ),
        }
        ok = dim_ok and enough and finite and nondeg
        return "confirmed" if ok else "no_signal", json.dumps(result, ensure_ascii=False, sort_keys=True)
    # generic: маркеры EMP_VERDICT/RESULT_JSON могут оказаться ВЫШЕ последних 25 строк (torch/варнинги
    # печатаются после результата) → ищем по ВСЕМУ логу, иначе verdict молча дефолтит в no_signal и
    # узел ложно помечается refuted. Нет маркера вовсе → invalid (сбой), а не no_signal.
    full = subprocess.run(
        ["ssh", rec["server"], f"grep -aE '^(EMP_VERDICT|RESULT_JSON):' {rec['log']}"],
        capture_output=True, text=True, errors="replace", timeout=20).stdout or tail
    verdict, rj, seen = "no_signal", "", False
    for line in full.splitlines():
        if line.startswith("EMP_VERDICT:"):
            verdict = line.split(":", 1)[1].strip(); seen = True
        elif line.startswith("RESULT_JSON:"):
            rj = line.split(":", 1)[1].strip()
    if not seen:
        return "invalid", json.dumps({"error": "no EMP_VERDICT marker in log"}, ensure_ascii=False)
    return verdict, rj


def _save_full_exp_log(cfg: ARConfig, rec: Dict[str, Any], verdict: str) -> Tuple[str, str]:
    """Тянет ПОЛНЫЙ лог эксперимента с сервера и кладёт его в проект (git) → идейные агенты видят
    ВСЕ per-seed данные, а не только итоговый вердикт. Возвращает (относит.путь-в-проекте, все
    RESULT_JSON-строки одной пачкой). Без этого multi-scale negative-таблица (per-seed rows +
    bootstrap-CI) не собиралась — сырые числа оставались в логах на серверах."""
    nid = rec.get("nid", "exp")
    try:
        full = subprocess.run(["ssh", rec["server"], f"cat {rec['log']}"],
                              capture_output=True, text=True, errors="replace", timeout=30).stdout or ""
    except (subprocess.TimeoutExpired, OSError):
        full = ""
    if not full.strip():
        return "", ""
    # сырьё для multi-scale таблицы: итоговый RESULT_JSON + ПОСТРОЧНЫЕ per-seed итоги (эксп-скрипты
    # печатают {"row": true, "seed": N, "selector":..., "source_forget":..., "target_loss":...} на
    # каждый seed×selector — это и есть per-seed rows, которых просит критик; берём их тоже).
    _rj = [l for l in full.splitlines() if l.startswith("RESULT_JSON:")]
    _rows = [l for l in full.splitlines() if '"row": true' in l.lower() or '"row":true' in l.lower()]
    per_seed = "\n".join(_rj + _rows[:60])  # до 60 per-seed rows (масштабы × seeds × selectors)
    root = os.path.abspath(cfg.project_root)
    logs_dir = os.path.join(root, "Results", "exp_logs")
    os.makedirs(logs_dir, exist_ok=True)
    kind = rec.get("kind") or (rec.get("exp_spec") or {}).get("kind") or "exp"
    fname = f"{_safe(nid)}__{_safe(str(kind))}.log"
    path = os.path.join(logs_dir, fname)
    # ограничиваем разумно (последние 4000 строк) — per-seed JSON + хвост, без гигабайтов
    lines = full.splitlines()
    body = "\n".join(lines[-4000:]) if len(lines) > 4000 else full
    header = (f"# exp log: node={nid} kind={kind} verdict={verdict} server={rec.get('server')} "
              f"gpu={rec.get('gpu')}\n# env={json.dumps(rec.get('env') or {}, ensure_ascii=False)}\n"
              f"# ВСЕ per-seed RESULT_JSON ниже + полный хвост лога\n\n")
    try:
        open(path, "w", encoding="utf-8").write(header + body)
    except OSError:
        return "", per_seed[:20000]
    return os.path.join("Results", "exp_logs", fname), per_seed[:20000]


def _exp_plan(rec: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
    """Из env эксперимента: pre/ft шагов, число сидов, opt/LR cells, всего прогонов."""
    env = rec.get("env") or {}
    def _i(k, d):
        try:
            return int(env.get(k, d))
        except (ValueError, TypeError):
            return d
    kind = rec.get("kind") or (rec.get("exp_spec") or {}).get("kind") or ""
    if kind == "rho_predictor":
        return 0, 0, 1, 1, 1
    if kind in {"llama_checkpoint_pool_selector", "llama_firewall_calibration_audit",
                "llama_alignment_bridge_audit"}:
        pre = _i("PRE_STEPS", 240)
        ft = _i("FT_STEPS", 120)
        seeds = str(env.get("SEEDS") or "0,1,2")
        selectors = str(env.get("SELECTORS") or "c1,loss,time,gn_proxy")
        n_seeds = max(1, len([s for s in seeds.split(",") if s.strip()]))
        n_selectors = max(1, len([s for s in selectors.split(",") if s.strip()]))
        return pre, ft, n_seeds, n_selectors, n_seeds * n_selectors
    pre = _i("PRE_STEPS", 0 if kind != "legacy_llama_forgetting" else 300)
    ft = _i("FT_STEPS", _i("STEPS", _i("FORGET_STEPS", _i("GATE_STEPS", _i("MECH_STEPS", 150)))))
    seeds = str(env.get("SEEDS") or env.get("FORGET_SEEDS") or env.get("GATE_SEEDS")
                or env.get("MECH_SEEDS") or "0,1,2")
    opts = str(env.get("METHODS") or env.get("OPTS") or env.get("FORGET_OPTS") or env.get("GATE_OPTS")
               or env.get("MECH_OPTS") or "sgd,muon,adam")
    n_seeds = max(1, len([s for s in seeds.split(",") if s.strip()]))
    methods = [s.strip().lower() for s in opts.split(",") if s.strip()]
    def _grid_count(method: str) -> int:
        raw = str(env.get("LR_GRID_" + method.upper()) or "")
        vals = [x for x in raw.split(",") if x.strip()]
        return max(1, len(vals))
    n_cells = sum(_grid_count(m) for m in methods) or 1
    return pre, ft, n_seeds, n_cells, n_cells * n_seeds


def _hhmm(ts: float) -> str:
    from datetime import datetime, timezone, timedelta
    return datetime.fromtimestamp(ts, timezone(timedelta(hours=3))).strftime("%H:%M")


def _timing(rec: Dict[str, Any], rows_done: int) -> str:
    """Тайминг для карточки: запуск, сколько идёт, оценка остатка/конца по доле прогонов."""
    st = rec.get("started_at")
    if not st:
        return ""
    pre, ft, n_seeds, n_cells, total = _exp_plan(rec)
    el = max(0.0, time.time() - st)
    head = f"▶ запуск {_hhmm(st)} · идёт {int(el/60)}м · {rows_done}/{total} прогонов"
    if 0 < rows_done < total:
        eta = el * total / rows_done
        head += f" · ~ещё {max(0,int((eta-el)/60))}м (конец ~{_hhmm(st+eta)})"
    elif rows_done == 0:
        head += " (прогрев)"
    return f"{head} · {pre}+{ft}ш×{n_seeds}сид×{n_cells}opt/lr"


def _exp_dashboard_detail(rec: Dict[str, Any], note: str = "", rows_done: int = 0) -> str:
    """Readable one-card summary for the generic experimenter lane."""
    spec = rec.get("exp_spec") if isinstance(rec.get("exp_spec"), dict) else {}
    env = rec.get("env") if isinstance(rec.get("env"), dict) else {}
    kind = str(rec.get("kind") or spec.get("kind") or "?")
    nid = str(rec.get("nid") or "?")
    server = str(rec.get("server") or "?")
    gpu = rec.get("gpu", "?")
    model = str(rec.get("model") or spec.get("model") or "?")
    bits = [f"GPU busy: EXP_SPEC[{kind}] {nid} на {server} cuda:{gpu} ({model})"]
    timing = _timing(rec, rows_done)
    if timing:
        bits.append(timing)
    cfg_bits = []
    if env.get("AUDIT_KIND"):
        cfg_bits.append(f"audit={env['AUDIT_KIND']}")
    pre = env.get("PRE_STEPS")
    ft = env.get("FT_STEPS") or env.get("STEPS") or env.get("FORGET_STEPS")
    if pre or ft:
        cfg_bits.append(f"steps={pre or '?'}+{ft or '?'}")
    seeds = env.get("SEEDS") or env.get("FORGET_SEEDS") or env.get("GATE_SEEDS")
    if seeds:
        cfg_bits.append(f"seeds={seeds}")
    methods = env.get("METHODS") or env.get("SELECTORS") or env.get("OPTS")
    if methods:
        cfg_bits.append(f"methods={methods}")
    size = []
    for key in ("N_LAYER", "N_HEAD", "N_EMBD", "SEQ", "BATCH"):
        if env.get(key) is not None:
            size.append(f"{key}={env[key]}")
    if size:
        cfg_bits.append("model_cfg=" + ",".join(size))
    grids = []
    for key in sorted(k for k in env if k.startswith("LR_GRID_")):
        grids.append(f"{key[8:].lower()}={env[key]}")
    if grids:
        cfg_bits.append("lr_grid=" + ";".join(grids[:5]))
    if cfg_bits:
        bits.append("config: " + " · ".join(cfg_bits))
    claim = str(spec.get("claim") or rec.get("idea") or rec.get("spec") or "").strip()
    if claim:
        bits.append("claim: " + claim[:260])
    confirm = str(spec.get("confirm_rule") or spec.get("success_metric") or "").strip()
    if confirm:
        bits.append("confirm: " + confirm[:260])
    if note:
        bits.append("monitor: " + note[:260])
    if rec.get("log"):
        bits.append(f"log={rec['log']}")
    return " | ".join(bits)[:1800]


def poll(cfg: ARConfig) -> List[Dict[str, Any]]:
    """Проверить все прогоны: waiting → разместить; running → не готов ли. Вернуть завершённые."""
    finished: List[Dict[str, Any]] = []
    for rec in _records(cfg):
        nid, short = rec["nid"], rec.get("short", "")
        st = rec.get("status")
        if st == "waiting":
            # ФИКС ВОЛНЫ 208 (изоляция записей): ветка `running` ниже свои ssh-сбои глотает, а
            # размещение — НЕТ, поэтому исключение в `_place` ОДНОЙ записи прерывало обход ВСЕХ
            # остальных (вызывающая сторона ловит уже снаружи цикла: «poll экспериментов упал»).
            # Одна запись не имеет права стоить цикла опроса остальным.
            try:
                _place(cfg, rec)  # появилась карта — поставит, иначе снова waiting
            except Exception as _pl_e:                 # noqa: BLE001 — изоляция записи, причина в лог
                logger.warning("[%s] размещение не выполнено в этой итерации: %s: %s",
                               nid, type(_pl_e).__name__, _pl_e)
            continue
        if st == "running":
            try:
                c = subprocess.run(["ssh", rec["server"], f"test -f {rec['done']} && echo DONE || echo WAIT"],
                                   capture_output=True, text=True, timeout=20).stdout
                tail = subprocess.run(["ssh", rec["server"], f"tail -40 {rec['log']}"],
                                      capture_output=True, text=True, errors="replace", timeout=20).stdout
            except (subprocess.TimeoutExpired, OSError):
                continue

            def _finalize(verdict_raw: str, rjv: str):
                norm = {"confirmed": "confirmed", "survives_control": "confirmed"}.get(
                    verdict_raw, verdict_raw or "no_signal")
                # OOM ≠ ОТРИЦАТЕЛЬНЫЙ РЕЗУЛЬТАТ. Автоген-скрипты при нехватке VRAM снижают BATCH и,
                # исчерпав лестницу, честно печатают oom:true по всем слайсам, но вердикт при этом
                # выходит `no_signal` → узел получает ФАЛЬШИВЫЙ негатив в граф и сжигает попытку, а
                # в статью попадает «no confirmed direction», хотя измерения не было вовсе.
                # Все слайсы в OOM → это инфра-сбой (invalid), а не свидетельство.
                if norm == "no_signal":
                    try:
                        _sc = (json.loads(rjv) or {}).get("scales") or []
                        if _sc and all(bool(s.get("oom")) for s in _sc):
                            norm = "invalid"
                    except (TypeError, ValueError, AttributeError):
                        pass
                # АНТИ-ПОДЛОГ ВЕРДИКТА (фикс волны 188). Скрипт может честно посчитать СВОЮ метрику и
                # напечатать `confirmed`, хотя контраст, заявленный спекой (руки METHODS, tuned
                # LR_GRID_*), в коде отсутствует — тогда «confirmed» относится к другому утверждению,
                # чем claim узла. Такой вердикт в графе хуже отсутствия результата: он закрывает узел,
                # тянет за собой statement в статью и разбивается о A*-рецензию («Claim не
                # проверяется»). Понижаем ТОЛЬКО путь подлога: no_signal/invalid идут как есть.
                if norm == "confirmed" and rec.get("_contrast_gap"):
                    _gapl = ", ".join(str(g) for g in (rec.get("_contrast_gap") or []))
                    norm = "invalid"
                    logger.warning("[%s] confirmed ПОНИЖЕН до invalid: скрипт не реализует контраст "
                                   "спеки (%s) — вердикт относился бы не к claim'у узла", nid, _gapl)
                    try:
                        _d = json.loads(rjv or "{}")
                        if isinstance(_d, dict):
                            _d["contrast_gap"] = rec.get("_contrast_gap")
                            _d["error"] = ("вердикт confirmed отклонён: скрипт не реализует "
                                           f"объявленный спекой контраст ({_gapl})")
                            rjv = json.dumps(_d, ensure_ascii=False, sort_keys=True)
                    except (TypeError, ValueError):
                        pass
                ok = norm == "confirmed"
                payload = rjv
                try:
                    payload = json.dumps({"exp_spec": rec.get("exp_spec"), "result": json.loads(rjv)},
                                         ensure_ascii=False, sort_keys=True)
                except (TypeError, ValueError):
                    payload = json.dumps({"exp_spec": rec.get("exp_spec"), "result_text": rjv},
                                         ensure_ascii=False, sort_keys=True)
                _card(cfg, nid, short, "passed" if ok else "failed",
                      f"эксперимент завершён: {verdict_raw}. {payload[:120]}", log=True)
                _bump(cfg, "confirmed" if ok else ("invalid" if norm == "invalid" else "no_signal"))
                if norm == "invalid":
                    _lesson_from_runtime_failure(
                        cfg, rec.get("kind") or "", nid, rjv,
                        str((rec.get("exp_spec") or {}).get("script") or ""))
                # ПОЛНЫЙ лог + ВСЕ per-seed RESULT_JSON в проект (git) → идейным агентам и reporter
                # доступны сырые числа (для multi-scale negative-таблицы). Делаем для ЛЮБОГО вердикта.
                log_file, per_seed = _save_full_exp_log(cfg, rec, norm)
                _del_rec(cfg, nid)
                finished.append({"nid": nid, "verdict": norm, "rj": payload,
                                 "prev_status": rec.get("prev_status"),
                                 "exp_spec": rec.get("exp_spec"),
                                 "exp_log_file": log_file, "per_seed": per_seed})

            if "DONE" in c:  # прогон штатно завершился → читаем вердикт и возвращаем идейному
                v, rj = _read_verdict(cfg, rec)
                _finalize(v, rj)
                continue
            # прогон ещё идёт → АГЕНТ-ЭКСПЕРИМЕНТАТОР смотрит лог, понимает идею, решает
            state, fix, note = _watch(cfg, rec, tail)
            if state in ("DEAD", "ABORT_HOPELESS"):
                _kill_tmux(rec)
                reason = ("прогон мёртв" if state == "DEAD"
                          else "ранний стоп: модель не учится / гипотеза не выполнится")
                _card(cfg, nid, short, "failed", f"экспериментатор: {reason} — {note}", log=True)
                _bump(cfg, "no_signal")
                # сохраняем ПОЛНЫЙ лог + per-seed и для no_signal (негатив — сырьё для multi-scale таблицы)
                log_file, per_seed = _save_full_exp_log(cfg, rec, "no_signal")
                _del_rec(cfg, nid)
                finished.append({"nid": nid, "verdict": "no_signal", "rj": f"{reason}: {note}",
                                 "prev_status": rec.get("prev_status"),
                                 "exp_spec": rec.get("exp_spec"),
                                 "exp_log_file": log_file, "per_seed": per_seed})
            elif state == "NEEDS_FIX" and fix and int(rec.get("fixes", 0)) < 3:
                _kill_tmux(rec)
                rec["env"] = {**(rec.get("env") or {}), **fix}
                rec["fixes"] = int(rec.get("fixes", 0)) + 1
                rec["status"] = "new"; rec["_counted"] = True  # перезапуск с новыми гиперами, не новый ran
                _card(cfg, nid, short, "experiment",
                      f"экспериментатор правит гиперы {fix} (#{rec['fixes']}) и перезапускает: {note}", log=True)
                _place(cfg, rec)
            elif state == "DONE":  # лог говорит «готово», done-файла нет → вердикт из лога
                v, rj = _read_verdict(cfg, rec)
                _finalize(v, rj)
            else:  # RUNNING — идёт нормально, экспериментатор отдаёт что видит на дашборд
                # ФИКС ВОЛНЫ 191: ПРЕДРЕШЁННО-НЕВАЛИДНЫЙ ПРОГОН НЕ ДОЖИВАЕТ ДО КОНЦА.
                # Анти-подлог в.188 понижает `confirmed`→`invalid`, если запущенная версия скрипта не
                # реализует контраст спеки. Пока это обнаруживалось ТОЛЬКО в финале, поэтому прогон,
                # чей вердикт уже предрешён, честно молол 10-12 ч, держал единственную большую карту
                # пула (44 ГБ) и не давал стартовать ни одному прогону этого же kind'а — включая тот,
                # который пересев только что сделал корректным. Снимаем такой прогон РОВНО тогда,
                # когда на диске уже лежит одобренная версия БЕЗ strict-разрыва контраста: обмен
                # заведомо-invalid результата на немедленный корректный запуск.
                if rec.get("_contrast_gap"):
                    try:
                        _sp2 = rec.get("exp_spec") or {}
                        _rel2 = str(_sp2.get("script") or "")
                        _fixed = bool(_rel2) and not _contrast_gap(
                            _sp2, _project_script_src(cfg, _rel2), strict=True)
                    except Exception as _pe:      # диагностика не должна ронять живой прогон
                        logger.info("[%s] проверка предрешённого invalid не выполнена: %s", nid, _pe)
                        _fixed = False
                    if _fixed:
                        _kill_tmux(rec)
                        _why = ("снят досрочно: запущенная версия скрипта не реализовывала контраст "
                                f"спеки ({', '.join(str(g) for g in rec.get('_contrast_gap') or [])}) "
                                "⇒ вердикт был предрешён invalid, а на диске уже есть одобренная "
                                "версия с контрастом — узел перезапустится на ней")
                        _card(cfg, nid, short, "failed", f"pre-empt: {_why}", log=True)
                        logger.warning("[%s] %s", nid, _why)
                        _bump(cfg, "invalid"); _del_rec(cfg, nid)
                        finished.append({"nid": nid, "verdict": "invalid", "rj": _why,
                                         "prev_status": rec.get("prev_status"),
                                         "exp_spec": rec.get("exp_spec")})
                        continue
                # WATCHDOG: _watch-LLM при неуверенности дефолтит в RUNNING → зависший прогон мог бы
                # висеть experiment_running ВЕЧНО (терминальный статус, узел не перевыбирается). Ловим
                # детерминированно: лог не растёт >WATCHDOG_STALE_S или wall-clock >WATCHDOG_WALL_S.
                now = time.time()
                try:
                    sz = int(subprocess.run(["ssh", rec["server"], f"wc -c < {rec['log']}"],
                                            capture_output=True, text=True, timeout=20).stdout.strip() or 0)
                except (subprocess.TimeoutExpired, OSError, ValueError):
                    sz = int(rec.get("_log_size", 0) or 0)
                if sz > int(rec.get("_log_size", -1) or -1):
                    rec["_log_size"] = sz; rec["_grow_ts"] = now; _save_rec(cfg, rec)
                grow_ts = float(rec.get("_grow_ts") or rec.get("started_at") or now)
                started0 = float(rec.get("started_at") or now)
                stale = (now - grow_ts) > WATCHDOG_STALE_S
                over_wall = (now - started0) > WATCHDOG_WALL_S
                if stale and not over_wall:
                    # Прежде чем убить прогон (это -N часов GPU-работы И invalid_attempt узлу),
                    # проверяем РЕАЛЬНУЮ живость: у этого семейства скриптов между config и
                    # финальным RESULT_JSON нет НИ ОДНОГО принта, так что мёртвый лог у живого
                    # 5-часового прогона — норма, а не зависание. Критерий = растёт ли CPU-время.
                    cpu = _cpu_seconds(rec)
                    prev_cpu = rec.get("_cpu_time")
                    alive = cpu is not None and (
                        cpu > float(prev_cpu) + 1.0 if prev_cpu is not None else cpu > 0.0)
                    if alive:
                        rec["_cpu_time"] = cpu; rec["_cpu_ts"] = now
                        rec["_grow_ts"] = now  # процесс демонстрируемо считает → новое окно STALE_S
                        _save_rec(cfg, rec)
                        logger.info("watchdog: лог тих %.1fч, но CPU-время растёт (%.0f с) → прогон жив, не убиваю: %s",
                                    (now - grow_ts) / 3600.0, cpu, nid)
                        stale = False
                if stale or over_wall:
                    _kill_tmux(rec)
                    why = (f"превышен wall-clock {WATCHDOG_WALL_S//3600}ч" if over_wall
                           else f"лог не растёт >{WATCHDOG_STALE_S//60}м и CPU-время не движется")
                    _card(cfg, nid, short, "failed", f"watchdog: {why} → invalid", log=True)
                    _bump(cfg, "invalid"); _del_rec(cfg, nid)
                    finished.append({"nid": nid, "verdict": "invalid", "rj": f"watchdog: {why}",
                                     "prev_status": rec.get("prev_status"), "exp_spec": rec.get("exp_spec")})
                    continue
                try:  # доля завершённых прогонов (строк "row") → оценка времени
                    rd = subprocess.run(["ssh", rec["server"], f"grep -c '\"row\"' {rec['log']} 2>/dev/null"],
                                        capture_output=True, text=True, timeout=20).stdout.strip()
                    rows_done = int(rd) if rd.isdigit() else 0
                except (subprocess.TimeoutExpired, OSError, ValueError):
                    rows_done = 0
                detail = _exp_dashboard_detail(rec, note or "идёт", rows_done)
                _card(cfg, nid, short, "experiment", detail)
                try:
                    agents_status.update(
                        cfg, "agent-experimenter", node="EXP_SPEC", short="typed experiment lane",
                        stage="experiment",
                        detail=detail,
                        engine="codex→GPU",
                        role="экспериментатор (typed EXP_SPEC)",
                    )
                    agents_status.write_snapshot(cfg)
                except Exception:
                    pass
    return finished


# --- ФИКС ВОЛНЫ 209 -----------------------------------------------------------------
# КЛАСС: «текст, дописанный в шаблон .format(), молча ломает ВЫЗОВ, а не только текст».
# Волна 208 дописала ревьюеру строку с литеральной JSON-формой `{"sha256": …}` внутри
# _EXP_REVIEW_PROMPT — а он рендерится через .format(), поэтому КАЖДЫЙ вызов код-ревью падал
# `KeyError: '"sha256"'` в daemon-потоке пересева. В логе это выглядело как одна загадочная
# строка «фоновый пересев kind=… упал: '"sha256"'», ни один скрипт не мог быть одобрен
# НИ ДЛЯ ОДНОГО kind'а (шаблон общий) => ось E закрыта наглухо при живом лупе и 0 LOST в
# инвентаре. Диагностика по строке отказа невозможна: ключ ошибки — это кусок ЧУЖОГО текста.
# Поэтому шаблоны проверяются НА ИМПОРТЕ: пробный рендер с фиктивными значениями превращает
# скрытую опечатку в громкую строку с ИМЕНЕМ шаблона и полем-виновником.
# Правило вахтёра: дописал в любой *_PROMPT фигурную скобку — она обязана быть удвоена ({{ }}),
# и проверяется это не глазами, а строкой инвентаря в.209.
_FORMATTED_PROMPTS: Dict[str, Tuple[str, ...]] = {
    "_EXP_REVIEW_PROMPT": ("claim", "success_metric", "confirm_rule", "env", "code"),
    "_EXP_SCRIPT_PROMPT": ("kind", "domain", "claim", "success_metric", "confirm_rule"),
    "_EXP_PROMPT": ("nid", "short", "idea", "derive", "domain"),
    "_WATCH_PROMPT": ("nid", "short", "idea", "spec", "env", "tail"),
}


def selfcheck_prompt_templates() -> List[str]:
    """Пробный рендер всех .format()-шаблонов модуля. Возвращает список поломок (пусто = норма)."""
    import string as _string
    problems: List[str] = []
    for name, fields in _FORMATTED_PROMPTS.items():
        tpl = globals().get(name)
        if not isinstance(tpl, str):
            problems.append(f"{name}: шаблон отсутствует")
            continue
        try:
            seen = {f for _, f, _, _ in _string.Formatter().parse(tpl) if f}
        except Exception as exc:  # несбалансированные скобки
            problems.append(f"{name}: шаблон не парсится ({exc})")
            continue
        extra = sorted(f for f in seen if f not in fields)
        if extra:
            problems.append(
                f"{name}: литеральная фигурная скобка в тексте не удвоена — поля {extra} "
                f"(правь текст на {{{{ }}}}, иначе .format() падает KeyError)")
            continue
        try:
            tpl.format(**{f: "" for f in fields})
        except Exception as exc:
            problems.append(f"{name}: пробный рендер упал {type(exc).__name__}: {exc}")
    return problems


for _p in selfcheck_prompt_templates():
    logger.error("⚠️ ШАБЛОН ПРОМПТА СЛОМАН (в.209): %s", _p)
