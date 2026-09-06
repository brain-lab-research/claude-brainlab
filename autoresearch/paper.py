"""Главный результат: A*-gated финальная СТАТЬЯ + отчёт по успешным кейсам.

Отличие от reporter.py (тот = широкий ОБЗОР, уровень канбана): здесь — то, что читает
пользователь как РЕЗУЛЬТАТ. Когда сюжет набрал высокий A*-балл у жёсткого критика, Opus
пишет полную связную статью (стиль paper/transition-law-ru.tex), Opus-критик (роль
NeurIPS/ICLR area-chair, no-mercy) ставит балл 1-10 и ревьюит готовый текст; ревайз↔критик
до порога; только при высоком балле статья «публикуется» (компилируется как финал).
Все агенты — Opus. honesty>hype.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import ARConfig
from .engines import engine_call
from . import exp_spec, graph_hygiene, graph_io as gio, engines
from .reporter import compile_pdf, _reports_dir, _UNI, _clause

logger = logging.getLogger(__name__)

_WINS = ("proven", "confirmed_emp")
_SURVIVED = ("proven", "confirmed_emp", "conditional")


def clean_latex(s: str) -> str:
    """Снять emoji/несовместимый unicode, СОХРАНИВ LaTeX (ascii) и кириллицу."""
    for u, r in _UNI.items():
        s = s.replace(u, r)
    return "".join(c for c in s if ord(c) < 0x0250 or 0x0400 <= ord(c) <= 0x04FF
                   or c in "\n\t")


def _branch_of(nid: str, by: Dict[str, Dict[str, Any]]) -> Optional[str]:
    for a in gio.ancestors(nid, by):
        if a.get("col") == 1:
            return a["id"]
    return nid if by.get(nid, {}).get("col") == 1 else None


def _edge_summary(edge: Any) -> str:
    if not isinstance(edge, dict):
        return str(edge)
    keys = ("mean", "CI", "excl0", "sign_stable", "values")
    return ", ".join(f"{k}={edge[k]}" for k in keys if k in edge)


def _exp_protocol_summary(n: Dict[str, Any]) -> str:
    spec = n.get("exp_spec") if isinstance(n.get("exp_spec"), dict) else {}
    if not spec:
        return ""
    bits = []
    for k in ("kind", "script", "model", "success_metric", "confirm_rule", "launch_command"):
        if spec.get(k):
            bits.append(f"{k}: {spec[k]}")
    env = spec.get("env") if isinstance(spec.get("env"), dict) else {}
    if env:
        preferred = (
            "AUDIT_KIND", "PRE_STEPS", "POOL_STEPS", "PROBE_FT_STEPS",
            "FT_STEPS", "SEEDS", "METHODS", "SELECTORS",
            "LR_GRID_GNMUON", "LR_GRID_MUON", "LR_GRID_ADAMW",
            "LR_GRID_FULL_GN", "N_LAYER", "N_HEAD", "N_EMBD",
            "SEQ", "BATCH", "COV_BATCHES", "EVAL_BATCHES",
        )
        env_bits = [f"{k}={env[k]}" for k in preferred if env.get(k) is not None]
        extra = [
            f"{k}={v}" for k, v in sorted(env.items())
            if k not in preferred and v is not None
        ]
        bits.append("env: " + ", ".join((env_bits + extra)[:24]))
    markers = spec.get("required_markers")
    if markers:
        bits.append(f"required_markers: {markers}")
    return " | ".join(bits)


def _refuted_summary(n: Dict[str, Any], limit: int = 2400) -> str:
    protocol = _exp_protocol_summary(n)
    compact = n.get("exp_refuted_summary")
    if compact:
        text = str(compact)
        if protocol:
            text = f"{text} | protocol: {protocol}"
        return text[:limit]
    # raw RESULT_JSON (полный payload с per-seed rows/CI) даёт богатый провенанс для appendix (W1);
    # exp_refuted хранит лишь summary → предпочитаем exp_result_json, если он есть.
    payload = n.get("exp_result_json") or n.get("exp_refuted")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            text = payload
            if protocol:
                text = f"{text} | protocol: {protocol}"
            return text[:limit]
    if not isinstance(payload, dict):
        return str(payload or "")[:limit]
    bits = []
    for k in ("kind", "model", "claim", "success_metric"):
        if payload.get(k):
            bits.append(f"{k}: {payload[k]}")
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    final = result.get("FINAL") if isinstance(result.get("FINAL"), dict) else result
    verdict = result.get("verdict") or final.get("verdict") or result.get("pass")
    if verdict is not None:
        bits.append(f"verdict/pass: {verdict}")
    matched = final.get("matched_target")
    if isinstance(matched, dict):
        if matched.get("core_cross_rate"):
            bits.append(f"matched_target.core_cross_rate={matched['core_cross_rate']}")
        if matched.get("target_tau") is not None:
            bits.append(f"matched_target.target_tau={matched['target_tau']}")
        if matched.get("edge") is not None:
            bits.append(f"matched_target.edge: {_edge_summary(matched['edge'])}")
    edges = final.get("edges") if isinstance(final.get("edges"), dict) else {}
    for key in ("target_edge", "wallclock_edge", "retention_edge"):
        edge = final.get(key) or edges.get(key)
        if edge is not None:
            bits.append(f"{key}: {_edge_summary(edge)}")
    for key in ("primary_confirmed", "P_delta", "target_noninferior",
                "target_noninferiority_eps"):
        if final.get(key) is not None:
            bits.append(f"{key}={final[key]}")
    for key in ("Delta_C1", "target_edge_neg_gap"):
        edge = final.get(key)
        if edge is not None:
            bits.append(f"{key}: {_edge_summary(edge)}")
    if final.get("limitations"):
        bits.append(f"limitations: {final['limitations']}")
    if protocol:
        bits.append(f"protocol: {protocol}")
    if not bits:
        bits.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return " | ".join(bits)[:limit]


def _refuted_exp_kinds(data: Dict[str, Any], cfg: ARConfig,
                       headline_branch: Optional[str] = None) -> Dict[str, List[str]]:
    by = gio.index(data)
    out: Dict[str, List[str]] = {}
    for n in data.get("nodes", []):
        spec = n.get("exp_spec") if isinstance(n.get("exp_spec"), dict) else {}
        kind = str(spec.get("kind") or "").strip()
        if not kind or not n.get("exp_refuted"):
            continue
        if n.get("col") in (0, None):
            continue
        if not graph_hygiene.report_eligible(n):
            continue
        if not graph_hygiene.project_policy_eligible(n, cfg):
            continue
        if headline_branch and not (
            n.get("id") == headline_branch
            or _branch_of(n["id"], by) == headline_branch
            or any(a.get("id") == headline_branch for a in gio.ancestors(n["id"], by))
            or n.get("parent") == headline_branch
        ):
            continue
        out.setdefault(kind, []).append(str(n.get("id")))
    return out


def _training_code_commit(cfg: ARConfig) -> str:
    """РЕАЛЬНЫЙ commit-хэш кода экспериментов (`Results/scripts`).

    A*-рецензент гейтит ось R на поле `training_git_commit` в submission-registry, а папка проекта
    лежит в Obsidian/Syncthing-вольте и НЕ под git → писатель честно писал `null` («unavailable»),
    из-за чего R не могла стать pass НИКОГДА (структурный потолок балла). Держим git-dir ВНЕ вольта
    (`~/.autoresearch_training_code/<project>.git`, work-tree = `Results/scripts`, поэтому в вольт
    ничего не пишется и Syncthing не видит .git), коммитим текущее состояние скриптов и отдаём хэш.
    Значение проверяемо: `git --git-dir=<gd> --work-tree=<wt> show --stat <sha>`.
    Любая ошибка → "" (контекст просто без этой строки, писатель снова честно скажет unavailable).
    """
    try:
        import subprocess
        wt = os.path.join(os.path.abspath(cfg.project_root), "Results", "scripts")
        if not os.path.isdir(wt):
            return ""
        name = re.sub(r"[^a-zA-Z0-9_.-]", "_", os.path.basename(os.path.abspath(cfg.project_root)))
        gd = os.path.expanduser(os.path.join("~/.autoresearch_training_code", name + ".git"))
        env = dict(os.environ, GIT_DIR=gd, GIT_WORK_TREE=wt,
                   GIT_AUTHOR_NAME="autoresearch", GIT_AUTHOR_EMAIL="autoresearch@local",
                   GIT_COMMITTER_NAME="autoresearch", GIT_COMMITTER_EMAIL="autoresearch@local")

        def g(*a):
            return subprocess.run(["git", *a], env=env, capture_output=True, text=True, timeout=90)

        if not os.path.isdir(gd):
            os.makedirs(os.path.dirname(gd), exist_ok=True)
            g("init", "-q")
        g("add", "-A")
        g("commit", "-q", "-m", "training code snapshot")  # no-op, если нечего коммитить
        r = g("rev-parse", "HEAD")
        sha = (r.stdout or "").strip()
        return sha if r.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", sha) else ""
    except Exception:
        return ""


def _finalize_submission_registry(cfg: ARConfig, tex: str, score: int) -> Optional[str]:
    """Один submission registry, привязанный К ФАКТИЧЕСКИ ПРЕДЪЯВЛЯЕМОМУ манускрипту (ось R).

    W4 рецензии 07:54 MSK: «текущий manuscript `_wip__paper_wip_i4fo950x.tex` с SHA256 … не имеет
    matching submission-registry entry; полный environment лежит в `submission_registry_rwsnb6j3.json`,
    привязанном к ДРУГОМУ manuscript path/hash» ⇒ R=partial. Причина структурная, не небрежность
    составителя: registry строится ВНУТРИ одного раунда и хэширует тогдашний артефакт
    `_wip__paper_wip_<suffix>.tex`, после чего идут раунды правок (новый артефакт, новый хэш), а
    ratchet вообще может отправить в отчёт СОХРАНЁННЫЙ лучший драфт — то есть текст, которого нет ни
    в одном wip-файле. Хэш в registry отстаёт ВСЕГДА.

    Поэтому финальный registry выпускает сам луп, после того как текст отчёта окончательно выбран:
    фиксируем предъявляемые байты в неизменяемый `provenance/manuscript_submitted.tex`, считаем по
    НИМ sha256/длину и переносим остальные поля из последнего registry составителя (ledger, artifacts,
    audit-команды). Ничего не выдумываем: environment/fingerprint — из `_env_provenance`, commit — из
    `_training_code_commit`, `containerized=false`, `container_digest=null`. Путь стабилен
    (`provenance/submission_registry_final.json`), поэтому статья может ссылаться на него заранее.
    Любая ошибка — молча без registry (отчёт важнее)."""
    import hashlib
    try:
        prov = os.path.join(cfg.workdir, "provenance")
        os.makedirs(prov, exist_ok=True)
        prev: Dict[str, Any] = {}
        prev_rel = ""
        cands: List[Tuple[float, str]] = []
        for root, _dirs, files in os.walk(prov):
            for fn in files:
                if fn.startswith("submission_registry") and fn.endswith(".json") \
                        and fn != "submission_registry_final.json":
                    p = os.path.join(root, fn)
                    try:
                        cands.append((os.path.getmtime(p), p))
                    except OSError:
                        pass
        if cands:
            p = max(cands)[1]
            try:
                with open(p, encoding="utf-8") as f:
                    prev = json.load(f) or {}
                prev_rel = os.path.relpath(p, cfg.workdir)
            except (OSError, ValueError):
                prev, prev_rel = {}, ""
        body = tex if isinstance(tex, str) else ""
        man_abs = os.path.join(prov, "manuscript_submitted.tex")
        with open(man_abs, "w", encoding="utf-8") as f:
            f.write(body)
        raw = body.encode("utf-8")
        reg = dict(prev) if isinstance(prev, dict) else {}
        reg.update({
            "registry_version": 6,
            "registry_role": "final submission registry for the manuscript actually reported "
                             "in this cycle (emitted by the pipeline after the last revision "
                             "and after the best-draft ratchet, so the hash cannot lag)",
            "manuscript_path": "provenance/manuscript_submitted.tex",
            "manuscript_sha256": hashlib.sha256(raw).hexdigest(),
            "manuscript_characters": len(body),
            "manuscript_bytes": len(raw),
            "manuscript_verification_command":
                "python3 -c \"import hashlib;print(hashlib.sha256("
                "open('provenance/manuscript_submitted.tex','rb').read()).hexdigest())\"",
            "reported_score": score,
            "containerized": False,
            "container_digest": None,
            "container_note": "runs are not containerized; environment identified by the "
                              "fingerprint below",
        })
        # ФИКС ВОЛНЫ 188 (ось R была структурно закрыта ВТОРЫМ путём). Поля рабочей копии
        # (`working_manuscript_path`/`working_copy_sha256`) наследовались из registry СОСТАВИТЕЛЯ и
        # указывали на мутабельный `_wip__paper_wip_*.tex`, а его же verify-скрипт требовал
        # побайтового равенства с предъявляемым манускриптом. Файл переписывается следующей ревизией
        # (наблюдение: registry 12:25:21, тот же wip перезаписан 12:31:00, 30233 vs 26051 байт) ⇒
        # `python3 provenance/verify_submission_final.py` падал на `AssertionError` НАВСЕГДА, то есть
        # по чеклисту рецензента R не мог стать pass ни при каком качестве статьи. Теперь рабочая
        # копия — тоже НЕИЗМЕНЯЕМЫЙ файл в `provenance/`, побайтово равный предъявляемому.
        _wc_rel = "provenance/manuscript_working_copy.tex"
        _wc_abs = os.path.join(prov, "manuscript_working_copy.tex")
        _inherited = str(reg.get("working_manuscript_path") or "")
        try:
            _stale_wc = (not _inherited) or os.path.basename(_inherited).startswith("_wip_") or (
                open(os.path.join(cfg.workdir, _inherited), "rb").read() != raw
                if os.path.exists(os.path.join(cfg.workdir, _inherited)) else True)
        except OSError:
            _stale_wc = True
        if _stale_wc:
            with open(_wc_abs, "w", encoding="utf-8") as f:
                f.write(body)
            reg["working_manuscript_path"] = _wc_rel
            reg["working_copy_sha256"] = reg["manuscript_sha256"]
            reg["working_copy_note"] = (
                "immutable byte-identical copy of the submitted manuscript; the writer's transient "
                f"`{_inherited or 'n/a'}` is overwritten by later revisions and must not be "
                "referenced by any verifier")
            logger.info("[R] рабочая копия манускрипта переведена на неизменяемый %s "
                        "(было: %s)", _wc_rel, _inherited or "нет")
        if prev_rel:
            reg["derived_from_round_registry"] = prev_rel
        env = _env_provenance(cfg) or {}
        if env.get("servers"):
            reg["environment"] = env["servers"]
            reg["environment_fingerprint"] = env.get("environment_fingerprint")
        commit = _training_code_commit(cfg)
        if commit:
            reg["training_git_commit"] = commit
        _launch = _recorded_launch_commands(cfg)
        if _launch and not str(reg.get("training_command") or "").strip():
            reg["training_command"] = _launch
            reg["training_command_status"] = (
                "exact launch commands as issued by the loop for the runs recorded below "
                "(env + interpreter + argv); historical runs launched before the loop began "
                "persisting commands remain unreconstructed and are marked as such")
            logger.info("[R] training_command заполнен фактическими командами запуска (%d)",
                        len(_launch))
        out = os.path.join(prov, "submission_registry_final.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(reg, f, ensure_ascii=False, indent=1)
        logger.info("submission registry финального манускрипта: %s (sha256 %s…, %d байт)",
                    os.path.relpath(out, cfg.workdir), reg["manuscript_sha256"][:12], len(raw))
        _run_submission_verifier(cfg)
        return out
    except (OSError, ValueError, TypeError) as e:
        logger.warning("финальный submission registry не выпущен: %s", e)
        return None


def _recorded_launch_commands(cfg: ARConfig, limit: int = 12) -> Dict[str, str]:
    """ФАКТИЧЕСКИЕ команды запуска прогонов (ось R, фикс волны 189).

    W4 рецензии: «Exact training launch command отсутствует и честно указан как unreconstructed» —
    одна из четырёх причин R=partial. Команду собирает `empirical._place`, и до этого фикса она
    жила только в cmdline процесса на GPU-сервере: registry писал `training_command=null`, а
    составитель честно писал «command absent» в колонке protocol-таблицы. Теперь берём её из
    записей живых прогонов и из `exp_spec` узлов графа (туда она доезжает с вердиктом).
    Ничего не реконструируем: если команды нет — ключ не появляется."""
    out: Dict[str, str] = {}
    try:
        rdir = os.path.join(cfg.workdir, ".run", "experiments")
        for fn in sorted(os.listdir(rdir)) if os.path.isdir(rdir) else []:
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(rdir, fn), encoding="utf-8") as f:
                    rec = json.load(f) or {}
            except (OSError, ValueError):
                continue
            c = str(rec.get("launch_command") or "").strip()
            if c and rec.get("nid"):
                out[str(rec["nid"])] = c
    except OSError as e:
        logger.info("[R] команды запуска из записей не прочитаны: %s", e)
    try:
        from . import graph_io as _gio
        data = _gio.load_graph(cfg) or {}
        _ns = data.get("nodes") or []
        if isinstance(_ns, dict):
            _ns = list(_ns.values())
        for n in _ns:
            if not isinstance(n, dict):
                continue
            spec = n.get("exp_spec") if isinstance(n.get("exp_spec"), dict) else {}
            c = str((spec or {}).get("launch_command") or "").strip()
            if c and n.get("id"):
                out.setdefault(str(n["id"]), c)
    except Exception as e:                                     # граф не читается — не ломаем registry
        logger.info("[R] команды запуска из графа не прочитаны: %s", e)
    return dict(sorted(out.items())[:limit])


def _run_submission_verifier(cfg: ARConfig) -> str:
    """Прогнать end-to-end верификатор провенанса и ЗАПИСАТЬ вердикт в лог (фикс волны 188).

    До этого верификатор, который составитель пишет сам, никто никогда не запускал: он молча падал
    месяцами (`AssertionError` на побайтовом равенстве с мутабельным `_wip_*`), а рецензент честно
    держал ось R на partial. Теперь каждый отчётный цикл печатает `[R] verify … PASS/FAIL` с текстом
    падения — это и видно вахтёру, и попадает в контекст следующего составителя через лог."""
    import subprocess
    prov = os.path.join(cfg.workdir, "provenance")
    cands = sorted(
        (fn for fn in os.listdir(prov) if fn.startswith("verify_submission") and fn.endswith(".py")),
        key=lambda fn: os.path.getmtime(os.path.join(prov, fn)), reverse=True) \
        if os.path.isdir(prov) else []
    if not cands:
        logger.info("[R] verify_submission*.py отсутствует — верификатор не запускался")
        return ""
    rel = os.path.join("provenance", cands[0])
    try:
        r = subprocess.run(["python3", rel], cwd=cfg.workdir, capture_output=True, text=True,
                           timeout=120)
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning("[R] verify %s не запустился: %s", rel, e)
        return ""
    if r.returncode == 0:
        logger.info("[R] verify %s PASS: %s", rel, (r.stdout or "").strip()[-200:])
        return "PASS"
    tail = ((r.stderr or "") + (r.stdout or "")).strip().splitlines()
    why = " | ".join(tail[-3:])[:300]
    logger.warning("[R] verify %s FAIL (rc=%s): %s — ось R не может стать pass, пока верификатор "
                   "падает; ссылки на мутабельные `_wip_*` запрещены контрактом составителя",
                   rel, r.returncode, why)
    return "FAIL"


def _env_provenance(cfg: ARConfig) -> Dict[str, Any]:
    """ОТПЕЧАТОК СРЕДЫ ПРОГОНОВ (ось R, фикс волны 175).

    Рецензент держит R=partial с формулировкой «submission registry/container отсутствуют»: в
    registry нет ни версий интерпретатора/библиотек, ни идентификатора среды, по которому прогон
    можно воспроизвести. Контейнера у проекта НЕТ (прогоны идут в conda-env на GPU-серверах), и
    выдумывать `container_digest` нельзя — это integrity-нарушение. Поэтому собираем ЧЕСТНЫЙ
    отпечаток: python/torch/CUDA/драйвер/GPU каждого сервера пула + sha256 по этому JSON как
    `environment_fingerprint`. Кэш в `.run/env_provenance.json` (обновляем раз в сутки), любая
    ошибка ssh → просто нет блока (писатель честно пишет unavailable, как раньше).
    """
    import hashlib
    import json as _json
    import shlex
    import subprocess
    import time
    cache = os.path.join(cfg.workdir, ".run", "env_provenance.json")
    try:
        if os.path.exists(cache) and (time.time() - os.path.getmtime(cache)) < 86400:
            return _json.load(open(cache, encoding="utf-8"))
    except (OSError, ValueError):
        pass
    probe = ("import json,platform,sys;d={'python':platform.python_version(),"
             "'executable':sys.executable};\n"
             "try:\n import torch;d['torch']=torch.__version__;d['cuda']=torch.version.cuda;"
             "d['gpu']=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None\n"
             "except Exception as e: d['torch']='unavailable: %s' % e\n"
             "print(json.dumps(d))")
    out: Dict[str, Any] = {"containerized": False,
                           "note": "прогоны идут не в контейнере, а в conda-env на GPU-серверах; "
                                   "идентификатор среды = environment_fingerprint ниже",
                           "servers": {}}
    for srv in (getattr(cfg, "gpu_servers", None) or [getattr(cfg, "gpu_server", "")]):
        if not srv:
            continue
        py = cfg.gpu_python_map.get(srv, cfg.gpu_python)
        try:
            r = subprocess.run(["ssh", srv, f"{py} -c {shlex.quote(probe)}; nvidia-smi "
                                            "--query-gpu=driver_version --format=csv,noheader | head -1"],
                               capture_output=True, text=True, timeout=90)
            lines = [l for l in (r.stdout or "").strip().split("\n") if l.strip()]
            rec: Dict[str, Any] = {"python_path": py}
            for l in lines:
                if l.lstrip().startswith("{"):
                    try:
                        rec.update(_json.loads(l))
                    except ValueError:
                        pass
                elif re.fullmatch(r"[\d.]+", l.strip()):
                    rec["nvidia_driver"] = l.strip()
            if len(rec) > 1:
                out["servers"][srv] = rec
        except (subprocess.TimeoutExpired, OSError):
            continue
    if not out["servers"]:
        return {}
    out["environment_fingerprint"] = "sha256:" + hashlib.sha256(
        _json.dumps(out["servers"], sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    try:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w", encoding="utf-8") as f:
            _json.dump(out, f, ensure_ascii=False, indent=1)
    except OSError:
        pass
    return out


def cluster_context(data: Dict[str, Any], cfg: ARConfig, branch: Optional[str] = None,
                    statuses: Tuple[str, ...] = _SURVIVED, headline_branch: Optional[str] = None) -> str:
    by = gio.index(data)
    nodes = [n for n in data["nodes"]
             if n.get("status") in statuses and n.get("col") not in (0, None)
             and graph_hygiene.report_eligible(n)
             and graph_hygiene.project_policy_eligible(n, cfg)]
    # raw confirmed эмпирика = узлы с РЕАЛЬНЫМ typed confirmed-экспериментом. Включаем и
    # writeup_exhausted: эксп confirmed (RESULT_JSON есть), но derive-writeup не прошёл придирку
    # контракт-гейта за 2 попытки → by design упал в conditional. Результат РЕАЛЕН → отчёт обязан
    # его учесть как эмпирику (иначе confirmed-эксп теряется, E остаётся partial, потолок 6).
    raw_confirmed = [n for n in data["nodes"]
                     if (graph_hygiene.is_typed_writeup(n) or graph_hygiene.writeup_exhausted(n))
                     and n.get("col") not in (0, None)
                     and graph_hygiene.report_eligible(n)
                     and graph_hygiene.project_policy_eligible(n, cfg)]
    refuted = [n for n in data["nodes"]
               if n.get("exp_refuted") and n.get("exp_spec") and n.get("col") not in (0, None)
               and graph_hygiene.report_eligible(n)
               and graph_hygiene.project_policy_eligible(n, cfg)]
    if branch:
        nodes = [n for n in nodes if _branch_of(n["id"], by) == branch]
        raw_confirmed = [n for n in raw_confirmed if _branch_of(n["id"], by) == branch]
        refuted = [n for n in refuted if _branch_of(n["id"], by) == branch]
    elif headline_branch:  # режим B: ветка headline ЦЕЛИКОМ + confirmed_emp/proven из ЛЮБЫХ веток
        def _in_headline_lineage(n: Dict[str, Any]) -> bool:
            return (
                n.get("id") == headline_branch
                or _branch_of(n["id"], by) == headline_branch
                or any(a.get("id") == headline_branch for a in gio.ancestors(n["id"], by))
                or n.get("parent") == headline_branch
            )
        nodes = [n for n in nodes if _in_headline_lineage(n)
                 or graph_hygiene.is_headline_win(n)]
        raw_confirmed = [n for n in raw_confirmed if _in_headline_lineage(n)
                         or graph_hygiene.is_headline_win(n)]
        refuted = [n for n in refuted if _in_headline_lineage(n)]
    # confirmed_emp/proven — ВПЕРЁД и отдельным блоком: это headline-материал (подтверждено на сети).
    # Иначе сильный результат (напр. GN-Muon) тонет среди десятков условных и писатель берёт слабую идею.
    def _fmt(n):
        exp = n.get("exp_confirmed_summary") or n.get("exp_confirmed")
        exp_line = f"\n    typed experiment: {str(exp)[:1200]}" if exp and n.get("exp_spec") else ""
        # ПРОВЕНАНС для осей E/R: критик требует цепочку «число → RESULT_JSON/лог → script/env» —
        # без путей к артефактам E=fail (числа «не прослеживаются»), хотя логи реально лежат на диске.
        logs = n.get("exp_log_files") or []
        prov = ""
        if logs and n.get("exp_spec"):
            kind = str((n.get("exp_spec") or {}).get("kind", ""))
            env = (n.get("exp_spec") or {}).get("env") or {}
            env_s = ", ".join(f"{k}={v}" for k, v in list(env.items())[:8])
            # SHA256 артефактов: критик (директива A*) требует хэши в провенанс-таблице, а писатель
            # сам их посчитать не может — считаем при сборке контекста (файлы локальны, дёшево).
            def _h(rel):
                try:
                    import hashlib
                    p = rel if os.path.isabs(str(rel)) else os.path.join(cfg.project_root, str(rel))
                    return f"{rel} (sha256:{hashlib.sha256(open(p,'rb').read()).hexdigest()[:12]})"
                except OSError:
                    return str(rel)
            script = str((n.get("exp_spec") or {}).get("script", "") or "")
            prov = (f"\n    provenance: kind={kind}; script={script or '(типовой kind-скрипт)'}; env: {env_s}; "
                    f"артефакты (пути+sha256 для манифеста): {', '.join(_h(p) for p in logs[-3:])}")
        return (f"- [{n['id']}, {n.get('status')}] {n.get('short','')}\n"
                f"    идея: {n.get('idea','')}\n    вердикт: {graph_hygiene.safe_verdict(n)}"
                f"{exp_line}{prov}")
    strong = [n for n in nodes if graph_hygiene.is_headline_win(n)]
    rest = [n for n in nodes if not graph_hygiene.is_headline_win(n)]
    L = [f"КОРНЕВАЯ ИДЕЯ: {data.get('root_idea','')}",
         f"ОБЩИЙ ВЕРДИКТ: {data.get('overall_verdict','')}"]
    # ПРОВЕНАНС КОДА ОБУЧЕНИЯ (ось R): реальный, проверяемый commit скриптов экспериментов.
    # Без него registry вечно писал training_git_commit=null → R не могла стать pass.
    _tc = _training_code_commit(cfg)
    if _tc:
        _tcwt = os.path.join(os.path.abspath(cfg.project_root), "Results", "scripts")
        _tcname = re.sub(r"[^a-zA-Z0-9_.-]", "_", os.path.basename(os.path.abspath(cfg.project_root)))
        _tcgd = os.path.expanduser(os.path.join("~/.autoresearch_training_code", _tcname + ".git"))
        L += ["", "TRAINING CODE PROVENANCE (РЕАЛЬНЫЙ, проверяемый — используй ИМЕННО это значение в "
              f"submission-registry как training_git_commit, НЕ null): {_tc}",
              f"  (git-dir={_tcgd}, work-tree={_tcwt}; проверка: "
              f"git --git-dir={_tcgd} --work-tree={_tcwt} show --stat {_tc[:12]})"]
    # ОТПЕЧАТОК СРЕДЫ (ось R, фикс волны 175): рецензент требует container/environment provenance.
    # Контейнера нет — даём ЧЕСТНЫЙ отпечаток версий вместо выдуманного digest'а.
    _envp = _env_provenance(cfg)
    if _envp:
        L += ["", "ENVIRONMENT PROVENANCE (реальные версии среды прогонов — вставь ЭТИ значения в "
              "submission-registry полем `environment` и `environment_fingerprint`; контейнера нет, "
              "поэтому `container_digest` пиши null и рядом честно: «runs are not containerized; "
              "environment identified by the fingerprint below», НЕ выдумывай digest):",
              json.dumps(_envp, ensure_ascii=False)[:1200]]
    if refuted:
        L += ["", "BINDING REFUTED EXP_SPEC KINDS (hard constraint for A*-critic):"]
        for kind, ids in _refuted_exp_kinds(data, cfg, headline_branch).items():
            L.append(f"- kind={kind} already_refuted_by={', '.join(ids[:6])}; "
                     "do not request this kind again as CORNERSTONE")
        L += ["", "ОТРИЦАТЕЛЬНЫЕ TYPED ЭКСПЕРИМЕНТЫ (binding constraints — НЕ требуй их как missing; "
              "используй для честного сужения claims и limitations):"]
        for n in refuted:
            L.append(f"- [{n['id']}, exp_refuted] {n.get('short','')}\n"
                     f"    результат: {_refuted_summary(n)}")
    # per-seed для multi-scale scale-up узлов из ВСЕГО графа (НЕ только refuted — scale-up узлы обычно
    # conditional/open). Вне блока refuted, чтобы работало всегда. Макс 6×700 симв (защита от раздувания
    # контекста → writer-codex давится → writer_failed).
    _ms = [n for n in data.get("nodes", []) if n.get("exp_per_seed")
           and ("_emp_" in n.get("id", "") or "scale" in n.get("id", "").lower())][:6]
    if _ms:
        L += ["", "PER-SEED ДЛЯ MULTI-SCALE NEGATIVE-ТАБЛИЦЫ (реальные числа + bootstrap-CI; собери ОДНУ "
              "таблицу масштаб×результат×CI×контроли — это критик требует для E=pass; НЕ выдумывай числа):"]
        for n in _ms:
            L.append(f"- {n['id']}: {str(n.get('exp_per_seed'))[:700]}")
    L += ["", "ПОДТВЕРЖДЕНО НА РЕАЛЬНОЙ СЕТИ (headline-материал — построй статью ВОКРУГ СИЛЬНЕЙШЕГО из них, "
          "особенно если критик указал пивот; это твой actionable-вывод):"]
    L += [_fmt(n) for n in strong] or ["  (пока нет подтверждённых на сети)"]
    if raw_confirmed:
        L += ["", "СЫРЫЕ CONFIRMED TYPED ЭКСПЕРИМЕНТЫ (ещё ждут post-exp writeup; "
              "можно учитывать как реальные числа, но нельзя изображать как прошедший полный proof/code gate):"]
        L += [_fmt(n) for n in raw_confirmed]
    # RELATED WORK для оси N: related_work узлов (библиотекарь litreview) раньше НЕ доходил до
    # писателя → секции Related Work в статье не было в принципе и ось N не могла стать pass.
    # Блоки headline/strong-узлов первыми, суммарный кап 5000 (весь контекст режется [:35000] ниже).
    rw_seen, rw_lines, rw_total = set(), [], 0
    for n in (strong + rest):
        rw = str(n.get("related_work") or "").strip()
        if not rw or n["id"] in rw_seen:
            continue
        rw_seen.add(n["id"])
        chunk = f"- [{n['id']}]: {rw[:1200]}"
        if rw_total + len(chunk) > 5000:
            break
        rw_lines.append(chunk); rw_total += len(chunk)
    if rw_lines:
        L += ["", "RELATED WORK (собрано библиотекарем по узлам; напиши по нему секцию Related Work — "
              "позиционируй вклад ПРОТИВ этих работ: что уже известно, чем наш результат отличается. "
              "Ссылайся текстом (название/авторы/arXiv id), БЕЗ выдуманных bibtex-ключей и без \\cite):"]
        L += rw_lines
    L += ["", "ПОДДЕРЖКА (условное/теоретическое — НЕ несущая конструкция, бери только если усиливает headline):"]
    L += [_fmt(n) for n in rest] or ["  (нет)"]
    out = "\n".join(L)
    # ПРЕДОХРАНИТЕЛЬ: большой контекст → writer-codex давится, выдаёт обрывок (writer_failed).
    # base_draft(30k)+ctx(35k)+обвязка ≈ 70k промпт — терпимо для codex. headline/per-seed идут ВПЕРЁД.
    return out[:35000]


# ---------------- Opus писатель статьи ----------------

_WRITER_PROMPT = """\
Ты ПИШЕШЬ ФИНАЛЬНУЮ НАУЧНУЮ СТАТЬЮ (главный результат проекта): {domain}
ЦЕЛЬ — НЕ свалка фактов, а A*-статья вокруг ОДНОЙ интересной НОВОЙ идеи с ПРАКТИЧЕСКОЙ
пользой. Читатель должен понять В ЧЁМ СМЫСЛ и КАК ЭТО ПРИМЕНИТЬ. Обязательно дай хотя бы один
ACTIONABLE-вывод уровня: когда остановить предобучение; какой оптимизатор даёт лучшие финальные
веса для дообучения; какой чекпоинт брать для finetune, чтобы меньше забывать.
ЖЁСТКИЕ ПРАВИЛА A*:
- КАЖДЫЙ символ определён при ПЕРВОМ вхождении (никаких C, G, Σ без расшифровки).
- Одна сильная связная история, не россыпь лемм; бери из материала ТОЛЬКО то, что её усиливает,
  слабое/нерелевантное НЕ включай (качество > количество, мало страниц — ок).
- honesty>hype: негативы/границы прямо, без overclaim; 0 выдуманных цитат и чисел.
- Если в материале есть блок "ОТРИЦАТЕЛЬНЫЕ TYPED ЭКСПЕРИМЕНТЫ", это обязательные факты:
  явно упомяни их в Экспериментах или Ограничениях с ключевыми числами. Не пиши, что такой
  audit "missing"; пиши, что он уже выполнен и сузил или опроверг широкий claim.
- Такой audit нельзя оставлять как непрозрачную "typed cell": в Экспериментах обязательно
  дай compact protocol/baseline table по полям из `protocol/env`: kind/script/model, \(K\)
  или FT_STEPS, PRE/POOL steps, seeds, methods/selectors, LR grids, model size/class
  (N_LAYER/N_HEAD/N_EMBD/SEQ/BATCH), bootstrap unit/CI rule, baselines/controls и
  required markers. Если таблица не помещается, вынеси её в appendix-style paragraph,
  но не опускай воспроизводимые параметры.
- ПРОВЕНАНС-МАНИФЕСТ (обязателен для E/R): для КАЖДОГО headline-числа/CI дай прослеживаемую
  цепочку — kind/script, env-параметры и ПУТЬ к логу/RESULT_JSON (бери из строк `provenance:` в
  материале). Оформи компактным абзацем-манифестом в Экспериментах или appendix-style в конце:
  «число X: kind=…, env=…, лог=Results/exp_logs/…». Число, для которого пути НЕТ в материале, —
  либо в Ограничения с честной пометкой «без immutable-артефакта», либо вон из main text.
- ФИНАЛЬНЫЙ SUBMISSION REGISTRY ВЫПУСКАЕТ САМ ПАЙПЛАЙН в `provenance/submission_registry_final.json`
  (он фиксирует предъявляемый текст в `provenance/manuscript_submitted.tex` и хэширует ИМЕННО его,
  уже ПОСЛЕ последней правки и ratchet, плюс несёт environment/fingerprint/training commit/
  `containerized=false`/`container_digest=null`). В прозе про воспроизводимость ссылайся именно на
  этот путь как на submission registry текущей статьи и на `manuscript_verification_command` из него;
  свой per-round registry можешь строить как приложение, но НЕ выдавай его за финальный и НЕ пиши,
  что финальный отсутствует.
- ЕСЛИ строишь submission_registry / numeric_ledger / verify_submission (end-to-end верификатор):
  САМЫМ ПОСЛЕДНИМ ДЕЙСТВИЕМ скопируй финальный текст статьи ПОБАЙТОВО в
  `provenance/manuscript_working_copy.tex` и ставь `manuscript_path`/`working_manuscript_path`
  ИМЕННО на этот путь; sha256/длину считай по НЕМУ. **ЗАПРЕЩЕНО ссылаться в registry и в
  verify-скрипте на рабочий файл `_wip_*.tex` (по пути, по имени в assert или по его байтам):** этот
  файл перезаписывается следующей ревизией/следующим циклом, поэтому любой assert вида
  `wip.read_bytes() == submitted.read_bytes()` или `registry["working_manuscript_path"] ==
  "_wip__paper_wip_XXX.tex"` становится ЛОЖНЫМ через минуты — верификатор падает НАВСЕГДА и ось R
  структурно не может стать pass (наблюдалось: registry 12:25, тот же wip перезаписан в 12:31,
  `AssertionError` на третьем assert). Все пути верификатора должны указывать на НЕИЗМЕНЯЕМЫЕ файлы
  внутри `provenance/`. sha256 пересчитай в самом конце, иначе hash отстанет от текста. Пиши в файл ТОЛЬКО чистый LaTeX (первый
  символ `\\begin{{abstract}}`), без комментариев вокруг — предъявляемый текст должен побайтово
  совпадать с захэшированным. Все пути в registry — относительно рабочей папки, без dangling-ссылок;
  мысленно прогони свой верификатор по каждому assert, он ОБЯЗАН пройти.
- GATE-МЕТКИ В ТАБЛИЦАХ строго по сохранённому `confirm_rule`: если правило записано на среднем
  (mean ≤ eps) — считай СРЕДНЕЕ, а не границу CI, и наоборот. Подмена статистики = integrity-нарушение
  и автоматический потолок балла. Не смешивай pre-finetune matching-gap с post-finetune target
  non-inferiority: это РАЗНЫЕ величины, называй каждую своим именем, а непроверенную честно помечай
  «не проверено».
- Математика настоящим LaTeX; без тире-эм и точек-с-запятой в прозе; без \\textbf/\\emph.
- ДЛИНА: целевой объём статьи 5-6 СТРАНИЦ (~18000-20000 знаков LaTeX-body, жёсткий потолок 25000).
  Спланируй так, чтобы статья ПОЛНОСТЬЮ уместилась в этот объём и ЗАВЕРШИЛАСЬ (Заключение + все секции
  доведены до конца). Полнота > детализация; НИКОГДА не обрывай на полуслове. Пишешь в файл — лимита
  вывода нет, но держи 5-6 стр.: компактная завершённая статья лучше длинной с водой.
Структура \\section{{}}: Введение (мотивация + практический вопрос); Related Work (по блоку
RELATED WORK из материала: что уже известно и чем наш вклад отличается, 1-2 абзаца; ссылки
текстом — название/авторы/arXiv id, БЕЗ выдуманных \\cite-ключей; если блока нет — короткий
честный абзац позиционирования); Теория (формулировки + наброски); Эксперименты (которые
РЕАЛЬНО подтверждают практический вывод); Ограничения; Заключение (что даёт на практике).
Перед Введением — \\begin{{abstract}}...\\end{{abstract}}.
ГЛАВНОЕ ДЛЯ A*: если в материале есть узлы со статусом confirmed_emp (подтверждены РЕАЛЬНЫМ
LLM-экспериментом на сети, с bootstrap-CI исключающим ноль) — построй ACTIONABLE-вывод и всю
историю ВОКРУГ НИХ, приведи их реальные числа+CI как ГЛАВНОЕ доказательство практической
пользы (matched-step, не matched-loss). НЕ прячь подтверждённый результат среди недоказанных
лемм и НЕ подавай вывод, который сам же дезавуируешь («эффект мал / лишь как baseline») —
такой actionable критик не примет.

МАТЕРИАЛ (выжившие результаты + binding typed refutations; бери ТОЛЬКО релевантное идее):
{context}
{feedback}
ПИВОТ (ОБЯЗАТЕЛЬНО): если в критике/директивах сказано сменить headline на конкретный подтверждённый
результат (напр. «ПИВОТ headline на X») — СДЕЛАЙ это: именно X в \\begin{{abstract}} и Введении как
ГЛАВНЫЙ actionable-вывод, вокруг него вся история. Идею, которую критик назвал слабой/non-actionable/
опровергнутой, НЕ выноси в headline — оставь как поддержку или в Ограничения. Не повторяй прицел,
который критик уже забраковал.
ОБЯЗАТЕЛЬНЫЙ ЧЕК-ЛИСТ ПЕРЕД ВЫВОДОМ (дешёвые оси C, R, T, N — всё, что закрывается ТЕКСТОМ, без нового
эксперимента; закрывай ВСЕГДА, даже если критик параллельно требует дорогой новый прогон: балл считается
по ЧИСЛУ partial-осей, поэтому текстовые оси решают его сами по себе):
- C (полнота символов): КАЖДЫЙ практический объект определён при первом употреблении СО ВСЕМИ
  зависимостями. Если объект зависит от кандидата/оптимизатора/числа шагов/итогового состояния —
  впиши это в определение (не пиши его функцией только начального состояния). Где это существенно,
  явно различай точный population-знак (directional bit) и одну случайную (sampled) метку. Проектор
  определяй как проектор на ИМЕНОВАННОЕ подпространство (укажи, на какое именно), не «проектор матрицы».
  Пропиши независимость/связь (coupling) величин, если критик на это указал.
- R (МАШИННО ПРОВЕРЯЕМЫЙ ПРОТОКОЛ): протокол всех прогонов — РОВНО ОДНА registry-backed таблица
  (не три), и в тексте ОБЯЗАНЫ дословно встречаться: `CI-unit` (или «bootstrap unit»), число
  бутстрап-репликаций (`replicates`/`resamples`), `RNG`, `sha256`, `confirm rule`, `wall-clock`,
  `LR grid`. Колонки таблицы: kind/model, PRE/POOL/PFT/FT steps, seeds-or-task-IDs, LR-grid,
  CI-unit, replicates, RNG, controls, command, artifact sha256, confirm rule. Команду бери из поля
  `launch_command:` строки `protocol:` в материале (это ФАКТИЧЕСКАЯ команда запуска, а не
  реконструкция) — если она там есть, «command absent» писать ЗАПРЕЩЕНО; помечай «не сохранена»
  ТОЛЬКО те прогоны, у которых `launch_command` действительно нет. Эти требования проверяются
  автоматически перед отправкой критику, и непройденное вернётся тебе на адресный раунд правки.
- R (провенанс headline-чисел): НИ ОДНОГО headline-числа без прослеживаемого пути к логу/RESULT_JSON из
  материала. Число, для которого такого пути НЕТ (или immutable-артефакт утрачен), НЕ оставляй в
  main text/headline — перенеси в Ограничения с честной пометкой «без immutable-артефакта» либо убери.
  Лучше меньше чисел в main text, но все прослеживаемы. Не выдумывай hashes/команды, которых нет.
- T (условия теорем): у КАЖДОЙ леммы/теоремы в ФОРМУЛИРОВКЕ стоят условия, которые доказательство
  реально использует (интегрируемость, конечные вторые моменты, ограниченность, измеримость,
  независимость). Если в доказательстве берётся ожидание квадрата, делится на дисперсию или
  выбирается «достаточно малый параметр» — соответствующее предположение ОБЯЗАНО быть выписано, а не
  подразумеваться. Слова `matched`/`tight`/`optimal` допустимы ТОЛЬКО когда доказана пара
  lower+upper с совпадающими константами/скоростями на ОДНОМ явно заданном классе; иначе просто
  УБЕРИ слово (снятие необоснованного эпитета — законная мгновенная правка, результат не теряется).
- N (новизна): явное сравнение «ближайшая известная теорема vs наша» (таблица или абзац) с точным
  указанием, ЧТО именно ново (конструкция / точное значение / объединение гарантий) и чем это
  отличается от классических инструментов, из которых собрано доказательство. Честно локализовать
  новизну лучше, чем умолчать: умолчание критик читает как её отсутствие.
Прогони этот чек-лист по СВОЕМУ выводу ПЕРЕД тем как его вернуть; не откладывай C/R ради headline-пивота.
=== ФОРМАТ ВЫВОДА (СТРОГО) ===
Выведи ТОЛЬКО готовый LaTeX-body статьи ЦЕЛИКОМ, начиная с \\begin{{abstract}} и заканчивая
последним \\section. Преамбула T2A/babel-russian добавится снаружи.
ЗАПРЕЩЕНО: любой текст вне LaTeX — НЕ пиши «вот что я изменил», «ошибка в …», «краткое резюме
правок», markdown-заборы ```; НЕ комментируй правки. Выведи всю статью ЦЕЛИКОМ как единый
документ (полный LaTeX, НЕ diff); если выше дан ТЕКУЩИЙ ЧЕРНОВИК — доработай ЕГО инкрементально,
НЕ сочиняй с нуля и НЕ теряй уже готовые части. ОБЯЗАТЕЛЬНО заверши все секции без обрыва.
Первый символ ответа = \\begin{{abstract}}."""


_R_TABLE_REQS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("единица бутстрапа (CI-unit)", ("ci-unit", "ci unit", "bootstrap unit", "bootstrap-unit",
                                     "единица бутстрапа")),
    ("число бутстрап-репликаций", ("replicat", "resample", "ресэмпл", "репликац")),
    ("RNG/seed бутстрапа", ("rng", "random-number-generator")),
    ("sha256 артефактов", ("sha256", "sha-256")),
    ("правило подтверждения (confirm rule)", ("confirm rule", "confirm_rule",
                                              "правило подтверждения")),
    ("per-arm wall-clock", ("wall-clock", "wall clock", "per_arm_time_s")),
    ("LR-сетка", ("lr grid", "lr-grid", "lr_grid", "lr сетка", "lr-сетка")),
)
_R_TABLE_COLS = ("kind", "model", "seed", "lr", "ci", "command", "sha", "confirm", "step")


def _r_protocol_violations(tex: str, context: str = "") -> List[str]:
    """ДЕТЕРМИНИРОВАННАЯ проверка оси R по тексту статьи (фикс волны 189).

    Балл стоит на 6 циклами, потому что из четырёх partial-осей ДВЕ закрываются ЧИСТО ТЕКСТОМ, и
    R — самая близкая: W4 рецензии просит РОВНО одну registry-backed protocol-таблицу с колонками
    kind/model/PRE/POOL/PFT/FT/seeds/LR-grid/CI-unit/replicates/RNG/controls/command/artifact-hash/
    confirm-rule и требует убрать числа без immutable-артефакта. Составитель директиву читает, но
    выполняет частично (наблюдение волны 189: `Набросок доказательства` он убрал по W2, а протокол
    так и остался ТРЕМЯ таблицами без CI-unit/hash-колонок и со строкой «command absent»).
    Мягкий судья такое пропускает — значит нужен машинный чек, как `_mech_contract_violations`
    для скриптов (волна 185). НЕ блокирует цикл: даёт составителю ОДИН адресный раунд правки."""
    t = (tex or "").lower()
    ctx = (context or "").lower()
    out: List[str] = []
    for name, variants in _R_TABLE_REQS:
        if not any(v in t for v in variants):
            out.append(f"в protocol-таблице нет {name}")
    tabs = re.findall(r"\\begin\{tabular\}(.{0,4000}?)\\end\{tabular\}", t, re.DOTALL)
    if not tabs:
        out.append("нет ни одной protocol-таблицы (tabular)")
    elif max(sum(1 for c in _R_TABLE_COLS if c in tb) for tb in tabs) < 5:
        out.append("протокол размазан по нескольким таблицам: НИ ОДНА не содержит хотя бы 5 колонок "
                   "из kind/model/steps/seeds/LR/CI/command/sha/confirm — слей их в ОДНУ "
                   "registry-backed protocol-таблицу")
    if "launch_command" in ctx or "команда запуска" in ctx:
        # «command absent» у ИСТОРИЧЕСКИХ прогонов — честно и остаётся; флажим только когда в тексте
        # НЕТ ни одной фактической команды, хотя материал их даёт.
        if not any(v in t for v in ("launch command", "launch_command", "команда запуска",
                                    "ssh ")):
            out.append("точные команды запуска ЕСТЬ в материале (`launch_command:` в protocol), но в "
                       "тексте их нет — впиши их в колонку command (пометку «не сохранена» оставь "
                       "ТОЛЬКО для прогонов, у которых команды в материале нет)")
    return out


def _strip_fences(s: str) -> str:
    """Снять markdown-заборы ```latex ... ``` если составитель обернул в них."""
    m = re.search(r"```(?:latex|tex)?\s*\n(.*?)```", s, re.DOTALL)
    return m.group(1) if m else s


def _extract_body(out: str) -> str:
    out = _strip_fences(out)
    i = out.find(r"\begin{abstract}")
    if i < 0:
        i = out.find(r"\section")
    return clean_latex(out[i:].strip() if i >= 0 else out.strip())


def _looks_like_paper(tex: str) -> bool:
    """Это реальная статья, а не болтовня/правки/обрывок/ЭХО ПРОМПТА? (иначе критику слать нельзя)."""
    if len(tex) < 1500:
        return False
    # PROMPT-ECHO GUARD: codex иногда копирует ТЕКСТ ЗАДАНИЯ вместо статьи (`\begin{abstract}...` +
    # инструкции промпта). Такой «драфт» сохранялся как best_draft и травил все отчёты. Отсекаем по
    # харрактерным фразам промпта writer'а.
    echo_markers = ("ГЛАВНОЕ ДЛЯ A*", "ACTIONABLE-вывод и всю", "построй ACTIONABLE",
                    "\\begin{abstract}...", "первый символ =", "не прячь подтверждённый")
    if any(m in tex for m in echo_markers):
        return False
    has_abstract = r"\begin{abstract}" in tex and r"\end{abstract}" in tex and "abstract}..." not in tex
    n_sections = tex.count(r"\section")
    return has_abstract or n_sections >= 2


def write_paper(context: str, cfg: ARConfig, call: Callable[[str], str], feedback: str = "",
                base_draft: str = "") -> str:
    """Составитель пишет/ДОРАБАТЫВАЕТ статью; при болтовне вместо LaTeX — один строгий ретрай.
    base_draft: текущий лучший черновик — если дан, писатель ИНКРЕМЕНТАЛЬНО правит ЕГО (не с нуля),
    что убирает обрывы и накапливает прогресс между циклами. Возвращает '' если не выдал статью."""
    fb = f"\n\nКРИТИК отклонил прошлую версию, ИСПРАВЬ по замечаниям (но выведи СТАТЬЮ ЦЕЛИКОМ, " \
         f"не комментарий правок):\n{feedback}" if feedback else ""
    if base_draft and _looks_like_paper(base_draft):
        # base ЦЕЛИКОМ (persist-guard держит его <=90k): reporter в файл-режиме НЕ обрезает вывод,
        # поэтому показываем ВЕСЬ черновик ВКЛЮЧАЯ Заключение — иначе генератор видел только начало
        # (без концовки) и «дорабатывал» обрыв → C=fail → скор 4 (качели 4↔6). [:90000] = совпадает с
        # persist size-guard: НИКОГДА не режем реальный драфт посреди таблицы.
        fb += ("\n\n=== ТЕКУЩИЙ ЛУЧШИЙ ЧЕРНОВИК СТАТЬИ — ДОРАБОТАЙ ЕГО ИНКРЕМЕНТАЛЬНО, НЕ ПИШИ С НУЛЯ ===\n"
               + base_draft[:90000] +
               "\n=== КОНЕЦ ЧЕРНОВИКА ===\n"
               "Внеси ТОЧЕЧНЫЕ правки по замечаниям прямо в этот черновик: СОХРАНИ всё рабочее "
               "(структуру, формулировки теорем, доказательства, числа, таблицы), исправляй ТОЛЬКО "
               "помеченные места и доводи неполные секции до конца. "
               "ДЛИНА: целевой объём 5-6 СТРАНИЦ (~18000-20000 знаков LaTeX-body, потолок 25000). Пишешь "
               "в файл — обрезки вывода нет, но держи этот объём. Если "
               "черновик длиннее — СОКРАТИ прозу (короче формулировки, убери повторы/воду), но СОХРАНИ "
               "все секции, теоремы, числа и таблицы. ПОЛНОТА > ОБЪЁМ: лучше сжато и ЦЕЛИКОМ, чем "
               "длинно и с обрывом. Выведи ПОЛНЫЙ LaTeX (не diff), ОБЯЗАТЕЛЬНО заверши ВСЕ секции "
               "и последним \\section — никакого обрыва на полуслове.")
    # reporter=codex → файл-режим: статья пишется ПРЯМО В ФАЙЛ (не в обрезаемое сообщение),
    # иначе sol обрывает Заключение на полуслове → критик C=fail → скор 4 (разброс 4↔6).
    _codex_rep = cfg.engines.reporter == "codex" and engines.codex_available()
    def _gen(p: str) -> str:
        # keep=True: составитель в этом же вызове строит submission_registry/verify_submission,
        # привязанные к пути+sha256 своего файла-артефакта. Удаление файла делало верификатор
        # НЕПРОХОДИМЫМ навсегда (dangling `_wip__paper_wip_*.tex`) → R=fail, потолок балла 4.
        return engines.codex_write_artifact(p, cfg, "_paper_wip.tex", keep=True) if _codex_rep else call(p)
    out = _gen(_WRITER_PROMPT.format(domain=cfg.domain, context=context[:35000], feedback=fb))
    tex = _extract_body(out)
    if not _looks_like_paper(tex):
        logger.warning("составитель выдал не-статью (%d chars) — строгий ретрай", len(tex))
        strict = ("\n\nКРИТИЧНО: предыдущий ответ не был статьёй. Выведи РОВНО LaTeX-документ, "
                  "первый символ = \\begin{abstract}, никакого текста-комментария.")
        out = _gen(_WRITER_PROMPT.format(domain=cfg.domain, context=context[:35000], feedback=fb + strict))
        tex = _extract_body(out)
    if not _looks_like_paper(tex):
        return ""
    # МАШИННЫЙ ЧЕК ОСИ R (фикс волны 189): один адресный раунд правки, если протокол/провенанс не
    # добирает требований W4. Ворота НЕ ужесточаются: не прошло второй раз — отдаём текст как есть
    # (пусто вернуть нельзя, иначе цикл потеряет статью), но пишем в лог, что именно осталось.
    viol = _r_protocol_violations(tex, context)
    if viol:
        logger.info("[R] машинный чек протокола не пройден (%d): %s", len(viol), "; ".join(viol)[:400])
        rfix = ("\n\n=== МАШИННАЯ ПРОВЕРКА ОСИ R (провенанс/протокол) НЕ ПРОЙДЕНА ===\n"
                "Рецензент держит R=partial ровно за это (W4: одна registry-backed protocol-таблица "
                "с колонками kind/model/PRE/POOL/PFT/FT/seeds-or-task-IDs/LR-grid/CI-unit/replicates/"
                "RNG/controls/command/artifact-hash/confirm-rule). Не пройдено:\n- "
                + "\n- ".join(viol) +
                "\nСДЕЛАЙ ТОЧЕЧНО: слей протокольные таблицы в ОДНУ с этими колонками (числа НЕ "
                "выдумывай — бери из `protocol/env/launch_command/provenance` материала; чего нет, "
                "помечай `не сохранено`), а числа без immutable-артефакта убери из main text в "
                "Ограничения. Остальной текст СОХРАНИ без изменений и уложись в потолок 25000 знаков.")
        out2 = _gen(_WRITER_PROMPT.format(domain=cfg.domain, context=context[:35000],
                                          feedback=fb + rfix))
        tex2 = _extract_body(out2)
        if _looks_like_paper(tex2):
            v2 = _r_protocol_violations(tex2, context)
            if len(v2) <= len(viol):
                if v2:
                    logger.warning("[R] после адресного раунда осталось %d требований: %s",
                                   len(v2), "; ".join(v2)[:300])
                else:
                    logger.info("[R] машинный чек протокола пройден после адресного раунда")
                tex = tex2
            else:
                logger.warning("[R] адресный раунд УХУДШИЛ протокол (%d→%d) — оставляю прежнюю версию",
                               len(viol), len(v2))
    return tex


# ---------------- Opus жёсткий A*-критик ----------------

_ASTAR_PROMPT = """\
Ты ЖЁСТКИЙ A*-РЕЦЕНЗЕНТ: роль NeurIPS/ICLR area-chair (optimization + theory of deep
learning), режим no-mercy. Принцип: honesty>hype. Проект: {domain}

КАЛИБРОВКА ДЛЯ CODEX-КРИТИКА:
- Будь не мягче прежнего Sonnet/Opus-критика. Не повышай балл за аккуратную подачу, если
  load-bearing evidence не закрывает killer-дыры.
- SCORE 8+ разрешён только если в тексте нет unresolved KILLER и есть один ясный actionable
  claim, подкреплённый matching real-net evidence/CI и честными baseline controls.
- Если есть хотя бы один unresolved KILLER, SCORE должен быть <=7. Если killer касается
  mismatch между claim и экспериментом, weak baseline, missing wall-clock или missing bridge,
  обычно SCORE 4-5.
- Не давай benefit of doubt. Promising direction != accepted paper. Красивый framing без
  решающего evidence не выше 5.
- ПРОВЕНАНС + INTEGRITY (сверяй ЯВНО, чеклист): каждое ключевое число/CI/утверждение о результате
  должно прослеживаться к РЕАЛЬНОМУ эксперименту (RESULT_JSON/лог), а не быть написанным в тексте.
  Считай KILLER и режь SCORE<=4 при любом из: выдуманные цитаты/baseline/методы; числа без прогона;
  claim шире того, что реально проверено; внутренние противоречия; метрика не соответствует
  confirm_rule; «подтверждение» на 1 сиде без bootstrap-CI. Честный негатив НЕ штрафуется.

ДЕТЕРМИНИРОВАННОСТЬ (КРИТИЧНО — один и тот же манускрипт ОБЯЗАН давать один и тот же балл;
разброс 5↔7 на одном тексте недопустим):
- SCORE = детерминированная функция ЧЕК-ЛИСТА ниже, НЕ впечатления. НЕ меняй балл из-за тона,
  длины или красоты подачи. Сначала заполни оси, потом выведи балл СТРОГО по таблице (МИНИМУМ
  из применимых строк). Оценивай текущий манускрипт как есть; не наказывай дважды за один дефект.

ЧЕК-ЛИСТ (5 осей, методология /astar-paper-review; покрой КАЖДУЮ секцию, оси = pass/partial/fail):
 T (теория): формулировки полны (observable O, правило r с доменом/кодоменом, label, допустимый
   класс, порядок кванторов); доказательства строги. ПРАВИЛО ЧАРИТИ (не фолс-флажь): прежде чем
   счесть шаг неверным, ПЕРЕВЫВЕДИ его благожелательно (авторы компетентны); корректное доказательство
   НЕ понижает балл.
 E (эксперименты/evidence): каждое число прослеживается к RESULT_JSON/CI; baseline/controls честны;
   вывод соответствует confirm_rule; сиды+bootstrap есть.
 C (ясность/нотация/полнота): КАЖДЫЙ символ определён при первом вхождении; НЕТ обрывов; все секции завершены.
 N (новизна/related work): вклад отделён от прежнего; без overclaim; строгий негатив/impossibility засчитывается как вклад.
 R (воспроизводимость/провенанс): protocol-таблица (kind/model/steps/seeds/CI-rule/controls) в тексте; артефакты названы.
Fail на несущей оси (T и E — несущие для theory/negative-статьи) = KILLER.

ДЕТЕРМИНИРОВАННАЯ ТАБЛИЦА БАЛЛА (SCORE = МИНИМУМ из всех применимых строк):
 - C=fail (обрыв/неполнота/неопределённые символы) ИЛИ выдуманные числа/цитаты → SCORE 4
 - >=1 KILLER на несущей оси (T=fail или E=fail) → SCORE 5
 - несущие оси >=partial, но >=2 оси в partial (незакрытые FIXABLE-killer'ы) → SCORE 6
 - все оси >=partial И <=1 оси в partial, история связна, есть actionable ИЛИ строгий негатив → SCORE 7
 - все 5 осей = pass, 0 unresolved killer, actionable/impossibility claim с matching evidence/CI → SCORE 8
 - плюс выдающаяся новизна/влияние → 9
ПЕРЕД строкой SCORE выведи оси явно: `AXES: T=.. E=.. C=.. N=.. R=..`.
Слабости — триплетами (claim → место/evidence → severity), severity-порядок (сначала топящая),
дедуп, ТОЛЬКО load-bearing verified (precision>recall, не длинный список полу-проверенных).

>>> ТРЕБОВАНИЯ РУКОВОДИТЕЛЯ (соблюдать ЖЁСТКО, это твой главный чек-лист):
{boss}
<<<

A* — это НЕ «набор корректных фактов с несложными доказательствами + эксперименты, которые
ничего особо не показывают». Это уровень «проект третьекурсника». A* статья ОБЯЗАНА иметь:
1. ПРАКТИЧЕСКУЮ ПОЛЬЗУ ИЛИ РЕШАЮЩИЙ НЕГАТИВНЫЙ/IMPOSSIBILITY РЕЗУЛЬТАТ. Положительное actionable:
   когда остановить предобучение; какой оптимизатор даёт лучшие финальные веса; какой чекпоинт брать.
   НО РАВНОЦЕННО: строгий ВОСПРОИЗВОДИМЫЙ негативный/impossibility результат — это ПОЛНОЦЕННАЯ
   A*-контрибуция, НЕ повод занижать балл. Пример нужного уровня: «target-only / GN-Muon НЕ улучшает
   перенос pretrain→finetune: Delta_C1<0 с bootstrap-CI на реальной сети (несколько сидов) + теория
   ПОЧЕМУ (impossibility: target-only сертификат недостаточен без source)». Это говорит полю, что не
   работает и где фундаментальный предел — полезно.
   ВАЖНО (авто-разворот): если положительное направление уже SETTLED-NEGATIVE (см. BINDING EVIDENCE:
   exp_refuted/no_signal по ключевым claim'ам) — НЕ требуй несуществующий положительный headline и не
   гоняй агентов «усилить» опровергнутое. Вместо этого ЯВНО направь историю: HEADLINE = строгий
   негатив/impossibility узел, DIRECTIVES = достроить это как несущую линию (теория предела + чистый
   прослеживаемый негатив). Провенанс к негативу обязателен так же, как к позитиву (RESULT_JSON/CI/сиды).
   НЕ A* только если нет НИ actionable-вывода, НИ строгого решающего негатива/impossibility.
2. КРУТУЮ СВЯЗНУЮ ИСТОРИЮ — одна сильная идея, а не россыпь лемм. Читатель должен понять, В
   ЧЁМ СМЫСЛ и зачем это нужно.
3. ПОНЯТНОСТЬ И ПОЛНОТУ — КАЖДЫЙ символ определён при первом вхождении (если встречается C,
   G, Σ и т.п. без определения — это провал полноты). Незнакомый человек должен всё понять.
4. СТРОГОСТЬ + ЭКСПЕРИМЕНТЫ, которые РЕАЛЬНО ЧТО-ТО ПОКАЗЫВАЮТ (подтверждают практический
   вывод на реальной сети, а не воспроизводят формулу сами на себе).
КАЧЕСТВО > КОЛИЧЕСТВО: лучше 3 страницы с одной новой полезной идеей и подтверждением, чем
20 страниц фактов. Мало страниц — не минус.

Дай разбор: Summary (в чём идея и её ПРАКТИЧЕСКАЯ польза); Strengths; Weaknesses killer→minor
(W1 (KILLER):…); отдельно: есть ли actionable-вывод? все ли символы определены? понятен ли
смысл? Затем СПИСОК слабых/лишних узлов, которые НЕ дотягивают до A* и которые стоит выкинуть
из истории.

КАРТА УЗЛОВ ПРОЕКТА (id | статус | краткое) — для WEAK_NODES бери ТОЧНЫЕ id ОТСЮДА, не выдумывай:
{node_map}

Ты НЕ ТОЛЬКО критикуешь — ты ГОВОРИШЬ, ЧТО КОНКРЕТНО СДЕЛАТЬ, чтобы тебя удовлетворить.
СНАЧАЛА классифицируй КАЖДОЕ killer-замечание в один из двух типов (это критично):
- FIXABLE (починяемо работой): не хватает доказательства/эксперимента/определения. Директива =
  что доказать или какой эксперимент прогнать (с измерением/CI).
- SETTLED-NEGATIVE (истинная негативная находка в данных, напр. «объект X не бьёт тривиальную базу»
  или «теория ломается в режиме Y»): это ПРАВДА, её НЕЛЬЗЯ «починить» переписыванием — попытка
  сделать ложное истинным = fabrication, запрещено. Директива здесь = РАЗВЕРНУТЬ историю прочь от
  мёртвого тезиса: либо честно подать как negative-result, либо ПЕРЕНАЦЕЛИТЬ статью на более
  сильный УЖЕ ПОДТВЕРЖДЁННЫЙ результат (confirmed_emp) из карты узлов. НЕ требуй от агентов
  снова и снова «усилить» то, что данные уже опровергли — это слив их раундов.
Если в карте есть сильная confirmed_emp-ветка, не используемая как хедлайн, — прямо укажи
пивот на неё. Дай 3-6 КОНКРЕТНЫХ ДЕЙСТВИЙ (идейным + составителю), каждое — выполнимый шаг.

BINDING EVIDENCE CONTEXT (не игнорировать): если здесь указан typed `exp_refuted` или
negative/no_signal result, НЕ называй соответствующий audit missing. Классифицируй это как
SETTLED-NEGATIVE или как уже выполненный отрицательный control, и требуй pivot/scope reduction,
а не повтор того же эксперимента.
Если BINDING EVIDENCE CONTEXT содержит `kind=<X> already_refuted_by=...`, запрещено писать
CORNERSTONE, который снова мапится в `<X>`. Такой CORNERSTONE является ошибкой рецензии.
Напиши `CORNERSTONE: нет`, если нет materially different llama-compatible experiment kind.
{evidence_context}

PROJECT EXPERIMENT POLICY:
{exp_policy}

КРОСС-ВЕТОЧНЫЙ СИНТЕЗ: если ДРУГАЯ ветка ОБЪЯСНЯЕТ/обосновывает headline (даёт механизм или
теорию-под него, напр. «геометрия слоёв ⇒ почему этот оптимизатор лучше») — укажи её как МОСТ и
вели вплести как «ПОЧЕМУ работает headline», УСИЛИВАЯ одну историю. НЕ склеивай две самостоятельные
истории в одну статью (две слабо связанные линии = провал, ты сам штрафуешь за это). Связи нет —
не притягивай за уши.

Если для подъёма балла нужен ОДИН решающий длинный эксперимент — «краеугольный» для всей истории
(например достоверно показать на реальной сети с узким CI ключевое сравнение оптимизаторов по
забыванию на бОльшем масштабе/числе сидов) — опиши его ОДНОЙ строкой: что с чем сравнить и какое
измерение/CI ждём. Команда поставит его как приоритетный и будет ждать результат. Иначе — 'нет'.

В САМОМ КОНЦЕ выведи РОВНО:
HEADLINE: <ТОЧНЫЙ id узла из карты — самый сильный несущий результат: положительный actionable
(обычно confirmed_emp) ЛИБО, если положительное направление settled-negative, строгий воспроизводимый
негатив/impossibility узел (напр. impossibility_targetonly) — вокруг него строится статья; или 'нет'>
DIRECTIVES: <нумерованный список 3-6 конкретных действий для агентов, чтобы поднять балл>
CORNERSTONE: <одно решающее сравнение для длинного эксперимента, или 'нет'>
WEAK_NODES: <НЕ БОЛЕЕ 3-5 САМЫХ слабых id из карты через запятую (не сноси десятки — это убивает
материал отчёта), или 'нет'>
DEMOTE_NODE: <id узла из карты, чей ЭМПИРИЧЕСКИЙ headline-claim (число/скорость/CI, напр. «13-19×»)
помечен как W1-KILLER непрослеживаемый (нет RESULT_JSON/логов/commit). Такой узел будет ДЕМОУТНУТ
proven/confirmed_emp→conditional, чтобы команда подкрепила его реальным экспериментом ИЛИ убрала
непрослеживаемое число. НЕ давай сюда узел, если его число реально прослеживается. Если таких нет — 'нет'>
SCORE: <целое 1-10>

СТАТЬЯ (LaTeX):
{paper}"""


def _suppress_refuted_cornerstone(review: str, refuted_kinds: Dict[str, List[str]]) -> str:
    if not refuted_kinds:
        return review
    cm = re.search(r"^(\s*CORNERSTONE\s*:\s*)(.+?)\s*$",
                   review, flags=re.IGNORECASE | re.MULTILINE)
    if not cm:
        return review
    spec = cm.group(2).strip()
    if not spec or "нет" in spec.lower()[:6]:
        return review
    # НЕ глушить материально-НОВЫЙ эксперимент. Причинные интервенции/ablation/transplant/subspace-
    # surgery НЕ равны опровергнутым checkpoint-selector/forgetting аудитам, но infer_kind по ключевым
    # словам ошибочно лумпит их в legacy-refuted kind и глушит → экспериментатор простаивает, хотя
    # критик просит новый прогон. Такие пропускаем: сеялка (_seed_cornerstone_candidate) уведёт их в
    # custom_llama (агент пишет свежий скрипт, эксп реально идёт на GPU).
    _novel = ("causal", "intervention", "surger", "transplant", "subspace", "ablation",
              "erasure", "protection", "counterfactual", "scrub", "интервенц", "причин",
              "трансплант", "абляц")
    if any(m in spec.lower() for m in _novel):
        return review
    try:
        kind = exp_spec.infer_kind(spec)
    except Exception:
        kind = ""
    if not kind or kind not in refuted_kinds:
        return review
    ids = ", ".join(refuted_kinds[kind][:6])
    note = (
        "DUPLICATE_REFUTED_CORNERSTONE: raw CORNERSTONE mapped to already-refuted "
        f"typed EXP_SPEC kind `{kind}` by node(s) {ids}. Treat this as "
        "SETTLED-NEGATIVE evidence and pivot/scope-reduce; do not request the same "
        "experiment again.\n"
    )
    review = review[:cm.start(2)] + "нет" + review[cm.end(2):]
    return note + review


_TRACE_FROZEN = ("proven", "confirmed_emp")


_THEORY_MARKERS = ("theorem", "impossib", "lemma", "no-free", "no_free", "identifiab",
                   "теорем", "невозможн", "лемм", "no_exp_needed", "push pass")


def _is_theory_headline(node: Dict[str, Any]) -> bool:
    """Узел — чистая теория / proof-gate-результат (нет эмпирического headline-числа для провенанса)?
    Такой узел НЕ подлежит провенанс-демоуту: доказанная теорема не требует RESULT_JSON."""
    if node.get("exp_confirmed"):
        return False  # есть реальное эмпирическое подтверждение → это эмпирика, не чистая теория
    blob = " ".join(str(node.get(k, "")) for k in
                    ("id", "short", "idea", "derive", "verdict", "last_gate")).lower()
    return any(m in blob for m in _THEORY_MARKERS)


def _apply_headline_demotion(review: str, data: Dict[str, Any], cfg: ARConfig) -> Optional[str]:
    """Разрыв doom-loop непрослеживаемого headline: если A*-критик пометил узел как несущий
    ЭМПИРИЧЕСКИЙ claim без провенанса (W1-KILLER), ДЕМОУТИМ его proven/confirmed_emp → conditional
    + директива. Тогда писатель перестаёт тащить непрослеживаемое число как «доказанное», а луп
    либо подкрепляет его реальным экспериментом (RESULT_JSON+CI), либо разворачивает в честный
    негатив. Возвращает id демоутнутого узла или None.
    Сигнал: явный `DEMOTE_NODE:` от критика; фолбэк — KILLER про провенанс → демоут HEADLINE-узла."""
    by = gio.index(data)
    nid = ""
    m = re.search(r"(?im)^\s*DEMOTE_NODE\s*:\s*(.+?)\s*$", review or "")
    if m:
        cand = m.group(1).strip()
        if cand and "нет" not in cand.lower()[:6]:
            nid = re.split(r"[\s,]+", cand)[0].strip()
    if not nid:  # фолбэк: KILLER про непрослеживаемость → демоутим сам HEADLINE-узел
        if re.search(r"(?is)KILLER.{0,140}(provenance|прослеж|непросл|traceab|RESULT_JSON)", review or ""):
            hm = re.search(r"(?im)^\s*HEADLINE\s*:\s*([A-Za-z0-9_\-]+)", review or "")
            if hm:
                nid = hm.group(1).strip()
    if not nid:
        return None
    node = by.get(nid)
    if not node or node.get("status") not in _TRACE_FROZEN:
        return None
    if _is_theory_headline(node):
        # Провенанс-демоут применим ТОЛЬКО к эмпирическому headline-числу (напр. «13-19×» без
        # RESULT_JSON). Чистая теория / proof-gate-теорема (NO_EXP_NEEDED, impossibility, лемма) НЕ
        # имеет эмпирического числа — требовать RESULT_JSON категориально неверно и порождает
        # doomed empirical-child. Не демоутим: доказанное остаётся headline.
        return None
    prev = node.get("status")
    node["status"] = "conditional"
    node["verdict"] = (
        f"ДЕМОУТ {prev}->conditional (A*-провенанс-KILLER): эмпирический headline-claim не "
        "прослеживается к RESULT_JSON/логам/commit. Подкрепить реальным typed EXP_SPEC-экспериментом "
        "(RESULT_JSON+bootstrap CI) ЛИБО убрать непрослеживаемое число и оставить только доказуемое."
    )
    node["open"] = (
        "Дать провенанс headline-числу (реальный эксперимент, RESULT_JSON+CI) или переформулировать "
        "узел без непрослеживаемого эмпирического claim."
    )
    return nid


def _cornerstone_kind(review: str) -> str:
    cm = re.search(r"(?im)^\s*CORNERSTONE\s*:\s*(.+?)\s*$", review or "")
    if not cm:
        return ""
    spec = cm.group(1).strip()
    if not spec or "нет" in spec.lower()[:6]:
        return ""
    try:
        return exp_spec.infer_kind(spec)
    except Exception:
        return ""


def _cornerstone_line(text: str) -> str:
    cm = re.search(r"(?im)^\s*CORNERSTONE\s*:\s*(.+?)\s*$", text or "")
    return cm.group(1).strip() if cm else ""


_CORNERSTONE_REQUERY = """\
Твой предложенный CORNERSTONE-эксперимент («{bad}») УЖЕ ПРОГНАН И ОПРОВЕРГНУТ (settled-negative).
Повторять опровергнутый эксперимент нельзя — это слив раундов команды. Уже опровергнутые типы: {refuted}.
Evidence-контекст:
{evidence}
Карта узлов (кратко):
{node_map}
Предложи ОДИН МАТЕРИАЛЬНО ДРУГОЙ решающий llama-эксперимент для A* headline: другой kind/метрика/
сравнение/конструкция, НЕ сводящийся к уже опровергнутым. Если реально нового решающего эксперимента
нет — честно ответь 'нет' (тогда историю надо развернуть: сделать из негатива результат, а не повторять refuted).
Ответь РОВНО одной строкой:
CORNERSTONE: <одно решающее сравнение НОВОГО эксперимента, или 'нет'>"""


def _requery_cornerstone_if_refuted(out: str, cfg: ARConfig, call: Callable[[str], str],
                                    node_map: str, evidence_context: str,
                                    refuted_kinds: Optional[Dict[str, List[str]]]) -> str:
    """Если A*-критик назвал уже опровергнутый CORNERSTONE — просим МАТЕРИАЛЬНО другой (не из refuted).
    Разблокирует поиск нового headline вместо повтора мёртвого эксперимента."""
    if not refuted_kinds:
        return out
    bad = _cornerstone_kind(out)
    if not bad or bad not in refuted_kinds:
        return out
    try:
        alt = call(_CORNERSTONE_REQUERY.format(
            bad=bad, refuted=", ".join(sorted(refuted_kinds)),
            evidence=(evidence_context or "(нет)")[:2000],
            node_map=(node_map or "(нет)")[:2000]))
        newc = _cornerstone_line(alt)
        if not newc or "нет" in newc.lower()[:6]:
            return out
        newk = ""
        try:
            newk = exp_spec.infer_kind(newc)
        except Exception:
            newk = ""
        if newk and newk not in refuted_kinds:
            logger.info("A* cornerstone re-query: refuted %s → новый %s (kind %s)", bad, newc[:60], newk)
            return re.sub(r"(?im)^\s*CORNERSTONE\s*:\s*.+$", "CORNERSTONE: " + newc, out, count=1)
    except Exception as e:
        logger.info("cornerstone re-query не удался: %s", e)
    return out


def astar_critic(paper_tex: str, cfg: ARConfig, call: Callable[[str], str],
                 node_map: str = "", evidence_context: str = "",
                 refuted_kinds: Optional[Dict[str, List[str]]] = None) -> Tuple[int, str, list]:
    from . import narrative as nar
    # [:90000] = persist size-guard, НЕ обрезает реальные драфты (целевой объём ~25k). Старый срез
    # [:16000] отрезал критику конец статьи (Table 2 хвост, Related Work, Conclusion, рецепт) →
    # критик честно писал «манускрипт оборван mid-table» → детерминированно C=fail → SCORE 4,
    # хотя полный текст он сам оценивал на 7-8. Критик ОБЯЗАН видеть статью ЦЕЛИКОМ.
    out = call(_ASTAR_PROMPT.format(domain=cfg.domain, paper=paper_tex[:90000],
                                    boss=nar.boss(cfg) or "(не задано)",
                                    node_map=(node_map or "(карта недоступна — WEAK_NODES не давай)")[:4000],
                                    evidence_context=evidence_context[:20000] or "(нет)",
                                    exp_policy=(
                                        "exp_kind=llama: CORNERSTONE, DIRECTIVES, and experiment requests must use "
                                        "llama-style evidence only. Do not name or request legacy checkpoint-size/model "
                                        "families such as old HF checkpoint family markers, 160m, or 410m. If no "
                                        "llama-compatible decisive experiment is available, write CORNERSTONE: нет."
                                        if str(getattr(cfg, "exp_kind", "") or "").strip().lower() == "llama"
                                        else "(no additional project experiment policy)"
                                    )))
    out = _requery_cornerstone_if_refuted(out, cfg, call, node_map, evidence_context, refuted_kinds)
    out = _suppress_refuted_cornerstone(out, refuted_kinds or {})  # safety net, если re-query не дал нового
    m = None
    for line in reversed(out.splitlines()):
        m = re.search(r"SCORE:\s*(\d+)", line)
        if m:
            break
    if m is None:
        # критик не вернул строку SCORE (движок завис/обрыв/CLAUDE_DISABLED) — это СБОЙ критика,
        # а НЕ балл 0. Сентинел -1 → generate_paper не запишет фейковый 0/10 и повторит/пропустит.
        return -1, out, []
    score = int(m.group(1))
    raw_score = score
    clamp_reasons = []
    if score >= 8 and re.search(r"\bKILLER\b", out, re.IGNORECASE):
        logger.warning("A* critic returned SCORE %d with KILLER blockers; clamping to 7", score)
        clamp_reasons.append("KILLER blockers present")
        score = 7
    cm = re.search(r"CORNERSTONE:\s*(.+)", out)
    if score >= 8 and cm:
        cornerstone = cm.group(1).splitlines()[0].strip().lower()
        if cornerstone and "нет" not in cornerstone[:12]:
            logger.warning("A* critic returned SCORE %d with unresolved CORNERSTONE; clamping to 7", score)
            clamp_reasons.append("unresolved CORNERSTONE")
            score = 7
    hm = re.search(r"HEADLINE:\s*(.+)", out)
    if score >= 8 and (not hm or "нет" in hm.group(1).lower()[:12]):
        logger.warning("A* critic returned SCORE %d without a concrete HEADLINE; clamping to 7", score)
        clamp_reasons.append("missing concrete HEADLINE")
        score = 7
    if score != raw_score:
        out = re.sub(r"(SCORE:\s*)\d+", lambda m: f"{m.group(1)}{score}", out)
        out = (f"STRICT_CLAMP: raw SCORE {raw_score} -> {score} "
               f"({'; '.join(clamp_reasons)})\n" + out)
    weak = []
    wm = re.search(r"WEAK_NODES:\s*(.+)", out)
    if wm and "нет" not in wm.group(1).lower():
        weak = [w.strip() for w in re.split(r"[,\s]+", wm.group(1)) if w.strip()]
    return min(max(score, 0), 10), out, weak


def astar_critic_median(paper_tex: str, cfg: ARConfig, call: Callable[[str], str],
                        node_map: str = "", evidence_context: str = "",
                        refuted_kinds: Optional[Dict[str, List[str]]] = None,
                        k: int = 3) -> Tuple[int, str, list]:
    """AIDE²-style public/private score: K НЕЗАВИСИМЫХ прогонов A*-критика на ОДНОМ tex → МЕДИАНА
    как устойчивый (private) балл. opus-критик недетерминирован (тот же контент → 4 или 6) → это
    и была причина пилы 4↔6; медиана убирает шум СТРУКТУРНО (не костылём anti-swing). Фидбек/weak
    берём от прогона, ближайшего к медиане (репрезентативный). k=1 → полностью старое поведение.
    Сбойные прогоны (SCORE=-1) отбрасываем; если валидных нет → -1 (как одиночный)."""
    # КЭШ ВЕРДИКТА ПО SHA ТЕКСТА (фикс 31-07): писатель регулярно выдаёт байт-в-байт тот же tex
    # (рэтчет бракует слабые попытки) → median-of-3 пере-суживал ОДИН И ТОТ ЖЕ текст по 3-4 раза
    # (юзер видел «один и тот же отчёт крутится»), жёг codex и добавлял шум пилы. Тот же текст ⇒
    # та же рецензия: отдаём кэшированный вердикт без вызова критика.
    import hashlib as _hl
    _ck = _hl.sha1(paper_tex.encode("utf-8", "ignore")).hexdigest()
    _cpath = os.path.join(cfg.workdir, ".run", "_critic_cache.json")
    try:
        _cc = json.load(open(_cpath))
        if _cc.get("sha") == _ck and isinstance(_cc.get("score"), int):
            logger.info("A* critic: tex не менялся с прошлого суда (sha %s) — кэшированный балл %d",
                        _ck[:8], _cc["score"])
            return int(_cc["score"]), str(_cc.get("review", "")), list(_cc.get("weak") or [])
    except (OSError, ValueError):
        pass
    if k <= 1:
        return astar_critic(paper_tex, cfg, call, node_map, evidence_context, refuted_kinds)
    runs = []
    for _ in range(k):
        s, out, weak = astar_critic(paper_tex, cfg, call, node_map, evidence_context, refuted_kinds)
        if s >= 0:
            runs.append((s, out, weak))
    if not runs:
        return -1, "", []
    runs.sort(key=lambda r: r[0])
    med = runs[len(runs) // 2][0]
    rep = min(runs, key=lambda r: abs(r[0] - med))  # фидбек от прогона ближе к медиане
    logger.info("A* critic median-of-%d: баллы=%s → медиана %d", len(runs),
                [r[0] for r in runs], med)
    try:
        import time as _t
        json.dump({"sha": _ck, "score": int(med), "review": rep[1], "weak": rep[2],
                   "ts": _t.time()}, open(_cpath, "w"), ensure_ascii=False)
    except OSError:
        pass
    return med, rep[1], rep[2]


def prune_weak_nodes(data: Dict[str, Any], weak: list, max_prune: int = 5) -> int:
    """Право A*-критика выкидывать узлы, не дотягивающие до A* (→ deferred).
    ОГРАНИЧЕНИЕ: не больше max_prune за отчёт — иначе критик сносит десятки узлов за раз,
    граф схлопывается к крошечному ядру и отчёт БЕДНЕЕТ (балл падал 5→4). Чистим постепенно.
    confirmed_emp/proven/conditional не трогаем — это уже доказанный материал, не россыпь."""
    by = {n["id"]: n for n in data["nodes"]}
    n = 0
    for nid in weak:
        if n >= max_prune:
            break
        node = by.get(nid)
        if not node:
            continue
        # Защищённый (proven/conditional/confirmed_emp или astar>=6) узел, который критик всё равно
        # называет weak → он реален, но ДИЛЛЮТИТ текущий headline. Статус не трогаем, а убираем из
        # МАТЕРИАЛА отчёта (report_excluded). Теорему-headline (чистую теорию) НИКОГДА не исключаем.
        try:
            ast = int(node.get("astar") or 0)
        except (ValueError, TypeError):
            ast = 0
        protected = ast >= 6 or node.get("status") in ("proven", "conditional", "confirmed_emp")
        if protected:
            if not _is_theory_headline(node) and not node.get("report_excluded"):
                node["report_excluded"] = True
                n += 1
            continue
        if node.get("status") not in ("open", "weak", "deferred"):
            continue
        node["status"] = "deferred"
        node["verdict"] = (node.get("verdict", "") + " | ").lstrip(" |") + "выкинут A*-критиком (не A*)"
        n += 1
    return n


# ---------------- оркестрация ----------------

def generate_paper(data: Dict[str, Any], cfg: ARConfig, ts: str, branch: Optional[str] = None,
                   min_score: int = 8, max_rounds: int = 3) -> Dict[str, Any]:
    """Написать финальную статью, прогнать жёсткого A*-критика, ревайз до порога, PDF.
    Эмитит статус двух главных report-агентов (составитель + A*-критик) на дашборд."""
    rcall = engine_call(cfg.engines.reporter, cfg)
    ccall = engine_call(cfg.engines.report_critic, cfg)
    reporter_engine = cfg.engines.reporter
    critic_engine = cfg.engines.report_critic
    reporter_role = f"составитель отчёта ({reporter_engine})"
    critic_role = f"супер-жёсткий A*-критик ({critic_engine})"
    def _emit(agent, stage, detail, role, engine=None):
        try:
            from . import agents_status
            agents_status.update(cfg, agent, node="ОТЧЁТ", short="отчёт по всем узлам",
                                 stage=stage, detail=detail,
                                 engine=engine or reporter_engine, role=role)
            agents_status.log_event(cfg, agent, f"{stage} · {detail[:90]}")
            agents_status.write_snapshot(cfg)
        except Exception:
            pass
    from . import narrative as _nar
    _nmap = _nar.node_map(data)  # карта реальных id узлов → критик даёт корректные WEAK_NODES для prune
    # HEADLINE прошлого цикла читаем ДО сборки материала → режим B: материал = ветка headline ЦЕЛИКОМ
    # + confirmed_emp/proven из любых веток (история строго про headline, с правом опереться на факты).
    seed_fb = ""
    headline_id = None
    try:
        fbp = os.path.join(cfg.workdir, "astar_feedback.md")
        if os.path.exists(fbp):
            fbtxt = graph_hygiene.scrub_project_policy_text(open(fbp, encoding="utf-8").read(), cfg)
            seed_fb = "УЧТИ ЭТУ КРИТИКУ С САМОГО НАЧАЛА:\n" + fbtxt[:2500]
            # DIRECTIVES критика стоят В КОНЦЕ рецензии → head-срез [:2500] их отрезал, и писатель
            # цикл за циклом не видел конкретных действий «что исправить». Дошиваем их явно.
            dmm = re.search(r"DIRECTIVES:.*?(?=\nCORNERSTONE:|\nWEAK_NODES:|\nDEMOTE_NODE:|\nSCORE:|\Z)",
                            fbtxt, re.DOTALL)
            if dmm and dmm.group(0)[:200] not in seed_fb:
                seed_fb += "\n\nДИРЕКТИВЫ КРИТИКА (обязательны к выполнению в этой ревизии):\n" + dmm.group(0)[:2500]
            hm = re.search(r"HEADLINE:\s*([A-Za-z0-9_\-]+)", fbtxt)
            if hm and "нет" not in hm.group(1).lower():
                headline_id = hm.group(1)
    except OSError:
        pass
    by_all = gio.index(data)
    hbranch = headline_id if (headline_id and headline_id in by_all) else None
    ctx = cluster_context(data, cfg, branch=branch, headline_branch=hbranch)
    refuted_kinds = _refuted_exp_kinds(data, cfg, hbranch)
    proven_thms = [n for n in data["nodes"]
                   if n.get("status") == "proven" and _is_theory_headline(n)
                   and n.get("col") not in (0, None)]
    if len(proven_thms) >= 2:  # matched-пара теорем → СОВМЕСТНЫЙ headline (не одну в «поддержку»)
        _ids = "; ".join(f"[{n['id']}] «{str(n.get('short',''))[:70]}»" for n in proven_thms[:4])
        ctx = (f"!!! ДОКАЗАНЫ НЕСКОЛЬКО ТЕОРЕМ ({len(proven_thms)}): {_ids}. Если они образуют "
               "MATCHED-ПАРУ (impossibility/lower bound + achievability/upper bound, либо "
               "necessity+sufficiency) — подай их СОВМЕСТНО как ЕДИНЫЙ главный результат (tight "
               "характеризация переноса), ОБЕ с ПОЛНЫМИ формулировками в разделе Теория; НЕ низводи "
               "ни одну до «поддержки». Это и есть A*-история negative-статьи.\n\n") + ctx
    elif headline_id and headline_id in by_all:  # жёсткий приказ писателю: статья строится ВОКРУГ узла
        hn = by_all[headline_id]
        ctx = (f"!!! ОБЯЗАТЕЛЬНЫЙ HEADLINE (приказ A*-критика): узел [{headline_id}] "
               f"«{hn.get('short','')}» — abstract, Введение и несущая история строятся ВОКРУГ НЕГО; "
               f"прочие узлы только в поддержку.\n\n") + ctx
    try:
        from . import agents_status as _as
        _as.report_flag(cfg, True, f"составитель ({reporter_engine}) пишет отчёт — это несколько минут")
    except Exception:
        pass
    # ИНКРЕМЕНТАЛЬНАЯ РЕВИЗИЯ: грузим лучший черновик прошлых циклов → писатель дорабатывает ЕГО,
    # а не пишет с нуля (убирает обрывы, копит прогресс). Ratchet: перезапишем только если балл не упал.
    best_draft_path = os.path.join(cfg.workdir, "paper_best_draft.tex")
    best_draft_score_path = os.path.join(cfg.workdir, "paper_best_draft.score")
    prev_draft, prev_draft_score = "", 0
    try:
        if os.path.exists(best_draft_path):
            prev_draft = open(best_draft_path, encoding="utf-8").read()
            if os.path.exists(best_draft_score_path):
                prev_draft_score = int((open(best_draft_score_path).read().strip() or "0"))
    except (OSError, ValueError):
        prev_draft, prev_draft_score = "", 0
    _emit("СОСТАВИТЕЛЬ-отчёта", "writing",
          f"[{reporter_engine}] дорабатываю лучший черновик (балл {prev_draft_score}/10) по критике"
          if prev_draft else f"[{reporter_engine}] пишу отчёт вокруг ОДНОЙ практ. идеи",
          reporter_role, engine=reporter_engine)
    tex = write_paper(ctx, cfg, rcall, feedback=seed_fb, base_draft=prev_draft)
    if not tex:
        # составитель не выдал валидную статью → НЕ зовём критика и НЕ пишем фейковый балл
        logger.warning("report-cycle: составитель не выдал статью → пропуск критика (без записи балла)")
        try:
            from . import agents_status as _as
            _as.report_flag(cfg, False)
            _as.update(cfg, "СОСТАВИТЕЛЬ-отчёта", node="ОТЧЁТ", short="отчёт по всем узлам",
                       stage="needswork", detail="составитель не смог выдать связную статью — повтор на след. цикле",
                       engine=reporter_engine, role=reporter_role)
            _as.log_event(cfg, "СОСТАВИТЕЛЬ-отчёта", "статья не собрана (не-LaTeX вывод) → балл НЕ записан")
            _as.write_snapshot(cfg)
        except Exception:
            pass
        return {"score": None, "published": False, "pdf": None, "tex": None, "review": None,
                "note": "writer_failed"}
    if prev_draft and tex.strip() == prev_draft.strip():
        # Писатель вернул тот же текст, что и рэтчет — судить/издавать нечего (фикс 31-07:
        # «отчёты не должны быть одинаковыми»). Ждём цикла с реальными правками.
        logger.info("составитель вернул текст БЕЗ изменений относительно рэтчета — суд и отчёт пропущены")
        try:
            from . import agents_status as _as
            _as.report_flag(cfg, False)
            _as.log_event(cfg, "СОСТАВИТЕЛЬ-отчёта",
                          "текст не изменился относительно лучшего драфта → суд/отчёт пропущены")
        except Exception:
            pass
        return {"score": None, "published": False, "pdf": None, "tex": None, "review": None,
                "note": "writer_no_change"}
    best = (0, tex, "", [])
    got_valid = False   # был ли хоть один валидный балл от критика (иначе не записываем фейковый 0/10)
    for rnd in range(max_rounds):
        try:
            from . import agents_status as _as
            _as.report_flag(cfg, True, f"A*-критик ({critic_engine}) судит отчёт, раунд {rnd+1}")
        except Exception:
            pass
        _emit("A*-критик-отчёта", "reviewing",
              f"[{critic_engine}] критикую отчёт как статью: практ. польза? понятность? история? (раунд {rnd+1})",
              critic_role, engine=critic_engine)
        _csamples = int(getattr(cfg, "critic_samples", 0) or 3)  # AIDE² median-of-K (шум критика→пила)
        score, review, weak = astar_critic_median(tex, cfg, ccall, node_map=_nmap,
                                                  evidence_context=ctx, refuted_kinds=refuted_kinds,
                                                  k=_csamples)
        if score < 0:  # критик не вернул SCORE (сбой движка) — НЕ балл 0; повторяем на том же tex
            logger.warning("A*-критик не вернул SCORE (сбой движка), раунд %d — без записи балла", rnd + 1)
            continue
        got_valid = True
        logger.info("A* critic round %d: SCORE %d/10 (порог %d), weak=%s", rnd + 1, score, min_score, weak)
        _emit("A*-критик-отчёта", "scored",
              f"[{critic_engine}] вынес вердикт A*={score}/10 · правки → агентам; выкинуть слабых: {', '.join(weak) or 'нет'}",
              critic_role, engine=critic_engine)
        if score > best[0]:
            best = (score, tex, review, weak)
        if score >= min_score or rnd == max_rounds - 1:
            break  # порог достигнут ИЛИ это последний раунд → НЕ переписывать (перерендер никто не судит,
                   # а opus-write блокировал бы главный поток до 40мин таймаута — пустой простой)
        _emit("СОСТАВИТЕЛЬ-отчёта", "writing",
              f"[{reporter_engine}] переписываю вокруг ОДНОЙ практ. идеи по правкам A*-критика (балл {score}/10)",
              reporter_role, engine=reporter_engine)
        new_tex = write_paper(ctx, cfg, rcall, feedback=review, base_draft=tex)
        if not new_tex:  # ревизия выродилась — оставляем лучшую версию, не зацикливаемся
            logger.warning("ревизия составителя выродилась — оставляю лучший драфт (%d/10)", best[0])
            break
        tex = new_tex
    if not got_valid:
        # критик ни разу не вернул валидный SCORE (движок завис/обрыв) → НЕ записываем фейковый 0/10
        logger.warning("report-cycle: A*-критик ни разу не вернул валидный SCORE → балл НЕ записан")
        try:
            from . import agents_status as _as
            _as.report_flag(cfg, False)
            _as.log_event(cfg, "A*-критик-отчёта", "критик не вернул SCORE (сбой движка) → балл НЕ записан")
        except Exception:
            pass
        return {"score": None, "published": False, "pdf": None, "tex": None, "review": None,
                "note": "critic_failed"}
    score, tex, review, weak = best
    # ANTI-КАЧЕЛИ: балл считается по СВЕЖЕ-регенерированному драфту каждый цикл, а критик (opus)
    # недетерминирован по осям (тот же контент → C=fail/4 или partial/6). Рэтчет уже хранит лучший
    # драфт на диске; если ЭТОТ цикл дал ХУЖЕ сохранённого — репортим/публикуем СОХРАНЁННЫЙ лучший
    # (он реально существует), а не шумный ре-ролл вниз. Балл становится монотонным → нет качелей 4↔6.
    attempt_lost = False
    if prev_draft_score > score and prev_draft and _looks_like_paper(prev_draft):
        # ПОПЫТКА ПРОИГРАЛА РЭТЧЕТУ (фикс 31-07). Раньше сюда подставлялся сохранённый лучший драфт
        # и издавался КАК НОВЫЙ отчёт → юзер видел ОДИН И ТОТ ЖЕ текст, пере-суженный по кругу.
        # Теперь проигравший цикл отчёт НЕ издаёт: рецензия попытки уходит в фидбек-петлю (директивы
        # писателю/агентам), публикация ждёт цикла, чей текст реально >= рэтчета.
        logger.info("anti-swing: попытка %d/10 < рэтчета %d/10 → отчёт-дубликат НЕ издаём; рецензия → фидбек",
                    score, prev_draft_score)
        attempt_lost = True
        score, tex = prev_draft_score, prev_draft
    accepted = score >= min_score
    # RATCHET: сохраняем лучший черновик, только если балл не упал (не теряем хорошую версию;
    # следующий цикл доработает ИМЕННО этот tex). Так прогресс копится, а не переписывается с нуля.
    try:
        # SIZE-GUARD: не сохраняем РАЗДУТЫЙ черновик как base (>90k = битый/накопил дампы; тогда
        # base_draft[:N] в следующем цикле даёт обрезок без Заключения → C=fail → качели 4↔6).
        # A*-статья компактна (~15-25k). Раздутый tex не persist-им, оставляем прошлый чистый base.
        if tex and _looks_like_paper(tex) and score >= prev_draft_score and len(tex) <= 90000:
            open(best_draft_path, "w", encoding="utf-8").write(tex)
            open(best_draft_score_path, "w").write(str(score))
            logger.info("persist best draft: score %d >= prev %d (%d chars) → сохранён для инкрем. доработки",
                        score, prev_draft_score, len(tex))
        elif tex and len(tex) > 90000:
            logger.warning("best draft НЕ сохранён: раздут (%d chars > 90k) — оставлен прошлый чистый base", len(tex))
    except OSError:
        pass
    try:
        from . import agents_status
        agents_status.record_score(cfg, score)  # для графика роста оценки
        agents_status.report_flag(cfg, False)    # report-cycle завершён → баннер снять
        agents_status.update(
            cfg, "A*-критик-отчёта", node="ОТЧЁТ", short="отчёт по всем узлам",
            stage=("accepted" if accepted else "scored"),
            detail=(f"[{critic_engine}] финальный вердикт A*={score}/10; "
                    f"weak: {', '.join(weak) or 'нет'}"),
            engine=critic_engine, role=critic_role, astar=str(score))
    except Exception:
        pass
    # АНТИ-КЛОББЕР (lost-update): data — снапшот, загруженный ДО ~часового цикла (writer + критики).
    # Сохранение ЕГО стирало всё, что воркеры записали в граф за время цикла (так пропадали свежие
    # proven-статусы: PASS узла ложился на диск, а сейв старого снапшота его затирал → узел «исчезал»).
    # Все мутации по итогам рецензии (prune/demote/elo-boost/index) применяем к СВЕЖЕМУ графу.
    data = gio.load_graph(cfg)
    gio.ensure_fields(data)
    # право A*-критика: выкинуть из истории узлы, не дотягивающие до A*
    if weak:
        pruned = prune_weak_nodes(data, weak)
        if pruned:
            gio.save_graph(data, cfg)   # gio — модульный; локальный re-import тут делал gio локальным во всей функции (UnboundLocalError выше)
            logger.info("A*-критик выкинул %d слабых узлов: %s", pruned, weak)
    # разрыв doom-loop: демоут непрослеживаемого headline-узла (proven→conditional) → писатель
    # перестаёт тащить фантомное число, луп подкрепляет экспериментом или разворачивает в негатив
    try:
        demoted = _apply_headline_demotion(review, data, cfg)
        if demoted:
            gio.save_graph(data, cfg)
            logger.info("A*-критик ДЕМОУТНУЛ непрослеживаемый headline-узел %s (proven→conditional)", demoted)
    except Exception as e:
        logger.warning("demote headline не удался: %s", e)
    try:
        from . import agents_status
        agents_status.update(
            cfg, "СОСТАВИТЕЛЬ-отчёта", node="ОТЧЁТ", short="отчёт по всем узлам",
            stage=("done" if accepted else "needswork"),
            detail=(f"отчёт принят как A* ({score}/10)" if accepted
                    else f"A*={score}/10 < порога {min_score} — НЕ A*, агенты дорабатывают теорию"),
            engine=reporter_engine, role=reporter_role, astar=str(score))
        agents_status.log_event(cfg, "A*-критик-отчёта",
                                (f"отчёт ПРИНЯТ A*={score}/10" if accepted
                                 else f"отчёт ОТКЛОНЁН A*={score}/10 (нужно >= {min_score}) → правки агентам"))
        agents_status.write_snapshot(cfg)
    except Exception:
        pass
    rd = _reports_dir(cfg)
    tag = "paper" if score >= min_score else f"paper-draft-{score}of10"
    name = f"{tag}-{ts}" + (f"-{branch}" if branch else "")
    if not attempt_lost:
        open(os.path.join(rd, name + ".tex"), "w", encoding="utf-8").write(tex)
        open(os.path.join(rd, name + ".review.md"), "w", encoding="utf-8").write(review)
    # индекс отчётов по узлам: тег = линии, которые покрыл отчёт; агент по этим узлам свериться
    try:
        if attempt_lost:
            raise ValueError("skip: попытка слабее рэтчета — отчёт не издан")
        from . import reports_index
        by = gio.index(data)
        covered = [n for n in data["nodes"]
                   if n.get("status") in _SURVIVED and n.get("col") not in (0, None)
                   and (not branch or _branch_of(n["id"], by) == branch)]
        node_ids = [n["id"] for n in covered]
        branch_ids = {_branch_of(n["id"], by) for n in covered}
        tags = [by[b].get("short", b) for b in branch_ids if b in by]
        summary = " ".join((review or "").split())[:400]
        reports_index.add(cfg, {"ts": ts, "score": score, "tex": name + ".tex",
                                "node_ids": node_ids, "tags": tags, "weak": weak,
                                "summary": summary})
    except Exception as e:
        logger.warning("report_index не записан: %s", e)
    # правки A*-критика → файл, подмешивается в память идейных агентов. ДИРЕКТИВЫ — В НАЧАЛО,
    # чтобы «что делать» дошло до агентов даже при обрезке памяти (не только killer-критика).
    # HEADLINE: узел, который критик назначил несущим → (1) запоминаем в astar_feedback.md
    # (следующий цикл строит статью вокруг него), (2) бустим Elo его ветки → идейные агенты копают её.
    hl_line = ""
    hlm = re.search(r"HEADLINE:\s*([A-Za-z0-9_\-]+)", review)
    if hlm and "нет" not in hlm.group(1).lower():
        hid = hlm.group(1)
        by_h = gio.index(data)
        if hid in by_h and graph_hygiene.project_policy_eligible(by_h[hid], cfg):
            hl_line = f"HEADLINE: {hid}\n\n"
            try:  # буст Elo ВСЕЙ ветки headline → селектор гонит идейных агентов развивать эту линию.
                # ИДЕМПОТЕНТНО: сперва снимаем прошлый буст со ВСЕХ узлов (поле hl_boost), потом ставим
                # на текущую ветку. Иначе буст копился бы каждый отчёт → вырождение в одну ветку.
                # БАЛАНС (энтропия): 100, НЕ 150 — при 150 ветка headline забирала 9/12 топа селектора
                # (мономания на одну линию); при 100 ядро проекта ведёт (7/12), линия headline держит
                # опору (3/12), но не доминирует. Подобрано симуляцией top-k по веткам.
                bid = _branch_of(hid, by_h)
                terminal = set(cfg.selector.terminal)
                BONUS = 100.0
                boosted = 0
                for n in data["nodes"]:
                    prev = float(n.pop("hl_boost", 0) or 0)   # снять прошлый headline-буст (не копим, переезжает)
                    if prev:
                        n["elo"] = float(n.get("elo") or cfg.elo.start) - prev
                    in_line = n["id"] in (hid, bid) or _branch_of(n["id"], by_h) == bid
                    if in_line and (n["id"] in (hid, bid) or n.get("status") not in terminal):
                        n["elo"] = float(n.get("elo") or cfg.elo.start) + BONUS
                        n["hl_boost"] = BONUS
                        boosted += 1
                from . import graph_io as _gio
                _gio.save_graph(data, cfg)
                logger.info("Elo-буст headline-ветки [%s]: идемпотентный +%d на %d узлов (прошлый снят)",
                            hid, int(BONUS), boosted)
            except Exception as e:
                logger.warning("Elo-буст headline не применён: %s", e)
    try:
        m = (re.search(r"DIRECTIVES:\s*(.+?)\nCORNERSTONE:", review, re.DOTALL)
             or re.search(r"DIRECTIVES:\s*(.+?)\nWEAK_NODES:", review, re.DOTALL))
        directives = m.group(1).strip() if m else "(критик не дал явных директив)"
        os.makedirs(cfg.workdir, exist_ok=True)
        feedback_text = (
            f"{hl_line}# ДИРЕКТИВЫ A*-КРИТИКА (балл {score}/10) — ДЕЛАЙ ЭТО, чтобы поднять балл:\n"
            f"{directives}\n\n# Полная рецензия (контекст):\n{review}\n")
        feedback_text = graph_hygiene.scrub_project_policy_text(feedback_text, cfg)
        open(os.path.join(cfg.workdir, "astar_feedback.md"), "w", encoding="utf-8").write(
            feedback_text + "\n")
    except (OSError, re.error):
        pass
    # КРАЕУГОЛЬНЫЙ эксперимент: критик может потребовать один длинный решающий прогон → supervisor
    # поставит его как приоритетный и скоординирует команду (мыслители отдыхают, пока он бежит).
    cornerstone = None
    cm = re.search(r"CORNERSTONE:\s*(.+)", review)
    if cm:
        spec = cm.group(1).splitlines()[0].strip()
        if spec and "нет" not in spec.lower()[:6]:
            if (str(getattr(cfg, "exp_kind", "") or "").strip().lower() == "llama"
                    and graph_hygiene.LEGACY_CHECKPOINT_TEXT_RE.search(spec)):
                logger.warning("policy-skip raw CORNERSTONE under exp_kind=llama: %s", spec[:160])
            else:
                cornerstone = spec
    title = f"{cfg.project_name}: финальная статья" + ("" if score >= min_score else " (черновик)")
    # Ось R: registry на ФАКТИЧЕСКИ предъявляемый текст (после ratchet, до сборки PDF) — см. W4.
    _finalize_submission_registry(cfg, tex, score)
    pdf = None if attempt_lost else compile_pdf(tex, cfg, name, title)
    logger.info("paper: score=%d/10 published=%s pdf=%s%s", score, score >= min_score, pdf,
                " [отчёт не издан: попытка слабее рэтчета]" if attempt_lost else "")
    # кнопка «Статья» в UI: положить в koi-structure/paper/ (только если опубликовано)
    if not attempt_lost and getattr(cfg, "store", "graph") == "koi" and score >= min_score:
        from . import koi_ui
        koi_ui.paper_to_koi(os.path.join(rd, name + ".tex"), pdf, cfg, score=score)
    # AIDE² FORK-ON-STALL: балл на плато K циклов подряд (не растёт над сохранённым) → сигналим
    # supervisor'у форкнуть фронтир под другим углом (evolve вне расписания). Файл-счётчик, порог 3.
    # НЕ трогает науку — только сигнал сменить стратегию (как AIDE²: застряла линия → новая арма).
    plateau = False
    if int(getattr(cfg, "fork_on_stall", 1) or 1):
        try:
            _pf = os.path.join(cfg.workdir, "paper_plateau.count")
            _prevc = 0
            if os.path.exists(_pf):
                _prevc = int((open(_pf).read().strip() or "0"))
            if score <= prev_draft_score:      # балл не вырос над лучшим → плато копится
                _c = _prevc + 1
                if _c >= 3:
                    plateau = True
                    logger.info("fork-on-stall: балл на плато %d/10 (%d циклов подряд) → сигнал форкнуть фронтир", score, _c)
                    _c = 0                      # сброс после сигнала
                open(_pf, "w").write(str(_c))
            else:
                open(_pf, "w").write("0")       # балл вырос → плато сброшено
        except (OSError, ValueError) as e:
            logger.warning("fork-on-stall счётчик упал: %s", e)
    return {"score": score, "published": (not attempt_lost) and score >= min_score, "pdf": pdf,
            "cornerstone": cornerstone, "plateau": plateau,
            "note": ("attempt_below_ratchet" if attempt_lost else None),
            "tex": None if attempt_lost else os.path.join(rd, name + ".tex"),
            "review": None if attempt_lost else os.path.join(rd, name + ".review.md")}


_SUCCESS_PROMPT = """\
Ты пишешь ОТЧЁТ ПО УСПЕШНЫМ КЕЙСАМ проекта: {domain}
Только ПОБЕДЫ (доказанное / подтверждённое на реальной сети). Для каждого: что доказано,
чем подтверждено (числа/CI), почему важно. БЕЗ «что рассматривалось» и без негативов —
это отдельный обзор. Связная проза по-русски, honesty>hype, без выдуманных чисел/цитат,
без тире-эм/точек-с-запятой. LaTeX-body (\\section{{}}/itemize), математика — $...$.

УСПЕШНЫЕ УЗЛЫ:
{context}
Выведи ТОЛЬКО LaTeX-body."""


def successes_report(data: Dict[str, Any], cfg: ARConfig, ts: str) -> Dict[str, Any]:
    rcall = engine_call(cfg.engines.reporter, cfg)
    ctx = cluster_context(data, cfg, statuses=_WINS)
    out = rcall(_SUCCESS_PROMPT.format(domain=cfg.domain, context=ctx[:14000]))
    i = out.find(r"\section")
    body = clean_latex(out[i:].strip() if i >= 0 else out.strip())
    pdf = compile_pdf(body, cfg, f"successes-{ts}", f"{cfg.project_name}: успешные кейсы")
    logger.info("successes report pdf=%s", pdf)
    return {"pdf": pdf}
