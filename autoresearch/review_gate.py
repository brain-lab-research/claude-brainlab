"""Фаза 2: ревью-гейт на git push (pre-receive).

Агент узла работает в ветке idea/<X>, пушит артефакты (derive.md, verify.py, ...).
На push серверный bare-repo запускает pre-receive хук → этот модуль: достаёт diff,
ЗАПУСКАЕТ изменённые *.py (численная проверка), и зовёт жёсткого Opus-гейткипера
(proof-логика + честность чисел + адверсариальная атака) → PASS/FAIL. FAIL → push
ОТКЛОНЯЕТСЯ (exit≠0) + фидбек в stderr; агент чинит и пушит снова. Git-нативная,
ПРИНУДИТЕЛЬНАЯ версия koi-report-review.

Установка bare-repo + хука: `ar gate-init --repo <path> --project <proj>`.
Ручной тест ревью на diff-файле: `ar gate <diff-file>`.
"""
from __future__ import annotations

import ast
import logging
import os
import re
import subprocess
import sys
import tempfile
from typing import Callable, List, Optional, Tuple

from .config import ARConfig
from .engines import engine_call, grep_tail
from . import exp_spec

logger = logging.getLogger(__name__)

DERIVE_PROMPT_LIMIT = 60000
OTHER_FILE_PROMPT_LIMIT = 18000
GATE_DIFF_PROMPT_LIMIT = 90000

# ТРОЙНЫЕ ВОРОТА: три независимых строгих критика. PASS только если ВСЕ три PASS.
_GATE_ROLES = {
    "proof": (
        "Ты ПРУФ-СУДЬЯ (как рецензент журнала). Оцени ТОЛЬКО логику доказательства: "
        "полнота (каждый шаг следует из предыдущих, нет пропусков/«схем»/«легко видеть»), "
        "корректность (нет циркулярности, кванторы и границы верны), все допущения явны, "
        "КАЖДЫЙ символ определён при первом вхождении (C, G, Σ и т.п. без расшифровки => FAIL), "
        "литеральная корректность названных частных случаев. Доказательство оборвано/неполно "
        "=> FAIL."),
    "code": (
        "Ты КОД-СУДЬЯ. Оцени численную проверку: код ЗАПУЩЕН (см. вывод), exit 0, числа "
        "РЕАЛЬНЫЕ, совпадают с текстом, тест проверяет где держится И где ломается, нет NaN/"
        "самоподтверждения (F из формулы теоремы и ей же проверяется => FAIL). Код упал/числа "
        "против текста => FAIL.\n"
        ">>> КОНТРАКТ ЭКСПЕРИМЕНТОВ: derive.md обязан содержать ровно один явный маркер: "
        "EXP_SPEC_REQUEST с JSON typed EXP_SPEC, если claim требует real-LLM проверки, или "
        "NO_EXP_NEEDED с причиной, если claim чисто алгебраический/аналитический. Если маркера "
        "нет — FAIL и требуй добавить контракт. Если есть валидный EXP_SPEC_REQUEST и toy-код "
        "не падает, вердикт должен быть EXPERIMENT, не PASS.\n"
        ">>> ГЛАВНОЕ ПРАВИЛО ЭКСПЕРИМЕНТА (применяй строго): если ЦЕНТРАЛЬНЫЙ claim узла "
        "касается ПОВЕДЕНИЯ РЕАЛЬНОЙ LLM — забывание/retention, сравнение оптимизаторов "
        "(Muon/SGD/AdamW), перенос pretrain→finetune, выбор чекпоинта, эффект на "
        "реальной сети — а verify.py это лишь toy-numpy/синтетика, то toy-проверки "
        "НЕДОСТАТОЧНО для подтверждения. В ЭТОМ СЛУЧАЕ выведи РОВНО GATE_VERDICT: EXPERIMENT "
        "и строку EXP_SPEC с JSON из allowlist kind: " + exp_spec.known_kinds() + ". "
        "(теория/логика ок, но claim про реальную сеть обязан быть проверен реальным прогоном). "
        "НЕ выдавай PASS эмпирическому claim'у только на основании toy-numpy. PASS оставляй "
        "для узлов, чей claim ЧИСТО алгебраический/аналитический (тождество, неравенство, "
        "свойство матрицы), где numpy — адекватная и полная проверка."),
    "critic": (
        "Ты СОСТЯЗАТЕЛЬНЫЙ КРИТИК узла + ХРАНИТЕЛЬ НАРРАТИВА. Калибровка: будь не мягче "
        "прежнего Sonnet/Opus-судьи. Не помогай автору «протащить» идею. При сомнении — FAIL.\n"
        "Узел — это КИРПИЧИК (лемма), НЕ вся статья: узкая честная КОРРЕКТНАЯ лемма может "
        "пройти, но только если она реально доказана, не дублирует старое, не создаёт overclaim "
        "и полезна для главной истории. A*-уровень собирается из совокупности на этапе отчёта.\n"
        ">>> ТРЕБОВАНИЯ РУКОВОДИТЕЛЯ (учитывай ВСЕГДА):\n{boss}\n<<<\n"
        "ЕДИНАЯ ИСТОРИЯ ПРОЕКТА (в неё собираем кирпичи):\n{narrative}\n\n"
        "УЖЕ СУЩЕСТВУЮЩИЕ УЗЛЫ (для проверки дубля):\n{node_map}\n\n"
        "PASS разрешён только если все условия выполнены: (a) claim точен и уже не шире "
        "доказанного/проверенного; (b) нет скрытого oracle/self-confirmation/mismatched verify; "
        "(c) узел закрывает конкретную дыру, усиливает headline или честно фиксирует negative; "
        "(d) feedback не содержит нерешённого blocker. Если хоть один пункт не доказан текстом "
        "или выводом кода — FAIL.\n"
        "FAIL ставь если: (1) ВХОД БЕССМЫСЛЕН или ДУБЛЬ уже доказанного/отвергнутого "
        "(см. список узлов) — нулевой вклад; (2) узел НЕ ВПИСЫВАЕТСЯ в историю и никак её не "
        "усиливает (не закрывает открытый вопрос, не связывает линии, не укрепляет тезис) — "
        "уводит в сторону; (3) OVERCLAIM (шире доказанного); (4) скрытое допущение/оракул/"
        "циркулярность/расхождение доказано-vs-проверено; (5) идея promising, но evidence "
        "ещё нет. НЕ ставь FAIL за узость/вспомогательность, если кирпич реально полезен. "
        "Overclaim → предложи сузить (путь к PASS). "
        "Оцени A*-ность 1-10 (информативно). В FEEDBACK укажи, как узел встаёт в историю. "
        "Добавь ПЕРЕД вердиктом строку: A_STAR: <1-10>"),
}

_GATE_PROMPT = """\
{role}
Проект: {domain}. Честность важнее хайпа: узкая доказуемая/негативная теорема лучше широкой с дырами.

КАЛИБРОВКА СТРОГОСТИ ДЛЯ CODEX-ONLY РЕЖИМА:
- Веди себя как старый Sonnet/Opus-гейт: default = REJECT, PASS только при явном evidence в тексте/коде.
- Не давай benefit of doubt. "Likely", "sketch", "needs more work", "not shown", "assuming" без доказательства = FAIL.
- FEEDBACK не должен быть "?". Назови конкретный blocker: missing definition/proof gap/self-confirming code/mismatched experiment/overclaim/duplicate.
- Если центральный empirical claim требует реальной LLM-проверки, а её нет, вердикт code-role обязан быть EXPERIMENT, не PASS.
- derive.md обязан содержать EXP_SPEC_REQUEST:{{...}} или NO_EXP_NEEDED:<причина>. Нет маркера
  => code-role FAIL с требованием добавить контракт.
- При GATE_VERDICT: EXPERIMENT добавь строку:
  EXP_SPEC: {{"kind":"<one of: {known_kinds}>","model":"<project-allowed model matching exp_kind>","claim":"<что проверяем>","success_metric":"<основная метрика>","confirm_rule":"EMP_VERDICT confirmed только если RESULT_JSON проверяет этот claim","env":{{}}}}

ИЗМЕНЕНИЯ (diff/файлы):
{diff}

ВЫВОД ЗАПУЩЕННОГО КОДА:
{run}

В КОНЦЕ выведи РОВНО:
FEEDBACK: <одна-две строки что чинить, или 'ok'>
GATE_VERDICT: PASS|FAIL|EXPERIMENT"""


def _has_no_exp_needed(text: str) -> bool:
    return bool(re.search(r"(?m)^\s*NO_EXP_NEEDED\s*:", text or ""))


def _has_exp_request(text: str) -> bool:
    return bool(re.search(r"(?m)^\s*EXP_SPEC_REQUEST\s*:", text or ""))


def _contract_markers(text: str) -> List[str]:
    return [line.strip() for line in (text or "").splitlines()
            if re.match(r"\s*(EXP_SPEC_REQUEST|NO_EXP_NEEDED)\s*:", line)]


def _strip_contract_marker_lines(text: str) -> str:
    return "\n".join(line for line in (text or "").splitlines()
                     if not re.match(r"\s*(EXP_SPEC_REQUEST|NO_EXP_NEEDED)\s*:", line))


def _excerpt(text: str, limit: int = 12000) -> str:
    """Bound LLM context while preserving the file tail where contracts live."""
    text = text or ""
    if len(text) <= limit:
        return text
    head = max(1, int(limit * 0.65))
    tail = max(1, limit - head)
    return (text[:head].rstrip()
            + "\n\n[... omitted middle for gate prompt ...]\n\n"
            + text[-tail:].lstrip())


def _contract_marker_note(text: str) -> str:
    markers = _contract_markers(text)
    if not markers:
        return ""
    return ("[EXPERIMENT_CONTRACT_VISIBLE_TO_JUDGES: exactly one marker found "
            "in the complete derive.md]\n" + markers[0])


def _inject_contract_note(diff_text: str, contract_text: str) -> str:
    note = _contract_marker_note(contract_text)
    if not note:
        return diff_text
    scrubbed = _strip_contract_marker_lines(diff_text)
    marker = "=== derive.md ==="
    pos = scrubbed.find(marker)
    if pos < 0:
        return note + "\n\n" + scrubbed
    start = pos + len(marker)
    if scrubbed[start:start + 1] == "\n":
        start += 1
    return scrubbed[:start] + note + "\n" + scrubbed[start:]


def _gate_files(old: str, new: str, gitdir: str) -> List[str]:
    files = list(_changed_files(old, new, gitdir))
    for f in ("derive.md", "verify.py"):
        if f not in files and _file_at(new, f, gitdir):
            files.append(f)
    return files


def _file_section(diff_text: str, filename: str) -> Optional[str]:
    marker = f"=== {filename} ===\n"
    start = diff_text.find(marker)
    if start < 0:
        return None
    start += len(marker)
    nxt = re.search(r"\n=== [^=\n]+ ===\n", diff_text[start:])
    end = start + nxt.start() if nxt else len(diff_text)
    return diff_text[start:end]


def _verify_sanity(verify_text: Optional[str]) -> Optional[str]:
    if verify_text is None:
        return None
    try:
        tree = ast.parse(verify_text or "", filename="verify.py")
    except SyntaxError as e:
        return f"[code] verify.py is not valid Python: {e.msg}"
    body = list(tree.body)
    if body and isinstance(body[0], ast.Expr):
        val = getattr(body[0], "value", None)
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            body = body[1:]
    executable = [
        st for st in body
        if not isinstance(st, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                              ast.AsyncFunctionDef, ast.ClassDef))
    ]
    if not executable:
        return (
            "[code] verify.py has no module-level executable verification. "
            "Comments/transcripts/imports/helper definitions are not a verifier."
        )
    return None


def review(diff_text: str, run_output: str, cfg: ARConfig,
           call: Optional[Callable[[str], str]] = None, node: str = "",
           narrative: str = "", node_map: str = "",
           contract_text: Optional[str] = None,
           verify_text: Optional[str] = None) -> Tuple[str, str]:
    """Тройные ворота: proof+code+critic. PASS ⟺ все три PASS.
    critic получает НАРРАТИВ проекта + карту узлов (судит вписываемость и дубль)."""
    contract_source = diff_text if contract_text is None else contract_text
    markers = _contract_markers(contract_source)
    if len(markers) != 1:
        return "fail", (
            f"[code] derive.md must contain exactly one experiment contract marker, "
            f"found {len(markers)}"
        )
    declared_exp_spec = None
    if _has_exp_request(contract_source):
        raw_spec = exp_spec.extract(contract_source)
        if raw_spec is None:
            return "fail", "[code] EXP_SPEC_REQUEST present but JSON object was not parsed"
        try:
            declared_exp_spec = exp_spec.canonicalize(
                raw_spec,
                context=f"node={node} diff={contract_source[:2000]}",
                exp_kind=getattr(cfg, "exp_kind", ""),
            )
        except ValueError as e:
            return "fail", f"[code] invalid EXP_SPEC_REQUEST: {e}"
    elif not _has_no_exp_needed(contract_source):
        return "fail", (
            "[code] missing experiment contract: add either "
            "EXP_SPEC_REQUEST:{...} for real-LLM claims or "
            "NO_EXP_NEEDED:<reason> for a purely algebraic/theoretical node"
        )
    else:
        required_kind = exp_spec.no_exp_needed_invalid_kind(
            contract_source,
            exp_kind=getattr(cfg, "exp_kind", ""),
        )
        if required_kind:
            return "fail", (
                "[code] NO_EXP_NEEDED is invalid for this real-LLM/checkpoint-selector "
                "claim. The same idea agent must add "
                f"EXP_SPEC_REQUEST with kind={required_kind!r}; experimenter only "
                "runs/monitors typed specs produced by the branch."
            )
    verify_fail = _verify_sanity(
        verify_text if verify_text is not None else _file_section(diff_text, "verify.py")
    )
    if verify_fail:
        return "fail", verify_fail

    engines = {"proof": cfg.engines.gate_proof, "code": cfg.engines.gate_code,
               "critic": cfg.engines.gate_critic}
    fails: List[str] = []
    astar = ""
    needs_exp = declared_exp_spec is not None
    needs_exp_spec = declared_exp_spec
    visible_diff = _inject_contract_note(diff_text, contract_source)
    for role_key, role_text in _GATE_ROLES.items():
        if role_key == "critic":
            from . import narrative as _nar
            role_text = role_text.format(narrative=(narrative or "(нарратив не задан)")[:3000],
                                         node_map=(node_map or "(карта пуста)")[:3000],
                                         boss=(_nar.boss(cfg) or "(не задано)")[:2000])
        rcall = call or engine_call(engines[role_key], cfg)
        out = rcall(_GATE_PROMPT.format(role=role_text, domain=cfg.domain,
                                        known_kinds=exp_spec.known_kinds(),
                                        diff=_excerpt(visible_diff, GATE_DIFF_PROMPT_LIMIT),
                                        run=(run_output or "(кода не было)")[:4000]))
        v = grep_tail(out, "GATE_VERDICT:").upper()
        ok = v.startswith("PASS")
        fb = grep_tail(out, "FEEDBACK:")
        if role_key == "code" and v.startswith("EXPERIMENT"):
            if declared_exp_spec is None:
                # Узел объявил NO_EXP_NEEDED и это прошло контракт-проверку выше (required_kind==""),
                # т.е. эксперимент для него НЕ требуется. Судья НЕ вправе молча переклассифицировать
                # его в EXPERIMENT: иначе чисто-теоретический узел (напр. scope-pivot) получает реальный
                # прогон → refuted → клинит pressure-механизм. Требуем сменить контракт ЯВНО.
                ok = False
                fb = ("code-role требует EXPERIMENT, но derive.md объявляет NO_EXP_NEEDED. Либо смени "
                      "контракт на EXP_SPEC_REQUEST:{...} с валидным typed kind, либо оставь "
                      "NO_EXP_NEEDED и сузь claim так, чтобы real-run не подразумевался.")
            else:
                needs_exp = True; ok = True  # теория ок, но нужен реальный LLM-эксперимент
                try:
                    needs_exp_spec = exp_spec.canonicalize(
                        exp_spec.extract(out),
                        context=f"node={node} diff={contract_source[:2000]} feedback={fb}",
                        exp_kind=getattr(cfg, "exp_kind", ""),
                    )
                except ValueError as e:
                    ok = False
                    fb = f"invalid EXP_SPEC: {e}"
        if role_key == "critic":
            astar = grep_tail(out, "A_STAR:")
            if ok:
                try:
                    int(str(astar).strip().split("/")[0])
                except (TypeError, ValueError):
                    ok = False
                    fb = "critic did not provide numeric A_STAR; strict gate rejects malformed verdict"
        if not ok:
            fails.append(f"[{role_key}] {fb}")
    tag = f" A*={astar}/10" if astar and astar != "?" else ""
    if fails:
        return "fail", " | ".join(fails) + tag
    if needs_exp:
        if getattr(cfg, "empirical_in_loop", False):
            spec_s = exp_spec.dump(needs_exp_spec or exp_spec.canonicalize(
                None, context=f"node={node} diff={contract_source[:2000]}",
                exp_kind=getattr(cfg, "exp_kind", "")))
            return "experiment", f"EXP_SPEC: {spec_s} | теория прошла toy-гейт, нужен real-LLM прогон" + tag
        return "fail", "[code] реальный LLM-эксперимент обязателен, но empirical_in_loop=false; PASS запрещён" + tag
    return "pass", f"ok (proof+code+critic все PASS){tag}"


# ---------------- git plumbing (вызывается из pre-receive хука) ----------------

def _git(args: List[str], cwd: Optional[str] = None) -> str:
    r = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True, errors="replace")
    return r.stdout


def _changed_files(old: str, new: str, gitdir: str) -> List[str]:
    base = "4b825dc642cb6eb9a060e54bf8d69288fbee4904" if set(old) == {"0"} else old  # empty tree
    out = _git(["--git-dir", gitdir, "diff", "--name-only", base, new])
    return [f for f in out.splitlines() if f.strip()]


def _file_at(rev: str, path: str, gitdir: str) -> str:
    return _git(["--git-dir", gitdir, "show", f"{rev}:{path}"])


def _run_py(files: List[str], rev: str, gitdir: str, timeout: int = 120) -> str:
    """Выписать изменённые *.py во временную папку и запустить (численная проверка)."""
    pys = [f for f in files if f.endswith(".py")]
    if not pys:
        return ""
    chunks = []
    with tempfile.TemporaryDirectory() as td:
        for p in pys:
            dst = os.path.join(td, os.path.basename(p))
            open(dst, "w", encoding="utf-8").write(_file_at(rev, p, gitdir))
        for p in pys:
            dst = os.path.join(td, os.path.basename(p))
            try:
                r = subprocess.run(["python3", dst], cwd=td, capture_output=True,
                                   text=True, errors="replace", timeout=timeout)
                tail = (r.stdout + "\n" + r.stderr).strip()[-1500:]
                chunks.append(f"# {p} (exit {r.returncode})\n{tail}")
            except subprocess.TimeoutExpired:
                chunks.append(f"# {p} TIMEOUT >{timeout}s")
            except OSError as e:
                chunks.append(f"# {p} run error: {e}")
    return "\n\n".join(chunks)


def hook_main() -> int:
    """pre-receive: читает stdin '<old> <new> <ref>'; гейтит ветки idea/*."""
    from . import config as cfgmod
    gitdir = os.environ.get("GIT_DIR") or os.getcwd()
    try:
        cfg = cfgmod.load(os.environ.get("AUTORESEARCH_PROJECT"))
    except Exception as e:
        sys.stderr.write(f"[gate] нет конфига проекта: {e}\n")
        # Ветки идей БЕЗ конфига не пропускаем: return 0 принимал idea/*-пуши ВООБЩЕ БЕЗ ревью
        # (сломанный конфиг = тихо выключенный гейт). Прочие ветки не блокируем, как раньше.
        rc = 0
        for line in sys.stdin:
            parts = line.split()
            if len(parts) == 3 and parts[2].startswith("refs/heads/idea/"):
                sys.stderr.write(f"[gate {parts[2].rsplit('/', 1)[-1]}] REJECT — гейт не сконфигурен, "
                                 "ревью невозможно; почини конфиг (AUTORESEARCH_PROJECT/autoresearch.json)\n")
                rc = 1
        return rc
    rc = 0
    for line in sys.stdin:
        parts = line.split()
        if len(parts) != 3:
            continue
        old, new, ref = parts
        if not ref.startswith("refs/heads/idea/"):
            continue  # гейтим только ветки идей
        node = ref.rsplit("/", 1)[-1]
        files = _gate_files(old, new, gitdir)
        full_chunks = []
        prompt_chunks = []
        derive_contract = ""
        for f in files:
            content = _file_at(new, f, gitdir)
            if f == "derive.md":
                derive_contract = content
                limit = DERIVE_PROMPT_LIMIT
            else:
                limit = OTHER_FILE_PROMPT_LIMIT
            full_chunks.append(f"=== {f} ===\n{content}")
            prompt_chunks.append(f"=== {f} ===\n{_excerpt(content, limit)}")
        full_diff = "\n\n".join(full_chunks) or "(пусто)"
        diff = "\n\n".join(prompt_chunks) or "(пусто)"
        run = _run_py(files, new, gitdir)
        narrative, node_map = "", ""
        try:
            from . import graph_io as gio, narrative as nar
            data = gio.load_graph(cfg)
            narrative = nar.ensure(cfg, data)
            node_map = nar.node_map(data)
        except Exception:
            pass
        status, fb = review(
            diff, run, cfg, node=node, narrative=narrative,
            node_map=node_map, contract_text=derive_contract or full_diff,
            verify_text=_file_section(full_diff, "verify.py"),
        )
        # pass → accept; fail/experiment → push отклонён (но experiment = маркер «ставь эксп»)
        label = {"pass": "PASS", "fail": "REJECT", "experiment": "EXPERIMENT"}.get(status, "REJECT")
        sys.stderr.write(f"\n[gate idea/{node}] {label} — {fb}\n")
        if status != "pass":
            rc = 1
    return rc


# ---------------- установка bare-repo + хука ----------------

_HOOK = """#!/usr/bin/env bash
# токен headless-claude (CLAUDE_CODE_OAUTH_TOKEN) + прочие секреты — из файла, не в коде
[ -f "$HOME/.autoresearch_env" ] && . "$HOME/.autoresearch_env"
export PYTHONPATH="{claude}:$PYTHONPATH"
export AUTORESEARCH_PROJECT="{project}"
export GIT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
exec {python} -m autoresearch.review_gate hook
"""


def gate_init(repo: str, project: str, python: str = "python3",
              claude_home: str = "~/.claude") -> str:
    repo = os.path.abspath(os.path.expanduser(repo))
    if not os.path.exists(os.path.join(repo, "HEAD")):
        subprocess.run(["git", "init", "--bare", repo], check=True, capture_output=True)
    hook = os.path.join(repo, "hooks", "pre-receive")
    os.makedirs(os.path.dirname(hook), exist_ok=True)
    open(hook, "w").write(_HOOK.format(
        claude=os.path.expanduser(claude_home),
        project=os.path.abspath(os.path.expanduser(project)), python=python))
    os.chmod(hook, 0o755)
    logger.info("gate bare-repo + pre-receive: %s", repo)
    return repo


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "hook":
        sys.exit(hook_main())
