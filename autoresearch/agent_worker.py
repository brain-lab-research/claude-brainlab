"""Идейный агент-член команды: ведёт один узел в своей git-ветке idea/<X>.

НЕ эфемерный сабагент: процесс-воркер берёт узел от supervisor, делает работу
(Opus пишет derive.md + verify.py из памяти-дерева), коммитит в ветку idea/<X>,
ПУШИТ в bare-repo гейта → срабатывает pre-receive ревью-гейт (proof/code/critic) →
accept/reject. На reject читает фидбек, исправляет, пушит снова (bounded). Может звать
пиров через inbox. На accept — обновляет статус узла в store.
"""
from __future__ import annotations

import logging
import json
import os
import re
import subprocess
import tempfile
from typing import Any, Callable, Dict, List, Optional

from datetime import datetime

from .config import ARConfig
from .engines import engine_call
from . import graph_hygiene, inbox, agents_status, node_memory, exp_spec, review_gate, litreview, engines


def _codex_writer(cfg: ARConfig) -> bool:
    """derive/verify пишет codex → используем файл-режим (пишет в файл, не в обрезаемое сообщение)."""
    return cfg.engines.writer == "codex" and engines.codex_available()

logger = logging.getLogger(__name__)

# Маркеры чистой теоремы: если такой узел ПРОШЁЛ тройной гейт (proof+code+critic) и это НЕ
# typed-эксперимент, он ДОКАЗАН → должен стать proven (headline-материал), а не висеть вечно
# conditional (иначе не входит в статью как headline_win — был баг: achievability застревал).
_THEORY_PROMOTE_MARKERS = (
    "theorem", "impossib", "lemma", "no-free", "no_free", "identifiab", "achievab",
    "upper bound", "lower bound", "теорем", "невозможн", "лемм", "no_exp_needed",
)


def _looks_like_theory(n: Dict[str, Any]) -> bool:
    blob = (str(n.get("short", "")) + " " + str(n.get("idea", ""))).lower()
    return any(m in blob for m in _THEORY_PROMOTE_MARKERS)


# derive и verify пишутся РАЗНЫМИ вызовами — чтобы доказательству достался ВЕСЬ бюджет вывода и
# оно не обрывалось на полуслове (раньше оба блока в одном ответе → длинный proof упирался в лимит).
_DERIVE_PROMPT = """\
Ты ИДЕЙНЫЙ АГЕНТ-член команды по узлу [{nid}]. Проект: {domain}
ПАМЯТЬ (дерево/предки/тупики/фронтир):
{memory}

УЗЕЛ: {short}
идея: {idea}
открытые вопросы: {open}
план: {plan}
{related_work}
{feedback}{prev}
ЗАДАЧА: ОДНА точная теорема по узлу (компактно, не серия T1..T5) + ПОЛНОЕ доказательство
из первых принципов, ДО КОНЦА (заверши QED/выводом — НЕ обрывайся на полуслове). Лучше УЗКАЯ
доказуемая или негативная теорема, чем широкая с пропусками.
СТРОГОСТЬ (частая причина реджекта — пробел, НЕ обрыв текста): для ЛЮБОГО неравенства/оценки/
оптимальности докажи ОБЕ стороны — и нижнюю, и ВЕРХНЮЮ оценку; для любой LMO/дуальности/KKT/
сертификата докажи FEASIBILITY (допустимость) и достижимость явно. Не пиши QED, пока каждая
заявленная оценка/граница не имеет своего доказательства в тексте. Если верхнюю оценку доказать
нельзя — честно сузь утверждение до того, что доказуемо, а не пропускай шаг.
КОНТРАКТ ЭКСПЕРИМЕНТОВ (обязателен в конце derive.md после доказательства):
- Если узел делает или поддерживает claim про реальную LLM/finetune/forgetting/retention/
  оптимизаторы/checkpoint choice/wall-clock/baselines — добавь строку
	  `EXP_SPEC_REQUEST: {{...}}` с JSON EXP_SPEC. Минимальные поля: kind, model, claim, success_metric, confirm_rule, env.
  ПРЕДПОЧИТАЙ подходящий готовый kind: {known_kinds}. Но ты НЕ ограничен списком — если НИ ОДИН не
  подходит под нужный тебе эксперимент, задай СВОЙ описательный kind (напр. "wallclock_scaling_ladder")
  и подробный claim/success_metric/confirm_rule: экспериментатор САМ напишет самодостаточный llama-скрипт
  под него (такой kind станет custom_llama, модель llama, размер из exp_env ~60M). Никогда не выбирай
  NO_EXP_NEEDED только потому, что «нет готового kind» — если claim эмпирический, опиши эксперимент.
- Модель llama (проект exp_kind=llama; pythia запрещена). Для готовых типизированных kind бери
  llama-compatible; для своего kind просто model="llama" — скрипт напишется на llama_exp_kit.
- МАСШТАБ: проект задаёт дефолтный размер модели/шаги (exp_env). Если claim становится
  убедительным ТОЛЬКО на бо́льшем масштабе (напр. ревьюер требует крупные прогоны, эффект может
  проявиться лишь при большей модели) — укажи в env бо́льшие N_EMBD/N_LAYER/PRE_STEPS/FT_STEPS:
  они ПЕРЕОПРЕДЕЛЯЮТ дефолт проекта. Обоснуй масштаб в claim; не раздувай без нужды (GPU-время).
- Если узел чисто алгебраический/аналитический и реальный LLM-прогон для него не нужен — добавь
  `NO_EXP_NEEDED: <одна конкретная причина>`.
Без одного из этих двух маркеров code-судья отклонит push и отправит тебя доделывать.
ОБЗОР ЛИТЕРАТУРЫ: связанные работы уже собраны библиотекарем (блок СВЯЗАННЫЕ РАБОТЫ выше). Если
их НЕДОСТАТОЧНО или подозреваешь, что твоя идея уже где-то сделана — добавь в derive строку
`RW_SEARCH: <конкретный запрос>` (одну). Будет запущен точечный веб+библиотека поиск, а результат
допишется в related_work узла (увидят все агенты). Не злоупотребляй — только при реальной нехватке.
ЦЕЛЬ КАЧЕСТВА: результат должен УСИЛИВАТЬ единую историю проекта и по возможности нести
ПРАКТИЧЕСКИЙ смысл (когда остановить претрейн / какой оптимизатор даёт лучшие финальные веса /
какой чекпоинт брать). Каждый символ определи при первом вхождении (никаких C, G, Σ без
расшифровки). Не плоди оторванные мелкие факты. honesty>hype; устрани каждое замечание ревьюера.
У тебя ВЕСЬ бюджет вывода на доказательство — не экономь, но и не лей воду; доведи до конца.
Выведи РОВНО ОДИН блок:
```markdown
<derive.md: формулировка одной теоремы + все допущения явно + ПОЛНОЕ доказательство до QED>
```"""

_VERIFY_PROMPT = """\
Ты ИДЕЙНЫЙ АГЕНТ. Проект: {domain}. По узлу [{nid}]: {short}
Вот уже написанная теорема и доказательство (derive.md):
{derive}
{prevv}
ЗАДАЧА: напиши verify.py — самодостаточный numpy-скрипт, проверяющий ЭТУ теорему. Выведи РОВНО
ОДИН блок:
```python
<verify.py: ЗАПУСКАЕТСЯ без ошибок (exit 0). Проверяет, где теорема держится И где должна
ломаться; много сидов; печатает РЕАЛЬНЫЕ числа; assert ТОЛЬКО на том, что теорема реально
гарантирует (избегай NaN/деления на ноль в вырожденных κ≈1). Числа в коде и в тексте СОВПАДАЮТ.>
```
Внутри блока должен быть только валидный Python-код без prose/placeholders. Код обязан иметь
module-level executable verification: например, вычисления/печать/assert в `if __name__ == "__main__":`
или на верхнем уровне. Одних imports, comments, helper functions/classes недостаточно.
Перед ответом мысленно проверь, что `ast.parse(verify.py)` проходит.
Выведи РОВНО один fenced python block и ничего после него."""


_CONTINUE_PROMPT = """\
	Это ПРОДОЛЖЕНИЕ незаконченного derive.md по узлу [{nid}]: {short}. Текст оборвался по длине.
	ПРОДОЛЖИ строго с места обрыва и доведи доказательство до конца (QED). НЕ повторяй уже написанное,
	НЕ начинай заново.
	ВАЖНО: после QED в конце продолжения обязательно добавь ровно один экспериментальный контракт:
	`EXP_SPEC_REQUEST: {{...}}` с JSON typed EXP_SPEC allowlist, если claim требует real-LLM проверки,
	или `NO_EXP_NEEDED: <одна конкретная причина>`, если узел чисто алгебраический/аналитический.
		Известные EXP_SPEC kind: {known_kinds}. Если project exp_kind=llama, используй только
		llama-compatible typed harness/target. Без этого code-судья снова отклонит push.
	Выведи ТОЛЬКО продолжение в ОДНОМ блоке:
	```markdown
	<только продолжение, с того символа, где оборвалось, до QED, затем EXP_SPEC_REQUEST или NO_EXP_NEEDED>
	```
	КОНЕЦ уже написанного (продолжай отсюда):
	{tail}"""


def _md_truncated(out: str) -> bool:
    """Доказательство оборвалось (codex не дописал) → нужно продолжение до QED.
    Два случая:
    (1) есть ```markdown, но он НЕ закрыт → обрыв внутри фенса (старый случай);
    (2) фенса НЕТ вообще (codex пишет пруф без обёртки) — тогда судим по маркеру завершения:
        нет QED/∎/вывода в конце → пруф не доведён → обрыв. Без этого continuation-loop не
        срабатывал для бесфенсовых выводов и обрезанные пруфы летели в гейт («обрывается»)."""
    if "```markdown" in out:
        return not re.search(r"```markdown\s*\n.*?\n```", out, re.DOTALL)
    # обрезанные пруфы НЕ доходят до маркера завершения — у них его НЕТ НИГДЕ (codex обрезал до QED);
    # завершённые содержат QED/∎ где-то (даже если в конце идёт обсуждение). Проверка по всему выводу
    # чисто разделяет их (проверено на реальных derive). re.search по ASCII-вариантам + символам.
    return not re.search(r"QED|∎|◻|□|\bq\.?e\.?d\.?\b|ч\.?\s*т\.?\s*д", out, re.IGNORECASE)


def _extract(out: str, lang: str) -> str:
    # GREEDY до ПОСЛЕДНЕГО ``` — внутри derive бывают ВЛОЖЕННЫЕ ```-блоки (LaTeX/код/пример);
    # non-greedy обрезал бы доказательство на первом вложенном фенсе → гейт видел «оборвано на
    # полуслове» (это был НЕ реальный обрыв по длине, а мис-экстракция). Greedy берёт всё тело.
    m = re.search(r"```" + lang + r"\s*\n(.*)\n```", out, re.DOTALL)
    if m:
        return m.group(1).strip()
    # незакрытый (реальный обрыв по длине) — берём от ```lang до конца, чтобы не терять работу
    m = re.search(r"```" + lang + r"\s*\n(.*)$", out, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def _has_experiment_contract(text: str) -> bool:
    # РОВНО один маркер — как на воротах (review_gate требует exactly one; локальный чек «>=1»
    # пропускал derive с ДВУМЯ маркерами, и раунд гарантированно сгорал на пуше: «found 2»).
    n = len(re.findall(r"(?m)^\s*EXP_SPEC_REQUEST\s*:", text or "")) + \
        len(re.findall(r"(?m)^\s*NO_EXP_NEEDED\s*:", text or ""))
    return n == 1


def _git(args, cwd=None, check=False):
    return subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True,
                          errors="replace", check=check)


def _produce(node: Dict[str, Any], memory: str, cfg: ARConfig,
             call: Callable[[str], str], feedback: str = "",
             prev: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    fb = f"\nФИДБЕК РЕВЬЮЕРА (исправь):\n{feedback}\n" if feedback else ""
    prev = prev or {}
    pd = (prev.get("derive.md") or "").strip()
    # ПАМЯТЬ: прошлый derive агента → продолжить/дописать (если оборвался) / исправить, НЕ с нуля
    prevblock = ("\nТВОЙ ПРОШЛЫЙ derive.md — НЕ начинай с нуля: ПРОДОЛЖИ/ДОВЕДИ до QED (если он "
                 "оборвался на полуслове — допиши с места обрыва) и ИСПРАВЬ по фидбеку:\n" + pd[:12000]
                 + "\n") if pd else ""
    # 1) derive.md — ОТДЕЛЬНЫЙ вызов: весь бюджет вывода на доказательство (нет обрыва).
    # codex → файл-режим (пишет пруф ПРЯМО В ФАЙЛ, не в обрезаемое сообщение → длинное доказательство целиком).
    d_prompt = _DERIVE_PROMPT.format(
        nid=node["id"], domain=cfg.domain, memory=memory[:9000], short=node.get("short", ""),
        idea=node.get("idea", ""), open=node.get("open", ""), plan=node.get("plan", ""),
        related_work=litreview.rw_for_prompt(node),
        feedback=fb, prev=prevblock, known_kinds=exp_spec.known_kinds())
    if _codex_writer(cfg):
        d_out = engines.codex_write_artifact(d_prompt, cfg, "_derive_wip.md")
    else:
        d_out = call(d_prompt)
    derive = _extract(d_out, "markdown") or d_out.strip()
    # БЕЗ ЛИМИТА ДЛИНЫ: если derive оборвался по длине — дописываем с места обрыва до закрытия блока
    # (≤4 продолжений), чтобы гейт всегда видел ПОЛНОЕ доказательство до QED, а не огрызок.
    cont_tries = 0
    while _md_truncated(d_out) and cont_tries < 4:
        d_out = call(_CONTINUE_PROMPT.format(
            nid=node["id"], short=node.get("short", ""), tail=derive[-3000:],
            known_kinds=exp_spec.known_kinds()))
        more = _extract(d_out, "markdown") or d_out.strip()
        if not more.strip():
            break
        derive = derive.rstrip() + "\n" + more.lstrip()
        cont_tries += 1
    pv = (prev.get("verify.py") or "").strip()
    prevv = ("\nТВОЙ ПРОШЛЫЙ verify.py (исправь/дополни, не с нуля):\n" + pv[:6000] + "\n") if pv else ""
    # 2) verify.py — ОТДЕЛЬНЫЙ вызов, видит готовый derive. codex → файл-режим (без обрезки кода).
    v_prompt = _VERIFY_PROMPT.format(
        nid=node["id"], domain=cfg.domain, short=node.get("short", ""),
        # [:30000] вместо [:9000]: у длинной теоремы финальные оценки/константы живут в хвосте derive —
        # verify-писатель их не видел и проверял не то, что доказано (гейт ловил рассинхрон → сгоревшие раунды).
        derive=derive[:30000], prevv=prevv)
    if _codex_writer(cfg):
        v_out = engines.codex_write_artifact(v_prompt, cfg, "_verify_wip.py")
    else:
        v_out = call(v_prompt)
    v_body = _extract(v_out, "python")
    if not v_body.strip() and v_out.strip():
        # файл-режим: codex пишет СЫРОЙ Python в артефакт-файл (без markdown-фенса — это .py!),
        # _extract без фенса возвращал "" → verify.py пустой → LOCAL_VERIFY_CHECK «no module-level
        # executable» → сгоревшие раунды. derive имеет фолбэк `or d_out.strip()` (стр.195), verify
        # не имел. Принимаем сырой контент, только если он реально парсится как Python.
        import ast as _ast
        try:
            _ast.parse(v_out)
            v_body = v_out.strip()
        except SyntaxError:
            pass
    return {"derive.md": derive, "verify.py": v_body}


def _launch_experiment(node, derive, cfg, _st, spec=None):
    """АСИНХРОННО: код-гейт потребовал реальный LLM-эксперимент → ставим прогон на GPU и
    СРАЗУ освобождаем идейный слот (не блокируем). Мониторит и подбирает вердикт supervisor.poll
    каждую итерацию → пока эксп считается, supervisor берёт НОВЫЙ узел => всегда 2 идейных агента."""
    from . import empirical
    rec = empirical.launch(node, derive, cfg, spec=spec)  # пишет карточку сам
    waiting = rec.get("status") == "waiting"
    blocked = rec.get("status") == "blocked_no_spec"
    if blocked:
        _st("failed", f"эксперимент не запущен: {rec.get('error','нет валидного EXP_SPEC')}",
            engine="codex→GPU", role="экспериментатор (GPU, typed router)", log=True)
        return rec
    _st("waiting_gpu" if waiting else "experiment",
        ("нет свободной карты в пуле — встал в очередь на GPU (поставлю как освободится)" if waiting
         else f"запущен real-LLM эксп [{rec.get('kind','?')}] на {rec.get('server')} cuda:{rec.get('gpu')}; "
              "слот освобождён, мониторит supervisor"),
        engine="codex→GPU", role="экспериментатор (GPU, async typed)", log=True)
    return rec


def work_node(node: Dict[str, Any], memory: str, cfg: ARConfig, gate_repo: str,
              max_rounds: int = 2, call: Optional[Callable[[str], str]] = None,
              lineage_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """Полный проход агента по узлу: produce → commit idea/<node> → push в гейт → revise."""
    call = call or engine_call(cfg.engines.writer, cfg)  # идейный агент = codex (как в сетапе)
    nid = node["id"]; branch = f"idea/{nid}"
    lineage_ids = lineage_ids or [nid]
    tree_branch = node_memory.tree_ref(lineage_ids)
    aname = f"agent-{nid}"
    short = node.get("short", "")
    weng = cfg.engines.writer  # codex
    post_exp_writeup = graph_hygiene.needs_writeup_priority(node)
    base_role = ("экспериментатор (writeup typed EXP_SPEC)"
                 if post_exp_writeup else "идейный агент (теория+эксперимент)")
    def _st(stage, detail="", engine=weng, role="", astar="", log=False):
        try:
            agents_status.update(cfg, aname, node=nid, short=short, stage=stage,
                                 detail=detail, engine=engine, role=role or base_role, astar=astar)
            if log:
                agents_status.log_event(cfg, aname, f"{stage} · {short}"
                                        + (f" A*={astar}/10" if astar else "") + f" — {detail[:80]}")
            agents_status.write_snapshot(cfg)
        except Exception:
            pass
    def _astar(fb):
        import re as _re
        m = _re.search(r"A\*?=\s*(\d+)\s*/?\s*10", fb or "")
        return m.group(1) if m else ""
    res = {"node": nid, "branch": branch, "tree_branch": tree_branch,
           "pushed": False, "gate_passed": False,
           "rounds": 0, "feedback": ""}
    _st("thinking", f"[{weng}] читаю память дерева (предки, тупики, фронтир), формулирую ОДНУ теорему по узлу «{short}»")
    with tempfile.TemporaryDirectory() as work:
        r = _git(["clone", "-q", gate_repo, work])
        if r.returncode != 0 and "empty" not in (r.stderr or ""):
            res["error"] = f"clone failed: {r.stderr[:200]}"; return res
        _git(["config", "user.email", "agent@autoresearch"], work)
        _git(["config", "user.name", f"agent-{nid}"], work)
        # если у узла УЖЕ есть ветка (прошлые пуши) — встаём на неё, чтобы прошлый derive.md/verify.py
        # оказались на диске (иначе -B сбрасывает на HEAD клона и память теряется).
        if _git(["rev-parse", "--verify", "-q", f"origin/{branch}"], work).returncode == 0:
            _git(["checkout", "-q", "-B", branch, f"origin/{branch}"], work)
        else:
            _git(["checkout", "-q", "-B", branch], work)
        # ПАМЯТЬ АГЕНТА: его прошлая работа уже закоммичена в ветку idea/<nid> — читаем с диска, чтобы
        # ПРОДОЛЖИТЬ/ИСПРАВИТЬ (в т.ч. дописать оборванное доказательство), а не писать с нуля.
        prev_files: Dict[str, str] = {}
        for fn in ("derive.md", "verify.py"):
            p = os.path.join(work, fn)
            if os.path.exists(p):
                try:
                    prev_files[fn] = open(p, encoding="utf-8").read()
                except OSError:
                    pass
        feedback = ""
        for rnd in range(max_rounds):
            res["rounds"] = rnd + 1
            if post_exp_writeup:
                detail = (f"[{weng}] вписываю подтверждённый typed EXP_SPEC в derive.md "
                          f"и синхронизирую verify.py — writeup round {rnd+1}")
            else:
                detail = (f"[{weng}] пишу derive.md (теорема+доказательство) и verify.py "
                          f"(numpy-проверка) — раунд {rnd+1}")
            detail += (f"; правлю по фидбеку: {feedback[:120]}" if feedback else "")
            detail += (" (продолжаю прошлый derive)" if prev_files.get("derive.md") else "")
            _st("writing", detail, log=True)
            files = _produce(node, memory, cfg, call, feedback, prev=prev_files)
            prev_files = files  # следующий раунд продолжает с этого
            # агент САМ просит доп-поиск литературы (маркер RW_SEARCH в derive) → точечный веб+библиотека,
            # результат дописывается в related_work узла (персист в supervisor под glock). Один раз на узел.
            try:
                _q = litreview.extract_rw_search(files.get("derive.md", ""))
                if _q and not res.get("rw_append"):
                    _st("lit_review", f"агент запросил доп-поиск литературы: {_q[:60]}",
                        engine=cfg.engines.librarian, role="идейный агент → доп-RW")
                    res["rw_append"] = litreview.make_related_work(cfg, node, query=_q)
            except Exception:
                pass
            if not _has_experiment_contract(files.get("derive.md", "")):
                feedback = ("LOCAL_CONTRACT_CHECK: derive.md must contain EXACTLY ONE contract "
                            "marker: either EXP_SPEC_REQUEST:{...} for real-LLM claims or "
                            "NO_EXP_NEEDED:<reason> for a purely algebraic/theoretical node "
                            "(zero markers = missing; two+ markers = duplicate, remove extras)")
                res["feedback"] = feedback
                res["local_contract_failed"] = True
                _st("rejected", "локально не пушу: нет EXP_SPEC_REQUEST/NO_EXP_NEEDED; даю агенту ещё раунд",
                    engine=weng, log=True)
                continue
            verify_fail = review_gate._verify_sanity(files.get("verify.py", ""))
            if verify_fail:
                feedback = "LOCAL_VERIFY_CHECK: " + verify_fail
                res["feedback"] = feedback
                res["local_verify_failed"] = True
                rec = node_memory.record_rejected(cfg, node, lineage_ids, branch, rnd + 1,
                                                  files, feedback, feedback)
                res["last_rejected_attempt"] = rec.get("attempt_id")
                res["last_rejected_raw"] = rec.get("raw_dir")
                _st("rejected", f"локально не пушу: verify.py не проходит preflight: {verify_fail}; даю агенту ещё раунд",
                    engine=weng, log=True)
                continue
            for fn, content in files.items():
                if content:
                    open(os.path.join(work, fn), "w", encoding="utf-8").write(content)
            with open(os.path.join(work, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(node_memory.branch_meta(node, branch, lineage_ids, rnd + 1),
                          f, ensure_ascii=False, indent=2, sort_keys=True)
                f.write("\n")
            _git(["add", "-A"], work)
            commit_res = _git(["commit", "-qm", f"{nid}: round {rnd+1}"], work)
            if commit_res.returncode != 0:
                raw = ((commit_res.stdout or "") + (commit_res.stderr or "")).strip()
                feedback = (
                    "LOCAL_GIT_CHECK: git commit failed before gate ran; "
                    f"worker must not record PASS without a commit. {raw[:500]}"
                )
                res["feedback"] = feedback
                res["local_git_failed"] = True
                rec = node_memory.record_rejected(cfg, node, lineage_ids, branch, rnd + 1,
                                                  files, feedback, raw)
                res["last_rejected_attempt"] = rec.get("attempt_id")
                res["last_rejected_raw"] = rec.get("raw_dir")
                _st("rejected", feedback[:160], engine=weng, log=True)
                continue
            _st("gate", f"пуш idea/{nid} → тройной гейт судит: proof-судья "
                        f"[{cfg.engines.gate_proof}] логику, code-судья [{cfg.engines.gate_code}] "
                        f"запускает verify.py, critic [{cfg.engines.gate_critic}] атакует",
                engine=f"{cfg.engines.gate_proof}+{cfg.engines.gate_code}+{cfg.engines.gate_critic}",
                role="у ревьюеров (тройной A*-гейт)")
            push = _git(["push", "origin", branch], work)
            out = (push.stdout or "") + (push.stderr or "")
            res["pushed"] = push.returncode == 0
            gate_fb = ""
            m = re.search(r"\[gate [^\]]+\][^\n]*", out)
            if m:
                gate_fb = m.group(0)
            # код-гейт решил: нужен реальный LLM-эксперимент (не numpy) → запускаем АСИНХРОННО и
            # выходим (освобождаем слот). Вердикт подберёт supervisor.poll, тогда узел до-обработается.
            # Матчим ТОЧНЫЙ label хука `[gate idea/X] EXPERIMENT — …`: голое `"EXPERIMENT" in out`
            # ложно срабатывало на REJECT-фидбек, содержащий слово EXPERIMENT (напр. «code-судья
            # требует EXPERIMENT, но derive объявляет NO_EXP_NEEDED») → GPU-эксп на отклонённом узле.
            if re.search(r"\[gate [^\]]+\]\s*EXPERIMENT", out):
                try:
                    spec = exp_spec.canonicalize(
                        exp_spec.extract(out),
                        context=f"node={nid} short={short} idea={node.get('idea','')} gate={out[:2000]}",
                        exp_kind=getattr(cfg, "exp_kind", ""),
                    )
                except ValueError as e:
                    feedback = f"{gate_fb} | invalid EXP_SPEC: {e}"
                    res["feedback"] = feedback
                    _st("rejected", feedback[:160], engine=weng, log=True)
                    continue
                rec = _launch_experiment(node, files.get("derive.md", ""), cfg, _st, spec=spec)
                if rec.get("status") == "blocked_no_spec":
                    feedback = f"{gate_fb} | {rec.get('error','typed router refused EXP_SPEC')}"
                    res["feedback"] = feedback
                    # EXP_SPEC router отверг скрипт (напр. impossibility 2-world честно не строится
                    # автогеном) → помечаем res, чтобы apply_result СЧИТАЛ router-блоки на узле и после
                    # порога запарковал эмпирическую линию (policy_blocked_exp). Иначе тот же узел
                    # переизбирается вечно и жжёт слоты экспериментатора (был трэш 52 отказа).
                    res["exp_router_blocked"] = True
                    continue
                res["experiment_pending"] = True
                res["experiment_status"] = rec.get("status")
                res["exp_spec"] = spec
                res["gate_passed"] = False
                return res
            if push.returncode != 0 and "PASS" in gate_fb and "REJECT" not in gate_fb:
                feedback = (
                    "LOCAL_GIT_CHECK: git push failed even though the hook emitted PASS; "
                    "worker must not treat this as accepted gate progress. "
                    f"{out[:700]}"
                )
                res["feedback"] = feedback
                res["local_git_failed"] = True
                rec = node_memory.record_rejected(cfg, node, lineage_ids, branch, rnd + 1,
                                                  files, feedback, out)
                res["last_rejected_attempt"] = rec.get("attempt_id")
                res["last_rejected_raw"] = rec.get("raw_dir")
                logger.warning("[%s] git push failed after hook PASS (round %d): %s",
                               nid, rnd + 1, out[:300])
                _st("rejected", feedback[:160], engine=weng, log=True)
                continue
            if "declined" in out or "rejected" in out:
                feedback = gate_fb; res["feedback"] = gate_fb
                rec = node_memory.record_rejected(cfg, node, lineage_ids, branch, rnd + 1,
                                                  files, gate_fb, out)
                res["last_rejected_attempt"] = rec.get("attempt_id")
                res["last_rejected_raw"] = rec.get("raw_dir")
                logger.info("[%s] gate REJECT (round %d): %s", nid, rnd + 1, gate_fb[:120])
                _st("rejected", f"гейт отклонил (раунд {rnd+1}): {gate_fb} — переписываю и пушу снова",
                    engine=weng, astar=_astar(gate_fb), log=True)
                continue  # исправляем и пушим снова
            if push.returncode != 0:
                feedback = (
                    "LOCAL_GIT_CHECK: git push failed before gate accepted the branch; "
                    f"worker must not record PASS. {out[:700]}"
                )
                res["feedback"] = feedback
                res["local_git_failed"] = True
                rec = node_memory.record_rejected(cfg, node, lineage_ids, branch, rnd + 1,
                                                  files, feedback, out)
                res["last_rejected_attempt"] = rec.get("attempt_id")
                res["last_rejected_raw"] = rec.get("raw_dir")
                logger.warning("[%s] git push failed before gate PASS (round %d): %s",
                               nid, rnd + 1, out[:300])
                _st("rejected", feedback[:160], engine=weng, log=True)
                continue
            pass_summary = gate_fb or "PUSH_ACCEPTED: gate accepted branch; no explicit gate summary emitted."
            res["gate_passed"] = True; res["feedback"] = pass_summary; res["astar"] = _astar(pass_summary)
            commit = _git(["rev-parse", "HEAD"], work).stdout.strip()
            res["commit"] = commit
            alias = _git(["push", "origin", f"{commit}:refs/heads/{tree_branch}"], work)
            if alias.returncode != 0:
                alias_out = (alias.stderr or "") + (alias.stdout or "")
                feedback = (
                    f"LOCAL_GIT_CHECK: tree branch push failed for {tree_branch}; "
                    "worker must not record PASS until the accepted commit is visible to tree/report memory. "
                    f"{alias_out[:700]}"
                )
                res["feedback"] = feedback
                res["gate_passed"] = False
                res["local_git_failed"] = True
                rec = node_memory.record_rejected(cfg, node, lineage_ids, branch, rnd + 1,
                                                  files, feedback, alias_out)
                res["last_rejected_attempt"] = rec.get("attempt_id")
                res["last_rejected_raw"] = rec.get("raw_dir")
                logger.warning("[%s] tree branch push failed (%s): %s",
                               nid, tree_branch, alias_out[:300])
                _st("rejected", feedback[:160], engine=weng, log=True)
                continue
            node_memory.record_pass(cfg, node, lineage_ids, branch, rnd + 1, commit, pass_summary)
            logger.info("[%s] gate PASS (round %d)", nid, rnd + 1)
            _st("passed", f"теорема прошла тройной A*-гейт (proof+code+critic) ✓ за {rnd+1} раунд(ов)",
                engine=weng, astar=_astar(pass_summary), log=True)
            break
    if not res["gate_passed"]:
        # все раунды исчерпаны и не прошёл → ФИНАЛ (узел оставлен), не «правит»
        if res.get("local_contract_failed"):
            failed_by = "локальным контрактом"
        elif res.get("local_verify_failed"):
            failed_by = "локальным verify.py preflight"
        else:
            failed_by = "гейтом"
        _st("failed", f"отклонён {failed_by} за {max_rounds} раунд(ов), узел оставлен: {res.get('feedback','')[:140]}",
            engine=weng, astar=_astar(res.get("feedback", "")), log=True)
    return res


def apply_result(data: Dict[str, Any], res: Dict[str, Any]) -> None:
    """Обновить статус узла в store по итогу гейта."""
    by = {n["id"]: n for n in data["nodes"]}
    n = by.get(res["node"])
    if not n:
        return
    was_typed_writeup = graph_hygiene.is_typed_writeup(n)
    n["attempts"] = int(n.get("attempts", 0) or 0) + 1
    if was_typed_writeup:
        n["writeup_attempts"] = graph_hygiene.writeup_attempts(n) + 1
    if res.get("astar"):
        n["astar"] = res["astar"]
    if res.get("gate_passed"):
        n.pop("last_gate", None)
        n["last_pass_ts"] = datetime.now().isoformat(timespec="seconds")
        if res.get("feedback"):
            n["last_pass"] = res["feedback"][:400]
        if res.get("commit"):
            n["last_pass_commit"] = res["commit"]
    if res.get("experiment_pending"):
        # узел ушёл на реальный GPU-эксперимент → не перевыбираем, пока supervisor.poll не вернёт вердикт
        n["status"] = "experiment_running"
        if res.get("exp_spec"):
            n["exp_spec"] = res["exp_spec"]
        return
    if res.get("exp_router_blocked") and not res.get("gate_passed"):
        # EXP_SPEC-роутер отверг скрипт узла (напр. impossibility 2-world честно не строится автогеном).
        # Копим счётчик; после порога паркуем ЭМПИРИЧЕСКУЮ линию (policy_blocked_exp читают и selector,
        # и confirm-lane) → экспериментаторские слоты не жгутся вечно на неавтогенируемом скрипте
        # (был трэш 52 отказа), они идут на achiev/scale, которые реально бегут. Узел как формальная
        # теория остаётся живым для идейных агентов.
        n["exp_router_blocks"] = int(n.get("exp_router_blocks", 0) or 0) + 1
        if n["exp_router_blocks"] >= 3 and not n.get("policy_blocked_exp"):
            n["policy_blocked_exp"] = True
            n["verdict"] = (n.get("verdict", "") + " | " if n.get("verdict") else "") + \
                (f"эмпирическая линия запаркована после {n['exp_router_blocks']} отказов EXP_SPEC-роутера "
                 "(скрипт не строится автогеном); узел остаётся как формальная теория")
            logger.info("[park] %s: эмп-линия запаркована после %d router-блоков (policy_blocked_exp)",
                        n["id"], n["exp_router_blocks"])
    if res.get("gate_passed") and was_typed_writeup:
        # узел дописал РЕАЛЬНЫЕ эксп-числа в derive И прошёл гейт → полноценно подтверждён
        n["status"] = "confirmed_emp"
        n["verdict"] = (n.get("verdict", "") + " | " if n.get("verdict") else "") + \
            "derive с реальными typed EXP_SPEC числами прошёл тройной гейт"
    elif res.get("gate_passed") and n.get("status") == "needs_writeup":
        n["status"] = "conditional"
        n["verdict"] = (n.get("verdict", "") + " | " if n.get("verdict") else "") + \
            "writeup passed, but empirical evidence was not typed EXP_SPEC; kept conditional"
    elif res.get("gate_passed") and not was_typed_writeup and _looks_like_theory(n) \
            and n.get("status") in ("open", "weak", "deferred", "conditional", None):
        # Чистая теорема прошла тройной гейт proof+code+critic → это ДОКАЗАНО. Без промоушена в
        # proven theory-узел вечно висит conditional и не входит в статью как headline_win.
        n["status"] = "proven"
        n["verdict"] = (n.get("verdict", "") + " | " if n.get("verdict") else "") + \
            f"PROVEN: теорема прошла тройной гейт proof+code+critic ({res['branch']}" + \
            (f", A*={res['astar']}/10" if res.get('astar') else "") + ")"
    elif res.get("gate_passed") and n.get("status") in ("open", "weak", "deferred", None):
        n["status"] = "conditional"
        n["verdict"] = (n.get("verdict", "") + " | " if n.get("verdict") else "") + \
            f"agent push PASS ({res['branch']}" + (f", A*={res['astar']}/10" if res.get('astar') else "") + ")"
    elif not res.get("gate_passed"):
        # узел НЕ потерян: сохраняем замечания гейта, чтобы при ВОЗВРАТЕ продолжить с них.
        if res.get("feedback"):
            n["last_gate"] = res["feedback"][:400]
        att = int(n.get("attempts", 0) or 0)
        wu_att = graph_hygiene.writeup_attempts(n)
        if att >= 4 and n.get("status") == "open":
            n["status"] = "weak"  # selector отодвинет ещё дальше, но не хоронит
        elif wu_att >= 2 and was_typed_writeup:
            # эксп подтвердил, но writeup N раз НЕ прошёл тройной гейт → не держим +5-приоритет-слот
            # вечно (иначе очередь needs_writeup монополизирует воркеров и фронтир голодает).
            # Фиксируем conditional; реальные эксп-числа сохранены в exp_confirmed.
            n["status"] = "conditional"
            n["verdict"] = (n.get("verdict", "") + " | ").lstrip(" |") + \
                "эксп подтвердил, но writeup не прошёл тройной гейт за неск. попыток — оставлен conditional"


def request_peer(cfg: ARConfig, to_agent: str, from_node: str, text: str, kind: str = "help") -> None:
    """Агент зовёт пира через inbox (нужна лемма/ревью/мнение)."""
    inbox.post(cfg, to_agent, {"from": from_node, "type": kind, "node": from_node, "text": text})
