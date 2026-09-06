"""Агент-отчётчик: всегда держит подробный PDF-отчёт о состоянии проекта.

Стиль — твой Reports/STATUS.md: объект+тезис → ранжированные идеи (с исходами) →
теор-реестр (таблица) → главные негативы (с классом провала) → фронтир → числа.
honesty>hype. Поток:
  1) детерминированный СКЕЛЕТ из koi-store (всегда компилится, fallback без LLM);
  2) Opus-отчётчик пишет полную честную прозу в LaTeX поверх скелета;
  3) Opus-КРИТИК ДО компиляции (overclaim/честность/полнота/выдуманные числа) — гейт,
     revise-loop до PASS;
  4) pdflatex (×2) → Reports/STATUS-<ts>.pdf (+ STATUS.md рядом).
Все агенты — Opus (cfg.engines.claude_model).
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import ARConfig
from .engines import engine_call, grep_tail
from . import failure_taxonomy as ftax, graph_hygiene

logger = logging.getLogger(__name__)

_SURVIVED = ("proven", "confirmed_emp", "conditional")
_DEAD = ("rejected", "refuted_breadth")
EMOJI = {"proven": "✅", "confirmed_emp": "🟩", "conditional": "🟡", "open": "🔵",
         "weak": "🟠", "deferred": "⚪", "rejected": "❌", "refuted_breadth": "❌"}

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[T2A]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[russian,english]{babel}
\usepackage{lmodern}
% expansion=false: font expansion требует scalable шрифты; если lmodern недоступен и
% pdflatex падает на bitmap Computer Modern, expansion ломает сборку. protrusion безопасен.
\usepackage[expansion=false]{microtype}
\usepackage{amsmath,amssymb,longtable,booktabs,geometry,hyperref,xcolor,enumitem}
\geometry{a4paper,margin=2.6cm}
\setlength{\parskip}{5pt}\setlength{\parindent}{0pt}
\linespread{1.05}
\setlist{nosep,leftmargin=1.4em}
\usepackage{titlesec}
\titleformat{\section}{\large\bfseries\color{black!85}}{\thesection}{0.6em}{}
\titlespacing*{\section}{0pt}{12pt}{6pt}
\hypersetup{colorlinks=true,linkcolor=blue!45!black,urlcolor=blue!45!black}
"""


def _reports_dir(cfg: ARConfig) -> str:
    base = os.path.dirname(cfg.koi_dir or cfg.gdir)
    d = os.path.join(base, "Reports")
    os.makedirs(d, exist_ok=True)
    return d


def _clause(s: str, n: int = 160) -> str:
    s = (s or "").strip().replace("\n", " ")
    for sep in (". ", "; "):
        if sep in s:
            s = s.split(sep)[0]; break
    return s[:n]


# ---------------- детерминированный скелет (markdown, для LLM + STATUS.md) ----------------

def skeleton_md(data: Dict[str, Any], cfg: ARConfig) -> str:
    nodes = data["nodes"]
    surv = [n for n in nodes if n.get("status") in _SURVIVED]
    rank = sorted(surv, key=lambda n: (_SURVIVED.index(n.get("status", "conditional")),
                                       -float(n.get("elo") or 1500)))
    reg = [n for n in nodes if n.get("col") not in (0, None)]
    L = [f"# {cfg.project_name} — отчёт о состоянии проекта",
         f"> Подробный честный срез из koi-store. Цель: A*. honesty>hype.", "",
         "## 0. Объект и центральный тезис", data.get("root_idea", ""), "",
         data.get("overall_verdict", ""), "",
         "## 1. Самые удачные идеи (по статусу+Elo)"]
    for i, n in enumerate(rank[:12], 1):
        L.append(f"{i}. {EMOJI.get(n['status'],'')} **{n.get('short','')}** "
                 f"[{n['id']}, {n.get('status')}] — "
                 f"{_clause(graph_hygiene.safe_verdict(n) or n.get('idea',''))}")
    L += ["", "## 2. Теоретический реестр (все узлы-результаты)",
          "| id | статус | суть / исход |", "|---|---|---|"]
    for n in reg:
        L.append(f"| {n['id']} | {EMOJI.get(n.get('status'),'')} {n.get('status')} | "
                 f"{_clause(graph_hygiene.safe_verdict(n) or n.get('idea',''), 120)} |")
    tax = ftax.taxonomy(data)
    L += ["", "## 3. Негативы и тупики (классы провалов)"]
    for cls, cnt, ids in tax:
        L.append(f"- [{cnt}×] {cls}: {', '.join(ids[:6])}")
    fr = [n for n in nodes if n.get("status") in ("open", "conditional", "weak") and (n.get("open") or "").strip()]
    L += ["", "## 4. Фронтир (открытые вопросы)"]
    for n in fr[:15]:
        L.append(f"- [{n['id']}] {n.get('short','')}: {_clause(n.get('open',''), 140)}")
    from collections import Counter
    c = Counter(n.get("status") for n in reg)
    L += ["", "## 5. Числа", " · ".join(f"{EMOJI.get(s,'')}{s}={c[s]}" for s in c),
          f"всего узлов-результатов: {len(reg)}; выжило: {len(surv)}"]
    return "\n".join(L)


_UNI = {"→": " -> ", "↔": " <-> ", "⇒": " => ", "⊗": " (x) ", "≤": " <= ", "≥": " >= ",
        "≈": " ~ ", "×": "x", "·": ".", "∝": " prop ", "√": "sqrt", "∞": "inf",
        "⭐": "", "✅": "[OK]", "🟩": "[OK+]", "🟡": "[~]", "🔵": "[o]", "🟠": "[w]",
        "⚪": "[d]", "❌": "[X]", "◻️": "", "🔄": "", "🧱": "", "📊": "", "🎨": "", "⚠️": "(!)"}


def _esc(s: str) -> str:
    s = s or ""
    for u, r in _UNI.items():
        s = s.replace(u, r)
    # выкинуть всё, что вне ASCII+латин-расш.+греческого+кириллицы (emoji/CJK/редкие символы).
    # Греческий диапазон 0x0370-0x03FF обязателен: фильтр молча удалял α/β/λ/Δ из реестра теорем,
    # искажая математику в отчёте без предупреждения.
    s = "".join(c for c in s if ord(c) < 0x0250 or 0x0370 <= ord(c) <= 0x03FF or 0x0400 <= ord(c) <= 0x04FF)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("$", r"\$"), ("^", r"\^{}"),
                 ("~", r"\~{}")):
        s = s.replace(a, b)
    return s


def registry_table_latex(data: Dict[str, Any], cfg: ARConfig) -> str:
    """Детерминированный полный теор-реестр (longtable) — подставляется всегда,
    чтобы Opus не воспроизводил таблицу на 150 строк (источник обрыва вывода)."""
    reg = [n for n in data["nodes"] if n.get("col") not in (0, None)]
    out = [r"\section*{Теоретический реестр (полный, авто)}",
           r"\begin{longtable}{p{2.4cm}p{2.6cm}p{9.5cm}}",
           r"\textbf{id} & \textbf{статус} & \textbf{суть / исход}\\ \hline\endhead"]
    for n in reg:
        out.append(r"%s & %s & %s\\ \hline" % (
            _esc(n["id"]), _esc(str(n.get("status"))),
            _esc(_clause(graph_hygiene.safe_verdict(n) or n.get("idea", ""), 200))))
    out.append(r"\end{longtable}")
    return "\n".join(out)


def skeleton_latex(data: Dict[str, Any], cfg: ARConfig) -> str:
    """Гарантированно компилируемый LaTeX-body (fallback без LLM; hard-escape)."""
    nodes = data["nodes"]
    surv = [n for n in nodes if n.get("status") in _SURVIVED]
    rank = sorted(surv, key=lambda n: (_SURVIVED.index(n.get("status", "conditional")),
                                       -float(n.get("elo") or 1500)))
    out = [r"\section*{Объект и центральный тезис}", _esc(data.get("root_idea", "")), "",
           _esc(data.get("overall_verdict", "")), "",
           r"\section*{Самые удачные идеи}", r"\begin{enumerate}"]
    for n in rank[:12]:
        out.append(r"\item \textbf{%s} (%s) — %s" % (
            _esc(n.get("short", "")), _esc(n.get("status", "")),
            _esc(_clause(graph_hygiene.safe_verdict(n) or n.get("idea", "")))))
    out += [r"\end{enumerate}", registry_table_latex(data, cfg),
            r"\section*{Негативы и классы провалов}", r"\begin{itemize}"]
    for cls, cnt, ids in ftax.taxonomy(data):
        out.append(r"\item [%d раз] %s: %s" % (cnt, _esc(cls), _esc(", ".join(ids[:6]))))
    out += [r"\end{itemize}"]
    return "\n".join(out)


# ---------------- компиляция ----------------

def compile_pdf(latex_body: str, cfg: ARConfig, name: str, title: str) -> Optional[str]:
    if not __import__("shutil").which("pdflatex"):
        logger.error("pdflatex не найден — PDF не собран"); return None
    doc = (PREAMBLE + "\\title{%s}\n\\date{\\today}\n\\begin{document}\n\\maketitle\n"
           % title.replace("_", " ") + latex_body + "\n\\end{document}\n")
    rd = _reports_dir(cfg)
    with tempfile.TemporaryDirectory() as td:
        tex = os.path.join(td, name + ".tex")
        open(tex, "w", encoding="utf-8").write(doc)
        for _ in range(2):
            try:
                r = subprocess.run(["pdflatex", "-interaction=nonstopmode",
                                    "-output-directory", td, tex],
                                   capture_output=True, text=True, errors="replace", timeout=180)
            except subprocess.TimeoutExpired:
                # завис pdflatex не должен ронять весь generate() (в engines._run таймаут обработан,
                # здесь не был) — честно возвращаем «PDF не собран».
                logger.error("pdflatex завис (>180s) — PDF не собран"); return None
        pdf = os.path.join(td, name + ".pdf")
        if not os.path.exists(pdf):
            logger.error("pdflatex failed: %s", (r.stdout or "")[-600:]); return None
        dst = os.path.join(rd, name + ".pdf")
        __import__("shutil").copy2(pdf, dst)
    return dst


# ---------------- Opus агент-отчётчик + критик ----------------

_REPORT_PROMPT = """\
Ты АГЕНТ-ОТЧЁТЧИК проекта: {domain}
Напиши ПОДРОБНЫЙ ЧЕСТНЫЙ отчёт-ПРОЗУ о состоянии проекта в LaTeX (тело документа, БЕЗ
преамбулы/\\begin{{document}}). Стиль: плотно, по-русски, honesty>hype, цель A*. Разделы
\\section*{{}}: объект и центральный тезис; самые удачные идеи (ранжировано, с ЧЕСТНЫМ
исходом каждой — что доказано, что слабо/негатив); главные НЕГАТИВЫ и killers (прямо, не
пряча); фронтир (куда копать). Математику — настоящим LaTeX ($...$).
ВАЖНО: НЕ делай длинных таблиц (\\begin{{longtable}}/tabular) — полный теор-реестр на 150
строк подставится автоматически отдельно. Только связная проза + \\begin{{itemize}}/
\\begin{{enumerate}}. Никаких выдуманных чисел/цитат — только из данных ниже.

ДАННЫЕ (скелет из store):
{skeleton}

Выведи ТОЛЬКО валидный LaTeX-body (компилируется pdflatex; преамбула добавится снаружи)."""

_CRITIC_PROMPT = """\
Ты КРИТИК отчёта (как строгий ревьюер) ДО публикации. Проект: {domain}
Проверь LaTeX-отчёт по пунктам: (1) ЧЕСТНОСТЬ — нет overclaim, негативы не спрятаны,
слабый сигнал не продан за находку; (2) ПОЛНОТА — объект/тезис, удачные идеи с исходами,
реестр, killers, фронтир на месте; (3) ЧИСЛА — нет выдуманных, всё прослеживается;
(4) ЦИТАТЫ — нет выдуманных; (5) ОФОРМЛЕНИЕ — компилируемый LaTeX, термины определены.
В КОНЦЕ выведи РОВНО:
ISSUES: <через ; что исправить, или 'нет'>
REPORT_VERDICT: PASS|FAIL

ОТЧЁТ:
{report}"""


def reporter_agent(skeleton: str, cfg: ARConfig, call: Callable[[str], str]) -> str:
    out = call(_REPORT_PROMPT.format(domain=cfg.domain, skeleton=skeleton[:14000]))
    i = out.find(r"\section")
    return out[i:].strip() if i >= 0 else out.strip()  # отрезать болтовню до первого раздела


def critic_gate(report_latex: str, cfg: ARConfig,
                call: Callable[[str], str]) -> Tuple[bool, str]:
    out = call(_CRITIC_PROMPT.format(domain=cfg.domain, report=report_latex[:14000]))
    return grep_tail(out, "REPORT_VERDICT:").upper().startswith("PASS"), grep_tail(out, "ISSUES:")


def generate(data: Dict[str, Any], cfg: ARConfig, ts: str, live: bool = True,
             max_rounds: int = 2) -> Dict[str, Any]:
    """Сгенерировать отчёт: скелет → (Opus отчётчик → Opus критик-гейт) → PDF. ts извне."""
    md = skeleton_md(data, cfg)
    rd = _reports_dir(cfg)
    # STATUS-auto.md, чтобы НЕ затирать рукописный STATUS.md проекта
    open(os.path.join(rd, "STATUS-auto.md"), "w", encoding="utf-8").write(md)
    body, verdict, issues = skeleton_latex(data, cfg), "skeleton", ""
    if live:
        rcall = engine_call(cfg.engines.reporter, cfg)
        ccall = engine_call(cfg.engines.report_critic, cfg)
        table = registry_table_latex(data, cfg)   # детерминированный реестр (не от LLM)
        narrative = reporter_agent(md, cfg, rcall)
        for rnd in range(max_rounds):
            full = narrative + "\n\n" + table      # проза Opus + полный реестр
            ok, issues = critic_gate(full, cfg, ccall)
            logger.info("report critic round %d: %s (%s)", rnd + 1, "PASS" if ok else "FAIL", issues[:80])
            if ok:
                body, verdict = full, "PASS"; break
            narrative = reporter_agent(md + f"\n\nКРИТИК отклонил прошлую версию, исправь: {issues}",
                                       cfg, rcall)
        else:
            verdict = "FAIL"  # критик не принял -> компилируем скелет (детерминированный)
            body = skeleton_latex(data, cfg)
    pdf = compile_pdf(body, cfg, f"STATUS-{ts}", f"{cfg.project_name} — status {ts}")
    logger.info("report: verdict=%s pdf=%s", verdict, pdf)
    return {"md": os.path.join(rd, "STATUS-auto.md"), "pdf": pdf, "critic": verdict, "issues": issues}
