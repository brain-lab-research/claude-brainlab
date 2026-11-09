#!/usr/bin/env python3
"""Подготовить статью к разбору на своей машине. Zotero не нужен и не трогается.

Делает всю механику, где нечего судить, и печатает JSON для того, кто пишет разбор: тот же
договор, что у серверного конвейера (`text_file`, `note_file`, `example_note`,
`example_kind`, `figures`, `tables`, `assets_source`).

    ingest_local.py --arxiv 2608.27518 --theme adaptive-methods
    ingest_local.py --pdf-url https://…/paper.pdf --title "Название" --theme zo-estimators [--doi …]

Шаги:
1. тема → её клон литературы в `~/brainlab`; нет клона — `lab sync <тема>`;
2. дубль: номер arXiv или DOI уже стоит в какой-то заметке дерева — код 3 и путь к ней;
3. BibTeX только из `bibtex_fetch.py` (DOI → DBLP → arXiv), руками его не собирают;
4. PDF и текст в `~/.cache/brainlab/ingest/<id>/`: текст берёт `pdftotext`, без него PyMuPDF;
5. каркас заметки `<литература>/<Название>.md`: свойства, BibTeX, `## AI Explanation`;
6. рисунки и таблицы в `_attachments/<id>/`: из исходников arXiv, иначе из PDF;
7. эталонные разборы: из дерева, если тема склонирована, иначе с GitLab в кэш.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vault_paths import VAULT, tema_put, temy, vse_papki  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = Path(os.getenv("BRAINLAB_INGEST_CACHE", "~/.cache/brainlab/ingest")).expanduser()
CONF = Path("~/.config/brainlab").expanduser()
UA = {"User-Agent": "brainlab-paper-ingest/1.0"}
ARXIV_ID = re.compile(r"\d{4}\.\d{4,5}")
#: Потолок текста статьи для агента: при 60 тыс. знаков все статьи обрезались до результатов.
TEXT_LIMIT = 200_000
#: Ниже этого веса картинка почти наверняка логотип или обрезок вёрстки.
MIN_FIGURE_BYTES = 12_000
MAX_FIGURES = 10
#: Эталоны: эмпирический (SOMA) и теоретический. Путь — проект GitLab и файл в нём.
EXAMPLES = {
    "empirical": ("brainlab/matrix-optimization/muon-adaptive-preconditioning/literature",
                  "Sharpness-aware Orthogonalized Momentum Algorithms.md"),
    "theory": ("brainlab/optimization-foundations/adaptive-methods/literature",
               "Second-Moment Memory in Coordinatewise Adam.md"),
}


def sanitize(title: str) -> str:
    """Имя файла заметки: тире к дефису, запрещённые в путях знаки убрать, 150 знаков."""
    out = title.replace("—", "-").replace("–", "-").replace("−", "-")
    for bad in (":", "/", "\\", "$", "*", "?", '"', "<", ">", "|"):
        out = out.replace(bad, "")
    return " ".join(out.split())[:150]


def find_duplicate(arxiv_id: str, doi: str) -> str:
    """Где эта статья уже разобрана: заметка в дереве или страница в базе. Пусто — нигде.

    Дерево видит только склонированные темы, поэтому после него спрашивается поиск базы:
    точное попадание номера среди статей всей лаборатории. README темы перечисляет все её
    статьи, и он дублем не считается.
    """
    needles = [n for n in (arxiv_id, doi) if n]
    for folder in vse_papki():
        for note in folder.glob("*.md"):
            if note.name == "README.md":
                continue
            head = note.read_text(encoding="utf-8", errors="ignore")[:6000]
            if any(n in head for n in needles):
                return str(note)
    try:
        base = (CONF / "git-url").read_text().strip()
        token = (CONF / "git-token").read_text().strip()
    except OSError:
        return ""
    for needle in needles:
        query = urllib.parse.urlencode({"q": needle, "kind": "paper", "limit": 5})
        try:
            with urllib.request.urlopen(urllib.request.Request(
                    f"{base}/-/lab-search/api?{query}", headers={"PRIVATE-TOKEN": token}),
                    timeout=90) as answer:
                found = json.loads(answer.read().decode("utf-8"))["results"]
        except (OSError, ValueError, KeyError):
            continue
        hit = next((one for one in found if one.get("exact")), None)
        if hit:
            return hit.get("url") or f"{hit.get('project')}/{hit.get('path')}"
    return ""


def fetch_bibtex(arxiv_id: str, doi: str) -> dict:
    """BibTeX из `bibtex_fetch.py`. Без arXiv и DOI канонической записи нет."""
    if not (arxiv_id or doi):
        return {"bibtex": "", "title": ""}
    flag = ["--doi", doi] if doi else ["--arxiv", arxiv_id]
    run = subprocess.run([sys.executable, str(HERE / "bibtex_fetch.py"), *flag],
                         capture_output=True, text=True, timeout=120)
    if run.returncode != 0:
        raise SystemExit(f"bibtex_fetch не отработал: {run.stderr[:300]}")
    return json.loads(run.stdout)


def download(url: str, dest: Path, tries: int = 4) -> bool:
    for _ in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=120) as answer:
                data = answer.read()
        except OSError:
            continue
        if data[:4] == b"%PDF":
            dest.write_bytes(data)
            return True
    return False


def pdf_text(pdf: Path, txt: Path) -> str:
    """Текст статьи: `pdftotext`, без него PyMuPDF. Ни того, ни другого — пусто."""
    if shutil.which("pdftotext"):
        subprocess.run(["pdftotext", "-q", str(pdf), str(txt)], timeout=180, check=False)
    else:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return "не извлечён: нет ни pdftotext (poppler), ни PyMuPDF; читай PDF напрямую"
        with fitz.open(pdf) as doc:
            txt.write_text("\n".join(page.get_text() for page in doc), encoding="utf-8")
    if not txt.is_file():
        return "не извлечён"
    body = txt.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
    cut = body[:TEXT_LIMIT] + ("\n\n[…текст обрезан…]" if len(body) > TEXT_LIMIT else "")
    txt.write_text(cut, encoding="utf-8")
    return f"{len(body)} символов"


def write_skeleton(note: Path, title: str, theme: str, url: str, bibtex: str) -> None:
    body = [
        "---",
        f'title: "{title}"',
        f'url: "{url}"',
        'code_url: ""',
        'project_url: ""',
        'publication: "Unpublished (arXiv preprint)"',
        "tags:", f"  - {theme}",
        f'updated: "{datetime.now():%d-%m-%Y}"',
        "---", "",
        *(["## BibTeX", "", "```bibtex", bibtex.rstrip(), "```", ""] if bibtex else
          ["## BibTeX", "", "У работы нет ни номера arXiv, ни DOI, поэтому канонической "
           "библиографической записи не существует. Придумывать её нельзя: ключ "
           "цитирования берётся только из `bibtex_fetch.py`.", ""]),
        "## AI Explanation", "", "",
    ]
    note.write_text("\n".join(body), encoding="utf-8")


def assets(arxiv_id: str, pdf: Path, dest: Path, rel: str) -> tuple[list, list, str]:
    """Рисунки и таблицы: из исходников arXiv, иначе из PDF, иначе ничего."""
    dest.mkdir(parents=True, exist_ok=True)
    if arxiv_id:
        run = subprocess.run([sys.executable, str(HERE / "extract_tex_assets.py"),
                              "--arxiv", arxiv_id, "--out", str(dest)],
                             capture_output=True, text=True, timeout=600)
        try:
            data = json.loads(run.stdout) if run.returncode == 0 else {}
        except json.JSONDecodeError:
            data = {}
        if data.get("source") == "arxiv-tex" and data.get("figures"):
            for one in data["figures"]:
                one["path"] = f"{rel}/{one['file']}"
            return data["figures"], data.get("tables", []), "arxiv-tex"
    if not pdf.is_file():
        return [], [], "none"
    run = subprocess.run([sys.executable, str(HERE / "extract_pdf_figures.py"), "--pdf",
                          str(pdf), "--out", str(dest), "--min-size", "220", "--vector"],
                         capture_output=True, text=True, timeout=600)
    if run.returncode != 0:
        return [], [], "none"
    kept = []
    for img in sorted(dest.iterdir()):
        if img.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        if img.stat().st_size < MIN_FIGURE_BYTES:
            img.unlink(missing_ok=True)
        else:
            kept.append(img)
    kept = sorted(kept, key=lambda f: f.stat().st_size, reverse=True)[:MAX_FIGURES]
    try:
        captions = json.loads((dest / "captions.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        captions = {}
    return ([{"path": f"{rel}/{i.name}", "caption": str(captions.get(i.name) or ""),
              "label": "", "referenced_as": ""} for i in kept], [], "pdf")


def example(kind: str) -> str:
    """Эталонный разбор: из дерева, а если темы нет — с GitLab в кэш. Нет доступа — пусто."""
    project, name = EXAMPLES[kind]
    local = VAULT / project.removeprefix("brainlab/") / name
    if local.is_file():
        return str(local)
    cached = CACHE / "examples" / name
    if cached.is_file():
        return str(cached)
    try:
        base = (CONF / "git-url").read_text().strip()
        token = (CONF / "git-token").read_text().strip()
    except OSError:
        return ""
    url = (f"{base}/api/v4/projects/{urllib.parse.quote(project, safe='')}/repository/files/"
           f"{urllib.parse.quote(name, safe='')}/raw?ref=main")
    try:
        with urllib.request.urlopen(urllib.request.Request(
                url, headers={"PRIVATE-TOKEN": token}), timeout=60) as answer:
            data = answer.read()
    except OSError:
        return ""
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(data)
    return str(cached)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arxiv", default="", help="номер arXiv, например 2608.27518")
    ap.add_argument("--pdf-url", default="", help="прямая ссылка на PDF, если arXiv нет")
    ap.add_argument("--doi", default="")
    ap.add_argument("--title", default="", help="название; без arXiv и DOI обязательно")
    ap.add_argument("--theme", required=True, help="слаг темы: имя папки темы в дереве")
    ap.add_argument("--force", action="store_true",
                    help="готовить, даже если заметка уже есть (прежняя уходит в кэш)")
    args = ap.parse_args()
    arxiv_id = args.arxiv.removeprefix("arXiv:").strip()
    if arxiv_id and not ARXIV_ID.fullmatch(arxiv_id):
        raise SystemExit(f"не номер arXiv: {arxiv_id}")
    if not (arxiv_id or args.pdf_url):
        raise SystemExit("нужен --arxiv или --pdf-url")

    literature = tema_put(args.theme)
    if literature is None:
        known = ", ".join(sorted(temy())) or "ни одной"
        raise SystemExit(f"темы `{args.theme}` нет в дереве {VAULT}: `lab sync {args.theme}`. "
                         f"Склонированы: {known}")

    dup = find_duplicate(arxiv_id, args.doi)
    if dup and not args.force:
        print(json.dumps({"duplicate": dup}, ensure_ascii=False))
        return 3

    meta = fetch_bibtex(arxiv_id, args.doi)
    title = args.title or meta.get("title", "")
    if not title:
        raise SystemExit("название не нашлось: передай --title")
    key = arxiv_id or sanitize(title).lower().replace(" ", "-")[:60]
    work = CACHE / key
    work.mkdir(parents=True, exist_ok=True)

    pdf, txt = work / "paper.pdf", work / "paper.txt"
    pdf_url = args.pdf_url or f"https://arxiv.org/pdf/{arxiv_id}"
    text_state = pdf_text(pdf, txt) if download(pdf_url, pdf) else "PDF не скачался"

    note = literature / f"{sanitize(title)}.md"
    if note.is_file():
        if not args.force:
            print(json.dumps({"duplicate": str(note)}, ensure_ascii=False))
            return 3
        shutil.copy2(note, work / f"previous-{datetime.now():%Y%m%d-%H%M}.md")
        note.unlink()
    url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else args.pdf_url
    write_skeleton(note, title, args.theme, url, meta.get("bibtex", ""))

    # Путь от корня дерева, как у серверного конвейера: так вставку понимают и Obsidian,
    # и `fix_page.py`, который переводит её в вид GitLab.
    rel = f"{literature.relative_to(VAULT)}/_attachments/{key}"
    figures, tables, source = assets(arxiv_id, pdf, literature / "_attachments" / key, rel)
    try:
        (literature / "_attachments" / key).rmdir()   # пустую папку не оставляем
    except OSError:
        pass
    kind = "empirical" if (figures or tables) else "theory"
    print(json.dumps({
        "arxiv_id": arxiv_id, "title": title, "theme": args.theme,
        "text_file": str(txt) if txt.is_file() else "", "text": text_state,
        "pdf_file": str(pdf) if pdf.is_file() else "",
        "note_file": str(note), "example_note": example(kind),
        "example_kind": "эмпирическая статья" if kind == "empirical" else "статья без эмпирики",
        "assets_source": source, "figures": figures, "tables": tables,
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
