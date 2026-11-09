#!/usr/bin/env python3
"""Достать рисунки и таблицы из исходников статьи на arXiv.

Почему из исходников, а не из PDF. Извлечение из PDF даёт файлы вида
`fig_p03_i2.png` и тащит заодно логотипы и обрезки вёрстки: на 2608.27518 из
26 картинок настоящими рисунками были пять. По исходникам всё точно: имя файла
авторское, подпись берётся из `\\caption`, а по `\\ref` видно, в каком месте
текста на рисунок ссылаются. Тогда агент подписывает рисунок по существу, а не
угадывает по номеру страницы.

PDF остаётся запасным путём: часть авторов исходники не выкладывает.

    extract_tex_assets.py --arxiv 2608.27518 --out <папка вложений>
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

UA = {"User-Agent": "brainlab-paper-ingest/1.0"}
IMG_EXT = (".png", ".jpg", ".jpeg", ".pdf", ".eps")
#: Рисунок легче этого почти наверняка логотип или линейка.
MIN_BYTES = 6_000

FIGURE_ENV = re.compile(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", re.S)
TABLE_ENV = re.compile(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", re.S)
#: Пробел и перенос строки между `[...]` и `{файл}` законны в TeX: 08-11-2026 у 2610.16794 так
#: были записаны три рисунка, включая пример во введении, и разбор назвал его «недоступным».
INCLUDE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\s*\{([^}]+)\}")
#: Рисунок в обтекании текстом. У 2610.16794 так стояли результаты Countdown и качество критика.
WRAP_ENV = re.compile(r"\\begin\{wrapfigure\}(.*?)\\end\{wrapfigure\}", re.S)
CAPTION = re.compile(r"\\caption\{(.*?)\}\s*(?:\\label|\n\n|$)", re.S)
LABEL = re.compile(r"\\label\{([^}]+)\}")
TABULAR = re.compile(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}", re.S)


def fetch_source(arxiv_id: str, work: pathlib.Path) -> bool:
    """Скачать и разложить бандл исходников. False, если его нет."""
    req = urllib.request.Request(f"https://arxiv.org/e-print/{arxiv_id}", headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            blob = resp.read()
    except Exception:                        # noqa: BLE001 — исходников может не быть
        return False
    if not blob:
        return False
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tar:
            tar.extractall(work, filter="data")
        return True
    except (tarfile.TarError, TypeError):
        try:                                 # иногда это один .tex под gzip
            (work / "main.tex").write_bytes(gzip.decompress(blob))
            return True
        except OSError:
            return False


def main_tex(work: pathlib.Path) -> str:
    """Склеить все .tex: препринты часто разбиты по файлам через \\input."""
    chunks = []
    for tex in sorted(work.rglob("*.tex")):
        try:
            chunks.append(tex.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return "\n".join(chunks)


#: Авторские макросы. Статьи сплошь определяют свои сокращения, и в таблицу
#: они приезжают как есть: 03-09-2026 у 2608.11859 вся шапка Table 1 состояла
#: из `$\hparam{dim}$`, `$\hparam{n_layers}$`. Разворачиваем по преамбуле.
USER_MACROS: dict[str, tuple[int, str]] = {}


def collect_macros(tex: str) -> None:
    r"""Собрать \newcommand и \def из исходников. Только простые случаи."""
    USER_MACROS.clear()
    for m in re.finditer(r"\\(?:newcommand|renewcommand|providecommand)\*?\s*"
                         r"\{?\\([A-Za-z]+)\}?\s*(?:\[(\d+)\])?\s*(?:\[[^\]]*\])?\s*\{",
                         tex):
        body, _ = _braced(tex, m.end() - 1)
        USER_MACROS[m.group(1)] = (int(m.group(2) or 0), body)
    for m in re.finditer(r"\\def\s*\\([A-Za-z]+)\s*\{", tex):
        body, _ = _braced(tex, m.end() - 1)
        USER_MACROS.setdefault(m.group(1), (0, body))


def expand_macros(s: str, depth: int = 3) -> str:
    """Подставить авторские макросы. Вложенность разворачиваем до depth раз."""
    if not USER_MACROS:
        return s
    for _ in range(depth):
        changed = False
        for name, (argc, body) in USER_MACROS.items():
            i = 0
            while True:
                j = s.find("\\" + name, i)
                if j < 0:
                    break
                after = j + 1 + len(name)
                if after < len(s) and s[after].isalpha():   # \foo не часть \foobar
                    i = after
                    continue
                out, k = body, after
                for n in range(1, argc + 1):
                    while k < len(s) and s[k] in " \n\t":
                        k += 1
                    if k < len(s) and s[k] == "{":
                        arg, k = _braced(s, k)
                    elif k < len(s):
                        arg, k = s[k], k + 1
                    else:
                        arg = ""
                    out = out.replace(f"#{n}", arg)
                s = s[:j] + out + s[k:]
                changed = True
                i = j + len(out)
        if not changed:
            break
    return s


def strip_wrapper(s: str, name: str, nargs: int, keep: int) -> str:
    r"""Убрать \name{..}{..}{..}, оставив аргумент номер keep (с 1).

    Регулярка `\{([^}]*)\}` обрывает аргумент на первой закрывающей скобке, а в
    `\multicolumn{2}{c}{$\Delta\mathcal{L}\,(k{=}0)$}` она стоит внутри
    `\mathcal{L}`. Ячейка приезжала обрубком `$\Delta\mathcal{L\,(k{=}0`, то есть
    другой формулой. Считаем вложенность.
    """
    out, i = "", 0
    while True:
        j = s.find("\\" + name, i)
        if j < 0:
            return out + s[i:]
        after = j + 1 + len(name)
        if after < len(s) and s[after].isalpha():
            out += s[i:after]
            i = after
            continue
        args, k = [], after
        for _ in range(nargs):
            while k < len(s) and s[k] in " \n\t":
                k += 1
            if k < len(s) and s[k] == "{":
                arg, k = _braced(s, k)
                args.append(arg)
            else:
                args = []
                break
        if len(args) != nargs:
            out += s[i:after]
            i = after
            continue
        out += s[i:j] + (args[keep - 1] if 1 <= keep <= nargs else "")
        i = k


def clean_tex(s: str) -> str:
    """Подпись из \\caption в читаемый текст."""
    s = expand_macros(s)
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    # \multicolumn{2}{c}{Текст} -> Текст, иначе в таблице остаётся «2cТекст»
    s = strip_wrapper(s, "multicolumn", 3, 3)
    s = strip_wrapper(s, "multirow", 3, 3)
    # \cmidrule(lr){2-3} и родня — линейки, в тексте им места нет
    s = re.sub(r"\\cmidrule(?:\([a-z]+\))?\{[^}]*\}", "", s)
    for wrap in ("textbf", "textit", "emph", "texttt", "text"):
        s = strip_wrapper(s, wrap, 1, 1)
    s = re.sub(r"~", " ", s)
    for tex_cmd, sym in TEX_SYMBOLS.items():
        s = s.replace(tex_cmd + " ", sym).replace(tex_cmd, sym)
    # Формулы оставляем как есть: вырезание команд внутри $...$ превращало
    # `$O(\\epsilon^{-3})$` в `$O(^-3)$`, то есть в другую формулу.
    math = []
    def _keep(m):
        math.append(m.group(0))
        return f"\x00M{len(math) - 1}\x00"
    s = re.sub(r"\$[^$]*\$", _keep, s)
    s = re.sub(r"\\[a-zA-Z]+\*?", "", s)
    s = re.sub(r"[{}]", "", s)
    for k, frag in enumerate(math):
        s = s.replace(f"\x00M{k}\x00", frag)
    return " ".join(s.split())


def find_image(work: pathlib.Path, ref: str) -> pathlib.Path | None:
    ref = ref.strip().strip("./")
    cand = work / ref
    if cand.is_file():
        return cand
    for ext in IMG_EXT:                      # в \includegraphics расширение часто опущено
        if (work / (ref + ext)).is_file():
            return work / (ref + ext)
    stem = pathlib.Path(ref).name
    for f in work.rglob("*"):
        if f.is_file() and f.stem == pathlib.Path(stem).stem and f.suffix.lower() in IMG_EXT:
            return f
    return None


def to_png(src: pathlib.Path, dest_dir: pathlib.Path) -> pathlib.Path | None:
    """Растеризовать векторный рисунок; растровый просто скопировать."""
    out = dest_dir / (src.stem + ".png")
    if src.suffix.lower() in (".png", ".jpg", ".jpeg"):
        out = dest_dir / src.name
        shutil.copy2(src, out)
        return out
    try:
        import pymupdf
        doc = pymupdf.open(src)
        page = doc.load_page(0)
        page.get_pixmap(dpi=200).save(out)
        doc.close()
        return out
    except Exception:                        # noqa: BLE001 — eps без ghostscript и прочее
        return None


#: Математические макросы, которые нельзя просто выбросить: без них число
#: искажается. `1.2\pm0.3` превращалось в «1.20.3», то есть в таблице
#: появлялось несуществующее значение — хуже, чем отсутствие таблицы.
TEX_SYMBOLS = {
    r"\pm": "±", r"\mp": "∓", r"\times": "×", r"\cdot": "·",
    r"\leq": "≤", r"\geq": "≥", r"\neq": "≠", r"\approx": "≈",
    r"\ll": "≪", r"\gg": "≫", r"\infty": "∞", r"\rightarrow": "→",
    r"\to": "→", r"\uparrow": "↑", r"\downarrow": "↓",
    r"\dagger": "†", r"\ast": "*", r"\%": "%", r"\$": "\\$", r"\#": "#",
    r"\_": "_", r"\&": "&",
    # Денежную сумму `\$4.4K` нельзя превращать в голый `$`: в Obsidian он
    # откроет формулу и съест полстроки. Экранирование сохраняем.
}


def _braced(s: str, start: int) -> tuple[str, int]:
    """Содержимое {...} с учётом вложенности; start указывает на «{»."""
    if start >= len(s) or s[start] != "{":
        return "", start
    depth, i = 0, start
    while i < len(s):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1:i], i + 1
        i += 1
    return s[start + 1:], len(s)


def expand_spans(raw: str) -> list[str]:
    """Разбить строку таблицы на клетки, разворачивая \\multicolumn.

    Клетка с \\multicolumn{n} занимает n колонок, поэтому она отдаёт текст и
    ещё n-1 пустых клеток. Без этого шапка со сгруппированными колонками
    съезжает относительно тела, и вся таблица врёт.
    """
    cells, buf, i = [], "", 0
    while i < len(raw):
        ch = raw[i]
        if ch == "\\" and i + 1 < len(raw):        # экранированное — не разделитель
            buf += raw[i:i + 2]
            i += 2
            continue
        if ch == "{":
            inner, j = _braced(raw, i)
            buf += "{" + inner + "}"
            i = j
            continue
        if ch == "&":
            cells.append(buf)
            buf = ""
            i += 1
            continue
        buf += ch
        i += 1
    cells.append(buf)

    out = []
    for c in cells:
        m = re.search(r"\\multicolumn\s*\{(\d+)\}", c)
        span = int(m.group(1)) if m else 1
        out.append(clean_tex(c))
        out += [""] * (span - 1)
    return out


def find_tabular(env: str) -> tuple[str, str] | None:
    r"""Преамбула и тело таблицы с учётом вложенных скобок.

    Регулярка `\{([^}]*)\}` брала преамбулу до ПЕРВОЙ закрывающей скобки, а в
    `>{\raggedright}p{0.43\linewidth}lcc` первая закрывает `\raggedright`.
    Остаток преамбулы уезжал в тело и печатался как первая клетка шапки
    («lrrrrrrrrrr@ Checkpoint»).
    """
    m = re.search(r"\\begin\{(tabular\*?|tabularx|longtable)\}", env)
    if m is None:
        return None
    i = m.end()
    if m.group(1) == "tabularx" and env[i:i + 1] == "{":   # у tabularx сначала ширина
        _, i = _braced(env, i)
    while i < len(env) and env[i] in " \n\t":
        i += 1
    if env[i:i + 1] != "{":
        return None
    spec, i = _braced(env, i)
    end = re.search(r"\\end\{(?:tabular\*?|tabularx|longtable)\}", env[i:])
    body = env[i:i + end.start()] if end else env[i:]
    return spec, body


def spec_width(spec: str) -> int:
    """Сколько колонок описывает преамбула, с разворотом *{n}{...}."""
    spec = re.sub(r"@\{[^}]*\}", "", spec)
    spec = re.sub(r">\{[^}]*\}|<\{[^}]*\}", "", spec)
    def rep(m):
        return m.group(2) * int(m.group(1))
    spec = re.sub(r"\*\s*\{(\d+)\}\s*\{([^}]*)\}", rep, spec)
    spec = re.sub(r"[pmb]\{[^}]*\}", "p", spec)
    return len([c for c in spec if c in "lcrp"])


def markdown_table(spec: str, body: str) -> str:
    cols = spec_width(spec) or 2
    rows = []
    # Разделитель строки бывает с необязательным отступом: \\[2pt], \\*
    for raw in re.split(r"\\\\\*?(?:\[[^\]]*\])?", body):
        raw = re.sub(r"\\(?:hline|toprule|midrule|bottomrule)\b", "", raw)
        raw = re.sub(r"\\cmidrule(?:\([a-z]+\))?\s*\{[^}]*\}", "", raw)
        cells = expand_spans(raw)
        if any(c.strip() for c in cells):
            rows.append((cells + [""] * cols)[:cols])
    if len(rows) < 2:
        return ""
    head, *rest = rows
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(r) + " |" for r in rest]
    return "\n".join(out)



#: Стили конференций: их надо снимать целиком, иначе рисунок обрезается по
#: ширине полосы статьи и получает номер страницы.
ВЕНЮ = (r"icml|iclr|neurips|nips|aaai|acl|emnlp|naacl|cvpr|iccv|eccv|colm|tmlr"
        r"|jmlr|aistats|uai|kdd|sigir|www|coling|interspeech|miccai")
РАСТЯЖКА = re.compile(r"\\resizebox\s*\{[^}]*\}\s*\{[^}]*\}\s*\{%?\s*")


def без_растяжки(тело: str) -> str:
    r"""Снять `\resizebox{\linewidth}{!}{...}` вокруг рисунка.

    В отдельном документе ширина полосы статьи смысла не имеет, а рисунок по ней
    ужимается и обрезается: `standalone` меряет коробку растяжки, а TikZ рисует
    шире неё. Без растяжки рисунок сам задаёт свои границы, обрезка выходит тугой,
    и заодно пропадает потеря чёткости от уменьшения.
    """
    m = РАСТЯЖКА.search(тело)
    if not m:
        return тело
    начало, глубина, i = m.end(), 1, m.end()
    while i < len(тело) and глубина:
        if тело[i] == "{" and тело[i - 1] != "\\":
            глубина += 1
        elif тело[i] == "}" and тело[i - 1] != "\\":
            глубина -= 1
        i += 1
    return (тело[:m.start()] + тело[начало:i - 1] + тело[i:]).strip()


def preamble_of(work: pathlib.Path) -> str:
    """Преамбула статьи без её класса и стиля конференции.

    Берём целиком: построчная выборка обрезает многострочный `\\newcommand`, и
    сборка падает с `File ended while scanning use of \\@xargdef`.
    """
    # Преамбулу берём из ГЛАВНОГО файла, того, где есть \\documentclass. Склейка
    # всех .tex подряд обрывается на первом попавшемся \\begin{document} из
    # вспомогательного файла, и преамбула выходит в полтора килобайта вместо
    # настоящей — сборка падает с `Undefined control sequence`.
    голова = ""
    for f in sorted(work.rglob("*.tex")):
        try:
            текст = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "\\documentclass" not in текст:
            continue
        i = текст.find("\\begin{document}")
        if i > 0:
            голова = текст[:i]
            break
    голова = re.sub(r"(?m)^\s*\\documentclass[^\n]*\n", "", голова)
    # Стиль конференции убираем целиком, а не только icml. Он ставит свою ширину
    # полосы (`\textwidth 5.5 true in` у iclr2027), подключает `fancyhdr` и
    # `\pagestyle{fancy}`, и в отдельном документе от этого остаются номер
    # страницы прямо на картинке и обрезка по ширине полосы. У 2609.37899 так
    # обрезало обзор метода: пропали «context», «centroid router» и «sample next
    # token», а посреди рисунка стояла единица — номер страницы.
    голова = re.sub(r"(?mi)^\s*\\usepackage(?:\[[^\]]*\])?\{[^}]*(?:" + ВЕНЮ
                    + r")[^}]*\}[^\n]*\n", "", голова)
    # Команды класса конференции (`\icmltitlerunning`, `\iclrfinalcopy`) без него
    # не определены, и сборка падает на первой же из них.
    голова = re.sub(r"(?mis)^\s*\\(?:" + ВЕНЮ + r")[a-z@]*(?:\[[^\]]*\])?(?:\{.*?\})?\s*$\n?",
                    "", голова)
    return голова


#: Номера меток статьи: метка → «3.1». Заполняет `номера_меток` одной сборкой всей статьи.
НОМЕРА: dict[str, str] = {}


def номера_меток(work: pathlib.Path) -> None:
    """Собрать статью целиком один раз и взять номера теорем, рисунков и формул из `.aux`.

    Рисунок TikZ собирается отдельным документом, и ссылки в нём на теоремы статьи
    печатались «??»: 08-11-2026 так вышла вся схема доказательства у 2611.08414 (Theorem ??,
    Proposition A … ??). Ошибки сборки не мешают: `.aux` пишется по мере прохода.
    """
    for f in sorted(work.rglob("*.tex")):
        текст = f.read_text(encoding="utf-8", errors="ignore")
        if "\\documentclass" in текст and "\\begin{document}" in текст:
            сборка = work / "_full"
            сборка.mkdir(exist_ok=True)
            try:
                subprocess.run(["pdflatex", "-interaction=nonstopmode", "-draftmode",
                                "-output-directory", str(сборка), str(f)],
                               capture_output=True, timeout=300, cwd=str(f.parent))
            except (subprocess.TimeoutExpired, OSError):
                return
            for aux in сборка.glob("*.aux"):
                for м in re.finditer(r"\\newlabel\{([^}@]+)\}\{\{([^{}]*)\}",
                                     aux.read_text(encoding="utf-8", errors="ignore")):
                    НОМЕРА.setdefault(м.group(1), м.group(2))
            return


def ссылки_в_номера(тело: str) -> str:
    """`\\ref{x}` → номер из статьи; номера нет — пусто, но не «??»."""
    тело = re.sub(r"\\eqref\{([^}]+)\}",
                  lambda м: f"({НОМЕРА[м.group(1)]})" if м.group(1) in НОМЕРА else "", тело)
    return re.sub(r"\\(?:ref|autoref|cref|Cref|nameref)\*?\{([^}]+)\}",
                  lambda м: ", ".join(НОМЕРА.get(к.strip(), "") for к in м.group(1).split(",")).strip(", "),
                  тело)


def render_tikz(env: str, work: pathlib.Path, dest: pathlib.Path, имя: str) -> "pathlib.Path | None":
    """Собрать рисунок, нарисованный в TikZ, в PNG.

    Такие рисунки не лежат файлами: они вычисляются при сборке, и шаг копирования
    `\\includegraphics` их не видит. 30-09-2026 из-за этого у 2609.38095 пропали
    четыре рисунка из двенадцати, включая обзор метода, и в разборе осталась
    ссылка «по данным Figure 2» без самой Figure 2.
    """
    тело = re.sub(r"\A\s*\[[^\]]*\]", "", env)          # `[!t]` — указание вёрстки
    тело = re.sub(r"\\caption\{.*?\n\s*\}", "", тело, flags=re.S)
    тело = re.sub(r"\\caption\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", "", тело, flags=re.S)
    тело = re.sub(r"\\label\{[^}]*\}|\\centering|\\vskip[^\n]*|\\vspace\{[^}]*\}", "", тело)
    тело = ссылки_в_номера(тело.strip())
    тело = без_растяжки(тело)
    if not тело or ("tikzpicture" not in тело and "pgfplots" not in тело
                    and "axis" not in тело):
        return None
    док = ("\\documentclass[border=4pt]{standalone}\n" + preamble_of(work)
           + "\n\\begin{document}\n" + тело + "\n\\end{document}\n")
    сборка = work / "_tikz"
    сборка.mkdir(exist_ok=True)
    tex_файл = сборка / (имя + ".tex")
    tex_файл.write_text(док, encoding="utf-8")
    try:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                        "-output-directory", str(сборка), str(tex_файл)],
                       capture_output=True, timeout=180, cwd=str(work), check=True)
        pdf = сборка / (имя + ".pdf")
        if not pdf.is_file():
            return None
        png = dest / (имя + ".png")
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(["pdftoppm", "-png", "-r", "200", "-singlefile",
                        str(pdf), str(png)[:-4]], capture_output=True, timeout=180, check=True)
        return png if png.is_file() else None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Рисунки и таблицы из исходников arXiv")
    ap.add_argument("--arxiv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dest = pathlib.Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        work = pathlib.Path(tmp)
        if not fetch_source(args.arxiv, work):
            print(json.dumps({"source": "none", "figures": [], "tables": []},
                             ensure_ascii=False))
            return 0
        tex = main_tex(work)
        collect_macros(tex)
        # Номера нужны только рисункам TikZ, а сборка всей статьи — до пяти минут.
        if any(not INCLUDE.search(e) and "tikzpicture" in e for e in FIGURE_ENV.findall(tex)):
            номера_меток(work)

        figures = []
        for env in FIGURE_ENV.findall(tex) + WRAP_ENV.findall(tex):
            refs = INCLUDE.findall(env)
            cap = CAPTION.search(env)
            lab = LABEL.search(env)
            caption = clean_tex(cap.group(1)) if cap else ""
            label = lab.group(1) if lab else ""
            # где на рисунок ссылаются: строка вокруг \ref{label}
            where = ""
            if label:
                m = re.search(r"[^.]{0,220}\\(?:ref|autoref|Cref|cref)\{" + re.escape(label)
                              + r"\}[^.]{0,220}\.", tex)
                if m:
                    where = clean_tex(m.group(0))
            if not refs:
                # Рисунок без файла: он нарисован в TikZ прямо в исходнике.
                png = render_tikz(env, work, dest, label.replace(":", "_")
                                  or f"tikz_{len(figures) + 1}")
                if png is not None:
                    figures.append({"file": png.name, "caption": caption,
                                    "label": label, "referenced_as": where})
                continue
            for ref in refs:
                src = find_image(work, ref)
                if src is None or src.stat().st_size < MIN_BYTES:
                    continue
                png = to_png(src, dest)
                if png is None:
                    continue
                figures.append({"file": png.name, "caption": caption,
                                "label": label, "referenced_as": where})

        tables = []
        for env in TABLE_ENV.findall(tex):
            cap = CAPTION.search(env)
            tab = find_tabular(env)
            if not tab:
                continue
            md = markdown_table(tab[0], tab[1])
            if md:
                tables.append({"caption": clean_tex(cap.group(1)) if cap else "",
                               "markdown": md})

    print(json.dumps({"source": "arxiv-tex", "figures": figures, "tables": tables},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
