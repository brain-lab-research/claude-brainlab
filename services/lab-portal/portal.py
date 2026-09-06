#!/usr/bin/env python3
"""Раздача ключей к базе знания: одна ссылка, одна кнопка, один ключ.

Пароля здесь нет намеренно. Пароль защищал бы ровно одно — возможность перевыпустить ключ, не
беспокоя ведущего; доступ к самой базе даёт ключ, а не пароль. Для лаборатории в десяток человек
«попросить новую ссылку» — это несколько секунд ведущего, а пароль потянул бы за собой хранение
хешей, защиту от перебора, сессии и восстановление. Всё это — поверхность для атаки ради удобства,
которого никто не просил.

Что кабинет хранит: только приглашения, выданные ведущим, и отметку, что приглашение использовано.
Ни знания лаборатории, ни ключей: за ними он ходит в службу её же протоколом, а сами ключи живут у
людей.

Ключ выдаётся по нажатию кнопки, а не по открытию ссылки: почтовые клиенты и мессенджеры ходят по
ссылкам сами, чтобы показать предпросмотр, и приглашение сгорало бы, не дойдя до человека.

    LAB_PORTAL_DB=/root/lab-portal/portal.db \
    LAB_PORTAL_MCP=http://127.0.0.1:8001/mcp \
    LAB_PORTAL_TOKEN_FILE=/root/lab-portal/manager-token \
    LAB_PORTAL_PUBLIC=https://<адрес базы>/mcp \
    python3 portal.py --port 8088
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import secrets
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from contextlib import suppress
from pathlib import Path
from urllib.parse import unquote, urlparse

LABEL_OK = re.compile(r"^[0-9A-Za-z._-]{2,40}$")
EMAIL_OK = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def now() -> int:
    return int(time.time())


class Store:
    """Приглашения и ничего больше. Ни паролей, ни сессий — их здесь не бывает.

    Каждый запрос идёт в своём потоке, а соединение SQLite одно, и без замка оно ломается:
    восемь потоков на двухстах приглашениях дали пятьсот ошибок «bad parameter or other API
    misuse». Отсюда замок на каждом действии.
    """

    #: Ссылка, пролежавшая в почте неделю, доступа давать уже не должна.
    LIFETIME = 7 * 86400

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock, self.db:
            self.db.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS invites (
                    token TEXT PRIMARY KEY, email TEXT NOT NULL,
                    created_at INTEGER NOT NULL, used_at INTEGER);
                """
            )
        # В файле лежат неиспользованные приглашения: читать его посторонним незачем.
        with suppress(OSError):
            path.chmod(0o600)

    def invite(self, email: str) -> str:
        token = secrets.token_urlsafe(24)
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO invites (token, email, created_at) VALUES (?,?,?)",
                (token, email, now()),
            )
        return token

    def invited_email(self, token: str) -> str | None:
        """Кому годится это приглашение. Использованное и просроченное не годятся никому."""
        with self.lock:
            row = self.db.execute(
                "SELECT email, used_at, created_at FROM invites WHERE token=?", (token,)
            ).fetchone()
        if row is None or row["used_at"] or row["created_at"] < now() - self.LIFETIME:
            return None
        return str(row["email"])

    def claim(self, token: str) -> str | None:
        """Погасить приглашение и вернуть, чьё оно. Ровно один вызов из многих получит адрес.

        Гасить надо ДО похода в базу, а не после: между проверкой и записью проходит целый вызов
        службы, и за это время вторая вкладка успевает пройти ту же проверку. Так одна ссылка
        выдавала пять ключей, из которых два не доставались никому.
        """
        with self.lock, self.db:
            changed = self.db.execute(
                "UPDATE invites SET used_at=? WHERE token=? AND used_at IS NULL"
                " AND created_at > ?",
                (now(), token, now() - self.LIFETIME),
            ).rowcount
            if changed != 1:
                return None
            row = self.db.execute(
                "SELECT email FROM invites WHERE token=?", (token,)
            ).fetchone()
        return str(row["email"]) if row else None

    def release(self, token: str) -> None:
        """Вернуть приглашение к жизни: база не ответила, а человек в этом не виноват."""
        with self.lock, self.db:
            self.db.execute("UPDATE invites SET used_at=NULL WHERE token=?", (token,))


def _first_json_object(payload: str) -> dict:
    """Достать ответ из потока событий службы.

    Служба говорит streamable HTTP: ответ может приехать как обычный JSON, а может строками
    `data: {...}` вперемешку со служебными. Поиск первой фигурной скобки ломался о служебную
    строку с брекетом, и человек получал разорванное соединение вместо ответа.
    """
    for line in payload.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if line.startswith("{"):
            with suppress(ValueError):
                return json.loads(line)
    with suppress(ValueError):
        return json.loads(payload[payload.find("{"):]) if "{" in payload else {}
    return {}


class Base:
    """Разговор со службой знания тем же протоколом, каким с ней говорят агенты."""

    def __init__(self, url: str, token: str) -> None:
        self.url = url
        self.token = token

    def call(self, tool: str, arguments: dict) -> dict:
        body = json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": tool, "arguments": arguments}}
        ).encode()
        request = urllib.request.Request(
            self.url, data=body,
            headers={"Authorization": f"Bearer {self.token}",
                     "Content-Type": "application/json",
                     "Accept": "application/json, text/event-stream",
                     "X-Lab-Actor": "portal"},
        )
        with urllib.request.urlopen(request, timeout=60) as answer:
            payload = answer.read().decode("utf-8", "replace")
        data = _first_json_object(payload)
        result = data.get("result") or {}
        if result.get("isError"):
            text = ((result.get("content") or [{}])[0]).get("text", "служба отказала")
            raise RuntimeError(text)
        text = ((result.get("content") or [{}])[0]).get("text", "{}")
        return json.loads(text)


PAGE = """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · База знания лаборатории</title><style>
:root {{ color-scheme: light dark; --bg:#f2f5f5; --card:#fff; --ink:#10181b; --soft:#46585d;
  --rule:#d2dcdc; --lab:#1f7d6b; --bad:#a94a40; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0c1113; --card:#141b1e; --ink:#dfe7e7;
  --soft:#9fb2b5; --rule:#253134; --lab:#4fc0a6; --bad:#e0796c; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; min-height:100vh; display:flex; align-items:flex-start; justify-content:center;
  background:var(--bg); color:var(--ink); padding:48px 20px;
  font:400 16px/1.6 "IBM Plex Sans",system-ui,-apple-system,sans-serif; }}
main {{ width:100%; max-width:560px; display:flex; flex-direction:column; gap:20px; }}
.card {{ background:var(--card); border:1px solid var(--rule); border-radius:4px; padding:26px;
  display:flex; flex-direction:column; gap:16px; }}
h1 {{ margin:0; font:600 24px/1.2 "IBM Plex Serif",Georgia,serif; }}
h2 {{ margin:0; font:600 17px/1.3 "IBM Plex Serif",Georgia,serif; }}
p {{ margin:0; color:var(--soft); font-size:15px; }}
label {{ display:flex; flex-direction:column; gap:6px; font-size:13.5px; color:var(--soft); }}
input {{ font:400 15px/1.4 inherit; padding:10px 12px; border:1px solid var(--rule);
  border-radius:3px; background:var(--bg); color:var(--ink); }}
input:focus-visible {{ outline:2px solid var(--lab); outline-offset:1px; }}
button {{ font:500 15px/1 inherit; padding:11px 16px; border:0; border-radius:3px;
  background:var(--lab); color:#fff; cursor:pointer; }}
button.plain {{ background:none; color:var(--lab); padding:6px 0; text-align:left; }}
form {{ display:flex; flex-direction:column; gap:14px; }}
.row {{ display:flex; gap:10px; align-items:flex-end; }}
.row label {{ flex:1; }}
.bad {{ color:var(--bad); font-size:14px; }}
code, pre {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:13px; }}
pre {{ margin:0; padding:14px; background:var(--bg); border:1px solid var(--rule);
  border-radius:3px; overflow-x:auto; white-space:pre-wrap; word-break:break-all; }}
table {{ border-collapse:collapse; width:100%; font-size:14px; }}
th,td {{ text-align:left; padding:7px 0; border-bottom:1px solid var(--rule); }}
th {{ font:500 11.5px/1.3 "IBM Plex Mono",monospace; letter-spacing:.08em;
  text-transform:uppercase; color:var(--soft); }}
.muted {{ color:var(--soft); font-size:13.5px; }}
</style></head><body><main>{body}</main></body></html>"""


def page(title: str, body: str) -> bytes:
    return PAGE.format(title=html.escape(title), body=body).encode("utf-8")


def esc(value) -> str:
    return html.escape(str(value if value is not None else ""))


class Portal(BaseHTTPRequestHandler):
    server_version = "lab-portal"
    store: Store
    base: Base
    public_url: str

    def log_message(self, fmt: str, *args) -> None:
        """В журнал публичной машины путь приглашения попадать не должен: он и есть секрет.

        Решение принимается по разобранному пути, а не по подстроке: `/INVITE/…`, `//invite/…` и
        `%2F` обходили прежнюю проверку и уносили живую ссылку в журнал.
        """
        path = urlparse(self.path).path
        parts = [part for part in unquote(path).split("/") if part]
        safe = "/" + "/".join(parts[:1]) if parts else "/"
        if not parts or parts[0].casefold() != "invite":
            # Длинные куски пути тоже прячем: угадать, где ещё окажется секрет, нельзя.
            safe = "/" + "/".join("…" if len(part) > 20 else part for part in parts)
        else:
            safe = "/invite/…"
        print(f"{self.command} {safe}", flush=True)

    def _send(self, code: int, body: bytes, redirect: str | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'unsafe-inline'; frame-ancestors 'none'; "
            "form-action 'self'",
        )
        # На странице показывается ключ: ни в кэше браузера, ни у посредника ему делать нечего.
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Strict-Transport-Security", "max-age=31536000")
        if redirect:
            self.send_header("Location", redirect)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/health":
            self._send(200, b"ok")
        elif path.startswith("/invite/"):
            # Почтовые клиенты дописывают к ссылке косую и знаки препинания: годная ссылка
            # не должна из-за этого выглядеть использованной.
            self._offer(path.rstrip("/.,);").rsplit("/", 1)[-1])
        else:
            self._send(200, page("База знания", """<div class="card">
  <h1>База знания лаборатории</h1>
  <p>Доступ выдаётся по ссылке: попросите её у ведущего. Ссылка приходит письмом и годится один
  раз.</p>
</div>"""))

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path.startswith("/invite/"):
            self._issue(path.rstrip("/.,);").rsplit("/", 1)[-1])
        else:
            self._send(303, b"", redirect="/")

    def _offer(self, token: str, error: str = "") -> None:
        """Показать кнопку. Ключ выдаётся нажатием, а не открытием ссылки.

        Почтовые клиенты и мессенджеры ходят по ссылкам сами, чтобы нарисовать предпросмотр.
        Выдавай мы ключ на открытии — приглашение сгорало бы, не дойдя до человека.
        """
        email = self.store.invited_email(token)
        if not email:
            self._send(200, page("Ссылка не годится", """<div class="card">
  <h1>Ссылка уже использована</h1>
  <p>Или её никогда не было. Попросите у ведущего новую — это одна команда с его стороны.</p>
</div>"""))
            return
        body = f"""<div class="card">
  <h1>Ваш доступ к базе</h1>
  <p>Ключ для <code>{esc(email)}</code>. Одного ключа хватает на все ваши инструменты сразу:
  Claude Code, Codex, агентов на сервере.</p>
  {'<p class="bad">' + esc(error) + '</p>' if error else ''}
  <form method="post" action="/invite/{esc(token)}"><button type="submit">Показать ключ</button></form>
  <p class="muted">Ключ покажется один раз: в базе хранится только его отпечаток. Понадобится ещё
  один — попросите новую ссылку, прежние ключи при этом продолжат работать.</p>
</div>"""
        self._send(200, page("Доступ", body))

    @staticmethod
    def _next_label(existing: list[dict]) -> str:
        """Имя ключу по дате выдачи, а при повторе за день — с номером.

        Печатать имя человеку не приходится, но различать ключи в списке надо: иначе ведущий, гася
        потерянный, не поймёт, какой из трёх гасит.
        """
        today = time.strftime("%Y-%m-%d")
        taken = {item.get("label") for item in existing}
        if today not in taken:
            return today
        number = 2
        while f"{today}-{number}" in taken:
            number += 1
        return f"{today}-{number}"

    def _issue(self, token: str) -> None:
        """Выдать ключ по приглашению. Приглашение гасится до похода в базу.

        Порядок именно такой: сначала гасим, потом идём в службу. Иначе две вкладки проходят
        проверку обе и получают по ключу, а часть ключей не достаётся никому.
        """
        email = self.store.claim(token)
        if not email:
            self._send(303, b"", redirect="/")
            return
        try:
            existing = self.base.call("list_agent_tokens", {"email": email}).get("items", [])
            answer = self.base.call(
                "issue_agent_token", {"label": self._next_label(existing), "email": email}
            )
            key = answer["token"]
            label = answer.get("label", "")
        except (RuntimeError, urllib.error.URLError, TimeoutError, OSError,
                ValueError, KeyError) as failure:
            # Служба могла не ответить, ответить не тем или разорвать соединение. Во всех случаях
            # приглашение возвращается к жизни: человек не виноват, что база молчит.
            self.store.release(token)
            self._offer(token, f"Ключ не выдан: {failure}. Попробуйте ещё раз.")
            return

        codex = (
            f"export LAB_KNOWLEDGE_KEY={key}\n"
            f"codex mcp add lab-knowledge --url {self.public_url} "
            f"--bearer-token-env-var LAB_KNOWLEDGE_KEY"
        )
        claude = (
            f"claude mcp add --transport http lab-knowledge {self.public_url} "
            f'--header "Authorization: Bearer {key}"'
        )
        body = f"""<div class="card">
  <h1>Ключ для {esc(email)}</h1>
  <p>Скопируйте сейчас: второй раз он не покажется.</p>
  <pre>{esc(key)}</pre>
</div>
<div class="card">
  <h2>Codex</h2>
  <p>Переменную стоит дописать в свой профиль оболочки, иначе она пропадёт с закрытием окна.</p>
  <pre>{esc(codex)}</pre>
  <h2>Claude Code</h2>
  <p>Выполните в той папке, где работаете, или добавьте <code>--scope user</code>, чтобы сервер
  был доступен везде.</p>
  <pre>{esc(claude)}</pre>
  <p class="muted">Кто именно записал каждую строку, база определяет сама по тому, чем
  представился ваш инструмент. Расписываться вручную не нужно.</p>
  <p class="muted">Выданные раньше ключи продолжают работать: этот добавился к ним, а не заменил
  их. Ключ этой выдачи называется <code>{esc(label)}</code> — назовите его ведущему, если
  понадобится погасить именно его.</p>
</div>"""
        self._send(200, page("Ключ", body))


def main() -> int:
    parser = argparse.ArgumentParser(prog="lab-portal")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    token_file = Path(os.environ.get("LAB_PORTAL_TOKEN_FILE", "~/lab-portal/manager-token"))
    Portal.store = Store(
        Path(os.environ.get("LAB_PORTAL_DB", "~/lab-portal/portal.db")).expanduser()
    )
    Portal.base = Base(
        os.environ.get("LAB_PORTAL_MCP", "http://127.0.0.1:8001/mcp"),
        token_file.expanduser().read_text(encoding="utf-8").strip(),
    )
    Portal.public_url = os.environ.get("LAB_PORTAL_PUBLIC", "http://127.0.0.1:8001/mcp")
    server = ThreadingHTTPServer((args.host, args.port), Portal)
    print(f"раздача ключей слушает {args.host}:{args.port}, база {Portal.base.url}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
