#!/usr/bin/env python3
"""Отправка приглашения письмом.

Почему не SMTP. На публичной машине лаборатории провайдер закрыл исходящие 25, 587 и 465 — это
обычная защита от рассылок с дешёвых серверов, и обойти её нельзя. Поэтому письмо уходит через
HTTPS-API почтовой службы: единственный порт, который открыт, это 443.

Служба выбирается переменной, ключ лежит в файле с правами 600. Без ключа отправки нет вовсе, и
это не молчаливый пропуск: приглашение печатается на экран, чтобы человек передал ссылку сам, и
об этом честно говорится.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path


class MailNotConfigured(RuntimeError):
    """Ключа нет: письма не будет, ссылку придётся передать руками."""


class MailFailed(RuntimeError):
    """Служба письма не приняла. Причина приходит от неё и показывается как есть."""


LETTER = """Здравствуйте.

Вас пригласили в общую базу знания BRAIn Lab: лаборатория держит там свои утверждения, прогоны, измерения и решения, и оттуда же их читают ваши агенты.

Откройте ссылку и нажмите кнопку — получите свой ключ. Ссылка одноразовая, ключ показывается один раз.

{link}

Вставьте ключ в настройки вашего инструмента: этого достаточно, он работает сразу для всех ваших агентов — Claude Code, Codex, агентов на сервере. Как именно вставить, с проверенными командами, написано во вложении «{attachment}»: прочитайте его целиком, он короткий.
{repo}
Потеряли ключ или ссылка уже использована — попросите новую.
"""

REPO_LINE = """
Код и внутренние заметки лежат здесь, доступ вам уже открыт: {url}
"""


def _key() -> str:
    inline = os.environ.get("LAB_MAIL_API_KEY", "").strip()
    if inline:
        return inline
    # Ключ рядом с кабинетом — путь по умолчанию, чтобы отправка работала без переменных в
    # каждой команде. Файл с правами 600: в аргументах команды секрету не место, его видно всем
    # в списке процессов.
    path = os.environ.get("LAB_MAIL_KEY_FILE", "").strip() or str(
        Path(__file__).resolve().parent / "mail-key"
    )
    if Path(path).expanduser().is_file():
        return Path(path).expanduser().read_text(encoding="utf-8").strip()
    raise MailNotConfigured(
        "ключа почтовой службы нет: задайте LAB_MAIL_API_KEY или LAB_MAIL_KEY_FILE"
    )


def send_invite(to: str, link: str) -> str:
    """Отправить приглашение вместе с вводной запиской. Возвращает идентификатор письма."""
    key = _key()
    sender = os.environ.get("LAB_MAIL_FROM", "onboarding@resend.dev")
    endpoint = os.environ.get("LAB_MAIL_ENDPOINT", "https://api.resend.com/emails")
    repo_url = os.environ.get("LAB_REPO_URL", "").strip()
    guide = _guide()
    letter = LETTER.format(
        link=link,
        attachment=guide[0] if guide else "инструкция",
        repo=REPO_LINE.format(url=repo_url) if repo_url else "",
    )
    if not guide:
        # Без вводной записки письмо бесполезно: человек не знает ни зачем база, ни что с ней
        # делать. Молча отправить голую ссылку — значит отправить его в никуда.
        raise MailFailed(
            "вводной записки нет рядом с кабинетом: положите её или задайте LAB_GUIDE_FILE"
        )
    payload = {
        "from": f"BRAIn Lab <{sender}>",
        "to": [to],
        "subject": "Доступ к базе знания BRAIn Lab",
        "text": letter,
        "attachments": [{"filename": guide[0], "content": guide[1]}],
    }
    request = urllib.request.Request(
        endpoint, data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            # Без имени клиента защита перед почтовой службой отвечает 403 «error code: 1010»,
            # и выглядит это как отказ самой службы, хотя до неё запрос не доехал.
            "User-Agent": "brainlab-portal/1.0",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as answer:
            body = json.loads(answer.read().decode("utf-8", "replace") or "{}")
    except urllib.error.HTTPError as failure:
        detail = failure.read().decode("utf-8", "replace")[:300]
        raise MailFailed(f"{failure.code}: {detail}") from None
    except (urllib.error.URLError, TimeoutError) as failure:
        raise MailFailed(str(failure)) from None
    return str(body.get("id") or "отправлено")


def _guide() -> tuple[str, str] | None:
    """Вводная записка рядом с кабинетом: имя файла и содержимое в base64."""
    path = Path(os.environ.get("LAB_GUIDE_FILE", "") or
                Path(__file__).resolve().parent / "guide.md")
    if not path.is_file():
        return None
    return path.name, base64.b64encode(path.read_bytes()).decode("ascii")
