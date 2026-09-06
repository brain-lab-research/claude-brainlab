#!/usr/bin/env python3
"""Позвать человека в базу знания: завести участника и выдать одноразовую ссылку на кабинет.

Регистрация закрыта намеренно, поэтому это единственный вход. Скрипт делает две вещи: заводит
участника в самой базе (без записи о человеке токен некому предъявить) и создаёт приглашение в
кабинете. Ссылку передавайте так же, как передают пароли: она годится один раз.

Работает с той же машины, где стоит кабинет: в базу ходит её протоколом, а не скриптами сервера.

    LAB_PORTAL_DB=/root/lab-portal/portal.db \
    LAB_PORTAL_TOKEN_FILE=/root/lab-portal/manager-token \
    LAB_PORTAL_BASE_URL=https://<адрес кабинета> \
    python3 invite.py --email student@brainlab-ai.com
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mailer import MailFailed, MailNotConfigured, send_invite  # noqa: E402
from portal import EMAIL_OK, Base, Store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(prog="invite")
    parser.add_argument("--email", required=True)
    parser.add_argument(
        "--skip-base", action="store_true",
        help="человек уже заведён в базе, нужна только ссылка на кабинет",
    )
    parser.add_argument(
        "--no-mail", action="store_true",
        help="не отправлять письмо, только напечатать ссылку",
    )
    args = parser.parse_args()
    email = args.email.strip().lower()
    if not EMAIL_OK.match(email):
        parser.error("это не похоже на почту")

    if not args.skip_base:
        token_file = Path(os.environ.get("LAB_PORTAL_TOKEN_FILE", "/root/lab-portal/manager-token"))
        base = Base(
            os.environ.get("LAB_PORTAL_MCP", "http://127.0.0.1:8001/mcp"),
            token_file.expanduser().read_text(encoding="utf-8").strip(),
        )
        try:
            answer = base.call("invite_member", {"email": email})
        except Exception as failure:  # noqa: BLE001 — причину показываем как есть
            parser.error(f"участник в базе не завёлся: {failure}")
        print(f"участник заведён: {answer['email']}, роль {answer['role']}")

    store = Store(Path(os.environ.get("LAB_PORTAL_DB", "/root/lab-portal/portal.db")).expanduser())
    token = store.invite(email)
    # Адреса по умолчанию здесь нет намеренно: ссылка-приглашение должна вести на кабинет
    # этой лаборатории, а не на чужой, поэтому промолчать безопаснее, чем угадать.
    base_url = os.environ["LAB_PORTAL_BASE_URL"].rstrip("/")
    link = f"{base_url}/invite/{token}"
    if args.no_mail:
        print("\nссылка на один раз, отдайте её лично:")
        print(f"  {link}")
        return 0
    try:
        sent = send_invite(email, link)
    except MailNotConfigured as failure:
        # Молча не отправить хуже, чем не отправить громко: иначе человек ждёт письма, которого нет.
        print(f"\nписьмо не ушло — {failure}")
        print("ссылка на один раз, отдайте её лично:")
        print(f"  {link}")
        return 0
    except MailFailed as failure:
        print(f"\nпочтовая служба письмо не приняла — {failure}")
        print("ссылка всё равно годится, отдайте её лично:")
        print(f"  {link}")
        return 1
    print(f"\nписьмо отправлено на {email} ({sent})")
    print("ссылка одноразовая; если письмо не дойдёт, отдайте её лично:")
    print(f"  {link}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
