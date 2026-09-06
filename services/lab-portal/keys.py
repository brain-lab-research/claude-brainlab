#!/usr/bin/env python3
"""Чьи ключи живы и как погасить один из них.

Ключи накапливаются: человек настраивает ими и ноутбук, и сервер, и выдача нового прежние не
трогает. Значит нужен способ посмотреть список и убрать один — например, когда машину потеряли.
Сами ключи нигде не хранятся, видно только имя и дату выдачи.

    python3 keys.py --email student@university.edu
    python3 keys.py --email student@university.edu --revoke 2026-09-01
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from portal import Base  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(prog="keys")
    parser.add_argument("--email", required=True, help="чьи ключи смотрим")
    parser.add_argument("--revoke", default="", help="имя ключа, который надо погасить")
    args = parser.parse_args()

    token_file = Path(os.environ.get("LAB_PORTAL_TOKEN_FILE", "/root/lab-portal/manager-token"))
    base = Base(
        os.environ.get("LAB_PORTAL_MCP", "http://127.0.0.1:8001/mcp"),
        token_file.expanduser().read_text(encoding="utf-8").strip(),
    )
    email = args.email.strip().lower()

    if args.revoke:
        try:
            answer = base.call("revoke_agent_token", {"label": args.revoke, "email": email})
        except Exception as failure:  # noqa: BLE001 — причину показываем как есть
            parser.error(str(failure))
        killed = answer.get("revoked", 0)
        print(f"погашено ключей: {killed}" if killed else "такого живого ключа нет")

    try:
        items = base.call("list_agent_tokens", {"email": email}).get("items", [])
    except Exception as failure:  # noqa: BLE001
        parser.error(str(failure))
    if not items:
        print(f"у {email} живых ключей нет")
        return 0
    print(f"живые ключи {email}:")
    for item in items:
        print(f"  {item.get('label') or '—':<14} выдан {str(item.get('created_at'))[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
