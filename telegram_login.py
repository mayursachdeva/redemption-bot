"""One-time interactive login for telegram_signals.py.

Usage:
  source .env && ./venv/bin/python telegram_login.py send
  source .env && ./venv/bin/python telegram_login.py verify <code> [2fa_password]

Creates telegram_session.session — after that, telegram_signals.py reads
messages without any further login step.
"""
from __future__ import annotations

import os
import sys

from telethon.errors import SessionPasswordNeededError
from telethon.sync import TelegramClient

SESSION_NAME = "telegram_session"
CODE_HASH_FILE = ".telegram_code_hash"


def get_client() -> TelegramClient:
    api_id = int(os.environ["TELEGRAM_API_ID"])
    api_hash = os.environ["TELEGRAM_API_HASH"]
    client = TelegramClient(SESSION_NAME, api_id, api_hash)
    client.connect()
    return client


def send() -> None:
    client = get_client()
    phone = os.environ["TELEGRAM_PHONE"]
    sent = client.send_code_request(phone)
    with open(CODE_HASH_FILE, "w") as f:
        f.write(sent.phone_code_hash)
    print("code sent to", phone)


def verify(code: str, password: str | None) -> None:
    client = get_client()
    phone = os.environ["TELEGRAM_PHONE"]
    with open(CODE_HASH_FILE) as f:
        phone_code_hash = f.read().strip()
    try:
        client.sign_in(phone, code, phone_code_hash=phone_code_hash)
    except SessionPasswordNeededError:
        if not password:
            print("2FA password needed — rerun: verify <code> <password>")
            sys.exit(1)
        client.sign_in(password=password)
    os.remove(CODE_HASH_FILE)
    me = client.get_me()
    print("logged in as", me.username or me.phone)


if __name__ == "__main__":
    if sys.argv[1] == "send":
        send()
    elif sys.argv[1] == "verify":
        verify(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        print("usage: telegram_login.py send | verify <code> [password]")
        sys.exit(1)
