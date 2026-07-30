"""Reads recent messages from a paid premium-signals Telegram chat via
telethon, logged in as the user's own account (one-time login via
telegram_login.py). The Bot API can't read DMs a third-party bot sends you —
only messages sent to a bot you own — hence the user-account approach."""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

from telethon.sync import TelegramClient

SESSION_NAME = "telegram_session"

_cache: dict = {"messages": [], "fetched_at": 0.0}
_CACHE_TTL_SECONDS = 240  # a bit under the 5-min loop interval


def _fetch(limit: int = 50, minutes: int = 120) -> list[str]:
    chat_name = os.environ.get("TELEGRAM_SIGNAL_CHAT")
    api_id = os.environ.get("TELEGRAM_API_ID")
    api_hash = os.environ.get("TELEGRAM_API_HASH")
    if not (chat_name and api_id and api_hash):
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    texts: list[str] = []
    with TelegramClient(SESSION_NAME, int(api_id), api_hash) as client:
        entity = None
        for dialog in client.iter_dialogs():
            if dialog.name == chat_name:
                entity = dialog.entity
                break
        if entity is None:
            return []
        for msg in client.iter_messages(entity, limit=limit):
            if msg.date < cutoff:
                break
            if msg.text:
                texts.append(msg.text)
    return texts


def get_recent_telegram_messages() -> list[str]:
    """Cached (~4min TTL) recent messages from the configured signals chat."""
    now = time.time()
    if now - _cache["fetched_at"] > _CACHE_TTL_SECONDS:
        try:
            _cache["messages"] = _fetch()
        except Exception:
            pass  # ponytail: a Telegram hiccup shouldn't crash the trading loop
        _cache["fetched_at"] = now
    return _cache["messages"]


def get_telegram_signal_for_symbol(
    symbol: str, messages: list[str] | None = None
) -> list[str]:
    """Recent messages mentioning this symbol ($SYM or bare SYM as a word).
    ponytail: substring/word heuristic, not NLP — fine for a ticker-heavy pump
    channel; upgrade to regex ticker extraction if false-positives show up."""
    if messages is None:
        messages = get_recent_telegram_messages()
    needle_dollar = f"${symbol.lower()}"
    padded_needle = f" {symbol.lower()} "
    matches = []
    for text in messages:
        lower = text.lower()
        if needle_dollar in lower or padded_needle in f" {lower} ":
            matches.append(text)
    return matches


def _test_get_telegram_signal_for_symbol() -> None:
    messages = [
        "$PEPE about to send it 🚀",
        "unrelated message about BTC",
        "DOGE pump incoming",
    ]
    assert get_telegram_signal_for_symbol("PEPE", messages) == ["$PEPE about to send it 🚀"]
    assert get_telegram_signal_for_symbol("DOGE", messages) == ["DOGE pump incoming"]
    assert get_telegram_signal_for_symbol("SHIB", messages) == []
    print("telegram_signals self-check OK")


if __name__ == "__main__":
    _test_get_telegram_signal_for_symbol()
    msgs = get_recent_telegram_messages()
    print(f"fetched {len(msgs)} recent messages")
    for m in msgs[:5]:
        print(" -", m[:80].replace("\n", " "))
