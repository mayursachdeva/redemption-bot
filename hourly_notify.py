"""Hourly position check -> macOS notification. Run via launchd
(com.memecoin-scanner.hourly-check.plist), not directly — needs
BINANCE_API_KEY/SECRET sourced from .env first (see hourly_check.sh)."""
from __future__ import annotations

import subprocess

from execute import get_client, get_open_positions


def _applescript_quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def notify(title: str, message: str) -> None:
    script = f"display notification {_applescript_quote(message)} with title {_applescript_quote(title)}"
    subprocess.run(["osascript", "-e", script])


def main() -> None:
    client = get_client()
    positions = get_open_positions(client)
    if not positions:
        notify("Memecoin Bot", "No open positions.")
        return

    seen = set()
    total_cost = total_value = 0.0
    lines = []
    for p in positions:
        key = (p["symbol"], p["trade_id"])
        cost = float(p["qty"] * p["entry"])
        value = float(p["qty"] * p["current"])
        total_cost += cost
        total_value += value
        if key not in seen:
            seen.add(key)
            lines.append(f"{p['symbol']} {p['pnl_pct']:+.1f}%")

    total_pct = (total_value / total_cost - 1) * 100 if total_cost else 0.0
    message = f"{', '.join(lines)} | Total: {total_pct:+.1f}% (${total_value - total_cost:+.0f})"
    notify("Memecoin Bot — Hourly Check", message)


if __name__ == "__main__":
    main()
