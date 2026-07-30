"""Summarize trade_journal.jsonl: win rate, avg win/loss, by exit reason.
Run directly: ./venv/bin/python trade_stats.py"""
from __future__ import annotations

import json
import os

JOURNAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_journal.jsonl")


def load_journal() -> list[dict]:
    if not os.path.exists(JOURNAL_FILE):
        return []
    with open(JOURNAL_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(records: list[dict]) -> dict:
    scored = [r for r in records if r["pnl_pct"] is not None]
    wins = [r["pnl_pct"] for r in scored if r["pnl_pct"] > 0]
    losses = [r["pnl_pct"] for r in scored if r["pnl_pct"] <= 0]
    return {
        "total": len(records),
        "scored": len(scored),
        "win_rate": len(wins) / len(scored) if scored else None,
        "avg_win_pct": sum(wins) / len(wins) if wins else None,
        "avg_loss_pct": sum(losses) / len(losses) if losses else None,
        "by_reason": {
            reason: sum(1 for r in records if r["exit_reason"] == reason)
            for reason in {r["exit_reason"] for r in records}
        },
    }


def _test_summarize() -> None:
    records = [
        {"pnl_pct": 10.0, "exit_reason": "take_profit"},
        {"pnl_pct": -15.0, "exit_reason": "stop_loss"},
        {"pnl_pct": 5.0, "exit_reason": "early_exit_or_manual"},
        {"pnl_pct": None, "exit_reason": None},
    ]
    s = summarize(records)
    assert s["total"] == 4 and s["scored"] == 3
    assert s["win_rate"] == 2 / 3
    assert s["avg_win_pct"] == 7.5
    assert s["avg_loss_pct"] == -15.0
    print("trade_stats self-check OK")


if __name__ == "__main__":
    _test_summarize()
    records = load_journal()
    s = summarize(records)
    if s["total"] == 0:
        print("No closed trades logged yet — the journal starts from the moment it was built.")
    else:
        print(f"Closed legs: {s['total']} ({s['scored']} with a computable P&L)")
        print(f"Win rate: {s['win_rate']*100:.1f}%" if s["win_rate"] is not None else "Win rate: n/a")
        print(f"Avg win: {s['avg_win_pct']:+.2f}%" if s["avg_win_pct"] is not None else "Avg win: n/a")
        print(f"Avg loss: {s['avg_loss_pct']:+.2f}%" if s["avg_loss_pct"] is not None else "Avg loss: n/a")
        print(f"By exit reason: {s['by_reason']}")
