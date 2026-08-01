"""Build a daily equity curve from trade_journal.jsonl + short_trade_journal.jsonl
(realized) plus current open positions (unrealized-as-of-today), then hand it to
quantstats for a full risk/performance tearsheet (Sharpe, max drawdown, Kelly
criterion, Monte Carlo bust probability, etc).

Run directly: ./venv/bin/python equity_curve.py
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

import pandas as pd
import quantstats as qs

STARTING_NAV = Decimal("10000")  # nominal baseline — only shape (Sharpe, drawdown %) matters, not the absolute number

SPOT_JOURNAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_journal.jsonl")
SHORT_JOURNAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "short_trade_journal.jsonl")

# Confirmed live (2026-07-31 session): get_open_positions() always records a
# leg's ORIGINAL order qty, never actual executed qty. When ESP leg1 partially
# filled (651 of 10006) and the stray remainder got cancelled to re-hedge, the
# journal reconciler logged the closure using the stale original qty — see
# execute.py:1001. Real filled qty was 651, not 10006. Overrides here correct
# known-bad entries by (symbol, trade_id, leg, closed_at) until the underlying
# bug in get_open_positions/_classify_and_log_closed_leg is fixed.
QTY_CORRECTIONS = {
    ("ESP", "17470ce026", 1, "2026-07-31T08:17:07.431000+00:00"): Decimal("651"),
}


def _D(v) -> Decimal | None:
    try:
        return Decimal(str(v))
    except InvalidOperation:
        return None


def _load(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def realized_pnl_by_date(spot_records: list[dict], short_records: list[dict]) -> dict[str, Decimal]:
    """date (YYYY-MM-DD, UTC) -> realized $ pnl that day, spot + shorts combined."""
    daily: dict[str, Decimal] = {}
    for r in spot_records:
        if r.get("pnl_pct") is None:
            continue
        qty = QTY_CORRECTIONS.get((r["symbol"], r["trade_id"], r.get("leg"), r["closed_at"]), _D(r["qty"]))
        entry, exitp = _D(r["entry"]), _D(r["exit_price"])
        if None in (qty, entry, exitp):
            continue
        pnl = qty * (exitp - entry)
        day = r["closed_at"][:10]
        daily[day] = daily.get(day, Decimal(0)) + pnl

    for r in short_records:
        if r.get("pnl_pct") is None:
            continue
        qty, entry, exitp = _D(r["qty"]), _D(r["entry"]), _D(r["exit_price"])
        if None in (qty, entry, exitp):
            continue
        pnl = qty * (entry - exitp)  # short: profit when price falls
        day = r["closed_at"][:10]
        daily[day] = daily.get(day, Decimal(0)) + pnl

    return daily


def current_unrealized_pnl() -> Decimal:
    """Live unrealized on today's open book — 0 if credentials/env aren't loaded
    (equity curve still works off realized history alone in that case)."""
    try:
        from execute import get_client, get_open_positions
        from execute_futures import get_futures_client, get_open_short_positions

        total = Decimal(0)
        for r in get_open_positions(get_client()):
            if r["entry"] is not None:
                total += r["qty"] * (r["current"] - r["entry"])
        for r in get_open_short_positions(get_futures_client()):
            total += r["qty"] * (r["entry"] - r["current"])
        return total
    except Exception as e:
        print(f"  (skipping live unrealized PnL — {e})")
        return Decimal(0)


def build_equity_curve() -> pd.Series:
    """Daily NAV series: STARTING_NAV + cumulative realized PnL, forward-filled
    across non-trading days, with today's live unrealized PnL as the final mark."""
    spot = _load(SPOT_JOURNAL)
    shorts = _load(SHORT_JOURNAL)
    daily_pnl = realized_pnl_by_date(spot, shorts)
    if not daily_pnl:
        raise ValueError("no closed trades in either journal — nothing to build a curve from")

    dates = sorted(daily_pnl)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    full_index = pd.date_range(dates[0], today, freq="D")

    nav = STARTING_NAV
    daily_nav = {}
    for day in full_index.strftime("%Y-%m-%d"):
        nav += daily_pnl.get(day, Decimal(0))
        daily_nav[day] = nav

    unrealized = current_unrealized_pnl()
    daily_nav[today] = daily_nav[today] + unrealized

    series = pd.Series({pd.Timestamp(d): float(v) for d, v in daily_nav.items()})
    series.index.name = "date"
    return series


def main() -> None:
    nav = build_equity_curve()
    returns = nav.pct_change().dropna()

    print(f"Equity curve: {nav.index[0].date()} -> {nav.index[-1].date()} ({len(nav)} days)")
    print(f"NAV: {nav.iloc[0]:.2f} -> {nav.iloc[-1]:.2f}  ({(nav.iloc[-1] / nav.iloc[0] - 1) * 100:+.2f}%)")
    print()
    print(f"Sharpe:          {qs.stats.sharpe(returns):.2f}")
    print(f"Sortino:         {qs.stats.sortino(returns):.2f}")
    print(f"Max drawdown:    {qs.stats.max_drawdown(returns) * 100:.2f}%")
    print(f"Kelly criterion: {qs.stats.kelly_criterion(returns) * 100:.2f}%")
    print(f"Win rate:        {qs.stats.win_rate(returns) * 100:.2f}%")
    print(f"Volatility (ann):{qs.stats.volatility(returns) * 100:.2f}%")

    # ponytail: qs.stats.montecarlo is main-branch-only as of quantstats 0.0.77
    # (the PyPI release), not in the installed package — skipped rather than
    # installing from git main for one function. Revisit if a release adds it.

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quantstats_report.html")
    qs.reports.html(returns, output=out_path, title="Memecoin Scanner — Equity Curve")
    print(f"\nFull tearsheet: {out_path}")


def _test_realized_pnl_by_date() -> None:
    spot = [{"symbol": "X", "trade_id": "t1", "leg": 0, "qty": 10, "entry": 1.0,
             "exit_price": 1.1, "pnl_pct": 10.0, "closed_at": "2026-01-01T00:00:00+00:00"}]
    shorts = [{"symbol": "Y", "trade_id": "t2", "qty": 5, "entry": 2.0,
               "exit_price": 1.8, "pnl_pct": 10.0, "closed_at": "2026-01-01T00:00:00+00:00"}]
    result = realized_pnl_by_date(spot, shorts)
    # spot: 10*(1.1-1.0)=1.0, short: 5*(2.0-1.8)=1.0 -> combined 2.0
    assert result["2026-01-01"] == Decimal("2.0"), result
    # qty correction applied when key matches
    corrected = [{"symbol": "ESP", "trade_id": "17470ce026", "leg": 1, "qty": 10006,
                  "entry": Decimal("0.07548208440802538554382656634"), "exit_price": 0.12302,
                  "pnl_pct": 62.98, "closed_at": "2026-07-31T08:17:07.431000+00:00"}]
    corrected_result = realized_pnl_by_date(corrected, [])
    assert abs(corrected_result["2026-07-31"] - Decimal("30.95")) < Decimal("0.1"), corrected_result
    print("ok")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        _test_realized_pnl_by_date()
    else:
        main()
