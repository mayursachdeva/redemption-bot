"""Simulated-leverage view of the real spot trade book: same entries/exits
as actually executed, replayed as if margin-traded at 3x/5x/10x. Spot
execution itself is untouched — this is a reporting artifact only.

Liquidation approximation: a leg is marked liquidated at a given leverage if
its spot P&L (at close, or current price if still open) breached
-100/leverage% — the price move that would wipe out that much margin. This
uses the close/current price, not intrabar low, so a leg that dipped past the
liquidation threshold and recovered before actually closing won't be flagged
even though real leveraged margin trading would have force-closed it there.
No funding/interest costs are modeled either — leveraged positions accrue
funding fees over time that spot doesn't pay, so real leveraged P&L would run
below every number in this sheet, not just diverge by the leverage multiple.

Auto-run every cycle via loop.py so the sheet stays current without asking.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd

from execute import TRADED_SYMBOLS_FILE, get_client

LEVERAGES = (3, 5, 10)
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leverage_journal.xlsx")


def _load_traded_symbols() -> list[str]:
    if os.path.exists(TRADED_SYMBOLS_FILE):
        with open(TRADED_SYMBOLS_FILE) as f:
            return json.load(f)
    return []


def fmt_time(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def build_trade_rows(client) -> list[dict]:
    """FIFO-match every BUY/SELL fill per symbol into discrete trades.
    Filters on executedQty > 0, not status=='FILLED' — Binance marks an
    order's overall status CANCELED once any unfilled remainder is canceled,
    even when part of it genuinely executed (see execute.py's
    _classify_and_log_closed_leg for the same fix, found live on REZ)."""
    rows = []
    for sym in _load_traded_symbols():
        try:
            orders = client.get_orders(symbol=f"{sym}USDT", limit=1000)
        except Exception:
            continue
        filled = [o for o in orders if Decimal(o["executedQty"]) > 0]
        filled.sort(key=lambda o: o["time"])

        lots: list[list] = []  # [price, qty_remaining, buy_time]
        for o in filled:
            qty = Decimal(o["executedQty"])
            quote = Decimal(o["cummulativeQuoteQty"])
            price = quote / qty
            if o["side"] == "BUY":
                lots.append([price, qty, o["time"]])
            else:
                remaining = qty
                while remaining > 0 and lots:
                    lot = lots[0]
                    matched = min(remaining, lot[1])
                    rows.append({
                        "symbol": sym, "qty": matched, "buy_price": lot[0], "sell_price": price,
                        "buy_time": lot[2], "sell_time": o["updateTime"], "open": False,
                    })
                    lot[1] -= matched
                    remaining -= matched
                    if lot[1] <= 0:
                        lots.pop(0)
        for lot in lots:
            if lot[1] > Decimal("0.00000001"):
                current = Decimal(client.ticker_price(symbol=f"{sym}USDT")["price"])
                rows.append({
                    "symbol": sym, "qty": lot[1], "buy_price": lot[0], "sell_price": current,
                    "buy_time": lot[2], "sell_time": None, "open": True,
                })
    rows.sort(key=lambda r: r["buy_time"])
    return rows


def simulate_leverage(row: dict, leverage: int) -> dict:
    """See module docstring for what this approximates and doesn't."""
    spot_pnl_pct = (row["sell_price"] / row["buy_price"] - 1) * 100
    liq_threshold_pct = Decimal(-100) / leverage
    liquidated = spot_pnl_pct <= liq_threshold_pct
    if liquidated:
        return {"pnl_pct": Decimal(-100), "liquidated": True}
    return {"pnl_pct": spot_pnl_pct * leverage, "liquidated": False}


def simulate_leverage_short(row: dict, leverage: int) -> dict:
    """Mirror of simulate_leverage: same entry/exit prices, opposite
    direction (sell high, buy back low). A long's -X% move is a short's +X%
    move and vice versa, so a long liquidation (drop past -100/leverage%)
    can never coincide with a short liquidation (rise past +100/leverage%)
    on the same trade — the two are mutually exclusive by construction."""
    spot_pnl_pct = (row["sell_price"] / row["buy_price"] - 1) * 100
    short_pnl_pct = -spot_pnl_pct
    liq_threshold_pct = Decimal(-100) / leverage
    liquidated = short_pnl_pct <= liq_threshold_pct
    if liquidated:
        return {"pnl_pct": Decimal(-100), "liquidated": True}
    return {"pnl_pct": short_pnl_pct * leverage, "liquidated": False}


def build_dataframe(rows: list[dict]) -> pd.DataFrame:
    records = []
    for i, r in enumerate(rows, 1):
        capital = r["qty"] * r["buy_price"]
        spot_pnl_pct = (r["sell_price"] / r["buy_price"] - 1) * 100
        spot_pnl_dollars = capital * spot_pnl_pct / 100
        record = {
            "#": i, "Symbol": r["symbol"], "Status": "OPEN" if r["open"] else "CLOSED",
            "Qty": float(r["qty"]), "Buy Price": float(r["buy_price"]), "Sell Price": float(r["sell_price"]),
            "Opened (UTC)": fmt_time(r["buy_time"]),
            "Closed (UTC)": fmt_time(r["sell_time"]) if r["sell_time"] else "",
            "Capital Allocated ($)": round(float(capital), 2),
            "Spot P&L (%)": round(float(spot_pnl_pct), 2),
            "Spot P&L ($)": round(float(spot_pnl_dollars), 2),
        }
        for lev in LEVERAGES:
            sim = simulate_leverage(r, lev)
            record[f"{lev}x P&L (%)"] = round(float(sim["pnl_pct"]), 2)
            record[f"{lev}x P&L ($)"] = round(float(capital * sim["pnl_pct"] / 100), 2)
            record[f"{lev}x Liquidated?"] = "YES" if sim["liquidated"] else ""
        records.append(record)
    return pd.DataFrame(records)


def build_shorts_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Same trades, mirrored as shorts (sell at 'buy_price', cover at
    'sell_price') — what the book would look like if every position had been
    the opposite direction. Not a recommendation to short; see the earlier
    chat note on short-specific risks (squeezes, unbounded loss)."""
    records = []
    for i, r in enumerate(rows, 1):
        capital = r["qty"] * r["buy_price"]
        short_pnl_pct = (r["buy_price"] / r["sell_price"] - 1) * 100
        short_pnl_dollars = capital * short_pnl_pct / 100
        record = {
            "#": i, "Symbol": r["symbol"], "Status": "OPEN" if r["open"] else "CLOSED",
            "Qty": float(r["qty"]), "Short Entry": float(r["sell_price"]), "Cover Price": float(r["buy_price"]),
            "Opened (UTC)": fmt_time(r["buy_time"]),
            "Closed (UTC)": fmt_time(r["sell_time"]) if r["sell_time"] else "",
            "Capital Allocated ($)": round(float(capital), 2),
            "Short P&L (%)": round(float(short_pnl_pct), 2),
            "Short P&L ($)": round(float(short_pnl_dollars), 2),
        }
        for lev in LEVERAGES:
            sim = simulate_leverage_short(r, lev)
            record[f"{lev}x P&L (%)"] = round(float(sim["pnl_pct"]), 2)
            record[f"{lev}x P&L ($)"] = round(float(capital * sim["pnl_pct"] / 100), 2)
            record[f"{lev}x Liquidated?"] = "YES" if sim["liquidated"] else ""
        records.append(record)
    return pd.DataFrame(records)


def build_summary(df: pd.DataFrame, shorts_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append({
        "Metric": "Spot (long)", "Total P&L ($)": round(df["Spot P&L ($)"].sum(), 2),
        "Win Rate (%)": round((df["Spot P&L ($)"] > 0).mean() * 100, 1),
        "Liquidations": "",
    })
    for lev in LEVERAGES:
        col_dollars, col_liq = f"{lev}x P&L ($)", f"{lev}x Liquidated?"
        rows.append({
            "Metric": f"{lev}x leverage, long (simulated)",
            "Total P&L ($)": round(df[col_dollars].sum(), 2),
            "Win Rate (%)": round((df[col_dollars] > 0).mean() * 100, 1),
            "Liquidations": int((df[col_liq] == "YES").sum()),
        })
    rows.append({
        "Metric": "Spot (mirrored short)", "Total P&L ($)": round(shorts_df["Short P&L ($)"].sum(), 2),
        "Win Rate (%)": round((shorts_df["Short P&L ($)"] > 0).mean() * 100, 1),
        "Liquidations": "",
    })
    for lev in LEVERAGES:
        col_dollars, col_liq = f"{lev}x P&L ($)", f"{lev}x Liquidated?"
        rows.append({
            "Metric": f"{lev}x leverage, short (simulated)",
            "Total P&L ($)": round(shorts_df[col_dollars].sum(), 2),
            "Win Rate (%)": round((shorts_df[col_dollars] > 0).mean() * 100, 1),
            "Liquidations": int((shorts_df[col_liq] == "YES").sum()),
        })
    return pd.DataFrame(rows)


def write_excel(df: pd.DataFrame, shorts_df: pd.DataFrame, summary: pd.DataFrame) -> None:
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        df.to_excel(writer, sheet_name="Trades", index=False)
        shorts_df.to_excel(writer, sheet_name="Shorts", index=False)


def main() -> None:
    client = get_client()
    rows = build_trade_rows(client)
    if not rows:
        print("No trades to log yet.")
        return
    df = build_dataframe(rows)
    shorts_df = build_shorts_dataframe(rows)
    summary = build_summary(df, shorts_df)
    write_excel(df, shorts_df, summary)
    print(f"leverage_journal.xlsx updated: {len(df)} trades logged.")


if __name__ == "__main__":
    main()
