"""Short-side execution on Binance USD-M Futures Testnet — a SEPARATE
account/API from the Spot testnet the long-only bot (execute.py) uses.
GAMBLING EXPERIMENT ONLY.

Mirrors execute.py's structure (sizing, brackets, early-exit, journal) but
for shorts: worst-momentum candidate instead of best, get_llm_short_verdict
instead of get_llm_verdict, SELL-to-open/BUY-to-close instead of the reverse.

Binance migrated conditional orders (STOP_MARKET/TAKE_PROFIT_MARKET) to a
separate "Algo Order" API in late 2025 (POST/DELETE /fapi/v1/algoOrder,
error -4120 on the old /fapi/v1/order path) — found live while building
this. binance-futures-connector 4.2.0 doesn't wrap it yet, so those two
calls go through the client's own sign_request() directly. No endpoint
lists "all open algo orders" (only single lookups by algoId), so open
brackets are tracked in SHORT_POSITIONS_FILE, not re-derived from the API
the way execute.py's get_open_positions() re-derives everything from
clientOrderId tags.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from decimal import Decimal

from binance.um_futures import UMFutures

from execute import (
    DAILY_LOSS_LIMIT_PCT,
    MIN_REWARD_RISK_RATIO,
    _record_traded_symbol,
    _round_step,
    check_daily_loss_limit,
    compute_atr_stop_loss_pct,
    compute_atr_take_profit_pcts,
    compute_vwap_deviation_pct,
    is_counter_trend,
    is_extended_from_vwap,
    kill_switch_active,
    make_trade_id,
    meets_min_reward_risk,
    obv_warns_against,
)
from scanner import (
    get_binance_momentum_short,
    get_binance_universe,
    get_cmc_movers,
    get_dexscreener_trend,
    get_fear_greed_index,
    get_google_trends,
    get_kronos_forecast,
    get_llm_short_verdict,
    get_obv_divergence,
    get_rsi,
    get_vwap,
    rank_symbols,
)

DEX_ENRICH_TOP_N = 15  # how many worst-momentum symbols get the (slower) DEXScreener/Trends/CMC enrichment

FUTURES_LEVERAGE = int(os.environ.get("FUTURES_LEVERAGE", "3"))
FUTURES_MAX_PORTFOLIO_PCT = Decimal(os.environ.get("FUTURES_MAX_PORTFOLIO_PCT", "0.3"))
FUTURES_POSITION_SPLIT = int(os.environ.get("FUTURES_POSITION_SPLIT", "5"))
FUTURES_STOP_LOSS_PCT = Decimal(os.environ.get("FUTURES_STOP_LOSS_PCT", "0.15"))
FUTURES_TAKE_PROFIT_PCT = Decimal(os.environ.get("FUTURES_TAKE_PROFIT_PCT", "0.30"))
FUTURES_RSI_OVERSOLD_THRESHOLD = Decimal(os.environ.get("FUTURES_RSI_OVERSOLD_THRESHOLD", "25"))

SHORT_POSITIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "short_positions.json")
SHORT_CONFIRM_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_worst_candidate.json")
SHORT_JOURNAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "short_trade_journal.jsonl")
DAILY_LOSS_FILE_FUTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "daily_loss_futures.json")


def get_futures_client() -> UMFutures:
    return UMFutures(
        key=os.environ["BINANCE_FUTURES_API_KEY"],
        secret=os.environ["BINANCE_FUTURES_API_SECRET"],
        base_url=os.environ.get("BINANCE_FUTURES_BASE_URL", "https://testnet.binancefuture.com"),
    )


def get_futures_symbol_filters(exchange_info: dict, symbol: str) -> dict:
    row = next((s for s in exchange_info["symbols"] if s["symbol"] == symbol), None)
    if row is None:
        raise ValueError(f"{symbol} not tradeable on futures testnet")
    filters = {f["filterType"]: f for f in row["filters"]}
    max_qty = Decimal(filters["LOT_SIZE"]["maxQty"])
    market_lot = filters.get("MARKET_LOT_SIZE")
    if market_lot and Decimal(market_lot["maxQty"]) > 0:
        max_qty = min(max_qty, Decimal(market_lot["maxQty"]))
    return {
        "step_size": Decimal(filters["LOT_SIZE"]["stepSize"]),
        "min_qty": Decimal(filters["LOT_SIZE"]["minQty"]),
        "max_qty": max_qty,
        "tick_size": Decimal(filters["PRICE_FILTER"]["tickSize"]),
        "min_notional": Decimal(filters["MIN_NOTIONAL"]["notional"]),
    }


def _new_algo_order(client: UMFutures, symbol: str, side: str, order_type: str, trigger_price: Decimal, client_algo_id: str) -> dict:
    return client.sign_request("POST", "/fapi/v1/algoOrder", {
        "algoType": "CONDITIONAL", "symbol": symbol, "side": side, "type": order_type,
        "triggerPrice": str(trigger_price), "closePosition": "true",
        "workingType": "CONTRACT_PRICE", "clientAlgoId": client_algo_id,
    })


def _cancel_algo_order(client: UMFutures, algo_id: int) -> dict:
    return client.sign_request("DELETE", "/fapi/v1/algoOrder", {"algoId": algo_id})


def _load_short_positions() -> dict:
    if os.path.exists(SHORT_POSITIONS_FILE):
        with open(SHORT_POSITIONS_FILE) as f:
            return json.load(f)
    return {}


def _save_short_positions(positions: dict) -> None:
    with open(SHORT_POSITIONS_FILE, "w") as f:
        json.dump(positions, f)


def open_short(
    client: UMFutures, exchange_info: dict, symbol: str, usdt_margin: Decimal,
    leverage: int = FUTURES_LEVERAGE, stop_loss_pct: Decimal = FUTURES_STOP_LOSS_PCT,
    take_profit_pct: Decimal = FUTURES_TAKE_PROFIT_PCT,
) -> dict:
    """Open a short with a market SELL, then bracket it with a STOP_MARKET
    (above entry) and TAKE_PROFIT_MARKET (below entry), both closePosition
    orders via the algo-order API. Persists the position + both algoIds to
    SHORT_POSITIONS_FILE for later reconciliation."""
    pair = f"{symbol}USDT"
    try:
        client.change_leverage(symbol=pair, leverage=leverage)
    except Exception as e:
        print(f"  leverage set failed for {pair} (continuing with existing leverage): {e}")

    filters = get_futures_symbol_filters(exchange_info, pair)
    price = Decimal(client.ticker_price(symbol=pair)["price"])
    notional = usdt_margin * leverage
    qty = _round_step(notional / price, filters["step_size"])
    if qty > filters["max_qty"]:
        qty = _round_step(filters["max_qty"], filters["step_size"])
    if qty < filters["min_qty"] or qty * price < filters["min_notional"]:
        raise ValueError(
            f"{usdt_margin} USDT margin @ {leverage}x too small for {pair} "
            f"(min_notional={filters['min_notional']}, min_qty={filters['min_qty']})"
        )

    order = client.new_order(symbol=pair, side="SELL", type="MARKET", quantity=str(qty))
    time.sleep(1)  # MARKET order fills async on futures — status is NEW in the immediate response
    filled = client.query_order(symbol=pair, orderId=order["orderId"])
    if filled["status"] != "FILLED":
        raise RuntimeError(f"Short open for {pair} did not fill — status={filled['status']}")
    entry_price = Decimal(filled["avgPrice"])

    stop_price = _round_step(entry_price * (1 + stop_loss_pct), filters["tick_size"])
    take_profit = _round_step(entry_price * (1 - take_profit_pct), filters["tick_size"])
    trade_id = make_trade_id()
    sl = _new_algo_order(client, pair, "BUY", "STOP_MARKET", stop_price, f"fs{trade_id}SL")
    tp = _new_algo_order(client, pair, "BUY", "TAKE_PROFIT_MARKET", take_profit, f"fs{trade_id}TP")

    positions = _load_short_positions()
    positions[trade_id] = {
        "symbol": symbol, "qty": str(qty), "entry_price": str(entry_price),
        "leverage": leverage, "sl_algo_id": sl["algoId"], "tp_algo_id": tp["algoId"],
        "sl_price": str(stop_price), "tp_price": str(take_profit),
        "opened_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_short_positions(positions)
    _record_traded_symbol(symbol)
    return {"trade_id": trade_id, "symbol": symbol, "qty": qty, "entry_price": entry_price,
            "stop_price": stop_price, "take_profit": take_profit, "leverage": leverage}


def get_open_short_positions(client: UMFutures) -> list[dict]:
    """Cross-checks the persisted registry against live position risk —
    anything with positionAmt==0 has closed (via SL/TP fill or a manual
    close) since it was last checked, and gets dropped + logged here."""
    positions = _load_short_positions()
    rows = []
    changed = False
    for trade_id, pos in list(positions.items()):
        pair = f"{pos['symbol']}USDT"
        risk = client.get_position_risk(symbol=pair)
        amt = Decimal(risk[0]["positionAmt"]) if risk else Decimal("0")
        if amt == 0:
            _log_closed_short(client, trade_id, pos)
            del positions[trade_id]
            changed = True
            continue
        current = Decimal(risk[0]["markPrice"])
        entry = Decimal(pos["entry_price"])
        pnl_pct = (entry / current - 1) * 100  # short: profit when price falls
        rows.append({
            "trade_id": trade_id, "symbol": pos["symbol"], "qty": Decimal(pos["qty"]),
            "entry": entry, "current": current, "pnl_pct": pnl_pct,
            "leverage": pos["leverage"], "sl_algo_id": pos["sl_algo_id"], "tp_algo_id": pos["tp_algo_id"],
        })
    if changed:
        _save_short_positions(positions)
    return rows


def _log_closed_short(client: UMFutures, trade_id: str, pos: dict) -> None:
    """Figure out how a short closed (SL/TP algo fill, or an explicit market
    close) by checking the two algo orders' status, then append to the short
    journal and cancel whichever bracket order is still dangling."""
    pair = f"{pos['symbol']}USDT"
    entry = Decimal(pos["entry_price"])
    exit_price = exit_reason = None
    for algo_id, reason in ((pos["sl_algo_id"], "stop_loss"), (pos["tp_algo_id"], "take_profit")):
        try:
            algo = client.sign_request("GET", "/fapi/v1/algoOrder", {"algoId": algo_id})
        except Exception:
            continue
        if algo.get("algoStatus") == "FINISHED" and Decimal(algo.get("executedQty", "0")) > 0:
            exit_price = Decimal(algo["avgPrice"])
            exit_reason = reason
        elif algo.get("algoStatus") not in ("FINISHED", "CANCELED"):
            try:
                _cancel_algo_order(client, algo_id)
            except Exception:
                pass  # already gone, fine

    if exit_price is None:
        # neither bracket filled -> closed via close_short()'s market order
        trades = client.get_account_trades(symbol=pair, limit=5)
        buys = [t for t in trades if t["side"] == "BUY"]
        if buys:
            latest = max(buys, key=lambda t: t["time"])
            exit_price = Decimal(latest["price"])
        exit_reason = "early_exit_or_manual"

    pnl_pct = float((entry / exit_price - 1) * 100) if exit_price else None
    record = {
        "symbol": pos["symbol"], "trade_id": trade_id, "qty": pos["qty"], "entry": pos["entry_price"],
        "exit_price": float(exit_price) if exit_price else None, "exit_reason": exit_reason,
        "pnl_pct": pnl_pct, "leverage": pos["leverage"],
        "closed_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(SHORT_JOURNAL_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    msg = f"{pnl_pct:+.2f}%" if pnl_pct is not None else "unknown P&L"
    print(f"  SHORT JOURNAL: {pos['symbol']} closed via {exit_reason} — {msg}")


def close_short(client: UMFutures, symbol: str, trade_id: str) -> Decimal:
    """Market-close a short early (mirrors execute.py's close_position_early)."""
    positions = _load_short_positions()
    pos = positions[trade_id]
    pair = f"{symbol}USDT"
    for algo_id in (pos["sl_algo_id"], pos["tp_algo_id"]):
        try:
            _cancel_algo_order(client, algo_id)
        except Exception:
            pass
    qty = Decimal(pos["qty"])
    client.new_order(symbol=pair, side="BUY", type="MARKET", quantity=str(qty))
    return qty


EARLY_EXIT_MIN_PROFIT_PCT = Decimal("2")
EARLY_EXIT_FORECAST_THRESHOLD = Decimal("5")  # Kronos predicted % change ABOVE this = short thesis fading (price bouncing back up)


def _should_exit_short_early(pnl_pct: Decimal, forecast_pct_change: Decimal | None) -> bool:
    if forecast_pct_change is None:
        return False
    return pnl_pct >= EARLY_EXIT_MIN_PROFIT_PCT and forecast_pct_change >= EARLY_EXIT_FORECAST_THRESHOLD


def manage_short_positions(client: UMFutures) -> None:
    positions = get_open_short_positions(client)
    for pos in positions:
        forecast = get_kronos_forecast(pos["symbol"])
        forecast_pct = Decimal(str(forecast["predicted_pct_change"])) if forecast else None
        if _should_exit_short_early(pos["pnl_pct"], forecast_pct):
            print(f"  SHORT EARLY EXIT: {pos['symbol']} up {pos['pnl_pct']:+.2f}%, Kronos now predicts "
                  f"{forecast_pct:+.2f}% — locking in profit before the bounce.")
            qty = close_short(client, pos["symbol"], pos["trade_id"])
            print(f"  Bought back {qty} {pos['symbol']} at market.")


def _load_worst_confirm_pool() -> list[str]:
    if os.path.exists(SHORT_CONFIRM_FILE):
        with open(SHORT_CONFIRM_FILE) as f:
            return json.load(f).get("candidates", [])
    return []


def _save_worst_confirm_pool(symbols: list[str]) -> None:
    with open(SHORT_CONFIRM_FILE, "w") as f:
        json.dump({"candidates": symbols}, f)


def run_once_short() -> None:
    """One scan -> worst-momentum -> short-verdict -> maybe-short cycle,
    called alongside (not instead of) execute.py's long-side run_once().
    Walks the full worst-momentum-first list, not just the single worst
    symbol — same fix as run_once()'s fall-through, same reason (one
    dominant candidate shouldn't lock out everything else for a cycle)."""
    client = get_futures_client()
    print("Checking open short positions for early-exit opportunities...")
    manage_short_positions(client)

    if kill_switch_active():
        print("KILL_SWITCH active — skipping new short evaluation this cycle.")
        return

    fear_greed = get_fear_greed_index()  # macro overlay, same for every candidate this cycle
    if fear_greed is not None:
        print(f"  Fear & Greed Index: {fear_greed}")

    max_price = float(os.environ.get("MAX_PRICE_USD", "5.0"))
    universe = get_binance_universe(max_price=max_price)
    momentum = get_binance_momentum_short(universe, window="1h")

    worst_by_momentum = sorted(momentum, key=lambda s: momentum[s]["pct_change_24h"])[:DEX_ENRICH_TOP_N]
    if not worst_by_momentum:
        print("No candidates this cycle.")
        return
    dex_scores = {sym: get_dexscreener_trend(sym) for sym in worst_by_momentum}
    try:
        trends = get_google_trends(worst_by_momentum)
    except Exception as e:
        print(f"  Google Trends skipped: {e}")
        trends = {}
    try:
        cmc_movers = get_cmc_movers()
    except Exception as e:
        print(f"  CoinMarketCap movers skipped: {e}")
        cmc_movers = {}
    # rank_symbols sorts highest-score-first (long convention) — reverse it
    # for shorts, since the most bearish combined score is what we want first
    scored = rank_symbols({sym: momentum[sym] for sym in worst_by_momentum}, dex_scores, trends, cmc_movers)
    ranked = [sym for sym, _ in reversed(scored)]

    prev_candidates = _load_worst_confirm_pool()
    _save_worst_confirm_pool(ranked)
    open_symbols = {p["symbol"] for p in get_open_short_positions(client)}

    from execute import compute_opportunity_score

    chosen = None
    for sym in ranked:
        score = momentum[sym]["pct_change_24h"]
        print(f"Evaluating worst-momentum: {sym} ({score:+.2f}%)")

        if sym not in prev_candidates:
            print(f"  {sym} wasn't in last cycle's candidate pool — needs one more cycle to confirm.")
            continue

        if sym in open_symbols:
            print(f"  already holding an open {sym} short, skipping.")
            continue

        rsi = get_rsi(sym)
        if rsi is not None and rsi < float(FUTURES_RSI_OVERSOLD_THRESHOLD):
            print(f"  {sym} RSI {rsi:.1f} is oversold (<{FUTURES_RSI_OVERSOLD_THRESHOLD}), bounce risk, skipping.")
            continue

        daily = get_binance_momentum_short([sym], window="1d").get(sym)
        daily_pct = Decimal(str(daily["pct_change_24h"])) if daily else None
        if is_counter_trend(daily_pct, is_short=True):
            print(f"  {sym}: 1h signal fighting a bullish 24h trend ({daily_pct:+.1f}%), counter-trend, skipping.")
            continue

        current_price = Decimal(client.ticker_price(symbol=f"{sym}USDT")["price"])
        vwap = get_vwap(sym)
        vwap_dev = compute_vwap_deviation_pct(current_price, Decimal(str(vwap))) if vwap else None
        if is_extended_from_vwap(vwap_dev, is_short=True):
            print(f"  {sym}: price {vwap_dev:+.1f}% below rolling VWAP, overextended, likely to revert, skipping.")
            continue

        obv_div = get_obv_divergence(sym)
        if obv_warns_against(obv_div, is_short=True):
            print(f"  {sym}: {obv_div} OBV divergence — volume isn't confirming the price move, skipping.")
            continue

        kronos = get_kronos_forecast(sym)
        verdict = get_llm_short_verdict(sym, {"rank_score": score, **momentum[sym]}, kronos=kronos)
        if verdict is None:
            print("Ollama not running — `ollama serve` or open the app")
            return
        print(f"  Short verdict: {verdict['verdict'].upper()} (confidence {verdict['confidence']:.2f}) — {verdict['reasoning']}")
        if verdict["verdict"] != "short" or verdict["confidence"] < 0.5:
            print(f"  {sym}: verdict is skip or low confidence.")
            continue

        kronos_pct = Decimal(str(kronos["predicted_pct_change"])) if kronos else None
        opportunity = compute_opportunity_score(
            Decimal(str(verdict["confidence"])), kronos_pct, fear_greed=fear_greed, is_short=True,
        )
        print(f"  Short opportunity score: {opportunity['reasoning']}")
        if not opportunity["passes"]:
            print(f"  {sym}: opportunity score fails the gate.")
            continue

        stop_loss_pct = compute_atr_stop_loss_pct(sym, current_price)
        take_profit_pct = compute_atr_take_profit_pcts(stop_loss_pct)[0]  # shorts aren't laddered, leg 1 only
        if not meets_min_reward_risk(take_profit_pct, stop_loss_pct):
            print(f"  {sym}: R:R {take_profit_pct / stop_loss_pct:.2f}:1 (TP {take_profit_pct * 100:.0f}% / "
                  f"stop {stop_loss_pct * 100:.1f}%) below minimum {MIN_REWARD_RISK_RATIO}:1, skipping.")
            continue

        chosen = {"symbol": sym, "score": score, "verdict": verdict, "stop_loss_pct": stop_loss_pct, "take_profit_pct": take_profit_pct}
        break

    if chosen is None:
        print("No short candidate this cycle cleared every gate.")
        return
    worst_symbol, worst_score, verdict = chosen["symbol"], chosen["score"], chosen["verdict"]
    stop_loss_pct = chosen["stop_loss_pct"]
    take_profit_pct = chosen["take_profit_pct"]

    account = client.account()
    usdt = next((a for a in account["assets"] if a["asset"] == "USDT"), None)
    total_value = Decimal(usdt["marginBalance"]) if usdt else Decimal("0")
    balance = Decimal(usdt["availableBalance"]) if usdt else Decimal("0")
    deployed = sum(
        (Decimal(p["qty"]) * Decimal(client.ticker_price(symbol=f"{p['symbol']}USDT")["price"])) / p["leverage"]
        for p in _load_short_positions().values()
    )
    room = (FUTURES_MAX_PORTFOLIO_PCT * total_value) - deployed
    print(f"Futures portfolio: ${total_value:.2f} total, ${deployed:.2f} margin deployed "
          f"({FUTURES_MAX_PORTFOLIO_PCT * 100:.0f}% cap)")
    if room <= 0:
        print("Not shorting — already at or over the futures exposure cap.")
        return

    daily_halted, daily_pnl_pct = check_daily_loss_limit(total_value, DAILY_LOSS_FILE_FUTURES)
    print(f"Today's futures P&L: {daily_pnl_pct:+.2f}% (limit -{DAILY_LOSS_LIMIT_PCT}%)")
    if daily_halted:
        print(f"Not shorting — daily loss limit hit ({daily_pnl_pct:+.2f}% <= -{DAILY_LOSS_LIMIT_PCT}%). Resets at UTC midnight.")
        return

    per_trade_cap = (FUTURES_MAX_PORTFOLIO_PCT * total_value) / FUTURES_POSITION_SPLIT
    margin = min(balance, room, per_trade_cap)
    if margin < 5:
        print(f"Not shorting — available margin (${margin:.2f}) too small to be worth a trade.")
        return

    exchange_info = client.exchange_info()
    print(f"ATR-based stop-loss: {stop_loss_pct * 100:.1f}% (vs flat {FUTURES_STOP_LOSS_PCT * 100:.0f}% default)")
    print(f"ATR-based take-profit: {take_profit_pct * 100:.1f}% (vs flat {FUTURES_TAKE_PROFIT_PCT * 100:.0f}% default)")
    result = open_short(client, exchange_info, worst_symbol, margin, leverage=FUTURES_LEVERAGE,
                         stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct)
    print(f"\nShorted {result['qty']} {worst_symbol} @ ~{result['entry_price']} "
          f"({FUTURES_LEVERAGE}x, margin ${margin:.2f}). "
          f"SL={result['stop_price']} TP={result['take_profit']}")


def _test_should_exit_short_early() -> None:
    assert _should_exit_short_early(Decimal("9.8"), Decimal("6")) is True
    assert _should_exit_short_early(Decimal("9.8"), Decimal("2")) is False
    assert _should_exit_short_early(Decimal("0.5"), Decimal("6")) is False
    assert _should_exit_short_early(Decimal("-3.0"), Decimal("6")) is False
    assert _should_exit_short_early(Decimal("9.8"), None) is False


def _test_confirm_worst_candidate(tmp_path: str) -> None:
    global SHORT_CONFIRM_FILE
    original = SHORT_CONFIRM_FILE
    SHORT_CONFIRM_FILE = tmp_path
    try:
        assert _load_worst_confirm_pool() == []
        _save_worst_confirm_pool(["AAA", "BBB"])
        assert "AAA" in _load_worst_confirm_pool()
        assert "CCC" not in _load_worst_confirm_pool()
    finally:
        SHORT_CONFIRM_FILE = original
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    import tempfile
    _test_should_exit_short_early()
    _test_confirm_worst_candidate(os.path.join(tempfile.gettempdir(), "_test_short_confirm.json"))
    run_once_short()
