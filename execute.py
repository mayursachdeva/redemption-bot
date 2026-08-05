"""Testnet order execution off the scanner's ranked list + LLM verdict.
GAMBLING EXPERIMENT ONLY. Uses its own Binance testnet account — separate
from trading-system's M6 soak (see .env.example); never point this at that
account's credentials.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from decimal import Decimal

from binance.spot import Spot

from scanner import (
    MEME_SYMBOLS,
    equal_profit_fractions,
    get_atr,
    get_binance_momentum_short,
    get_binance_universe,
    get_cmc_movers,
    get_dexscreener_trend,
    get_fear_greed_index,
    get_fibonacci_levels,
    get_google_trends,
    get_kronos_forecast,
    get_llm_verdict,
    get_market_breadth,
    get_obv_divergence,
    get_rsi,
    get_vwap,
    rank_symbols,
)

DEX_ENRICH_TOP_N = 15  # how many top-momentum symbols get the (slower) DEXScreener lookup


def get_client() -> Spot:
    return Spot(
        api_key=os.environ["BINANCE_API_KEY"],
        api_secret=os.environ["BINANCE_API_SECRET"],
        base_url=os.environ.get("BINANCE_BASE_URL", "https://testnet.binance.vision"),
    )


def _round_step(value: Decimal, step: Decimal) -> Decimal:
    """Round down to the nearest exchange-mandated step (quantity/price tick)."""
    return (value // step) * step


def get_symbol_filters(client: Spot, symbol: str) -> dict:
    info = client.exchange_info(symbol=symbol)["symbols"][0]
    filters = {f["filterType"]: f for f in info["filters"]}
    notional = filters.get("NOTIONAL") or filters["MIN_NOTIONAL"]
    # MARKET_LOT_SIZE caps how big a single MARKET order can be — separate
    # from (and usually much smaller than) LOT_SIZE's general max, since a
    # huge market order would move the price too much. Real limit hit live:
    # a $22k market buy on DIA exceeded its 80,807-unit MARKET_LOT_SIZE cap
    # (LOT_SIZE's own cap was 922,327 — the general limit wasn't the binder).
    market_lot = filters.get("MARKET_LOT_SIZE")
    max_qty = Decimal(filters["LOT_SIZE"]["maxQty"])
    if market_lot and Decimal(market_lot["maxQty"]) > 0:
        max_qty = min(max_qty, Decimal(market_lot["maxQty"]))
    return {
        "step_size": Decimal(filters["LOT_SIZE"]["stepSize"]),
        "min_qty": Decimal(filters["LOT_SIZE"]["minQty"]),
        "max_qty": max_qty,
        "tick_size": Decimal(filters["PRICE_FILTER"]["tickSize"]),
        "min_notional": Decimal(notional["minNotional"]),
    }


def get_usdt_balance(client: Spot) -> Decimal:
    for bal in client.account()["balances"]:
        if bal["asset"] == "USDT":
            return Decimal(bal["free"])
    return Decimal("0")


def get_testnet_symbols(client: Spot) -> set[str]:
    """Base assets with a TRADING USDT pair on testnet. Mainnet's symbol
    list (used by get_binance_universe for discovery) is not 1:1 with
    testnet's — filter any discovery result through this before touching it."""
    info = client.exchange_info()
    return {s["baseAsset"] for s in info["symbols"] if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"}


def has_ask_liquidity(client: Spot, symbol: str) -> bool:
    """Testnet order books for meme pairs are thin and can be empty on
    either side — check before betting, not just after a failed fill."""
    book = client.depth(symbol=f"{symbol}USDT", limit=5)
    return len(book.get("asks", [])) > 0


_TRADE_ID_RE = re.compile(r"^g(?P<tid>[0-9a-f]{10})L\d[AB]$")


def make_trade_id() -> str:
    return uuid.uuid4().hex[:10]


TRADE_STOP_PCTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_stop_pcts.json")


def _record_trade_stop_pct(trade_id: str, stop_loss_pct: Decimal) -> None:
    """get_open_positions() infers entry price from the resting stop-loss
    order (there's no other cheap way to read it back from Binance's OCO
    orders), which requires knowing the exact % used at placement — no
    longer a safe constant now that stop_loss_pct is ATR-derived and varies
    per trade. Persisted here, read back there."""
    pcts = {}
    if os.path.exists(TRADE_STOP_PCTS_FILE):
        with open(TRADE_STOP_PCTS_FILE) as f:
            pcts = json.load(f)
    pcts[trade_id] = str(stop_loss_pct)
    with open(TRADE_STOP_PCTS_FILE, "w") as f:
        json.dump(pcts, f)


def _get_trade_stop_pct(trade_id: str) -> Decimal:
    if os.path.exists(TRADE_STOP_PCTS_FILE):
        with open(TRADE_STOP_PCTS_FILE) as f:
            pcts = json.load(f)
        if trade_id in pcts:
            return Decimal(pcts[trade_id])
    return DEFAULT_STOP_LOSS_PCT  # pre-ATR trade or missing record — best available guess


def count_open_positions(client: Spot, symbols: list[str] = MEME_SYMBOLS) -> int:
    """Number of open GAMBLE TRADES (not order-lists) across our symbols.
    A laddered trade places multiple OCO order-lists for one logical bet —
    counting raw orderListId over-counts it as N positions instead of 1.
    Each leg's clientOrderId is tagged with a shared trade-id (see
    make_trade_id/place_oco_exit); group by that instead."""
    trade_ids = set()
    for sym in symbols:
        for order in client.get_open_orders(symbol=f"{sym}USDT"):
            m = _TRADE_ID_RE.match(order.get("clientOrderId", ""))
            if m:
                trade_ids.add(m.group("tid"))
    return len(trade_ids)


_LEG_RE = re.compile(r"^g(?P<tid>[0-9a-f]{10})L(?P<leg>\d)[AB]$")


def get_open_positions(client: Spot) -> list[dict]:
    """One row per OCO leg. entry is IMPLIED from the stop-loss level using
    that trade's actual stop_loss_pct (ATR-derived, varies per trade — see
    _get_trade_stop_pct), not the literal fill price — an approximation,
    not a ledger. Scans ALL open orders account-wide (no symbol list) since
    the bot trades a dynamic universe now — a fixed list here would miss
    any newly-discovered token's position."""
    legs: dict[tuple, dict] = {}
    for o in client.get_open_orders():
        m = _LEG_RE.match(o.get("clientOrderId", ""))
        if not m:
            continue
        sym = o["symbol"].removesuffix("USDT")
        key = (sym, m.group("tid"), int(m.group("leg")))
        rec = legs.setdefault(key, {})
        if o["type"] == "LIMIT_MAKER":
            rec["qty"] = Decimal(o["origQty"])
            rec["tp"] = Decimal(o["price"])
        elif o["type"] == "STOP_LOSS_LIMIT":
            rec["sl"] = Decimal(o["stopPrice"])

    price_cache: dict[str, Decimal] = {}
    rows = []
    for (sym, tid, leg_idx), rec in legs.items():
        if "qty" not in rec:
            continue  # incomplete leg (race with a just-filled order), skip
        if sym not in price_cache:
            price_cache[sym] = Decimal(client.ticker_price(symbol=f"{sym}USDT")["price"])
        current = price_cache[sym]
        if "sl" not in rec:
            # OCO cancels the SL leg the instant the TP leg starts filling —
            # a partial TP fill leaves the unsold remainder with no downside
            # protection at all. Surface it instead of silently dropping it
            # (previously invisible: REZ sat here for hours before being
            # noticed and re-hedged).
            rows.append({
                "symbol": sym, "trade_id": tid, "leg": leg_idx,
                "qty": rec["qty"], "entry": None, "tp": rec.get("tp"), "sl": None,
                "current": current, "pnl_pct": None, "unprotected": True,
            })
            continue
        stop_pct = _get_trade_stop_pct(tid)
        entry = rec["sl"] / (1 - stop_pct)
        pnl_pct = (current / entry - 1) * 100
        rows.append({
            "symbol": sym, "trade_id": tid, "leg": leg_idx,
            "qty": rec["qty"], "entry": entry, "tp": rec.get("tp"), "sl": rec["sl"],
            "current": current, "pnl_pct": pnl_pct, "unprotected": False,
        })
    return sorted(rows, key=lambda r: (r["symbol"], r["trade_id"], r["leg"]))


def get_portfolio_exposure(client: Spot) -> tuple[Decimal, Decimal]:
    """(total_portfolio_usdt, deployed_usdt). deployed = current USD value of
    OUR OWN open positions — derived from our tagged OCO leg orders, NOT raw
    account balances. Binance testnet seeds every account with a large
    faucet basket (429 unrelated assets on this account: 1 WBTC ≈ $65k,
    10k USDC, 10k FDUSD, 1 ETH, PAXG, XAUT, ...) that has nothing to do with
    the bot; summing all balances inflated 'total' to ~$326k when the real
    number (our USDT + our two actual positions) was a small fraction of
    that. total = free USDT + deployed."""
    usdt_free = Decimal("0")
    for b in client.account()["balances"]:
        if b["asset"] == "USDT":
            usdt_free = Decimal(b["free"])
            break

    qty_by_symbol: dict[str, Decimal] = {}
    for o in client.get_open_orders():
        if o["type"] != "STOP_LOSS_LIMIT" or not _TRADE_ID_RE.match(o.get("clientOrderId", "")):
            continue  # STOP_LOSS_LIMIT and LIMIT_MAKER legs share the same qty — count once
        sym = o["symbol"].removesuffix("USDT")
        qty_by_symbol[sym] = qty_by_symbol.get(sym, Decimal("0")) + Decimal(o["origQty"])

    deployed = Decimal("0")
    for sym, qty in qty_by_symbol.items():
        price = Decimal(client.ticker_price(symbol=f"{sym}USDT")["price"])
        deployed += qty * price

    return usdt_free + deployed, deployed


def place_oco_exit(
    client: Spot,
    symbol: str,
    qty: Decimal,
    fill_price: Decimal,
    stop_loss_pct: Decimal = Decimal("0.15"),
    take_profit_pct: Decimal = Decimal("0.30"),
    trade_id: str | None = None,
    leg_index: int = 0,
) -> dict:
    """Place a take-profit + stop-loss OCO sell bracket for an already-held
    qty. `trade_id`/`leg_index` tag both leg orders' clientOrderId so
    count_open_positions can group multi-leg (laddered) trades back into one
    logical position — see count_open_positions."""
    pair = f"{symbol}USDT"
    filters = get_symbol_filters(client, pair)
    qty = _round_step(qty, filters["step_size"])

    take_profit = _round_step(fill_price * (1 + take_profit_pct), filters["tick_size"])
    stop_price = _round_step(fill_price * (1 - stop_loss_pct), filters["tick_size"])
    # stop-limit sits a hair below the stop trigger so it actually fills on a fast drop
    stop_limit = _round_step(stop_price * Decimal("0.995"), filters["tick_size"])

    extra = {}
    if trade_id is not None:
        extra["aboveClientOrderId"] = f"g{trade_id}L{leg_index}A"
        extra["belowClientOrderId"] = f"g{trade_id}L{leg_index}B"

    oco = client.new_oco_order(
        symbol=pair,
        side="SELL",
        quantity=str(qty),
        aboveType="LIMIT_MAKER",
        abovePrice=str(take_profit),
        belowType="STOP_LOSS_LIMIT",
        belowStopPrice=str(stop_price),
        belowPrice=str(stop_limit),
        belowTimeInForce="GTC",
        **extra,
    )
    return {"oco": oco, "qty": qty, "take_profit": take_profit, "stop_price": stop_price}


def _net_qty_after_fees(executed_qty: Decimal, fills: list, symbol: str) -> Decimal:
    """executed_qty minus any commission taken in the same asset — the
    actual amount now available to sell. NEVER derive this from a fresh
    get_free_balance() read: testnet accounts can carry a pre-existing
    (faucet-seeded) balance in the same asset, and blindly selling "whatever
    the balance shows" sells that unrelated stash too. Confirmed live: a real
    trade bought 186 ESP but a get_free_balance()-based sell tried to sell
    8266 — the account already held ~8080 ESP before the buy. Same bug hit
    a MEME trade (37,807 bought, 56,253 offered — oversold by 18,446, a
    suspiciously round number, likely a fixed testnet faucet seed per asset)."""
    commission = sum(Decimal(f["commission"]) for f in fills if f.get("commissionAsset") == symbol)
    return executed_qty - commission


def place_gamble_trade(
    client: Spot,
    symbol: str,
    usdt_amount: Decimal,
    stop_loss_pct: Decimal = Decimal("0.15"),
    take_profit_pct: Decimal = Decimal("0.30"),
) -> dict:
    """Market-buy `usdt_amount` of SYMBOLUSDT, then place an OCO sell bracket
    (take-profit limit + stop-loss) for the filled quantity."""
    pair = f"{symbol}USDT"
    filters = get_symbol_filters(client, pair)

    price = Decimal(client.ticker_price(symbol=pair)["price"])
    qty = _round_step(usdt_amount / price, filters["step_size"])
    if qty > filters["max_qty"]:
        print(f"  {pair}: ${usdt_amount:.2f} exceeds MARKET_LOT_SIZE cap "
              f"({filters['max_qty']} units) — clipping to max, leaving the rest of the cap-room undeployed this trade.")
        qty = _round_step(filters["max_qty"], filters["step_size"])
    if qty < filters["min_qty"] or qty * price < filters["min_notional"]:
        raise ValueError(
            f"{usdt_amount} USDT too small for {pair} "
            f"(min_notional={filters['min_notional']}, min_qty={filters['min_qty']})"
        )

    buy = client.new_order(symbol=pair, side="BUY", type="MARKET", quantity=str(qty))
    executed_qty = Decimal(buy.get("executedQty", "0"))
    if executed_qty <= 0:
        # ponytail: testnet order books for meme pairs are thin and can be
        # empty on either side — a MARKET order with nothing to fill against
        # just expires (no exception raised). Fail loud instead of trying to
        # sell an asset we never actually acquired.
        raise RuntimeError(
            f"Buy for {pair} did not fill — status={buy.get('status')}, "
            f"reason={buy.get('expiryReason', 'unknown')}. "
            f"Testnet likely has no ask-side liquidity for this pair right now."
        )
    fills = buy.get("fills", [])
    fill_price = (
        sum(Decimal(f["price"]) * Decimal(f["qty"]) for f in fills)
        / sum(Decimal(f["qty"]) for f in fills)
        if fills
        else Decimal(buy["cummulativeQuoteQty"]) / executed_qty
    )

    net_qty = _round_step(_net_qty_after_fees(executed_qty, fills, symbol), filters["step_size"])
    exit_ = place_oco_exit(
        client, symbol, net_qty, fill_price, stop_loss_pct, take_profit_pct,
        trade_id=make_trade_id(), leg_index=0,
    )
    return {"buy": buy, "fill_price": fill_price, **exit_}


LADDER_TP_PCTS: tuple[Decimal, ...] = (Decimal("0.15"), Decimal("0.30"))


def place_gamble_trade_laddered(
    client: Spot,
    symbol: str,
    usdt_amount: Decimal,
    tp_pcts: tuple[Decimal, ...] = LADDER_TP_PCTS,
    stop_loss_pct: Decimal = Decimal("0.15"),
) -> dict:
    """Same single market-buy as place_gamble_trade, but splits the exit into
    multiple OCO brackets at different take-profit levels (scale-out)
    instead of one all-or-nothing target. Quantity per leg is sized so each
    leg banks the SAME profit — not equal quantity — via
    equal_profit_fractions: a farther target gets a smaller slice."""
    legs = tuple(zip(equal_profit_fractions(tp_pcts), tp_pcts))
    pair = f"{symbol}USDT"
    filters = get_symbol_filters(client, pair)

    price = Decimal(client.ticker_price(symbol=pair)["price"])
    qty = _round_step(usdt_amount / price, filters["step_size"])
    if qty > filters["max_qty"]:
        print(f"  {pair}: ${usdt_amount:.2f} exceeds MARKET_LOT_SIZE cap "
              f"({filters['max_qty']} units) — clipping to max, leaving the rest of the cap-room undeployed this trade.")
        qty = _round_step(filters["max_qty"], filters["step_size"])
    if qty < filters["min_qty"] or qty * price < filters["min_notional"]:
        raise ValueError(
            f"{usdt_amount} USDT too small for {pair} "
            f"(min_notional={filters['min_notional']}, min_qty={filters['min_qty']})"
        )

    buy = client.new_order(symbol=pair, side="BUY", type="MARKET", quantity=str(qty))
    executed_qty = Decimal(buy.get("executedQty", "0"))
    if executed_qty <= 0:
        raise RuntimeError(
            f"Buy for {pair} did not fill — status={buy.get('status')}, "
            f"reason={buy.get('expiryReason', 'unknown')}. "
            f"Testnet likely has no ask-side liquidity for this pair right now."
        )
    fills = buy.get("fills", [])
    fill_price = (
        sum(Decimal(f["price"]) * Decimal(f["qty"]) for f in fills)
        / sum(Decimal(f["qty"]) for f in fills)
        if fills
        else Decimal(buy["cummulativeQuoteQty"]) / executed_qty
    )

    held = _round_step(_net_qty_after_fees(executed_qty, fills, symbol), filters["step_size"])
    remaining = held
    placed_legs = []
    trade_id = make_trade_id()
    _record_trade_stop_pct(trade_id, stop_loss_pct)
    for i, (fraction, tp_pct) in enumerate(legs):
        leg_qty = remaining if i == len(legs) - 1 else _round_step(held * fraction, filters["step_size"])
        leg_qty = min(leg_qty, remaining)
        if leg_qty < filters["min_qty"] or leg_qty * fill_price < filters["min_notional"]:
            print(f"  leg {i} ({leg_qty} {symbol}) too small for min_notional, skipping")
            continue
        exit_ = place_oco_exit(
            client, symbol, leg_qty, fill_price, stop_loss_pct, tp_pct,
            trade_id=trade_id, leg_index=i,
        )
        placed_legs.append({"take_profit_pct": tp_pct, **exit_})
        remaining -= leg_qty

    return {"buy": buy, "fill_price": fill_price, "qty": held, "legs": placed_legs, "trade_id": trade_id}


EARLY_EXIT_MIN_PROFIT_PCT = Decimal("2")  # don't bother early-exiting for noise-level gains
EARLY_EXIT_FORECAST_THRESHOLD = Decimal("-5")  # Kronos predicted % change below this = "fading"

# Trailing profit-lock. A SECOND, independent profit-protecting exit that does
# NOT depend on Kronos agreeing — the existing early exit above only fires when
# the forecast specifically calls a reversal, so BANK rode +67.5% down to
# +17.5% (2026-08-04) with nothing triggering. Three floors are computed per
# open trade and the TIGHTEST (highest, in pnl-point terms) wins; falling
# through it market-closes the position.
#
# UNITS: pnl_pct throughout this codebase is PERCENTAGE POINTS (67.5), while
# stop_loss_pct is a FRACTION (0.08). ratchet_floor_pct converts between them —
# every other floor here works in points.
TRAILING_ARM_MIN_PROFIT_PCT = Decimal(os.environ.get("TRAILING_ARM_MIN_PROFIT_PCT", "10"))  # whole system stays dormant below this peak; the static stop-loss governs, as before
CHANDELIER_ATR_MULTIPLIER = Decimal(os.environ.get("CHANDELIER_ATR_MULTIPLIER", "3"))  # LeBeau's Chandelier Exit, the managed-futures standard: trail ATR x this off the peak
PEAK_GIVEBACK_PCT = Decimal(os.environ.get("PEAK_GIVEBACK_PCT", "0.35"))  # hard ceiling on retracement: never hand back more than this fraction of peak profit
RATCHET_STEP_R = Decimal(os.environ.get("RATCHET_STEP_R", "2"))  # R-multiple ratchet: floor sits this many R below the peak, so a +2R winner can never become a loser


def chandelier_floor_pct(
    entry: Decimal, peak_price: Decimal, atr: Decimal | None, is_short: bool = False
) -> Decimal | None:
    """Chandelier Exit (Chuck LeBeau) expressed as a pnl-point floor: trail
    CHANDELIER_ATR_MULTIPLIER x ATR off the best price the trade has seen.
    Volatility-adaptive — a wild coin gets room to breathe, a calm one is held
    tight — which is why it's the trend-following standard and why it fits a
    bot that already sizes stops off ATR everywhere else.

    `peak_price` is the best price for the position's direction: the HIGHEST
    price seen for a long, the LOWEST for a short. `atr` is in absolute price
    units (what scanner._compute_atr returns), None when unavailable this
    cycle — returns None then, and the other floors still apply.

    A long's floor can compute below zero on a huge ATR; that's left alone
    rather than special-cased, since a nonsense-low floor is simply never the
    max() winner in effective_trailing_floor and so never binds."""
    if atr is None or entry <= 0 or peak_price <= 0:
        return None
    offset = atr * CHANDELIER_ATR_MULTIPLIER
    if is_short:
        floor_price = peak_price + offset  # short profits as price falls, so the floor sits ABOVE the low-water mark
        return (entry / floor_price - 1) * 100
    floor_price = peak_price - offset
    return (floor_price / entry - 1) * 100


def ratchet_floor_pct(peak_pnl_pct: Decimal, stop_loss_pct: Decimal) -> Decimal | None:
    """R-multiple ratchet: floor = peak minus RATCHET_STEP_R units of the
    trade's own initial risk R (its stop_loss_pct). Encodes the systematic-fund
    rule "never let a +2R winner turn into a loser" — at exactly 2R the floor
    lands on breakeven, and it advances from there.

    Continuous rather than stepped (2R -> breakeven, 3R -> 1R, ...): identical
    at every step point, without the discretization edge cases.

    Returns None below RATCHET_STEP_R of peak profit (not armed yet), which is
    also what keeps the floor from ever computing negative."""
    if stop_loss_pct <= 0:
        return None
    r_points = stop_loss_pct * 100  # stop_loss_pct is a fraction; peak_pnl_pct is points
    if peak_pnl_pct < RATCHET_STEP_R * r_points:
        return None
    return peak_pnl_pct - RATCHET_STEP_R * r_points


def giveback_floor_pct(peak_pnl_pct: Decimal) -> Decimal | None:
    """Per-trade high-water-mark drawdown limit — the same shape as a fund's
    own drawdown rule, applied to one position's unrealized profit. Purely
    proportional, so unlike the Chandelier it doesn't care about volatility;
    it's the backstop that caps give-back when ATR is wide enough to let the
    Chandelier trail loosely."""
    if peak_pnl_pct <= 0:
        return None
    return peak_pnl_pct * (1 - PEAK_GIVEBACK_PCT)


def effective_trailing_floor(
    entry: Decimal, peak_price: Decimal, peak_pnl_pct: Decimal,
    stop_loss_pct: Decimal, atr: Decimal | None, is_short: bool = False,
) -> Decimal | None:
    """The tightest of the three floors, or None if the system isn't armed.

    Higher floor = tighter, so max() picks whichever mechanism would bail
    first; each floor independently returns None when its own precondition
    isn't met, and those are simply left out of the comparison. None overall
    means "no floor applies, leave this position to its static bracket"."""
    if peak_pnl_pct < TRAILING_ARM_MIN_PROFIT_PCT:
        return None
    floors = [
        f for f in (
            chandelier_floor_pct(entry, peak_price, atr, is_short),
            ratchet_floor_pct(peak_pnl_pct, stop_loss_pct),
            giveback_floor_pct(peak_pnl_pct),
        )
        if f is not None
    ]
    return max(floors) if floors else None


def _test_trailing_floors() -> None:
    # --- chandelier ---
    # long: entry 100, peak 150, ATR 5 -> floor price 150 - 15 = 135 -> +35%
    assert chandelier_floor_pct(Decimal("100"), Decimal("150"), Decimal("5")) == Decimal("35")
    # short: entry 100, peak (low) 50, ATR 5 -> floor price 50 + 15 = 65 -> 100/65-1 = +53.8%
    short_floor = chandelier_floor_pct(Decimal("100"), Decimal("50"), Decimal("5"), is_short=True)
    assert abs(short_floor - Decimal("53.846")) < Decimal("0.01"), short_floor
    # no ATR this cycle -> chandelier abstains, doesn't raise
    assert chandelier_floor_pct(Decimal("100"), Decimal("150"), None) is None
    assert chandelier_floor_pct(Decimal("0"), Decimal("150"), Decimal("5")) is None  # guard, no div-by-zero

    # --- ratchet --- (R = 8% -> 8 points)
    r = Decimal("0.08")
    assert ratchet_floor_pct(Decimal("10"), r) is None  # 1.25R peak, below the 2R arming bar
    assert ratchet_floor_pct(Decimal("16"), r) == Decimal("0")  # exactly 2R -> breakeven floor
    assert ratchet_floor_pct(Decimal("24"), r) == Decimal("8")  # 3R -> floor at 1R
    assert ratchet_floor_pct(Decimal("67.5"), r) == Decimal("51.5")  # the BANK case
    assert ratchet_floor_pct(Decimal("50"), Decimal("0")) is None  # no R -> can't ratchet

    # --- giveback ---
    assert giveback_floor_pct(Decimal("100")) == Decimal("65")  # 35% giveback allowed
    assert abs(giveback_floor_pct(Decimal("67.5")) - Decimal("43.875")) < Decimal("0.001")
    assert giveback_floor_pct(Decimal("-5")) is None  # never armed on a losing position

    # --- composition ---
    # below the arming bar -> no floor at all, static bracket still governs
    assert effective_trailing_floor(
        Decimal("100"), Decimal("105"), Decimal("5"), r, Decimal("1")
    ) is None
    # BANK: entry 0.06532, peak (low) 0.039, peak pnl +67.5%, R 8%, ATR ~0.002
    bank = effective_trailing_floor(
        Decimal("0.06532"), Decimal("0.039"), Decimal("67.5"), r, Decimal("0.002"), is_short=True,
    )
    assert bank is not None
    # tightest of: chandelier ~45.4, ratchet 51.5, giveback 43.9 -> ratchet wins
    assert abs(bank - Decimal("51.5")) < Decimal("0.01"), bank
    # ...and it would have fired: the position sank to +17.5%, far under that floor
    assert Decimal("17.5") < bank

    # armed, but ATR missing -> still floors off ratchet/giveback rather than abstaining
    no_atr = effective_trailing_floor(
        Decimal("100"), Decimal("150"), Decimal("50"), r, None,
    )
    assert no_atr == Decimal("34"), no_atr  # max(ratchet 50-16=34, giveback 32.5)

PEAK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio_peak.json")
DRAWDOWN_CIRCUIT_BREAKER_PCT = Decimal(os.environ.get("DRAWDOWN_CIRCUIT_BREAKER_PCT", "0.20"))
MIN_BET_FRACTION = Decimal(os.environ.get("MIN_BET_FRACTION", "0.5"))  # skip a trade if available room is below this fraction of the normal per-trade size
BREADTH_WEAK_UP_PCT = Decimal(os.environ.get("BREADTH_WEAK_UP_PCT", "0.35"))  # below this fraction of the universe up -> weak-breadth regime
BREADTH_SPLIT_MULTIPLIER = int(os.environ.get("BREADTH_SPLIT_MULTIPLIER", "2"))  # weak breadth doubles POSITION_SPLIT -> halves bet size, same mechanism as the drawdown circuit breaker but triggered by market conditions instead of our own P&L

ATR_MULTIPLIER = Decimal(os.environ.get("ATR_MULTIPLIER", "2"))
ATR_MIN_STOP_LOSS_PCT = Decimal(os.environ.get("ATR_MIN_STOP_LOSS_PCT", "0.08"))
ATR_MAX_STOP_LOSS_PCT = Decimal(os.environ.get("ATR_MAX_STOP_LOSS_PCT", "0.25"))
DEFAULT_STOP_LOSS_PCT = Decimal("0.15")  # fallback when ATR is unavailable this cycle
FIB_STOP_BUFFER_PCT = Decimal(os.environ.get("FIB_STOP_BUFFER_PCT", "0.02"))
FIB_MAX_SCALE_FACTOR = Decimal(os.environ.get("FIB_MAX_SCALE_FACTOR", "3"))


def compute_fib_stop_loss_pct(symbol: str, price: Decimal, is_short: bool = False) -> tuple[Decimal, Decimal] | None:
    """Stop-loss sized to the 23.6% Fibonacci retracement level instead of
    pure ATR volatility — ties the stop to actual price structure. Pushed
    FIB_STOP_BUFFER_PCT further out so the stop doesn't sit exactly on the
    level Fib traders themselves watch (mirrors place_oco_exit's
    stop_price * 0.995 "hair below the trigger" idea).

    Returns (stop_loss_pct, scale_factor): stop_loss_pct is clamped to
    [ATR_MIN_STOP_LOSS_PCT, ATR_MAX_STOP_LOSS_PCT] as before; scale_factor
    is how much that clamp stretched the raw (pre-clamp) distance — 1.0
    when the clamp didn't bind. compute_fib_take_profit_pcts stretches its
    own legs by this same factor so the bracket's ratio survives clamping
    intact (2026-08-01 whole-branch review, second pass).

    If scale_factor would exceed FIB_MAX_SCALE_FACTOR (default 3x, mirroring
    ATR_MAX_STOP_LOSS_PCT/ATR_MIN_STOP_LOSS_PCT's own 3.125x range), this
    returns None instead of an inflated pair: price sitting that close to
    the 23.6% level carries no real structural meaning, and force-scaling
    produced take-profits over 600% away that still passed the R:R gate
    (2026-08-01 whole-branch review, third pass — measured live up to
    104x scale / 647% TP / 80:1 "ratio" that the gate couldn't catch,
    since both sides of the ratio were inflated together). Returns None on
    a fetch failure or an over-cap scale — caller falls back to
    compute_atr_stop_loss_pct (and compute_atr_take_profit_pcts, together
    — never mix a Fib stop with an ATR take-profit or vice versa)."""
    if price <= 0:
        return None
    levels = get_fibonacci_levels(symbol, is_short=is_short)
    if levels is None:
        return None
    level_23_6 = Decimal(str(levels["retracements"][23.6]))
    raw_pct = abs(price - level_23_6) / price
    buffered_pct = raw_pct * (1 + FIB_STOP_BUFFER_PCT)
    clamped_pct = max(ATR_MIN_STOP_LOSS_PCT, min(ATR_MAX_STOP_LOSS_PCT, buffered_pct))
    scale = clamped_pct / buffered_pct if buffered_pct > 0 else Decimal("1")
    if scale > FIB_MAX_SCALE_FACTOR:
        return None
    return (clamped_pct, scale)


def compute_fib_take_profit_pcts(symbol: str, price: Decimal, scale: Decimal, is_short: bool = False) -> tuple[Decimal, Decimal] | None:
    """Take-profit legs sized to the 161.8%/261.8% Fibonacci extension
    levels, stretched by `scale` (see compute_fib_stop_loss_pct) so the
    stop/take-profit pair's ratio survives the stop's ATR-bound clamping
    intact. Always pass the scale_factor returned alongside the paired
    stop — never call this with a scale from a different symbol/cycle.
    Returns None if Fibonacci levels aren't available this cycle — caller
    falls back to compute_atr_take_profit_pcts."""
    if price <= 0:
        return None
    levels = get_fibonacci_levels(symbol, is_short=is_short)
    if levels is None:
        return None
    tp1 = Decimal(str(levels["extensions"][161.8]))
    tp2 = Decimal(str(levels["extensions"][261.8]))
    return (abs(tp1 - price) / price * scale, abs(tp2 - price) / price * scale)


FIB_SIGNAL_NUDGE = Decimal(os.environ.get("FIB_SIGNAL_NUDGE", "0.1"))


def fib_entry_signal(symbol: str, price: Decimal, is_short: bool = False) -> Decimal:
    """Small additive nudge into compute_opportunity_score's fib_score —
    same pattern and magnitude as the Fear & Greed nudge. Positive if price
    is holding on the "structurally intact" side of the 50% retracement
    (above it for a long, below it for a short); negative if price has
    already broken past the 61.8% level in the wrong direction (structure
    weakening, thesis in doubt). Neutral (0) if Fibonacci levels aren't
    available this cycle — never blocks, mirrors how fear_greed=None means
    no nudge in compute_opportunity_score."""
    if price <= 0:
        return Decimal("0")
    levels = get_fibonacci_levels(symbol, is_short=is_short)
    if levels is None:
        return Decimal("0")
    level_50 = Decimal(str(levels["retracements"][50.0]))
    level_61_8 = Decimal(str(levels["retracements"][61.8]))
    if is_short:
        if price <= level_50:
            return FIB_SIGNAL_NUDGE
        if price >= level_61_8:
            return -FIB_SIGNAL_NUDGE
    else:
        if price >= level_50:
            return FIB_SIGNAL_NUDGE
        if price <= level_61_8:
            return -FIB_SIGNAL_NUDGE
    return Decimal("0")


def compute_atr_stop_loss_pct(symbol: str, price: Decimal) -> Decimal:
    """Volatility-adjusted stop, replacing one flat % for every symbol —
    Turtle-style: stop = ATR_MULTIPLIER x ATR, expressed as a fraction of
    price. A calm coin (small ATR) gets a tight stop; a genuinely volatile
    one gets room to breathe before an ordinary swing reads as a reversal.
    Clamped to [ATR_MIN_STOP_LOSS_PCT, ATR_MAX_STOP_LOSS_PCT] so a near-zero
    or huge ATR reading can't produce a degenerate stop. Falls back to
    DEFAULT_STOP_LOSS_PCT if ATR can't be computed this cycle (used on
    both the spot-long and futures-short paths — same volatility logic
    either direction)."""
    atr = get_atr(symbol)
    if atr is None or price <= 0:
        return DEFAULT_STOP_LOSS_PCT
    raw_pct = (Decimal(str(atr)) / price) * ATR_MULTIPLIER
    return max(ATR_MIN_STOP_LOSS_PCT, min(ATR_MAX_STOP_LOSS_PCT, raw_pct))


def _test_compute_atr_stop_loss_pct() -> None:
    # ATR unavailable (bad symbol) -> falls back to the flat default
    assert compute_atr_stop_loss_pct("__NOPE__", Decimal("100")) == DEFAULT_STOP_LOSS_PCT
    # price <= 0 -> also falls back rather than dividing by zero
    assert compute_atr_stop_loss_pct("BTC", Decimal("0")) == DEFAULT_STOP_LOSS_PCT


def _test_compute_fib_stop_loss_pct() -> None:
    # bad symbol -> get_fibonacci_levels can't fetch -> None -> caller falls back to ATR
    assert compute_fib_stop_loss_pct("__NOPE__", Decimal("100")) is None
    # price <= 0 -> also None rather than dividing by zero
    assert compute_fib_stop_loss_pct("BTC", Decimal("0")) is None
    # price sitting a hair off the 23.6% level -> raw distance far under the
    # ATR_MIN_STOP_LOSS_PCT floor -> the clamp's scale factor blows past
    # FIB_MAX_SCALE_FACTOR -> None rather than an inflated pair (2026-08-01
    # review, third pass: unbounded scale reached 104x, producing 647%
    # take-profits that still passed the R:R gate because the inflated TP
    # inflated the ratio too)
    levels = get_fibonacci_levels("BTC")
    if levels is not None:  # skip when the klines fetch is unavailable
        near_level = Decimal(str(levels["retracements"][23.6])) * Decimal("1.005")
        assert compute_fib_stop_loss_pct("BTC", near_level) is None


def _test_compute_fib_take_profit_pcts() -> None:
    assert compute_fib_take_profit_pcts("__NOPE__", Decimal("100"), Decimal("1")) is None
    assert compute_fib_take_profit_pcts("BTC", Decimal("0"), Decimal("1")) is None


def _test_fib_entry_signal() -> None:
    assert fib_entry_signal("__NOPE__", Decimal("100")) == Decimal("0")
    assert fib_entry_signal("BTC", Decimal("0")) == Decimal("0")


SL_POLICY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sl_policy.json")


def _current_unrealized_pnl() -> Decimal:
    """Live unrealized $ pnl across spot + futures right now. Local-imports
    execute_futures (which itself imports from this module) to avoid a
    circular import at load time. Returns 0 if credentials/env aren't
    loaded — callers treat that as "no unrealized info available", not
    "no open positions"."""
    from execute_futures import get_futures_client, get_open_short_positions

    total = Decimal(0)
    try:
        for r in get_open_positions(get_client()):
            if r["entry"] is not None:
                total += r["qty"] * (r["current"] - r["entry"])
        for r in get_open_short_positions(get_futures_client()):
            total += r["qty"] * (r["entry"] - r["current"])
    except Exception as e:
        print(f"  (fixed-SL check: couldn't fetch live unrealized — {e})")
    return total


def activate_fixed_stop_loss() -> None:
    """Force flat DEFAULT_STOP_LOSS_PCT/FUTURES_STOP_LOSS_PCT on new trades
    instead of the ATR-scaled stop, until net $ pnl CHANGE since activation
    (realized pnl from trades closed after this point, plus the move in the
    currently-open book's unrealized value relative to its value right now)
    turns positive again. Snapshots today's unrealized as a baseline — the
    pre-existing open book's value at activation must not itself count as
    "profit since activation", only its change from here. Manual override
    after a run of wide ATR stops (the MMT 25%-stop loss, 2026-07-31) ate an
    outsized chunk of capital in one trade — see use_fixed_stop_loss."""
    with open(SL_POLICY_FILE, "w") as f:
        json.dump({
            "activated_at": datetime.now(timezone.utc).isoformat(),
            "baseline_unrealized": str(_current_unrealized_pnl()),
        }, f)


def _realized_pnl_since(since: str) -> Decimal:
    """Realized $ pnl from both journals for trades closed_at >= since."""
    from execute_futures import SHORT_JOURNAL_FILE

    total = Decimal(0)
    for path, is_short in ((TRADE_JOURNAL_FILE, False), (SHORT_JOURNAL_FILE, True)):
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("pnl_pct") is None or r["closed_at"] < since:
                    continue
                qty, entry, exitp = Decimal(str(r["qty"])), Decimal(str(r["entry"])), Decimal(str(r["exit_price"]))
                total += qty * (entry - exitp) if is_short else qty * (exitp - entry)
    return total


def use_fixed_stop_loss() -> bool:
    """True while the fixed-SL policy (see activate_fixed_stop_loss) is active
    and the account hasn't recovered to a net positive CHANGE since activation
    yet (realized-since + unrealized-now-vs-baseline). Self-deactivates
    (deletes the state file) the moment it finds recovery — flips back to
    ATR-scaled stops automatically, no manual step to turn it back off."""
    if not os.path.exists(SL_POLICY_FILE):
        return False
    with open(SL_POLICY_FILE) as f:
        policy = json.load(f)
    realized = _realized_pnl_since(policy["activated_at"])
    unrealized_change = _current_unrealized_pnl() - Decimal(policy["baseline_unrealized"])
    net = realized + unrealized_change
    if net > 0:  # strictly positive — a flat 0 (no change yet) must not read as "recovered"
        print(f"  fixed-SL policy: net change since activation is {net:+.2f} USDT — back in profit, reverting to ATR stops.")
        os.remove(SL_POLICY_FILE)
        return False
    print(f"  fixed-SL policy active: net change since activation is {net:+.2f} USDT — still using flat stop.")
    return True


def _test_use_fixed_stop_loss() -> None:
    import tempfile
    import execute as _execute_module

    global SL_POLICY_FILE
    original_file = SL_POLICY_FILE
    original_unrealized = _execute_module._current_unrealized_pnl
    original_realized = _execute_module._realized_pnl_since
    fd, SL_POLICY_FILE = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(SL_POLICY_FILE)  # exists()-check should treat a missing file as inactive
    try:
        assert use_fixed_stop_loss() is False  # no policy file -> ATR mode as normal

        # pre-existing open book is worth +200 unrealized at the moment we flip
        # the switch — that +200 must NOT itself count as "profit since
        # activation" once the policy is live (this was the real bug: the
        # first version folded the whole current unrealized snapshot in as if
        # it were new, so it self-deactivated the instant it was turned on).
        _execute_module._current_unrealized_pnl = lambda: Decimal("200")
        activate_fixed_stop_loss()
        assert os.path.exists(SL_POLICY_FILE)
        _execute_module._realized_pnl_since = lambda since: Decimal("0")
        assert use_fixed_stop_loss() is True  # no change yet -> stays active despite +200 baseline
        assert os.path.exists(SL_POLICY_FILE)

        # book unrealized dropped further, no new trades closed -> still active
        _execute_module._current_unrealized_pnl = lambda: Decimal("150")
        assert use_fixed_stop_loss() is True
        assert os.path.exists(SL_POLICY_FILE)

        # book recovers back above the +200 baseline -> net change positive -> reverts
        _execute_module._current_unrealized_pnl = lambda: Decimal("201")
        assert use_fixed_stop_loss() is False
        assert not os.path.exists(SL_POLICY_FILE)  # self-deactivated
    finally:
        _execute_module._current_unrealized_pnl = original_unrealized
        _execute_module._realized_pnl_since = original_realized
        if os.path.exists(SL_POLICY_FILE):
            os.remove(SL_POLICY_FILE)
        SL_POLICY_FILE = original_file


ATR_TP_RATIO_MULTIPLIERS = (
    Decimal(os.environ.get("ATR_TP_RATIO_1", "2")),
    Decimal(os.environ.get("ATR_TP_RATIO_2", "4")),
)


def compute_atr_take_profit_pcts(stop_loss_pct: Decimal) -> tuple[Decimal, Decimal]:
    """ATR-scaled take-profit legs, DERIVED FROM the already-computed ATR
    stop-loss rather than independently re-clamped from raw ATR.

    First attempt clamped stop and TP separately from their own
    [min, max] bounds — found live to be internally inconsistent: a calm
    coin clamps the stop to its 8% floor and the TP to its own 10% floor,
    giving a 1.25:1 ratio that doesn't match MIN_REWARD_RISK_RATIO at all;
    a volatile coin can push BOTH TP legs past their ceiling to the same
    clamped value, collapsing the two-leg ladder into one. Deriving TP as
    a straight multiple of the stop (ATR_TP_RATIO_1/2, default 2x/4x)
    fixes both: the ratio is exact by construction regardless of where the
    stop landed in its own clamp range, and leg 2 is always double leg 1's
    distance so the ladder never degenerates. 'Crypto Trading
    Strategies.md' asks for ATR-calibrated targets, not just stops — this
    achieves that while staying self-consistent with the R:R gate."""
    return tuple(stop_loss_pct * mult for mult in ATR_TP_RATIO_MULTIPLIERS)


def _test_compute_atr_take_profit_pcts() -> None:
    legs = compute_atr_take_profit_pcts(Decimal("0.10"))
    assert legs == (Decimal("0.20"), Decimal("0.40")), legs
    assert legs[1] == legs[0] * 2, "leg 2 must always be twice leg 1's distance"


COUNTER_TREND_THRESHOLD_PCT = Decimal(os.environ.get("COUNTER_TREND_THRESHOLD_PCT", "5"))


def is_counter_trend(daily_pct_change: Decimal | None, is_short: bool = False) -> bool:
    """'Crypto Trading Strategies.md': counter-trend trades are higher risk
    than reversals that align with the larger trend ("taking a counter-
    trend trade is higher risk... generally safer [to trade] in the
    direction of the macro trend"). A 1h-bullish long signal fighting a
    meaningfully bearish 24h trend (or the mirror for a 1h-bearish short
    fighting a meaningfully bullish 24h trend) is exactly that shape.
    None (no 24h data this cycle) never blocks — this is an extra caution
    layer on top of the other gates, not a required signal on its own."""
    if daily_pct_change is None:
        return False
    if is_short:
        return daily_pct_change >= COUNTER_TREND_THRESHOLD_PCT
    return daily_pct_change <= -COUNTER_TREND_THRESHOLD_PCT


def _test_is_counter_trend() -> None:
    # long fighting a meaningfully bearish 24h trend -> counter-trend
    assert is_counter_trend(Decimal("-8")) is True
    # long with a mildly negative 24h (noise, not a real trend) -> not flagged
    assert is_counter_trend(Decimal("-2")) is False
    # long with a positive/neutral 24h -> aligned, not counter-trend
    assert is_counter_trend(Decimal("3")) is False
    # no 24h data -> never blocks
    assert is_counter_trend(None) is False
    # short fighting a meaningfully bullish 24h trend -> counter-trend
    assert is_counter_trend(Decimal("8"), is_short=True) is True
    assert is_counter_trend(Decimal("-8"), is_short=True) is False


VWAP_DEVIATION_THRESHOLD_PCT = Decimal(os.environ.get("VWAP_DEVIATION_THRESHOLD_PCT", "10"))


def compute_vwap_deviation_pct(price: Decimal, vwap: Decimal) -> Decimal:
    """% price sits above (+) or below (-) the rolling VWAP."""
    return (price / vwap - 1) * 100


def is_extended_from_vwap(deviation_pct: Decimal | None, is_short: bool = False) -> bool:
    """'Crypto Trading Strategies.md': VWAP acts as a fair-value magnet —
    price stretched too far from it tends to revert. A long chasing price
    already far ABOVE VWAP, or a short chasing price already far BELOW
    VWAP, is buying/selling into that reversion instead of with it. None
    (no VWAP data this cycle) never blocks."""
    if deviation_pct is None:
        return False
    if is_short:
        return deviation_pct <= -VWAP_DEVIATION_THRESHOLD_PCT
    return deviation_pct >= VWAP_DEVIATION_THRESHOLD_PCT


def _test_vwap_deviation() -> None:
    dev = compute_vwap_deviation_pct(Decimal("110"), Decimal("100"))
    assert dev == Decimal("10"), dev
    assert is_extended_from_vwap(Decimal("12")) is True  # long, far above VWAP
    assert is_extended_from_vwap(Decimal("5")) is False  # long, mild stretch
    assert is_extended_from_vwap(Decimal("-12")) is False  # long, far below VWAP is fine (not chasing)
    assert is_extended_from_vwap(None) is False
    assert is_extended_from_vwap(Decimal("-12"), is_short=True) is True  # short, far below VWAP
    assert is_extended_from_vwap(Decimal("12"), is_short=True) is False


def obv_warns_against(divergence: str | None, is_short: bool = False) -> bool:
    """'Crypto Trading Strategies.md': an OBV divergence means volume isn't
    confirming price — the move is more fragile than it looks. A 'bearish'
    divergence (price up, volume distributing) warns against going long;
    the mirror 'bullish' divergence (price down, volume accumulating)
    warns against going short. None (no divergence, or no data) never
    blocks — this only fires when volume is actively disagreeing."""
    if divergence is None:
        return False
    return divergence == ("bullish" if is_short else "bearish")


def _test_obv_warns_against() -> None:
    assert obv_warns_against("bearish") is True  # long, price up unconfirmed
    assert obv_warns_against("bullish") is False  # long, no conflict
    assert obv_warns_against(None) is False
    assert obv_warns_against("bullish", is_short=True) is True  # short, price down unconfirmed
    assert obv_warns_against("bearish", is_short=True) is False


MIN_REWARD_RISK_RATIO = Decimal(os.environ.get("MIN_REWARD_RISK_RATIO", "2.0"))


def meets_min_reward_risk(tp_pct: Decimal, stop_loss_pct: Decimal, min_ratio: Decimal = MIN_REWARD_RISK_RATIO) -> bool:
    """Reward:risk gate ('Crypto Trading Strategies.md': demand >=3:1 R:R on
    high-risk momentum trades so even a sub-50% win rate nets positive).
    MIN_REWARD_RISK_RATIO defaults to 2.0, a bit below the source's 3:1, to
    stay compatible with the ladder's real TP1/stop shape rather than
    rejecting almost everything on day one — raise it toward 3.0 to match
    the source material more strictly once there's live data to tune from."""
    if stop_loss_pct <= 0:
        return False
    return (tp_pct / stop_loss_pct) >= min_ratio


def _test_meets_min_reward_risk() -> None:
    assert meets_min_reward_risk(Decimal("0.30"), Decimal("0.10")) is True  # 3:1
    assert meets_min_reward_risk(Decimal("0.15"), Decimal("0.10")) is False  # 1.5:1, below 2.0 default
    assert meets_min_reward_risk(Decimal("0.10"), Decimal("0")) is False  # no div-by-zero


KRONOS_NORMALIZE_PCT = Decimal(os.environ.get("KRONOS_NORMALIZE_PCT", "5"))  # a +/-5% Kronos forecast maps to +/-1.0 conviction
KRONOS_VETO_THRESHOLD = Decimal(os.environ.get("KRONOS_VETO_THRESHOLD", "-5"))  # hard reject below this, no matter how confident the LLM is
KRONOS_WEIGHT = Decimal(os.environ.get("KRONOS_WEIGHT", "0.3"))
LLM_WEIGHT = Decimal(os.environ.get("LLM_WEIGHT", "0.7"))
OPPORTUNITY_THRESHOLD = Decimal(os.environ.get("OPPORTUNITY_THRESHOLD", "0.3"))

FEAR_GREED_EXTREME_GREED = int(os.environ.get("FEAR_GREED_EXTREME_GREED", "75"))
FEAR_GREED_EXTREME_FEAR = int(os.environ.get("FEAR_GREED_EXTREME_FEAR", "25"))
FEAR_GREED_NUDGE = Decimal(os.environ.get("FEAR_GREED_NUDGE", "0.1"))


def compute_opportunity_score(
    llm_confidence: Decimal, kronos_pct: Decimal | None,
    fear_greed: int | None = None, is_short: bool = False, fib_score: Decimal = Decimal("0"),
) -> dict:
    """Two independent confidence scores blended into one buy/no-buy gate,
    instead of the LLM's own confidence being the only thing that decides.

    Real pattern from the trade journal: NIL, PROM, PUMP, COTI all lost after
    the verdict's own reasoning text said Kronos was forecasting a decline —
    and got bought anyway because raw momentum pushed LLM confidence over the
    old flat 0.5 bar. Kronos's DIRECTION is real information the LLM has been
    observed talking itself past; this makes that direction load-bearing
    instead of just one more line of prompt context.

    - kronos_score: signed conviction from the ML forecast, -1 (max bearish)
      to +1 (max bullish), scaled by KRONOS_NORMALIZE_PCT. No forecast this
      cycle -> neutral (0), not penalized or rewarded.
    - llm_score: the verdict's own confidence (0-1) — already required to be
      a "buy" at >=0.5 before this runs.
    - veto: Kronos alone can hard-block a trade if it's bearish enough
      (<= KRONOS_VETO_THRESHOLD%, same -5% bar used elsewhere for early-exit
      "fading" calls) — no LLM confidence talks its way past that.
    - fear_greed: small CONTRARIAN nudge from the Fear & Greed Index, not a
      per-symbol signal — a macro overlay. Extreme greed nudges a long DOWN
      (be fearful when others are greedy) and a short UP; extreme fear does
      the opposite. Anything in between is neutral (no nudge).
    - is_short: mirrors the WHOLE function for the short side (used to live
      as a hand-duplicated copy in execute_futures.py — folded back in here
      so there's one implementation, not two to keep in sync). Flips which
      Kronos direction counts as "good" (bearish is good for a short) and
      which direction the veto fires on (bullish enough to risk a squeeze).
    - fib_score: additive nudge from fib_entry_signal (see that function) —
      structural confirmation from Fibonacci retracement levels, on the
      same footing as the Fear & Greed nudge. Defaults to 0 (no effect) so
      every existing caller keeps working unchanged.
    """
    if kronos_pct is None:
        kronos_score = Decimal("0")
    else:
        signed_pct = -kronos_pct if is_short else kronos_pct
        kronos_score = max(Decimal("-1"), min(Decimal("1"), signed_pct / KRONOS_NORMALIZE_PCT))

    llm_score = llm_confidence
    opportunity_score = KRONOS_WEIGHT * kronos_score + LLM_WEIGHT * llm_score + fib_score

    fear_greed_note = ""
    if fear_greed is not None:
        if fear_greed >= FEAR_GREED_EXTREME_GREED:
            nudge = FEAR_GREED_NUDGE if is_short else -FEAR_GREED_NUDGE
            opportunity_score += nudge
            fear_greed_note = f" Fear&Greed {fear_greed} (extreme greed) contrarian nudge {nudge:+.2f}."
        elif fear_greed <= FEAR_GREED_EXTREME_FEAR:
            nudge = -FEAR_GREED_NUDGE if is_short else FEAR_GREED_NUDGE
            opportunity_score += nudge
            fear_greed_note = f" Fear&Greed {fear_greed} (extreme fear) contrarian nudge {nudge:+.2f}."

    if is_short:
        veto = kronos_pct is not None and kronos_pct >= -KRONOS_VETO_THRESHOLD
    else:
        veto = kronos_pct is not None and kronos_pct <= KRONOS_VETO_THRESHOLD
    passes = opportunity_score >= OPPORTUNITY_THRESHOLD and not veto

    direction = "bullish" if kronos_score > 0 else ("bearish" if kronos_score < 0 else "neutral")
    veto_reason = (
        "bullish enough to risk a squeeze" if is_short else "bearish enough to fade"
    )
    fib_note = f" Fib structure nudge {fib_score:+.2f}." if fib_score != 0 else ""
    reasoning = (
        f"Kronos: {kronos_score:+.2f} conviction ({direction}"
        + (f", predicts {kronos_pct:+.2f}%" if kronos_pct is not None else ", no forecast this cycle")
        + f") x weight {KRONOS_WEIGHT} = {KRONOS_WEIGHT * kronos_score:+.2f}. "
        f"LLM: {llm_score:.2f} confidence x weight {LLM_WEIGHT} = {LLM_WEIGHT * llm_score:+.2f}."
        f"{fear_greed_note}{fib_note} "
        f"Opportunity score {opportunity_score:+.2f} vs threshold {OPPORTUNITY_THRESHOLD} -> "
        + ("PASSES." if passes else f"FAILS" + (f" (Kronos veto — {veto_reason} regardless of LLM confidence)." if veto else "."))
    )
    return {
        "kronos_score": kronos_score, "llm_score": llm_score,
        "opportunity_score": opportunity_score, "veto": veto,
        "passes": passes, "reasoning": reasoning,
    }


def _test_compute_opportunity_score() -> None:
    # strong LLM confidence but Kronos strongly bearish -> vetoed regardless
    r = compute_opportunity_score(Decimal("0.90"), Decimal("-8"))
    assert r["veto"] is True and r["passes"] is False, r

    # moderate LLM confidence, Kronos bullish -> passes
    r = compute_opportunity_score(Decimal("0.65"), Decimal("3"))
    assert r["passes"] is True, r

    # no Kronos forecast this cycle -> neutral, decision rides on LLM alone
    r = compute_opportunity_score(Decimal("0.80"), None)
    assert r["kronos_score"] == 0 and r["passes"] is True, r

    # moderate LLM confidence dragged below threshold by Kronos bearishness
    # (this is the exact NIL-#2 shape: llm=0.70, kronos=-6.31%)
    r = compute_opportunity_score(Decimal("0.70"), Decimal("-6.31"))
    assert r["passes"] is False, r

    # fear & greed contrarian nudges
    base = compute_opportunity_score(Decimal("0.60"), Decimal("1"))["opportunity_score"]
    greedy_long = compute_opportunity_score(Decimal("0.60"), Decimal("1"), fear_greed=90)
    assert greedy_long["opportunity_score"] < base, "extreme greed should nudge a long DOWN"
    fearful_long = compute_opportunity_score(Decimal("0.60"), Decimal("1"), fear_greed=10)
    assert fearful_long["opportunity_score"] > base, "extreme fear should nudge a long UP"
    base_short = compute_opportunity_score(Decimal("0.60"), Decimal("1"), is_short=True)["opportunity_score"]
    greedy_short = compute_opportunity_score(Decimal("0.60"), Decimal("1"), fear_greed=90, is_short=True)
    assert greedy_short["opportunity_score"] > base_short, "extreme greed should nudge a short UP"
    neutral = compute_opportunity_score(Decimal("0.60"), Decimal("1"), fear_greed=50)
    assert neutral["opportunity_score"] == base, "neutral fear/greed should not nudge at all"

    # is_short: bearish Kronos is GOOD for a short (opposite of long)
    r = compute_opportunity_score(Decimal("0.70"), Decimal("-6"), is_short=True)
    assert r["passes"] is True, f"bearish Kronos should favor a short, got {r}"
    # is_short veto: bullish enough Kronos threatens a squeeze -> vetoed
    r = compute_opportunity_score(Decimal("0.90"), Decimal("8"), is_short=True)
    assert r["veto"] is True and r["passes"] is False, f"strongly bullish Kronos should veto a short, got {r}"

    # fib structure nudge
    base_fib = compute_opportunity_score(Decimal("0.60"), Decimal("1"))["opportunity_score"]
    bullish_fib = compute_opportunity_score(Decimal("0.60"), Decimal("1"), fib_score=Decimal("0.1"))
    assert bullish_fib["opportunity_score"] > base_fib, "positive fib_score should raise the opportunity score"
    bearish_fib = compute_opportunity_score(Decimal("0.60"), Decimal("1"), fib_score=Decimal("-0.1"))
    assert bearish_fib["opportunity_score"] < base_fib, "negative fib_score should lower the opportunity score"
    neutral_fib = compute_opportunity_score(Decimal("0.60"), Decimal("1"), fib_score=Decimal("0"))
    assert neutral_fib["opportunity_score"] == base_fib, "zero fib_score (the default) should be a no-op"
    print("compute_opportunity_score self-check OK")


TRADED_SYMBOLS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traded_symbols.json")


def _record_traded_symbol(symbol: str) -> None:
    """Append-only registry of every symbol ever bought — the robust source
    of truth for anything that needs to reconstruct the full trade history
    (e.g. leverage_journal.py) without fragile log-parsing or a hand-
    maintained list that goes stale the moment a new symbol trades."""
    symbols = set()
    if os.path.exists(TRADED_SYMBOLS_FILE):
        with open(TRADED_SYMBOLS_FILE) as f:
            symbols = set(json.load(f))
    if symbol not in symbols:
        symbols.add(symbol)
        with open(TRADED_SYMBOLS_FILE, "w") as f:
            json.dump(sorted(symbols), f)


GOLDEN_TRADES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_trade_ids.json")
GOLDEN_CONFIDENCE_THRESHOLD = Decimal(os.environ.get("GOLDEN_CONFIDENCE_THRESHOLD", "0.85"))
GOLDEN_RESERVE_PCT = Decimal(os.environ.get("GOLDEN_RESERVE_PCT", "0.20"))
GOLDEN_SPLIT = int(os.environ.get("GOLDEN_SPLIT", "4"))  # one golden trade uses at most 1/GOLDEN_SPLIT of the reserve, not all of it


def _load_golden_trade_ids() -> set[str]:
    if os.path.exists(GOLDEN_TRADES_FILE):
        with open(GOLDEN_TRADES_FILE) as f:
            return set(json.load(f))
    return set()


def _mark_golden_trade(trade_id: str) -> None:
    ids = _load_golden_trade_ids()
    ids.add(trade_id)
    with open(GOLDEN_TRADES_FILE, "w") as f:
        json.dump(sorted(ids), f)


def get_golden_deployed(client: Spot) -> Decimal:
    """Current $ value held in positions funded from the separate golden-trade
    reserve (GOLDEN_RESERVE_PCT of portfolio, only tapped for very-high-
    confidence verdicts) — tracked apart from the normal cap's deployed sum
    so the two pools stay independent: a golden trade never eats into normal
    trading room, and vice versa."""
    golden_ids = _load_golden_trade_ids()
    if not golden_ids:
        return Decimal("0")
    deployed = Decimal("0")
    seen = set()
    for pos in get_open_positions(client):
        key = (pos["symbol"], pos["trade_id"], pos["leg"])
        if pos["trade_id"] in golden_ids and key not in seen:
            seen.add(key)
            deployed += pos["qty"] * pos["current"]
    return deployed


def _test_golden_trade_marking(tmp_path: str) -> None:
    global GOLDEN_TRADES_FILE
    original = GOLDEN_TRADES_FILE
    GOLDEN_TRADES_FILE = tmp_path
    try:
        assert _load_golden_trade_ids() == set()
        _mark_golden_trade("abc123")
        assert _load_golden_trade_ids() == {"abc123"}
    finally:
        GOLDEN_TRADES_FILE = original
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
CIRCUIT_BREAKER_SPLIT_MULTIPLIER = 2  # drawdown breaker doubles POSITION_SPLIT -> halves bet size


def _update_peak_and_drawdown(total_value: Decimal) -> Decimal:
    """Kelly-criterion-style circuit breaker: track the portfolio's all-time
    high in a local file, return the current drawdown fraction from that peak
    (0 if at or above it). Fires size reduction in run_once(), not a trading
    halt — losing streaks should shrink bets, not stop the strategy."""
    peak = total_value
    if os.path.exists(PEAK_FILE):
        with open(PEAK_FILE) as f:
            peak = max(Decimal(json.load(f)["peak"]), total_value)
    with open(PEAK_FILE, "w") as f:
        json.dump({"peak": str(peak)}, f)
    return (peak - total_value) / peak if peak > 0 else Decimal("0")


DAILY_LOSS_LIMIT_PCT = Decimal(os.environ.get("DAILY_LOSS_LIMIT_PCT", "10"))
DAILY_LOSS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "daily_loss_spot.json")


def check_daily_loss_limit(current_value: Decimal, state_file: str) -> tuple[bool, Decimal]:
    """Hard binary halt ('Futureproof Wealth' Ch.9 — "daily loss limit" is
    named as its own risk rule, distinct from position-size throttling).
    Unlike the drawdown circuit breaker (which only shrinks bet size and
    never fully stops), this BLOCKS new trades outright once today's P&L
    drops below -DAILY_LOSS_LIMIT_PCT, resetting at UTC midnight. Separate
    state file per book (spot vs futures) since they're independent
    portfolios. Returns (halted, daily_pnl_pct)."""
    today = datetime.now(timezone.utc).date().isoformat()
    start_value = current_value
    if os.path.exists(state_file):
        with open(state_file) as f:
            state = json.load(f)
        if state.get("date") == today:
            start_value = Decimal(state["start_value"])
        else:
            with open(state_file, "w") as f:
                json.dump({"date": today, "start_value": str(current_value)}, f)
    else:
        with open(state_file, "w") as f:
            json.dump({"date": today, "start_value": str(current_value)}, f)

    daily_pnl_pct = (current_value / start_value - 1) * 100 if start_value > 0 else Decimal("0")
    halted = daily_pnl_pct <= -DAILY_LOSS_LIMIT_PCT
    return halted, daily_pnl_pct


def _test_check_daily_loss_limit(tmp_path: str) -> None:
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    try:
        # first call today -> establishes the baseline, never halted on the same value
        halted, pnl = check_daily_loss_limit(Decimal("1000"), tmp_path)
        assert halted is False and pnl == 0, (halted, pnl)
        # big drop from that baseline, same day -> halts
        halted, pnl = check_daily_loss_limit(Decimal("880"), tmp_path)
        assert halted is True and pnl == Decimal("-12"), (halted, pnl)
        # small drop, same day -> doesn't halt
        with open(tmp_path, "w") as f:
            json.dump({"date": datetime.now(timezone.utc).date().isoformat(), "start_value": "1000"}, f)
        halted, pnl = check_daily_loss_limit(Decimal("950"), tmp_path)
        assert halted is False, (halted, pnl)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


KILL_SWITCH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "KILL_SWITCH")


def kill_switch_active() -> bool:
    """Manual emergency stop ('Futureproof Wealth' Ch.9 — explicit advice to
    define a kill switch before running any automated strategy live).
    Presence of the file halts NEW trade evaluation only; existing-position
    management (early-exit, unprotected-leg re-hedge) keeps running so a
    halted bot doesn't leave open risk unmanaged. `touch KILL_SWITCH` / `rm
    KILL_SWITCH` to toggle, no restart needed."""
    return os.path.exists(KILL_SWITCH_FILE)


CONFIRM_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_top_candidate.json")
RSI_OVERBOUGHT_THRESHOLD = Decimal(os.environ.get("RSI_OVERBOUGHT_THRESHOLD", "75"))


def _load_confirm_pool() -> list[str]:
    """Two-cycle confirmation, generalized to the WHOLE ranked candidate
    pool (not just #1): a candidate must have also appeared in last cycle's
    pool, not just a single 5min snapshot — targets the REZ failure mode
    (Kronos+LLM bought a spike that reversed within one cycle). Used to
    matter less when only #1 was ever evaluated; now that run_once() walks
    the full ranked list, a symbol dropping in/out at #7 needs the same
    check as one dropping in/out at #1."""
    if os.path.exists(CONFIRM_FILE):
        with open(CONFIRM_FILE) as f:
            return json.load(f).get("candidates", [])
    return []


def _save_confirm_pool(symbols: list[str]) -> None:
    with open(CONFIRM_FILE, "w") as f:
        json.dump({"candidates": symbols}, f)


def _test_confirm_momentum(tmp_path: str) -> None:
    global CONFIRM_FILE
    original = CONFIRM_FILE
    CONFIRM_FILE = tmp_path
    try:
        assert _load_confirm_pool() == []
        _save_confirm_pool(["AAA", "BBB"])
        assert "AAA" in _load_confirm_pool()
        assert "CCC" not in _load_confirm_pool()
    finally:
        CONFIRM_FILE = original
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _test_update_peak_and_drawdown(tmp_path: str) -> None:
    global PEAK_FILE
    original = PEAK_FILE
    PEAK_FILE = tmp_path
    try:
        assert _update_peak_and_drawdown(Decimal("100")) == 0
        assert _update_peak_and_drawdown(Decimal("120")) == 0  # new peak
        dd = _update_peak_and_drawdown(Decimal("96"))  # down from 120 peak
        assert dd == Decimal("0.2"), dd
    finally:
        PEAK_FILE = original
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _should_exit_early(pnl_pct: Decimal, forecast_pct_change: Decimal | None) -> bool:
    """Lock in profit before the static take-profit bracket if momentum has
    turned: only fires when the position is already in meaningful profit AND
    the fresh Kronos forecast has turned meaningfully bearish. Doesn't touch
    losing positions — those stay governed by the static -15% stop-loss."""
    if forecast_pct_change is None:
        return False
    return pnl_pct >= EARLY_EXIT_MIN_PROFIT_PCT and forecast_pct_change <= EARLY_EXIT_FORECAST_THRESHOLD


def close_position_early(client: Spot, symbol: str, trade_id: str) -> Decimal:
    """Cancel every open OCO leg for this (symbol, trade_id) and market-sell
    the freed quantity in one shot. Returns the qty sold."""
    pair = f"{symbol}USDT"
    order_list_ids = set()
    total_qty = Decimal("0")
    for o in client.get_open_orders(symbol=pair):
        m = _LEG_RE.match(o.get("clientOrderId", ""))
        if m and m.group("tid") == trade_id:
            order_list_ids.add(o["orderListId"])
            if o["type"] == "STOP_LOSS_LIMIT":  # count each leg's qty once (both legs share it)
                total_qty += Decimal(o["origQty"])

    for lid in order_list_ids:
        client.cancel_oco_order(symbol=pair, orderListId=lid)

    filters = get_symbol_filters(client, pair)
    qty = _round_step(total_qty, filters["step_size"])
    client.new_order(symbol=pair, side="SELL", type="MARKET", quantity=str(qty))
    return qty


LAST_SEEN_LEGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_seen_legs.json")
TRADE_JOURNAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_journal.jsonl")
TRADE_PEAKS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_peaks.json")
TRADE_PEAKS_FILE_FUTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_peaks_futures.json")
REVERSAL_EXIT_MARKERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reversal_exit_markers.json")
REVERSAL_EXIT_MARKER_MAX_AGE_SECONDS = 1800  # 30min — outlives every leg of a laddered close in the same reconciliation pass, without growing the file unbounded if a marker is ever never consumed


def _load_trade_peaks(peaks_file: str) -> dict:
    if os.path.exists(peaks_file):
        with open(peaks_file) as f:
            return json.load(f)
    return {}


def _save_trade_peaks(peaks: dict, peaks_file: str) -> None:
    with open(peaks_file, "w") as f:
        json.dump(peaks, f)


def update_trade_peak(
    trade_id: str, pnl_pct: Decimal, price: Decimal, is_short: bool = False,
    peaks_file: str = TRADE_PEAKS_FILE,
) -> dict:
    """Ratchet this trade's high-water mark and return it as
    {"peak_pnl_pct": Decimal, "peak_price": Decimal}. Feeds
    effective_trailing_floor, which needs both.

    Only ever moves up. pnl_pct is already direction-normalized (positive means
    profit for a short too), and it's strictly monotonic in price for either
    direction, so "pnl made a new high" and "price hit a new best" are the same
    event — recording both together at that moment is what keeps peak_price
    guaranteed to be the price AT the peak, rather than two values maxed
    independently that could drift apart.

    `is_short` isn't needed for the comparison for exactly that reason; it's
    accepted so callers don't have to know that, and so the signature stays
    honest if the storage ever splits.

    `peaks_file` is per-book (spot vs futures) for the same reason
    check_daily_loss_limit takes its file: prune_closed_trade_peaks deletes
    anything outside the open set it's given, so a shared file would have each
    book wiping the other's peaks every cycle."""
    peaks = _load_trade_peaks(peaks_file)
    stored = peaks.get(trade_id)
    if stored is None or pnl_pct > Decimal(stored["peak_pnl_pct"]):
        peaks[trade_id] = {"peak_pnl_pct": str(pnl_pct), "peak_price": str(price)}
        _save_trade_peaks(peaks, peaks_file)
        return {"peak_pnl_pct": pnl_pct, "peak_price": price}
    return {"peak_pnl_pct": Decimal(stored["peak_pnl_pct"]), "peak_price": Decimal(stored["peak_price"])}


def get_trade_peak(trade_id: str, peaks_file: str = TRADE_PEAKS_FILE) -> dict | None:
    stored = _load_trade_peaks(peaks_file).get(trade_id)
    if stored is None:
        return None
    return {"peak_pnl_pct": Decimal(stored["peak_pnl_pct"]), "peak_price": Decimal(stored["peak_price"])}


def prune_closed_trade_peaks(open_trade_ids: set, peaks_file: str = TRADE_PEAKS_FILE) -> None:
    """Drop peaks for trades that are no longer open — the delete-on-close
    pattern short_positions.json uses, NOT the age-based prune the reversal
    markers use. A peak has to survive as long as its trade, which can be days;
    any TTL long enough for that wouldn't be bounding anything. Driven off the
    live open set instead, so it's self-maintaining with no guesswork.

    Deletes everything outside `open_trade_ids`, which is exactly why spot and
    futures must pass different `peaks_file`s — see update_trade_peak."""
    peaks = _load_trade_peaks(peaks_file)
    kept = {tid: v for tid, v in peaks.items() if tid in open_trade_ids}
    if kept != peaks:
        _save_trade_peaks(kept, peaks_file)


def _test_trade_peaks() -> None:
    import tempfile

    fd, spot_file = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(spot_file)  # missing file -> no peaks, not an error
    fd, futures_file = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(futures_file)
    try:
        assert get_trade_peak("t1", spot_file) is None

        # long ratcheting up: pnl and price recorded together
        p = update_trade_peak("t1", Decimal("5"), Decimal("105"), peaks_file=spot_file)
        assert p["peak_pnl_pct"] == Decimal("5") and p["peak_price"] == Decimal("105")
        p = update_trade_peak("t1", Decimal("12"), Decimal("112"), peaks_file=spot_file)
        assert p["peak_pnl_pct"] == Decimal("12") and p["peak_price"] == Decimal("112")

        # pullback does NOT lower the mark, and reports the retained peak
        p = update_trade_peak("t1", Decimal("3"), Decimal("103"), peaks_file=spot_file)
        assert p["peak_pnl_pct"] == Decimal("12"), p
        assert p["peak_price"] == Decimal("112"), p  # still the price at the peak, not the pullback price
        assert get_trade_peak("t1", spot_file)["peak_pnl_pct"] == Decimal("12")

        # short: profit rises as price FALLS, so peak_price ratchets DOWN
        update_trade_peak("t2", Decimal("10"), Decimal("90"), is_short=True, peaks_file=futures_file)
        p = update_trade_peak("t2", Decimal("30"), Decimal("70"), is_short=True, peaks_file=futures_file)
        assert p["peak_pnl_pct"] == Decimal("30") and p["peak_price"] == Decimal("70"), p
        p = update_trade_peak("t2", Decimal("20"), Decimal("80"), is_short=True, peaks_file=futures_file)
        assert p["peak_price"] == Decimal("70"), p  # bounce doesn't move it

        # the books are isolated: pruning one must not touch the other, which
        # is the entire reason peaks_file is a parameter
        prune_closed_trade_peaks(set(), peaks_file=spot_file)  # spot has nothing open
        assert get_trade_peak("t1", spot_file) is None
        assert get_trade_peak("t2", futures_file) is not None, "futures peak wiped by a spot prune"

        # prune keeps what's still open
        update_trade_peak("t3", Decimal("15"), Decimal("115"), peaks_file=spot_file)
        update_trade_peak("t4", Decimal("15"), Decimal("115"), peaks_file=spot_file)
        prune_closed_trade_peaks({"t3"}, peaks_file=spot_file)
        assert get_trade_peak("t3", spot_file) is not None
        assert get_trade_peak("t4", spot_file) is None
    finally:
        for f in (spot_file, futures_file):
            if os.path.exists(f):
                os.remove(f)


def _load_reversal_exit_markers() -> dict:
    if os.path.exists(REVERSAL_EXIT_MARKERS_FILE):
        with open(REVERSAL_EXIT_MARKERS_FILE) as f:
            return json.load(f)
    return {}


def _save_reversal_exit_markers(markers: dict) -> None:
    with open(REVERSAL_EXIT_MARKERS_FILE, "w") as f:
        json.dump(markers, f)


KRONOS_REVERSAL_EXIT = "kronos_reversal_exit"
TRAILING_EXIT = "trailing_exit"


def _marker_timestamp(entry) -> str:
    """Markers were originally {trade_id: iso_string} and are now
    {trade_id: {"reason": ..., "at": iso_string}}. Both shapes are read for as
    long as any pre-upgrade marker could still be on disk (<=30min), so a
    restart mid-flight neither crashes the prune nor mislabels a close."""
    return entry["at"] if isinstance(entry, dict) else entry


def mark_early_exit(trade_id: str, reason: str) -> None:
    """Records WHY trade_id is about to be market-closed (via
    close_position_early/close_short), so journal reconciliation
    (_classify_and_log_closed_leg/_log_closed_short) can label the close
    precisely instead of dropping it in the "manual" catch-all.

    `reason` is one of KRONOS_REVERSAL_EXIT / TRAILING_EXIT. The distinction
    matters: 4 of 31 losing trades in the 2026-08-03 loss-pattern analysis
    carried the old early_exit_or_manual catch-all with no way to tell which
    mechanism actually fired, which is what made the BANK give-back
    impossible to diagnose after the fact."""
    markers = _load_reversal_exit_markers()
    markers[trade_id] = {"reason": reason, "at": datetime.now(timezone.utc).isoformat()}
    _save_reversal_exit_markers(markers)


def get_early_exit_reason(trade_id: str) -> str | None:
    """The reason this trade was deliberately closed, or None if nothing
    marked it (i.e. a genuinely manual/unaccounted close).

    Peek, not consume — a trade_id can span multiple legs (laddered spot
    positions), each classified separately by _classify_and_log_closed_leg
    in its own reconciliation pass, so the marker must survive until every
    leg has been reconciled. Cleanup is prune_stale_reversal_markers'
    job, not this function's."""
    entry = _load_reversal_exit_markers().get(trade_id)
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry.get("reason", KRONOS_REVERSAL_EXIT)
    return KRONOS_REVERSAL_EXIT  # pre-upgrade marker: reversal was the only reason that existed


def mark_reversal_exit(trade_id: str) -> None:
    """Back-compat wrapper — see mark_early_exit."""
    mark_early_exit(trade_id, KRONOS_REVERSAL_EXIT)


def is_reversal_exit(trade_id: str) -> bool:
    """Back-compat wrapper — see get_early_exit_reason."""
    return get_early_exit_reason(trade_id) == KRONOS_REVERSAL_EXIT


def prune_stale_reversal_markers() -> None:
    """Call once per reconciliation pass (not per leg). Removes markers
    older than REVERSAL_EXIT_MARKER_MAX_AGE_SECONDS."""
    markers = _load_reversal_exit_markers()
    now = datetime.now(timezone.utc)
    fresh = {
        tid: entry for tid, entry in markers.items()
        if (now - datetime.fromisoformat(_marker_timestamp(entry))).total_seconds() < REVERSAL_EXIT_MARKER_MAX_AGE_SECONDS
    }
    if fresh != markers:
        _save_reversal_exit_markers(fresh)


def _test_reversal_exit_markers() -> None:
    import tempfile

    global REVERSAL_EXIT_MARKERS_FILE
    original_file = REVERSAL_EXIT_MARKERS_FILE
    fd, REVERSAL_EXIT_MARKERS_FILE = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(REVERSAL_EXIT_MARKERS_FILE)  # missing file -> no markers, not an error
    try:
        assert is_reversal_exit("abc123") is False  # no file yet -> not a reversal exit

        mark_reversal_exit("abc123")
        assert os.path.exists(REVERSAL_EXIT_MARKERS_FILE)
        assert is_reversal_exit("abc123") is True
        assert is_reversal_exit("other-trade") is False  # different trade_id, untouched

        # peek, not consume — a laddered close's leg0 and leg1 both classify
        # separately and both must see the same marker
        assert is_reversal_exit("abc123") is True

        prune_stale_reversal_markers()
        assert is_reversal_exit("abc123") is True  # fresh marker survives a prune pass

        # reasons are distinguishable, which is the whole point of the marker
        mark_early_exit("trail1", TRAILING_EXIT)
        assert get_early_exit_reason("trail1") == TRAILING_EXIT
        assert is_reversal_exit("trail1") is False  # a trailing exit is NOT a reversal exit
        assert get_early_exit_reason("abc123") == KRONOS_REVERSAL_EXIT
        assert get_early_exit_reason("never-marked") is None

        from datetime import datetime, timezone, timedelta

        # pre-upgrade marker shape ({trade_id: iso_string}) still reads as a
        # reversal exit rather than crashing or silently degrading to "manual"
        fresh_ts = datetime.now(timezone.utc).isoformat()
        _save_reversal_exit_markers({"legacy": fresh_ts})
        assert get_early_exit_reason("legacy") == KRONOS_REVERSAL_EXIT
        prune_stale_reversal_markers()  # must not choke on the old shape
        assert get_early_exit_reason("legacy") == KRONOS_REVERSAL_EXIT  # fresh, survives

        # simulate an old marker by writing one with a stale timestamp directly
        stale_ts = (datetime.now(timezone.utc) - timedelta(seconds=REVERSAL_EXIT_MARKER_MAX_AGE_SECONDS + 1)).isoformat()
        _save_reversal_exit_markers({"abc123": {"reason": KRONOS_REVERSAL_EXIT, "at": stale_ts}})
        prune_stale_reversal_markers()
        assert is_reversal_exit("abc123") is False  # pruned
        # ...and the same prune works on a stale marker in the legacy shape
        _save_reversal_exit_markers({"legacy": stale_ts})
        prune_stale_reversal_markers()
        assert get_early_exit_reason("legacy") is None
    finally:
        if os.path.exists(REVERSAL_EXIT_MARKERS_FILE):
            os.remove(REVERSAL_EXIT_MARKERS_FILE)
        REVERSAL_EXIT_MARKERS_FILE = original_file


def _leg_key(pos: dict) -> str:
    return f"{pos['symbol']}|{pos['trade_id']}|{pos['leg']}"


def _classify_and_log_closed_leg(client: Spot, key: str, snapshot: dict) -> None:
    """A leg that was open last cycle and isn't anymore. No local callback
    fires when Binance fills a resting TP/SL order, so the only way to know
    HOW it closed is to look up the leg's own orders after the fact: TP leg
    filled -> take_profit, SL leg filled -> stop_loss, neither -> must have
    been an explicit market sell (close_position_early). Appends one line to
    the trade journal either way."""
    symbol, trade_id, leg_idx = key.split("|")
    leg_idx = int(leg_idx)
    try:
        orders = client.get_orders(symbol=f"{symbol}USDT", limit=50)
    except Exception as e:
        print(f"  journal: couldn't look up {symbol} leg {leg_idx} closure: {e}")
        return

    tp_order = next((o for o in orders if o.get("clientOrderId") == f"g{trade_id}L{leg_idx}A"), None)
    sl_order = next((o for o in orders if o.get("clientOrderId") == f"g{trade_id}L{leg_idx}B"), None)

    # Binance marks an order's overall `status` CANCELED once any unfilled
    # remainder is canceled — even if part of it genuinely executed (OCO
    # cancels the sibling leg the instant one side starts filling, which can
    # leave the filling side's own remainder canceled too). Checking
    # executedQty instead of status is what actually tells you a fill
    # happened; using `price` would be wrong for that partial slice, so the
    # exit price comes from the real average fill (quote/qty), not the
    # resting limit price.
    exit_price = exit_reason = exit_time = None
    if tp_order and Decimal(tp_order["executedQty"]) > 0:
        exit_price = Decimal(tp_order["cummulativeQuoteQty"]) / Decimal(tp_order["executedQty"])
        exit_reason, exit_time = "take_profit", tp_order["updateTime"]
    elif sl_order and Decimal(sl_order["executedQty"]) > 0:
        exit_price = Decimal(sl_order["cummulativeQuoteQty"]) / Decimal(sl_order["executedQty"])
        exit_reason, exit_time = "stop_loss", sl_order["updateTime"]
    else:
        market_sells = [o for o in orders if o["side"] == "SELL" and o["type"] == "MARKET" and Decimal(o["executedQty"]) > 0]
        if market_sells:
            latest = max(market_sells, key=lambda o: o["time"])
            if Decimal(latest["executedQty"]) > 0:
                exit_price = Decimal(latest["cummulativeQuoteQty"]) / Decimal(latest["executedQty"])
            exit_reason = get_early_exit_reason(trade_id) or "manual"
            exit_time = latest["time"]

    entry = Decimal(str(snapshot["entry"])) if snapshot["entry"] is not None else None
    pnl_pct = float((exit_price / entry - 1) * 100) if (exit_price is not None and entry) else None

    record = {
        "symbol": symbol, "trade_id": trade_id, "leg": leg_idx,
        "qty": snapshot["qty"], "entry": snapshot["entry"],
        "exit_price": float(exit_price) if exit_price is not None else None,
        "exit_reason": exit_reason, "pnl_pct": pnl_pct,
        "closed_at": datetime.fromtimestamp(exit_time / 1000, tz=timezone.utc).isoformat() if exit_time else None,
    }
    with open(TRADE_JOURNAL_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    msg = f"{pnl_pct:+.2f}%" if pnl_pct is not None else "unknown P&L"
    print(f"  JOURNAL: {symbol} leg {leg_idx} closed via {exit_reason or 'unknown'} — {msg}")


def _reconcile_trade_journal(client: Spot, current_positions: list[dict]) -> None:
    """Diff this cycle's open legs against last cycle's snapshot; anything
    that vanished gets classified and appended to the trade journal."""
    last_seen = {}
    if os.path.exists(LAST_SEEN_LEGS_FILE):
        with open(LAST_SEEN_LEGS_FILE) as f:
            last_seen = json.load(f)

    current_keys = {_leg_key(p) for p in current_positions}
    for key, snapshot in last_seen.items():
        if key not in current_keys:
            _classify_and_log_closed_leg(client, key, snapshot)
    prune_stale_reversal_markers()

    new_snapshot = {
        _leg_key(p): {"entry": float(p["entry"]) if p["entry"] is not None else None, "qty": float(p["qty"])}
        for p in current_positions
    }
    with open(LAST_SEEN_LEGS_FILE, "w") as f:
        json.dump(new_snapshot, f)


def manage_open_positions(client: Spot) -> None:
    """Check every open position for fading momentum and lock in profit
    early instead of waiting for the static +15%/+30% take-profit bracket.
    Called once per run_once() cycle, before scanning for new trades."""
    from scanner import get_kronos_forecast  # local import: keeps the cross-venv subprocess call optional

    positions = get_open_positions(client)
    _reconcile_trade_journal(client, positions)
    checked_symbols: dict[str, dict | None] = {}
    seen_trades = set()
    for pos in positions:
        if pos.get("unprotected"):
            # OCO cancels the SL leg the instant the TP leg starts partially
            # filling, leaving the unsold remainder with no downside
            # protection (found live on REZ). Re-hedge it at the market
            # price rather than leaving it naked until next inspection.
            print(f"  RE-HEDGING: {pos['symbol']} leg {pos['leg']} has {pos['qty']} qty with no "
                  f"stop-loss (OCO partial-fill artifact) — placing a fresh protective bracket.")
            place_oco_exit(client, pos["symbol"], pos["qty"], pos["current"],
                            trade_id=pos["trade_id"], leg_index=pos["leg"])
            continue

        key = (pos["symbol"], pos["trade_id"])
        if key in seen_trades:
            continue  # one decision per (symbol, trade_id), not per leg
        seen_trades.add(key)

        if pos["symbol"] not in checked_symbols:
            checked_symbols[pos["symbol"]] = get_kronos_forecast(pos["symbol"])
        forecast = checked_symbols[pos["symbol"]]
        forecast_pct = Decimal(str(forecast["predicted_pct_change"])) if forecast else None

        # Logged on every check, firing or not — see execute_futures.py's
        # mirrored comment for why (found live on BANK, 2026-08-04).
        forecast_note = f"{forecast_pct:+.2f}%" if forecast_pct is not None else "unavailable"
        print(f"  {pos['symbol']} pos: pnl {pos['pnl_pct']:+.2f}%, Kronos predicts {forecast_note} "
              f"(exits if pnl>={EARLY_EXIT_MIN_PROFIT_PCT}% and forecast<={EARLY_EXIT_FORECAST_THRESHOLD}%)")

        if _should_exit_early(Decimal(str(pos["pnl_pct"])), forecast_pct):
            print(f"  EARLY EXIT: {pos['symbol']} up {pos['pnl_pct']:+.2f}%, Kronos now predicts "
                  f"{forecast_pct:+.2f}% — locking in profit instead of waiting for the static target.")
            mark_early_exit(pos["trade_id"], KRONOS_REVERSAL_EXIT)
            qty = close_position_early(client, pos["symbol"], pos["trade_id"])
            print(f"  Sold {qty} {pos['symbol']} at market.")
            continue  # closed; don't also run the trailing check on it

        # Trailing profit-lock: independent of Kronos, so it still fires when
        # the forecast never calls the reversal (the BANK failure mode).
        # Kronos gets first refusal above because it has richer information;
        # this is the mechanical backstop under it.
        if pos["entry"] is not None:
            peak = update_trade_peak(
                pos["trade_id"], Decimal(str(pos["pnl_pct"])), pos["current"],
            )
            atr = get_atr(pos["symbol"])
            floor = effective_trailing_floor(
                entry=pos["entry"],
                peak_price=peak["peak_price"],
                peak_pnl_pct=peak["peak_pnl_pct"],
                stop_loss_pct=_get_trade_stop_pct(pos["trade_id"]),
                atr=Decimal(str(atr)) if atr is not None else None,
            )
            if floor is not None:
                print(f"    trailing: peak {peak['peak_pnl_pct']:+.2f}%, floor {floor:+.2f}%")
            if floor is not None and Decimal(str(pos["pnl_pct"])) < floor:
                print(f"  TRAILING EXIT: {pos['symbol']} peaked at {peak['peak_pnl_pct']:+.2f}%, "
                      f"now {pos['pnl_pct']:+.2f}% — through the {floor:+.2f}% floor, banking it.")
                mark_early_exit(pos["trade_id"], TRAILING_EXIT)
                qty = close_position_early(client, pos["symbol"], pos["trade_id"])
                print(f"  Sold {qty} {pos['symbol']} at market.")

    prune_closed_trade_peaks({p["trade_id"] for p in positions})


def _test_round_step() -> None:
    assert _round_step(Decimal("1.2345"), Decimal("0.01")) == Decimal("1.23")
    assert _round_step(Decimal("100"), Decimal("10")) == Decimal("100")
    assert _round_step(Decimal("0.0049"), Decimal("0.01")) == Decimal("0.00")


def _test_should_exit_early() -> None:
    # in profit, forecast turned bearish -> exit
    assert _should_exit_early(Decimal("9.8"), Decimal("-11.5")) is True
    # in profit, but forecast still fine -> hold for the real target
    assert _should_exit_early(Decimal("9.8"), Decimal("2.0")) is False
    # forecast bearish, but profit too small to bother -> hold
    assert _should_exit_early(Decimal("0.5"), Decimal("-11.5")) is False
    # losing position, bearish forecast -> NOT this function's job (static stop-loss handles it)
    assert _should_exit_early(Decimal("-3.0"), Decimal("-11.5")) is False
    # no forecast available -> never act on missing data
    assert _should_exit_early(Decimal("9.8"), None) is False


def run_once() -> None:
    """One scan -> rank -> verdict -> maybe-bet cycle. Called directly or by loop.py.

    Walks the FULL ranked candidate list, not just #1 — previously a single
    dominant symbol (e.g. COTI staying top-ranked and RSI-overbought for 15+
    minutes straight) locked out every other candidate for the whole cycle,
    even a clean one (KAITO, RSI 62, unrated) sitting right below it. Falls
    through to the next candidate on any per-symbol gate failure; only a
    hard infra failure (Ollama down) aborts the whole cycle."""
    client = get_client()
    print("Checking open positions for early-exit opportunities...")
    manage_open_positions(client)

    if kill_switch_active():
        print(f"KILL_SWITCH active ({KILL_SWITCH_FILE}) — skipping new-trade evaluation this cycle. `rm {KILL_SWITCH_FILE}` to resume.")
        return

    fear_greed = get_fear_greed_index()  # macro overlay, fetched once per cycle, same for every candidate
    if fear_greed is not None:
        print(f"  Fear & Greed Index: {fear_greed}")

    print("Scanning whole Binance USDT market for new movers...")
    # get_binance_universe reads MAINNET (for real volume/discovery); testnet's
    # symbol list isn't 1:1 with mainnet's (e.g. leveraged tokens), so anything
    # not actually tradeable there must be dropped before we touch it, or
    # depth()/ticker_price() 400 with "Invalid symbol" against the testnet client.
    testnet_symbols = get_testnet_symbols(client)
    max_price = float(os.environ.get("MAX_PRICE_USD", "5.0"))
    universe = [s for s in get_binance_universe(max_price=max_price) if s in testnet_symbols]
    momentum = get_binance_momentum_short(universe, window="1h")  # lower timeframe: 1h, not 24h
    print(f"  {len(momentum)} symbols above volume floor (testnet-tradeable, 1h momentum)")

    # DEXScreener enrichment is one HTTP call per symbol — too slow/rate-limit-prone
    # to run against the whole market, so only enrich the top-N by raw momentum first.
    top_by_momentum = sorted(momentum, key=lambda s: momentum[s]["pct_change_24h"], reverse=True)[:DEX_ENRICH_TOP_N]
    dex_scores = {sym: get_dexscreener_trend(sym) for sym in top_by_momentum}
    try:
        trends = get_google_trends(top_by_momentum)
    except Exception as e:
        print(f"  Google Trends skipped: {e}")
        trends = {}
    try:
        cmc_movers = get_cmc_movers()
    except Exception as e:
        print(f"  CoinMarketCap movers skipped: {e}")
        cmc_movers = {}
    ranked = rank_symbols({sym: momentum[sym] for sym in top_by_momentum}, dex_scores, trends, cmc_movers)

    prev_candidates = _load_confirm_pool()
    _save_confirm_pool([sym for sym, _ in ranked])
    open_symbols = {p["symbol"] for p in get_open_positions(client)}

    fixed_sl_active = use_fixed_stop_loss()  # checked once per cycle, not once per candidate — it's a network call with a state-mutating side effect (2026-08-01 whole-branch review, Important #5)
    chosen = None
    for sym, score in ranked:
        if not has_ask_liquidity(client, sym):
            print(f"  {sym}: no testnet ask-side liquidity right now, skipping")
            continue
        print(f"Evaluating: {sym} (score {score:+.2f})")

        if sym not in prev_candidates:
            print(f"  {sym} wasn't in last cycle's candidate pool — needs one more cycle to confirm.")
            continue

        if sym in open_symbols:
            print(f"  already holding an open {sym} position, skipping to avoid stacking a second one.")
            continue

        rsi = get_rsi(sym)
        if rsi is not None and rsi > RSI_OVERBOUGHT_THRESHOLD:
            print(f"  {sym} RSI {rsi:.1f} is overbought (>{RSI_OVERBOUGHT_THRESHOLD}), likely chasing an exhausted move.")
            continue

        daily = get_binance_momentum_short([sym], window="1d").get(sym)
        daily_pct = Decimal(str(daily["pct_change_24h"])) if daily else None
        if is_counter_trend(daily_pct):
            print(f"  {sym}: 1h signal fighting a bearish 24h trend ({daily_pct:+.1f}%), counter-trend, skipping.")
            continue

        current_price = Decimal(client.ticker_price(symbol=f"{sym}USDT")["price"])
        vwap = get_vwap(sym)
        vwap_dev = compute_vwap_deviation_pct(current_price, Decimal(str(vwap))) if vwap else None
        if is_extended_from_vwap(vwap_dev):
            print(f"  {sym}: price {vwap_dev:+.1f}% above rolling VWAP, overextended, likely to revert, skipping.")
            continue

        obv_div = get_obv_divergence(sym)
        if obv_warns_against(obv_div):
            print(f"  {sym}: {obv_div} OBV divergence — volume isn't confirming the price move, skipping.")
            continue

        if (dex_scores.get(sym) or {}).get("pump_flag"):
            print(f"  {sym}: pump-signature detected (young/thin/spiking DEX pair) — flagged for the LLM.")
        kronos = get_kronos_forecast(sym)
        verdict = get_llm_verdict(sym, {"rank_score": score, **momentum[sym]}, kronos=kronos, dex_data=dex_scores.get(sym))
        if verdict is None:
            print("Ollama not running — `ollama serve` or open the app")
            return
        print(f"  Verdict: {verdict['verdict'].upper()} (confidence {verdict['confidence']:.2f}) — {verdict['reasoning']}")
        if verdict["verdict"] != "buy" or verdict["confidence"] < 0.5:
            print(f"  {sym}: verdict is skip or low confidence.")
            continue

        kronos_pct = Decimal(str(kronos["predicted_pct_change"])) if kronos else None
        fib_score = fib_entry_signal(sym, current_price)
        opportunity = compute_opportunity_score(
            Decimal(str(verdict["confidence"])), kronos_pct, fear_greed=fear_greed, fib_score=fib_score,
        )
        print(f"  Opportunity score: {opportunity['reasoning']}")
        if not opportunity["passes"]:
            print(f"  {sym}: opportunity score fails the Kronos+LLM blended gate.")
            continue

        if fixed_sl_active:
            # fixed-SL still wins over Fib for BOTH sides of the bracket, not
            # just the stop — pairing a flat stop with a structurally
            # independent Fib take-profit broke the drawdown-recovery
            # policy's original guaranteed-ratio behavior (2026-08-01
            # whole-branch review, Important #3).
            stop_loss_pct = DEFAULT_STOP_LOSS_PCT
            tp_pcts = compute_atr_take_profit_pcts(stop_loss_pct)
            sl_source, tp_source = "flat", "ATR-ratio (fixed-SL active)"
        else:
            fib_sl = compute_fib_stop_loss_pct(sym, current_price)
            fib_tp = compute_fib_take_profit_pcts(sym, current_price, fib_sl[1]) if fib_sl is not None else None
            if fib_sl is not None and fib_tp is not None:
                stop_loss_pct = fib_sl[0]
                tp_pcts = fib_tp
                sl_source, tp_source = "Fib", "Fib"
            else:
                stop_loss_pct = compute_atr_stop_loss_pct(sym, current_price)
                tp_pcts = compute_atr_take_profit_pcts(stop_loss_pct)
                sl_source, tp_source = "ATR", "ATR"
        if not meets_min_reward_risk(tp_pcts[0], stop_loss_pct):
            print(f"  {sym}: R:R {tp_pcts[0] / stop_loss_pct:.2f}:1 (TP1 {tp_pcts[0] * 100:.0f}% / stop "
                  f"{stop_loss_pct * 100:.1f}%) below minimum {MIN_REWARD_RISK_RATIO}:1, skipping.")
            continue

        chosen = {
            "symbol": sym, "score": score, "verdict": verdict, "stop_loss_pct": stop_loss_pct,
            "tp_pcts": tp_pcts, "sl_source": sl_source, "tp_source": tp_source,
        }
        break

    if chosen is None:
        print("No candidate this cycle cleared every gate.")
        return
    top_symbol, top_score, verdict = chosen["symbol"], chosen["score"], chosen["verdict"]
    stop_loss_pct = chosen["stop_loss_pct"]
    tp_pcts = chosen["tp_pcts"]
    print(f"{chosen['tp_source']}-based take-profit legs: {[f'{p * 100:.1f}%' for p in tp_pcts]}")
    print(f"{chosen['sl_source']}-based stop-loss: {stop_loss_pct * 100:.1f}%")

    max_portfolio_pct = Decimal(os.environ.get("MAX_PORTFOLIO_PCT", "0.3"))
    position_split = int(os.environ.get("POSITION_SPLIT", "5"))
    total_value, deployed_value = get_portfolio_exposure(client)
    deployed_pct = deployed_value / total_value if total_value else Decimal("1")
    room = (max_portfolio_pct * total_value) - deployed_value
    print(f"Portfolio: ${total_value:.2f} total, ${deployed_value:.2f} deployed "
          f"({deployed_pct * 100:.1f}%, cap {max_portfolio_pct * 100:.0f}%)")

    daily_halted, daily_pnl_pct = check_daily_loss_limit(total_value, DAILY_LOSS_FILE)
    print(f"Today's P&L: {daily_pnl_pct:+.2f}% (limit -{DAILY_LOSS_LIMIT_PCT}%)")
    if daily_halted:
        print(f"Not betting — daily loss limit hit ({daily_pnl_pct:+.2f}% <= -{DAILY_LOSS_LIMIT_PCT}%). Resets at UTC midnight.")
        return

    balance = get_usdt_balance(client)

    # Golden trade: a verdict this confident gets access to a SEPARATE
    # reserve (GOLDEN_RESERVE_PCT of portfolio) that sits outside the normal
    # 30% cap entirely — checked before the normal room<=0 gate below, since
    # the whole point is it still fires even when normal room is exhausted.
    # Not split by POSITION_SPLIT: one full-conviction shot, not five slots.
    if verdict["confidence"] >= GOLDEN_CONFIDENCE_THRESHOLD:
        golden_deployed = get_golden_deployed(client)
        golden_room = (GOLDEN_RESERVE_PCT * total_value) - golden_deployed
        golden_per_trade_cap = (GOLDEN_RESERVE_PCT * total_value) / GOLDEN_SPLIT
        print(f"Golden reserve: ${golden_deployed:.2f} deployed of ${GOLDEN_RESERVE_PCT * total_value:.2f} "
              f"({GOLDEN_RESERVE_PCT * 100:.0f}% of portfolio, 1/{GOLDEN_SPLIT} per trade = ${golden_per_trade_cap:.2f})")
        if golden_room >= 5:
            bet_size = min(balance, golden_room, golden_per_trade_cap)
            print(f"GOLDEN TRADE — confidence {verdict['confidence']:.2f} >= {GOLDEN_CONFIDENCE_THRESHOLD}, "
                  f"betting {bet_size:.2f} from the golden reserve (1/{GOLDEN_SPLIT} of it, bypasses the normal per-trade cap).")
            result = place_gamble_trade_laddered(client, top_symbol, bet_size, tp_pcts=tp_pcts, stop_loss_pct=stop_loss_pct)
            _mark_golden_trade(result["trade_id"])
            _record_traded_symbol(top_symbol)
            print(f"\nBought {result['qty']} {top_symbol} @ ~{result['fill_price']}.")
            for leg in result["legs"]:
                print(f"  leg tp={leg['take_profit_pct']}: qty={leg['qty']}, "
                      f"take-profit={leg['take_profit']}, stop-loss={leg['stop_price']}")
            return
        print("Golden reserve full — falling back to normal sizing.")

    if room <= 0:
        print("Not betting — already at or over the portfolio exposure cap.")
        return

    drawdown = _update_peak_and_drawdown(total_value)
    if drawdown >= DRAWDOWN_CIRCUIT_BREAKER_PCT:
        position_split *= CIRCUIT_BREAKER_SPLIT_MULTIPLIER
        print(f"Circuit breaker: portfolio down {drawdown * 100:.1f}% from peak "
              f"(threshold {DRAWDOWN_CIRCUIT_BREAKER_PCT * 100:.0f}%) — bet size halved (split={position_split}).")

    breadth = get_market_breadth(momentum)
    if breadth["up_pct"] is not None and breadth["up_pct"] < BREADTH_WEAK_UP_PCT:
        position_split *= BREADTH_SPLIT_MULTIPLIER
        print(f"Weak market breadth: only {breadth['up_pct'] * 100:.0f}% of {len(momentum)} scanned symbols up "
              f"(avg {breadth['avg_change']:+.2f}%, threshold {BREADTH_WEAK_UP_PCT * 100:.0f}%) — "
              f"bet size halved (split={position_split}). Reactive/lagging like any breadth measure — "
              f"can sit out the first leg of a recovery.")

    # 30% is the TOTAL portfolio ceiling, not one trade's size — cap each
    # individual bet at cap/POSITION_SPLIT so multiple positions can run
    # concurrently instead of one trade blowing the whole cap (that's what
    # happened with DIA: one oversized bet locked out every other signal
    # until it closed).
    per_trade_cap = (max_portfolio_pct * total_value) / position_split
    bet_size = min(balance, room, per_trade_cap)
    # Floor is a function of the intended trade size, not a flat $5 — a good
    # signal caught with the cap nearly full (e.g. DEXE got $83 of a normal
    # ~$4,300 bet, 18.8% return that should've been ~$800 turned into $16.87)
    # should wait for room to free up next cycle, not fire a scrap-sized bet
    # that wastes the signal. MIN_BET_FRACTION scales the floor with
    # per_trade_cap; the flat $5 stays as an absolute exchange-notional floor.
    min_viable_bet = max(Decimal("5"), per_trade_cap * MIN_BET_FRACTION)
    if bet_size < min_viable_bet:
        print(f"Not betting — remaining room (${bet_size:.2f}) is below {MIN_BET_FRACTION * 100:.0f}% of the "
              f"normal ${per_trade_cap:.2f} bet size (min viable ${min_viable_bet:.2f}). Waiting for room to free up.")
        return
    print(f"USDT balance: {balance}, betting: {bet_size:.2f} (1/{position_split} of cap, room left after: ${room - bet_size:.2f})")

    result = place_gamble_trade_laddered(client, top_symbol, bet_size, tp_pcts=tp_pcts, stop_loss_pct=stop_loss_pct)
    _record_traded_symbol(top_symbol)
    print(f"\nBought {result['qty']} {top_symbol} @ ~{result['fill_price']}.")
    for leg in result["legs"]:
        print(
            f"  leg tp={leg['take_profit_pct']}: qty={leg['qty']}, "
            f"take-profit={leg['take_profit']}, stop-loss={leg['stop_price']}"
        )


if __name__ == "__main__":
    _test_round_step()  # fails loudly if the rounding logic breaks
    _test_should_exit_early()
    _test_reversal_exit_markers()
    _test_trailing_floors()
    _test_trade_peaks()
    _test_update_peak_and_drawdown(os.path.join(tempfile.gettempdir(), "_test_peak.json"))
    _test_confirm_momentum(os.path.join(tempfile.gettempdir(), "_test_confirm.json"))
    _test_golden_trade_marking(os.path.join(tempfile.gettempdir(), "_test_golden.json"))
    _test_compute_opportunity_score()
    _test_compute_atr_stop_loss_pct()
    _test_compute_fib_stop_loss_pct()
    _test_compute_fib_take_profit_pcts()
    _test_fib_entry_signal()
    _test_use_fixed_stop_loss()
    _test_meets_min_reward_risk()
    _test_check_daily_loss_limit(os.path.join(tempfile.gettempdir(), "_test_daily_loss.json"))
    _test_compute_atr_take_profit_pcts()
    _test_is_counter_trend()
    _test_vwap_deviation()
    _test_obv_warns_against()
    run_once()
