"""Meme-coin momentum/hype scanner. Testnet gambling experiment only — see
project memory `project-memecoin-testnet-gamble`. Signal sources: Binance
public ticker (momentum), DEXScreener (DEX trend/volume), Google Trends
(search interest), Reddit (mention count), LunarCrush (social — gated
behind Individual+ tier, currently unavailable on this account).
"""
from __future__ import annotations

import json
import os
import subprocess
import time

import requests

# ponytail: hardcoded to the memecoin pairs confirmed tradeable on Binance
# testnet (config/environments/testnet.env mirrors mainnet's symbol list).
# Add more here if Binance lists new ones. Kept as the backtest's calibrated
# universe; see get_binance_universe() for dynamic "find new tokens" scanning.
MEME_SYMBOLS = ["DOGE", "SHIB", "PEPE", "FLOKI", "MEME", "BONK", "WIF", "TRUMP"]

# stablecoins only — these would show ~0% change and drown out anything
# real, regardless of price. Majors (BTC/ETH/etc) are no longer excluded by
# name: eligibility is decided by MAX_PRICE_USD alone, so a major trading
# under that threshold is fair game same as any alt/meme coin.
_EXCLUDE_STABLECOINS = {
    "USDT", "USDC", "BUSD", "TUSD", "FDUSD", "DAI", "USDP", "PAX", "EUR", "GBP", "TRY", "BRL",
}


def get_binance_universe(min_quote_volume: float = 500_000, max_price: float = 5.0) -> list[str]:
    """All Binance USDT pairs above a volume floor and under max_price, minus
    stablecoins — the actual 'find new tokens' mechanism. No hardcoded list
    and no meme/alt-only restriction: any token (majors included) trading
    under max_price is fair game, whether or not it's on MEME_SYMBOLS. Free,
    no key (mainnet ticker, which conveniently already includes lastPrice —
    no extra request needed for the price filter)."""
    resp = requests.get("https://api.binance.com/api/v3/ticker/24hr", timeout=10)
    resp.raise_for_status()
    out = []
    for row in resp.json():
        sym = row["symbol"]
        if not sym.endswith("USDT"):
            continue
        base = sym[:-4]
        if base in _EXCLUDE_STABLECOINS:
            continue
        if not base.isascii() or not base.isalnum():
            continue  # ponytail: a handful of symbols use non-ASCII names (e.g. Chinese characters) that break strict API param validation elsewhere
        if float(row["quoteVolume"]) < min_quote_volume:
            continue
        if float(row["lastPrice"]) >= max_price:
            continue
        out.append(base)
    return sorted(out)


def _pump_signature(age_hours: float | None, vol_liq_ratio: float, price_change_1h: float | None) -> bool:
    """Rough pump-and-dump signature: young pair + high volume relative to
    liquidity (thin market easily moved) + a strong recent price spike."""
    return bool(
        age_hours is not None and age_hours < 72
        and vol_liq_ratio > 2
        and (price_change_1h or 0) > 20
    )


def _test_pump_signature() -> None:
    assert _pump_signature(24, 5, 40) is True  # young, thin, spiking
    assert _pump_signature(200, 5, 40) is False  # too old
    assert _pump_signature(24, 0.5, 40) is False  # liquidity not thin enough
    assert _pump_signature(24, 5, 5) is False  # not spiking
    assert _pump_signature(None, 5, 40) is False  # no age data — can't confirm "young"


def get_dex_pump_watchlist(limit: int = 10) -> list[dict]:
    """Freshly-boosted/trending tokens from DEXScreener, with a rough
    pump-and-dump signature: young pair + high volume-to-liquidity ratio +
    strong 1h move. INFORMATIONAL ONLY — these are typically DEX-only
    launches (Solana/Base/etc via pump.fun and similar), not listed on
    Binance, so nothing here is tradeable through execute.py without adding
    real DEX execution (a much bigger scope change: no DEX testnet exists,
    it would need real wallets/gas on mainnet). Use for awareness, not bets."""
    resp = requests.get("https://api.dexscreener.com/token-boosts/latest/v1", timeout=10)
    resp.raise_for_status()
    boosts = resp.json()[:limit]
    addresses = ",".join(b["tokenAddress"] for b in boosts if b.get("tokenAddress"))
    if not addresses:
        return []
    pairs_resp = requests.get(f"https://api.dexscreener.com/latest/dex/tokens/{addresses}", timeout=10)
    pairs_resp.raise_for_status()
    pairs = pairs_resp.json().get("pairs") or []

    by_address: dict[str, dict] = {}
    for p in pairs:
        addr = p.get("baseToken", {}).get("address")
        if not addr:
            continue
        # keep the highest-liquidity pair per token address
        if addr not in by_address or (p.get("liquidity") or {}).get("usd", 0) > (by_address[addr].get("liquidity") or {}).get("usd", 0):
            by_address[addr] = p

    out = []
    for addr, p in by_address.items():
        liquidity = (p.get("liquidity") or {}).get("usd") or 0
        volume_24h = (p.get("volume") or {}).get("h24") or 0
        price_change_1h = (p.get("priceChange") or {}).get("h1")
        created_at = p.get("pairCreatedAt")
        age_hours = (time.time() * 1000 - created_at) / 3_600_000 if created_at else None
        vol_liq_ratio = volume_24h / liquidity if liquidity else 0
        pump_flag = _pump_signature(age_hours, vol_liq_ratio, price_change_1h)
        out.append({
            "symbol": p.get("baseToken", {}).get("symbol"),
            "chain": p.get("chainId"),
            "age_hours": age_hours,
            "price_change_1h": price_change_1h,
            "volume_24h": volume_24h,
            "liquidity_usd": liquidity,
            "vol_liq_ratio": round(vol_liq_ratio, 2),
            "pump_flag": pump_flag,
            "url": p.get("url"),
        })
    return sorted(out, key=lambda x: x["vol_liq_ratio"], reverse=True)


_KRONOS_DIR = os.path.dirname(__file__)
_KRONOS_PYTHON = os.path.join(_KRONOS_DIR, "kronos_venv", "bin", "python")
_KRONOS_SCRIPT = os.path.join(_KRONOS_DIR, "kronos_forecast.py")


def get_kronos_forecast(symbol: str, interval: str = "5m", pred_len: int = 12) -> dict | None:
    """Kronos foundation-model price forecast (github.com/shiyu-coder/Kronos)
    — runs as a subprocess in its own venv (kronos_venv, Python 3.11+, needs
    torch) since it requires Python 3.10+ and this project's main venv is
    3.9. Returns None on any failure (venv missing, model load error,
    timeout) so callers treat it as an optional signal, not a hard
    dependency."""
    if not os.path.exists(_KRONOS_PYTHON):
        return None
    try:
        result = subprocess.run(
            [_KRONOS_PYTHON, _KRONOS_SCRIPT, symbol, interval, str(pred_len)],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout.strip())
    except Exception:
        return None


def get_coinmarketcal_events(symbol: str) -> list[dict]:
    """Upcoming catalyst events for a coin, if covered. Free tier is
    top-100-coins-only and 24h-delayed — most meme symbols here won't have
    coverage; this is a best-effort background check, not core signal."""
    key = os.environ.get("COINMARKETCAL_API_KEY")
    if not key:
        return []
    resp = requests.get(
        "https://api.coinmarketcal.com/v2/events",
        headers={"x-api-key": key, "Accept": "application/json"},
        params={"coins": symbol.lower()},
        timeout=10,
    )
    if resp.status_code != 200:
        return []
    return resp.json().get("data", [])


def get_binance_momentum(symbols: list[str] = MEME_SYMBOLS) -> dict[str, dict]:
    """24h price-change % and quote volume from Binance's public ticker. No key needed."""
    resp = requests.get("https://api.binance.com/api/v3/ticker/24hr", timeout=10)
    resp.raise_for_status()
    by_symbol = {row["symbol"]: row for row in resp.json()}
    out = {}
    for sym in symbols:
        row = by_symbol.get(f"{sym}USDT")
        if row:
            out[sym] = {
                "pct_change_24h": float(row["priceChangePercent"]),
                "quote_volume": float(row["quoteVolume"]),
            }
    return out


def get_binance_momentum_short(symbols: list[str] = MEME_SYMBOLS, window: str = "1h") -> dict[str, dict]:
    """Same shape as get_binance_momentum (keys named pct_change_24h/
    quote_volume for drop-in compatibility with rank_symbols) but sourced
    from Binance's rolling-window ticker over `window` instead of a fixed
    24h — for lower-timeframe trading where a full day's change is too
    laggy a signal. Batched: this endpoint caps at 100 symbols/request."""
    out: dict[str, dict] = {}
    for i in range(0, len(symbols), 100):
        batch = symbols[i : i + 100]
        params = {
            "symbols": json.dumps([f"{s}USDT" for s in batch], separators=(",", ":")),
            "windowSize": window,
        }
        resp = requests.get("https://api.binance.com/api/v3/ticker", params=params, timeout=10)
        resp.raise_for_status()
        for row in resp.json():
            sym = row["symbol"].removesuffix("USDT")
            out[sym] = {
                "pct_change_24h": float(row["priceChangePercent"]),  # actually `window`, not 24h — see docstring
                "quote_volume": float(row["quoteVolume"]),
            }
    return out


def get_market_breadth(momentum: dict[str, dict]) -> dict:
    """Regime signal from the same momentum dict already fetched for ranking
    (get_binance_momentum_short's output) — no extra API calls. up_pct/
    down_pct measure how broadly the trading universe is participating right
    now, independent of any single token's own momentum score. Weak breadth
    (most of the universe red even if the top-ranked symbol looks fine) is a
    market-condition risk the per-symbol gates (RSI, Kronos, LLM) can't see."""
    if not momentum:
        return {"up_pct": None, "down_pct": None, "avg_change": None}
    changes = [v["pct_change_24h"] for v in momentum.values()]
    n = len(changes)
    up = sum(1 for c in changes if c > 0)
    down = sum(1 for c in changes if c < 0)
    return {"up_pct": up / n, "down_pct": down / n, "avg_change": sum(changes) / n}


def _test_get_market_breadth() -> None:
    momentum = {
        "A": {"pct_change_24h": 5.0}, "B": {"pct_change_24h": -3.0},
        "C": {"pct_change_24h": -1.0}, "D": {"pct_change_24h": 2.0},
    }
    b = get_market_breadth(momentum)
    assert b["up_pct"] == 0.5, b
    assert b["down_pct"] == 0.5, b
    assert abs(b["avg_change"] - 0.75) < 1e-9, b
    assert get_market_breadth({}) == {"up_pct": None, "down_pct": None, "avg_change": None}


def _compute_rsi(closes: list[float], period: int = 14) -> float | None:
    """Wilder-style RSI (0-100) over `closes`. Pure function, no I/O — the
    testable part; get_rsi() wraps it with the actual candle fetch."""
    if len(closes) < period + 1:
        return None
    gains = losses = 0.0
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains += max(delta, 0)
        losses += max(-delta, 0)
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def get_rsi(symbol: str, interval: str = "1h", period: int = 14) -> float | None:
    """RSI over the last `period` closed candles. None on a fetch failure or
    insufficient history — caller treats that as "can't tell", not a block."""
    try:
        resp = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": f"{symbol}USDT", "interval": interval, "limit": period + 1},
            timeout=10,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException:
        return None
    closes = [float(k[4]) for k in resp.json()]
    return _compute_rsi(closes, period)


def _test_compute_rsi() -> None:
    rising = [float(i) for i in range(15)]  # only gains -> RSI = 100
    assert _compute_rsi(rising) == 100.0
    falling = [float(15 - i) for i in range(15)]  # only losses -> RSI = 0
    assert _compute_rsi(falling) == 0.0
    assert _compute_rsi([1.0, 2.0]) is None  # not enough candles


def _compute_atr(candles: list[tuple[float, float, float]], period: int = 14) -> float | None:
    """Wilder's ATR (average true range, absolute price units) over
    `candles` — each a (high, low, close) tuple, oldest first. Pure
    function; get_atr() wraps it with the actual candle fetch. Used to size
    a stop-loss to how much a coin actually moves instead of one flat %
    for every symbol — a calm coin gets a tight stop, a volatile one gets
    room to breathe before being called a real reversal."""
    if len(candles) < period + 1:
        return None
    true_ranges = []
    for i in range(1, len(candles)):
        high, low, _ = candles[i]
        prev_close = candles[i - 1][2]
        true_ranges.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    return sum(true_ranges[-period:]) / period


def get_atr(symbol: str, interval: str = "1h", period: int = 14) -> float | None:
    """ATR in absolute price units over the last `period` closed candles.
    None on a fetch failure or insufficient history — caller treats that as
    "can't tell", falling back to a flat-% stop instead of blocking."""
    try:
        resp = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": f"{symbol}USDT", "interval": interval, "limit": period + 1},
            timeout=10,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException:
        return None
    candles = [(float(k[2]), float(k[3]), float(k[4])) for k in resp.json()]  # high, low, close
    return _compute_atr(candles, period)


def _test_compute_atr() -> None:
    # constant high-low range, flat closes -> ATR == that range
    flat = [(101.0, 99.0, 100.0) for _ in range(15)]
    assert abs(_compute_atr(flat) - 2.0) < 1e-9
    assert _compute_atr([(101.0, 99.0, 100.0)]) is None  # not enough candles


def get_dexscreener_trend(symbol: str) -> dict | None:
    """Top DEX pair by liquidity for a symbol, with 24h price change/volume.
    No key needed. Best-effort — DEXScreener 400s on some queries (e.g. a
    single-letter symbol like "A", seen live), so a request failure returns
    None instead of raising and killing the whole scan cycle."""
    try:
        resp = requests.get(
            "https://api.dexscreener.com/latest/dex/search",
            params={"q": symbol},
            timeout=10,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException:
        return None
    pairs = resp.json().get("pairs") or []
    if not pairs:
        return None
    top = max(pairs, key=lambda p: (p.get("liquidity") or {}).get("usd", 0))
    return {
        "pct_change_24h": (top.get("priceChange") or {}).get("h24"),
        "volume_24h": (top.get("volume") or {}).get("h24"),
        "liquidity_usd": (top.get("liquidity") or {}).get("usd"),
        "description": (top.get("info") or {}).get("description"),
    }


def get_google_trends(keywords: list[str] = MEME_SYMBOLS) -> dict[str, int]:
    """Latest Google search-interest score (0-100) per keyword. No key needed, can rate-limit."""
    from pytrends.request import TrendReq

    pytrends = TrendReq(hl="en-US", tz=0)
    out: dict[str, int] = {}
    # pytrends caps at 5 keywords per request
    for i in range(0, len(keywords), 5):
        batch = keywords[i : i + 5]
        pytrends.build_payload(batch, timeframe="now 1-d")
        df = pytrends.interest_over_time()
        for kw in batch:
            out[kw] = int(df[kw].iloc[-1]) if kw in df and len(df) else 0
        time.sleep(1)  # ponytail: avoid pytrends' aggressive rate limit
    return out


def get_reddit_mentions(
    keywords: list[str] = MEME_SYMBOLS,
    subreddits: tuple[str, ...] = ("CryptoMoonShots", "SatoshiStreetBets", "CryptoCurrency"),
    limit: int = 100,
) -> dict[str, int]:
    """Count of recent hot-post titles mentioning each keyword. Needs REDDIT_CLIENT_ID/SECRET."""
    if not os.environ.get("REDDIT_CLIENT_ID"):
        raise RuntimeError("REDDIT_CLIENT_ID not set — see .env.example")

    import praw

    reddit = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ.get("REDDIT_USER_AGENT", "memecoin-scanner/0.1"),
    )
    counts = {kw: 0 for kw in keywords}
    for sub in subreddits:
        for post in reddit.subreddit(sub).hot(limit=limit):
            title = post.title.upper()
            for kw in keywords:
                if kw.upper() in title:
                    counts[kw] += 1
    return counts


def get_lunarcrush_social(symbol: str) -> dict | None:
    """Galaxy Score / social volume. Requires LunarCrush Individual+ tier."""
    key = os.environ.get("LUNARCRUSH_API_KEY")
    if not key:
        return None
    resp = requests.get(
        f"https://lunarcrush.com/api4/public/coins/{symbol.lower()}/v1",
        headers={"Authorization": f"Bearer {key}"},
        timeout=10,
    )
    data = resp.json()
    if "error" in data:
        return None  # ponytail: tier-gated, not a hard failure — caller just skips this signal
    return data


def get_cmc_movers(limit: int = 100) -> dict[str, float]:
    """Whole-market 24h % change from CoinMarketCap listings/latest — catches
    meme coins not yet listed on Binance. Free Basic-tier endpoint."""
    key = os.environ.get("COINMARKETCAP_API_KEY")
    if not key:
        return {}
    resp = requests.get(
        "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest",
        headers={"X-CMC_PRO_API_KEY": key},
        params={"limit": limit, "sort": "percent_change_24h"},
        timeout=10,
    )
    resp.raise_for_status()
    return {
        row["symbol"]: row["quote"]["USD"]["percent_change_24h"]
        for row in resp.json()["data"]
    }


OLLAMA_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["buy", "skip"]},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "confidence", "reasoning"],
}


def get_llm_verdict(
    symbol: str, signals: dict, model: str = "llama3.2:3b", kronos: dict | None = "unset"
) -> dict | None:
    """Bull/bear reasoning over the ranked signals via a local Ollama model —
    free, no API key, no rate limit. `ollama pull llama3.2:3b` once first;
    `ollama serve` (or the Ollama app) must be running.

    `kronos`: pass an already-fetched forecast (from get_kronos_forecast) to
    avoid a second subprocess call — callers that also need the raw forecast
    for their own scoring (see execute.py's compute_opportunity_score) fetch
    it once and pass it in here. Leave unset to fetch it internally as before.
    """
    events = get_coinmarketcal_events(symbol)
    events_note = (
        f"Upcoming catalyst events: {json.dumps([e.get('title') for e in events])}."
        if events
        else "No known upcoming catalyst events (or not covered by free-tier data)."
    )
    if kronos == "unset":
        kronos = get_kronos_forecast(symbol)
    kronos_note = (
        f"Kronos foundation-model forecast: predicts {kronos['predicted_pct_change']:+.2f}% "
        f"over the next {kronos['pred_len']} {kronos['interval']} candles."
        if kronos
        else "Kronos forecast unavailable this cycle."
    )
    prompt = (
        f"Meme coin {symbol}. Signals: {json.dumps(signals)}. {events_note} {kronos_note} "
        "This is a short-term momentum gamble on testnet, not an investment "
        "thesis — give your honest read on whether the momentum looks real "
        "(fresh, still building) or already exhausted (spiked and fading). "
        "confidence must be a number between 0.0 and 1.0. reasoning must be "
        "1-2 non-empty sentences."
    )
    for attempt in range(2):  # small local model occasionally clips the forced-JSON output short
        try:
            resp = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "format": OLLAMA_VERDICT_SCHEMA,
                    "stream": False,
                },
                timeout=60,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            return None  # ponytail: Ollama not running — caller treats as "skip", not a crash
        verdict = json.loads(resp.json()["message"]["content"])
        if verdict.get("reasoning", "").strip():
            return verdict
    return None  # both attempts came back with empty reasoning — treat as unavailable, not a real verdict


OLLAMA_SHORT_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["short", "skip"]},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "confidence", "reasoning"],
}


def get_llm_short_verdict(
    symbol: str, signals: dict, model: str = "llama3.2:3b", kronos: dict | None = "unset"
) -> dict | None:
    """Mirror of get_llm_verdict for the short side: called on the WORST
    momentum performer (not the best), asks whether the decline still has
    room to fall (short) or is already oversold/likely to bounce (skip).
    Same Ollama call shape, different framing and schema (short/skip instead
    of buy/skip)."""
    events = get_coinmarketcal_events(symbol)
    events_note = (
        f"Upcoming catalyst events: {json.dumps([e.get('title') for e in events])}."
        if events
        else "No known upcoming catalyst events (or not covered by free-tier data)."
    )
    if kronos == "unset":
        kronos = get_kronos_forecast(symbol)
    kronos_note = (
        f"Kronos foundation-model forecast: predicts {kronos['predicted_pct_change']:+.2f}% "
        f"over the next {kronos['pred_len']} {kronos['interval']} candles."
        if kronos
        else "Kronos forecast unavailable this cycle."
    )
    prompt = (
        f"Meme coin {symbol} is one of the worst performers on the market right now. "
        f"Signals: {json.dumps(signals)}. {events_note} {kronos_note} "
        "This is a short-term momentum-short gamble on testnet, not an investment "
        "thesis — give your honest read on whether this decline still has real "
        "room to keep falling (fresh breakdown, not yet exhausted) or is already "
        "oversold and likely to bounce (bad time to short). "
        "confidence must be a number between 0.0 and 1.0. reasoning must be "
        "1-2 non-empty sentences."
    )
    for attempt in range(2):
        try:
            resp = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "format": OLLAMA_SHORT_VERDICT_SCHEMA,
                    "stream": False,
                },
                timeout=60,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            return None
        verdict = json.loads(resp.json()["message"]["content"])
        if verdict.get("reasoning", "").strip():
            return verdict
    return None


def get_text_sentiment(text: str) -> float:
    """VADER compound sentiment score, -1 (bearish) to +1 (bullish). Free,
    instant, no model download — tuned for short social-media-style text.
    ponytail: crude general-purpose lexicon, not finance-specific — swap for
    FinBERT (huggingface.co/ProsusAI/finbert) if this misreads crypto slang."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer().polarity_scores(text)["compound"]


def rank_symbols(
    momentum: dict[str, dict], dex_scores: dict[str, dict], trends: dict[str, int]
) -> list[tuple[str, float]]:
    """Combine signals into one ranked list, highest score first.

    Score = 24h price change % (Binance) + 24h price change % (DEX, halved —
    DEX pairs are noisier/thinner liquidity) + Google Trends score (0-100,
    scaled down 10x so it doesn't dominate). Pure function, no I/O — the
    thing that's actually worth a self-check.
    """
    scored = []
    for sym in momentum:
        score = momentum[sym]["pct_change_24h"]
        dex = dex_scores.get(sym)
        if dex and dex.get("pct_change_24h") is not None:
            score += dex["pct_change_24h"] * 0.5
        score += trends.get(sym, 0) * 0.1
        scored.append((sym, round(score, 2)))
    return sorted(scored, key=lambda x: x[1], reverse=True)


def equal_profit_fractions(tp_pcts: tuple) -> tuple:
    """Quantity fraction per take-profit leg so each leg banks the SAME
    profit amount, not the same quantity. A farther target needs a smaller
    slice to lock in the same profit as a closer one, so fractions are
    inversely weighted to tp_pct (e.g. +15%/+30% -> ~67%/~33%, not 50/50)."""
    inverse = [1 / tp for tp in tp_pcts]
    total = sum(inverse)
    return tuple(i / total for i in inverse)


def _test_rank_symbols() -> None:
    momentum = {"A": {"pct_change_24h": 10.0}, "B": {"pct_change_24h": 2.0}}
    dex = {"A": {"pct_change_24h": 20.0}}
    trends = {"B": 100}
    result = rank_symbols(momentum, dex, trends)
    assert result[0][0] == "A", f"expected A first (10 + 20*0.5=20), got {result}"
    assert result[0][1] == 20.0, result
    assert result[1] == ("B", 12.0), f"expected B=2+100*0.1=12, got {result}"


def _test_equal_profit_fractions() -> None:
    from decimal import Decimal

    fractions = equal_profit_fractions((Decimal("0.15"), Decimal("0.30")))
    assert abs(sum(fractions) - 1) < Decimal("0.0001"), fractions
    # each leg's profit contribution (fraction * tp_pct) must be equal
    profit_0 = fractions[0] * Decimal("0.15")
    profit_1 = fractions[1] * Decimal("0.30")
    assert abs(profit_0 - profit_1) < Decimal("0.0001"), (profit_0, profit_1)
    assert fractions[0] > fractions[1], "closer target should get the bigger slice"


if __name__ == "__main__":
    _test_rank_symbols()  # fails loudly if the scoring logic breaks
    _test_equal_profit_fractions()
    _test_pump_signature()
    _test_compute_rsi()
    _test_compute_atr()
    _test_get_market_breadth()

    print("Fetching Binance momentum...")
    momentum = get_binance_momentum()

    print("Fetching DEXScreener trend per symbol...")
    dex_scores = {sym: get_dexscreener_trend(sym) for sym in momentum}

    print("Fetching Google Trends (rate-limited, may take a bit)...")
    try:
        trends = get_google_trends(list(momentum))
    except Exception as e:  # pytrends 429s often
        print(f"  skipped: {e}")
        trends = {}

    ranked = rank_symbols(momentum, dex_scores, trends)
    print("\nRanked (score = Binance 24h% + 0.5*DEX 24h% + 0.1*Trends):")
    for sym, score in ranked:
        print(f"  {sym:6s} {score:+7.2f}  {momentum[sym]}")

    lunar = get_lunarcrush_social(ranked[0][0]) if ranked else None
    if lunar is None:
        print("\nLunarCrush: skipped (no key, or account below Individual tier)")

    cmc_movers = get_cmc_movers()
    if cmc_movers:
        print(f"\nCMC whole-market scan: {len(cmc_movers)} symbols, top mover "
              f"{max(cmc_movers, key=cmc_movers.get)} "
              f"({max(cmc_movers.values()):+.1f}%)")
    else:
        print("\nCoinMarketCap: skipped (no key)")

    print("\nVADER sentiment on top pair's DEXScreener description:")
    for sym, _ in ranked[:2]:
        desc = (dex_scores.get(sym) or {}).get("description")
        if desc:
            print(f"  {sym}: {get_text_sentiment(desc):+.2f}  ({desc[:80]}...)")
        else:
            print(f"  {sym}: no description text available")

    print("\nLLM bull/bear verdict on top 2 ranked (local Ollama, free):")
    for sym, score in ranked[:2]:
        verdict = get_llm_verdict(sym, {"rank_score": score, **momentum[sym]})
        if verdict is None:
            print("  skipped (Ollama not running — `ollama serve` or open the app)")
            break
        print(f"  {sym}: {verdict['verdict'].upper()} "
              f"(confidence {verdict['confidence']:.2f}) — {verdict['reasoning']}")
