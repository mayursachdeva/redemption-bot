# redemption-bot

A momentum-trading bot for meme coins and low-price altcoins, built and hardened over
one long iterative session on **Binance Spot Testnet** (long) and **Binance USD-M
Futures Testnet** (short) in parallel. All trades are on testnet — fake funds, real
order mechanics, real market data.

> **This is a gambling/research experiment, not an investment product.** Nothing here
> is trading advice. The strategy is deliberately high-variance (momentum chasing on
> illiquid small-cap tokens) and was built to stress-test risk controls under real
> conditions, not to demonstrate a reliable edge.

## Why "redemption"

Early in the session a sizing bug let a single trade consume the *entire* portfolio
cap in one bet (DIA, ~$14.6k) instead of splitting the cap across many smaller
positions. It stopped out for **-$2,226.45** within one second of opening — the
largest loss in the whole session, and for a long stretch the dominant reason the
book was net negative. Everything that follows in this repo — position-size caps,
the drawdown circuit breaker, the opportunity-score gate, ATR-based stops — exists
because that one bug got found, root-caused, and fixed live against a real account,
and the same discipline got applied to everything discovered after it. By the end of
the session the combined book (spot + futures) was net positive for the first time.

## What it does

Every 5 minutes (`loop.py`):

1. **Scans** the live market (Binance mainnet ticker data, used read-only for
   discovery — testnet doesn't have real volume) for any token trading under
   `MAX_PRICE_USD` (default $5), not just a fixed meme-coin list.
2. **Ranks** candidates by momentum (1h price/volume), DEXScreener trend, Google
   Trends interest.
3. **Walks the full ranked list**, not just #1 — falls through to the next
   candidate on any gate failure instead of one dominant symbol locking out
   everything else for a cycle.
4. **Gates** each candidate through, in order (cheapest checks first):
   - Two-cycle momentum confirmation (a candidate must have shown up last cycle too,
     not just a single 5-minute spike)
   - Dedup (don't stack a second position in a symbol already held)
   - RSI(14, 1h) overbought/oversold filter
   - **Kronos** foundation-model price forecast (ML, trained on raw OHLCV — not a
     hand-coded indicator)
   - A local LLM (Ollama, `llama3.2:3b`) reads all of the above plus upcoming
     catalyst events and gives a qualitative buy/skip (or short/skip) verdict with
     a confidence score
   - A blended **opportunity score** (`0.3 × Kronos conviction + 0.7 × LLM
     confidence`) with a hard veto if Kronos alone is bearish/bullish enough to
     threaten the trade thesis — added because the LLM was repeatedly observed
     overriding a clearly bearish Kronos forecast on raw momentum alone
5. **Sizes** the trade from the account's own portfolio value (not a fixed dollar
   amount), respecting a total exposure cap, a per-trade cap, a "golden trade"
   reserve for very-high-confidence signals, a drawdown circuit breaker, and a
   market-breadth regime filter — all described below.
6. **Executes** a laddered exit: multiple take-profit legs sized so each leg banks
   the *same* profit (not the same quantity), plus an ATR-derived stop-loss, plus
   an early-exit override that locks in gains the moment Kronos turns against an
   open winner.

The same pipeline runs mirrored for shorts on a separate Binance Futures Testnet
account (`execute_futures.py`) — worst-momentum candidate instead of best, a
short-framed LLM prompt, RSI-oversold instead of overbought, Kronos direction
negated in the opportunity score.

## Architecture

```
scanner.py          Signal gathering: Binance momentum, DEXScreener, Google Trends,
                     Reddit/LunarCrush/CoinMarketCap (social), CoinMarketCal (events),
                     Kronos forecast wrapper, RSI/ATR/market-breadth, local-LLM verdict
                     (both long and short framings)

execute.py           Spot long execution: position sizing, laddered OCO brackets,
                     early-exit, portfolio-exposure accounting, the opportunity-score
                     gate, drawdown circuit breaker, market-breadth regime filter,
                     ATR-based variable stop-loss, trade journal reconciliation

execute_futures.py   Mirror of execute.py for USD-M Futures shorts: leverage,
                     algo-order brackets (Binance migrated conditional orders to a
                     separate API in late 2025 — see code comments), short-specific
                     early-exit, its own journal

loop.py              Runs one long cycle + one short cycle + a leverage-journal
                     refresh every 5 minutes. Each stage is independently wrapped so
                     one failure doesn't kill the other two.

leverage_journal.py  Reconstructs the full trade history from live order data and
                     writes leverage_journal.xlsx: what every trade would have
                     returned at 3x/5x/10x, long and short (mirrored), including a
                     liquidation simulation. Reporting only — does not affect real
                     execution, which stays spot (unleveraged) for longs.

trade_stats.py       Win rate / avg win / avg loss / exit-reason breakdown from the
                     trade journal.

dashboard.py         Local HTML dashboard (positions, exposure, ranked signals,
                     recent verdicts) — generates dashboard.html, not published.

backtest.py          Historical backtest harness (Binance klines, daily and
                     intraday) for single-TP vs laddered-TP exit strategies.

telegram_signals.py / telegram_login.py
                     Reads a paid third-party premium-signals Telegram channel via
                     telethon (own-account login, since Bot API can't read DMs sent
                     by a bot you don't own). Built and reverse-engineered, then
                     backtested and found net-negative live — currently NOT wired
                     into the trading decision (see "Things that were tried and
                     dropped" below).
```

## The knowledge layer

Everything is direct API access — no browser/cookie scraping:

- **Market data**: Binance (momentum, klines for RSI/ATR), DEXScreener
- **Social**: LunarCrush (gated behind a paid tier), Reddit (official PRAW API),
  CoinMarketCap movers
- **Catalysts**: CoinMarketCal event calendar
- **Forecast**: Kronos foundation model (`shiyu-coder/Kronos`), local inference,
  no external data dependency
- **Reasoning**: local LLM via Ollama — free, no API cost, no rate limit

## Risk controls (all found necessary the hard way)

| Control | What it does | Why it exists |
|---|---|---|
| `MAX_PORTFOLIO_PCT` / `POSITION_SPLIT` | Total exposure cap, split across N concurrent slots | The DIA bug: one trade consuming the whole cap instead of sharing it |
| `MIN_BET_FRACTION` | Skip a trade if remaining cap room is too thin relative to a normal bet | A good signal (DEXE, 18.8% return) got a $83 scrap bet instead of ~$4,300 because the cap was nearly full |
| Drawdown circuit breaker | Doubles `POSITION_SPLIT` (halves bet size) once the portfolio is down X% from its peak | Kelly-criterion-style: shrink bets in a losing streak instead of trying to trade back to even |
| Market-breadth regime filter | Same mechanism, triggered by weak overall market breadth instead of own P&L | A dominant top-ranked symbol can look fine while the rest of the market is red |
| Opportunity-score gate | Blends Kronos forecast direction with LLM confidence; hard veto on strongly conflicting Kronos | Six real losses were bought despite the verdict's own text flagging a bearish Kronos read |
| ATR-based stop-loss | Stop width scales with each coin's actual volatility instead of one flat 15% | A flat stop got a genuinely volatile coin (BANK) stopped out right before a reversal that would have been +$330 |
| Per-symbol dedup | Won't buy/short a symbol already held | Found live: RIF and DGB both got bought twice in the same session |
| Unprotected-leg auto-heal | Detects and re-hedges a position left without a stop-loss after an OCO partial fill | Binance cancels the *sibling* order the instant one side partially fills — found live on REZ, sat unprotected for hours before being noticed |
| Golden trade reserve | A separate 20%-of-portfolio pool for very-high-confidence (≥0.85) verdicts, capped at 1/4 per trade | Lets an exceptional signal size up without touching the normal per-trade cap |

## Two books, one gate

Spot (long) and Futures (short) run through the same opportunity-score logic,
mirrored, but are otherwise independent accounts/portfolios/journals. They can end
up on opposite sides of the same symbol at the same time (e.g. long BANK on spot,
short BANK on futures) — not blocked, since each book is sized and risk-controlled
independently, but it does mean a move in either direction partially nets out
across the two books rather than compounding.

## Things that were tried and dropped

- **Reddit via Composio** — dropped for platform-policy reasons.
- **A third-party paid Telegram "premium signals" bot** — reverse-engineered (had
  to defeat a locked "Show signal" button and a mid-2025 Binance Algo-Order API
  migration to even read it), then backtested against real Binance history: net
  **-4.3% per trade** if followed literally, because it posts *after* the move
  already started. Built the read pipeline, then explicitly excluded it from the
  live decision loop once the backtest came back negative.
- **Hourly macOS notifications via launchd** — blocked by a TCC/Full-Disk-Access
  sandboxing restriction on processes spawned from a directory under `~/Documents`;
  the user chose to skip fixing it rather than grant broader disk access.

## Setup

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill in your own keys — see comments in the file
ollama pull llama3.2:3b && ollama serve   # local LLM, free, required
```

Needs **two separate Binance testnet accounts**: Spot testnet
(`testnet.binance.vision`) for longs, USD-M Futures testnet
(`testnet.binancefuture.com`, GitHub-login signup) for shorts. Kronos requires its
own Python 3.10+ venv (`kronos_venv/`, not included — see `kronos_forecast.py`) with
`shiyu-coder/Kronos` cloned locally. That same venv also runs FinBERT text
sentiment (`pip install transformers`, see `finbert_sentiment.py`) — no
separate venv needed, `torch` is already there for Kronos.

Run once: `./venv/bin/python -m execute` (long) or `execute_futures` (short).
Run continuously: `./venv/bin/python loop.py`.

## Honest limitations

- Small sample sizes throughout — win rates quoted in commit history and
  `trade_stats.py` output are not statistically robust yet.
- The early-exit mechanism only fires on positions already in profit, so any
  win-rate figure drawn from a journal dominated by early-exits is structurally
  optimistic — it excludes, by construction, every case where it would have fired
  on a loss.
- Leverage figures in `leverage_journal.xlsx` are a *simulation* against real spot
  price history — actual leveraged execution only happens on the futures short
  side; the long side never trades on margin.
- No backtested, statistically validated edge is claimed anywhere in this repo —
  every risk control here reduces the cost of being wrong, none of them make the
  underlying momentum signal reliably right.
