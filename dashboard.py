"""Local HTML dashboard for the meme-coin gambling bot. Reads the live
testnet account + scanner signals + loop.log, writes dashboard.html, opens
it in the browser. Not hosted — this reflects a real (testnet) trading
account, so it stays local rather than published anywhere.

Re-run this file any time to refresh; it's a snapshot, not a live page.
"""
from __future__ import annotations

import os
import re
import subprocess
from datetime import datetime
from decimal import Decimal

from execute import get_client, get_open_positions, get_portfolio_exposure, get_testnet_symbols, get_usdt_balance
from scanner import get_binance_universe, get_dexscreener_trend, get_binance_momentum, rank_symbols

_CYCLE_RE = re.compile(r"^=== (?P<ts>\S+) ===$")
_TOP_RE = re.compile(r"^Top tradeable: (?P<sym>\w+)")
_VERDICT_RE = re.compile(r"^Verdict: (?P<verdict>\w+) \(confidence (?P<conf>[\d.]+)\) — (?P<reasoning>.*)$")


def parse_recent_verdicts(log_path: str, limit: int = 15) -> list[dict]:
    if not os.path.exists(log_path):
        return []
    cycles: list[dict] = []
    current: dict | None = None
    with open(log_path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = _CYCLE_RE.match(line)
            if m:
                if current:
                    cycles.append(current)
                current = {"timestamp": m.group("ts"), "symbol": None, "verdict": None,
                           "confidence": None, "reasoning": None, "action": None}
                continue
            if current is None:
                continue
            if m := _TOP_RE.match(line):
                current["symbol"] = m.group("sym")
            elif m := _VERDICT_RE.match(line):
                current["verdict"] = m.group("verdict")
                current["confidence"] = float(m.group("conf"))
                current["reasoning"] = m.group("reasoning")
            elif line.startswith(("Bought ", "Not betting", "No ranked symbol", "Ollama not running")):
                current["action"] = line
    if current:
        cycles.append(current)
    return list(reversed(cycles))[:limit]


def _pct_class(value) -> str:
    if value is None:
        return "neutral"
    return "positive" if value > 0 else ("negative" if value < 0 else "neutral")


def render_html(data: dict) -> str:
    exposure_pct = float(data["deployed_pct"])
    cap_pct = float(data["cap_pct"])
    zone = "good" if exposure_pct < cap_pct * 0.7 else ("warning" if exposure_pct < cap_pct * 0.9 else "critical")

    position_rows = "".join(f"""
      <tr{' class="unprotected"' if p.get('unprotected') else ''}>
        <td>{p['symbol']}</td>
        <td>leg {p['leg']}</td>
        <td>{p['qty']}</td>
        <td>{f"{p['entry']:.6f}" if p['entry'] is not None else "—"}</td>
        <td>{p['current']:.6f}</td>
        <td>{f"{p['tp']:.6f}" if p['tp'] else "—"}</td>
        <td>{f"{p['sl']:.6f}" if p['sl'] is not None else "NO STOP-LOSS"}</td>
        <td class="{_pct_class(p['pnl_pct'])}">{f"{p['pnl_pct']:+.2f}%" if p['pnl_pct'] is not None else "—"}</td>
      </tr>""" for p in data["positions"]) or '<tr><td colspan="8" class="muted">No open positions</td></tr>'

    signal_rows = "".join(f"""
      <tr>
        <td>{s['symbol']}</td>
        <td>{s['score']:+.2f}</td>
        <td class="{_pct_class(s['momentum_pct'])}">{s['momentum_pct']:+.2f}%</td>
        <td class="{_pct_class(s['dex_pct'])}">{f"{s['dex_pct']:+.2f}%" if s['dex_pct'] is not None else "—"}</td>
        <td><span class="badge {'good' if s['liquid'] else 'critical'}">{'liquid' if s['liquid'] else 'no liquidity'}</span></td>
      </tr>""" for s in data["signals"])

    verdict_rows = "".join(f"""
      <tr>
        <td class="muted">{v['timestamp']}</td>
        <td>{v['symbol'] or '—'}</td>
        <td>{f'<span class="badge {"good" if v["verdict"]=="BUY" else "neutral"}">{v["verdict"]}</span>' if v['verdict'] else '—'}</td>
        <td>{f"{v['confidence']:.2f}" if v['confidence'] is not None else '—'}</td>
        <td class="reasoning">{(v['reasoning'] or '')[:140]}</td>
        <td class="muted">{v['action'] or ''}</td>
      </tr>""" for v in data["verdicts"]) or '<tr><td colspan="6" class="muted">No cycles logged yet</td></tr>'

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Meme Coin Gambling Dashboard</title>
<style>
  :root {{
    color-scheme: light;
    --surface: #fcfcfb; --surface-2: #f2f1ee; --border: #e4e2dd;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #8a887f;
    --blue: #2a78d6; --red: #e34948; --neutral-gray: #8a887f;
    --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #d03b3b;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      color-scheme: dark;
      --surface: #1a1a19; --surface-2: #242422; --border: #34332f;
      --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8a887f;
      --blue: #3987e5; --red: #e66767; --neutral-gray: #8a887f;
      --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #d03b3b;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 32px; background: var(--surface); color: var(--text-primary);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 1100px; margin-inline: auto;
  }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  .subtitle {{ color: var(--text-muted); font-size: 13px; margin-bottom: 28px; }}
  .stat-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }}
  .stat-tile {{
    background: var(--surface-2); border: 1px solid var(--border); border-radius: 8px;
    padding: 16px; overflow-x: auto;
  }}
  .stat-label {{ font-size: 12px; color: var(--text-secondary); margin-bottom: 6px; }}
  .stat-value {{ font-size: 22px; font-weight: 600; }}
  .exposure-bar-wrap {{ margin-bottom: 28px; }}
  .exposure-bar {{
    position: relative; height: 20px; background: var(--surface-2); border: 1px solid var(--border);
    border-radius: 10px; overflow: hidden;
  }}
  .exposure-fill {{ height: 100%; border-radius: 10px 0 0 10px; }}
  .exposure-fill.good {{ background: var(--good); }}
  .exposure-fill.warning {{ background: var(--warning); }}
  .exposure-fill.critical {{ background: var(--critical); }}
  .exposure-cap-marker {{
    position: absolute; top: -2px; bottom: -2px; width: 2px; background: var(--text-primary);
  }}
  .exposure-label {{ display: flex; justify-content: space-between; font-size: 12px; color: var(--text-secondary); margin-top: 4px; }}
  section {{ margin-bottom: 32px; }}
  h2 {{ font-size: 14px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--text-secondary); margin-bottom: 10px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--text-secondary); font-weight: 500; font-size: 12px; }}
  tr:hover td {{ background: var(--surface-2); }}
  .positive {{ color: var(--blue); }}
  .negative {{ color: var(--red); }}
  .neutral {{ color: var(--text-muted); }}
  .muted {{ color: var(--text-muted); }}
  .reasoning {{ color: var(--text-secondary); max-width: 320px; }}
  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600;
    color: white;
  }}
  .badge.good {{ background: var(--good); }}
  .badge.critical {{ background: var(--critical); }}
  .badge.neutral {{ background: var(--neutral-gray); }}
  .table-scroll {{ overflow-x: auto; }}
</style>
</head>
<body>
  <h1>Meme Coin Gambling Dashboard</h1>
  <div class="subtitle">Testnet only. Generated {data['generated_at']}. Re-run dashboard.py to refresh.</div>

  <div class="stat-row">
    <div class="stat-tile"><div class="stat-label">Total portfolio</div><div class="stat-value">${data['total']:.2f}</div></div>
    <div class="stat-tile"><div class="stat-label">Deployed</div><div class="stat-value">${data['deployed']:.2f}</div></div>
    <div class="stat-tile"><div class="stat-label">Deployed %</div><div class="stat-value">{exposure_pct:.1f}%</div></div>
    <div class="stat-tile"><div class="stat-label">Free USDT</div><div class="stat-value">${data['free_usdt']:.2f}</div></div>
  </div>

  <div class="exposure-bar-wrap">
    <div class="exposure-bar">
      <div class="exposure-fill {zone}" style="width:{min(exposure_pct,100):.1f}%"></div>
      <div class="exposure-cap-marker" style="left:{min(cap_pct,100):.1f}%"></div>
    </div>
    <div class="exposure-label">
      <span><span class="badge {zone}">{zone}</span> {exposure_pct:.1f}% deployed</span>
      <span>cap: {cap_pct:.0f}%</span>
    </div>
  </div>

  <section>
    <h2>Open Positions</h2>
    <div class="table-scroll">
    <table>
      <thead><tr><th>Symbol</th><th>Leg</th><th>Qty</th><th>Entry (≈)</th><th>Current</th><th>Take-profit</th><th>Stop-loss</th><th>P&L</th></tr></thead>
      <tbody>{position_rows}</tbody>
    </table>
    </div>
  </section>

  <section>
    <h2>Ranked Signals</h2>
    <div class="table-scroll">
    <table>
      <thead><tr><th>Symbol</th><th>Rank score</th><th>24h % (Binance)</th><th>DEX 24h %</th><th>Testnet</th></tr></thead>
      <tbody>{signal_rows}</tbody>
    </table>
    </div>
  </section>

  <section>
    <h2>Recent Verdicts</h2>
    <div class="table-scroll">
    <table>
      <thead><tr><th>Time</th><th>Symbol</th><th>Verdict</th><th>Confidence</th><th>Reasoning</th><th>Action</th></tr></thead>
      <tbody>{verdict_rows}</tbody>
    </table>
    </div>
  </section>
</body>
</html>"""


def gather_data() -> dict:
    client = get_client()
    total, deployed = get_portfolio_exposure(client)
    cap_pct = Decimal(os.environ.get("MAX_PORTFOLIO_PCT", "0.3")) * 100
    deployed_pct = (deployed / total * 100) if total else Decimal("0")

    # mirror execute.py's actual live logic: whole-market scan (filtered to
    # what's actually tradeable on testnet), then enrich only the top movers
    # (DEXScreener enrichment doesn't scale to 200+ symbols)
    testnet_symbols = get_testnet_symbols(client)
    universe = [s for s in get_binance_universe() if s in testnet_symbols]
    momentum = get_binance_momentum(universe)
    top_by_momentum = sorted(momentum, key=lambda s: momentum[s]["pct_change_24h"], reverse=True)[:15]
    dex_scores = {sym: get_dexscreener_trend(sym) for sym in top_by_momentum}
    ranked = rank_symbols({sym: momentum[sym] for sym in top_by_momentum}, dex_scores, {})
    signals = [
        {
            "symbol": sym,
            "score": score,
            "momentum_pct": momentum[sym]["pct_change_24h"],
            "dex_pct": (dex_scores.get(sym) or {}).get("pct_change_24h"),
            "liquid": len(client.depth(symbol=f"{sym}USDT", limit=5).get("asks", [])) > 0,
        }
        for sym, score in ranked
    ]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total": total,
        "deployed": deployed,
        "deployed_pct": deployed_pct,
        "cap_pct": cap_pct,
        "free_usdt": get_usdt_balance(client),
        "positions": get_open_positions(client),
        "signals": signals,
        "verdicts": parse_recent_verdicts(os.path.join(os.path.dirname(__file__), "loop.log")),
    }


if __name__ == "__main__":
    data = gather_data()
    out_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    with open(out_path, "w") as f:
        f.write(render_html(data))
    print(f"Wrote {out_path}")
    subprocess.run(["open", out_path])
