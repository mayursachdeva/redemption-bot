# Binance Testnet Spot Account Reset — 2026-08-06

Binance wiped and reseeded the **spot** testnet account. This was an
exchange-side action, not a trading event and not a bug in this bot. Recording
it because every spot number before this date belongs to a different account
than every number after it, and a PnL series that silently spans the boundary
is meaningless.

## Evidence

Each of these alone is suggestive; together they are conclusive.

- **Faucet-seed balances.** Exactly `1.00000000` BTC, `1.00000000` ETH,
  `1.00000000` BNB, `10000.00000000` USDT, `10000.00000000` USDC — and a
  uniform `18446` across hundreds of unrelated assets (GMT, NIGHT, PEOPLE,
  SAGA, XEC, ACH, ACT, AEVO, AI, AIXBT, …). Round, identical numbers across
  unrelated assets do not arise from trading.
- **Held quantities bear no relation to the positions.** XEC showed 18,446
  held against a 277,132,677 position; CFX 12,880 against 106,828.
- **Every OCO bracket vanished without executing.** Zero open spot orders, and
  order history shows no take-profit fill, no stop-loss fill, and no market
  sell for any of the seven trades. Orders were cancelled, not filled.
- **Futures was untouched** — marginBalance $4,968.55, all 14 shorts intact
  with their algo brackets. A wipe confined to one of two independent testnet
  accounts is an exchange action, not a market one.

## State immediately before the reset

| | Value |
|---|---|
| Spot portfolio total | **$71,213.56** (~$53,256 free USDT + ~$20,407 deployed) |
| Spot realized PnL, 48 closed legs | **+1,964.72 USDT** |
| Open spot positions | ATM, CFX, GMT, NIGHT, PEOPLE, SAGA, XEC (13 legs) |
| Futures marginBalance | $4,968.55 (unaffected) |

## State after

| | Value |
|---|---|
| Spot tradeable capital | **$10,000.00 USDT**, $0 deployed |
| Open spot positions | none |

`get_portfolio_exposure` (execute.py:209) correctly reports $10,000 rather than
counting the ~$85k faucet basket — it deliberately excludes seeded assets and
counts only free USDT plus the bot's own tagged positions. The 1 BTC
(≈$64,726) is not the bot's capital and is right to be ignored.

Position sizes will now be roughly **7× smaller** than before the reset, since
sizing is a fraction of portfolio value.

## Journal cleanup

The reconciler observed an empty order list and classified the wipe as 13
simultaneous leg closures, writing rows with `exit_price`, `exit_reason`,
`pnl_pct`, and `closed_at` all `null` — a phantom mass-closure. Those 13 rows
were removed from `trade_journal.jsonl`; the pre-cleanup file is preserved at
`trade_journal.jsonl.pre-reset-backup` (61 rows; 48 valid + the 13 nulls).

Removing them does not change any PnL figure: every analysis in this repo skips
rows where `pnl_pct is None`, so they contributed nothing but noise. Spot
realized PnL is +1,964.72 either way.

## Interpreting PnL across this boundary

The +1,964.72 spot realized figure was earned on a ~$71k account that no longer
exists. Do not add it to returns earned on the new $10k account, and do not
compute a percentage return spanning the reset — the denominators differ by
about 7×. Futures history is continuous and unaffected.

## Not fixed, deliberately

The reconciler will misread any future account wipe the same way. Hardening it
(for example: refuse to classify a closure when *every* tracked leg disappears
at once and no corresponding fill exists) is a real improvement, but it is a
distinct piece of work with its own failure modes and was not attempted here.
