# Data dictionary

Origin legend: **EX** = exchange-defined field from the bhavcopy (header spellings verified on real
`BhavCopyDateWise_*.csv` files, 2026-10-23: `Date, Instrument Name, Symbol, Expiry Date, Option Type,
Strike Price, Open, High, Low, Close, Previous Close, Volume(Lots), Volume(In 000's), Value(Lacs),
Open Interest(Lots)`),
**CALC** = calculated by AURUM, **META** = provenance, **CFG** = from config (status per field in
`config/contracts.yaml`).

## Processed observations (`data/processed/<mode>/observations.parquet`)

| Column | Origin | Meaning |
|---|---|---|
| date | EX | Trading date (parsed with explicit formats, day-first) |
| symbol | EX | Product code, stripped/upper-cased (GOLD, GOLDM, GOLDGUINEA, GOLDPETAL, GOLDTEN) |
| expiry | EX | Contract expiry date as given in the file |
| contract_id | CALC | `SYMBOL_YYYYMMDD` of expiry - the identity of one contract |
| open/high/low/close | EX | Day's prices in the product's quotation unit. `close` is NOT an executable fill |
| prev_close | EX | Previous close as reported |
| settle | EX | Settlement price if the file has it (often absent) |
| volume_lots | EX | Lots traded in the day. A liquidity proxy, not order-book depth |
| oi_lots | EX | Open interest in lots at end of day |
| value_lacs | EX | Traded value in Rs lakh (`Value(Lacs)`); used to cross-check lot multipliers |
| data_mode | META | REAL or SYNTHETIC |
| source_file / source_row | META | File name and 1-based line number of origin |

## Rejections (`rejections.csv`)
All raw columns plus `reason` ∈ {unparseable_date, unparseable_expiry, missing_close,
non_positive_close, negative_open/high/low, high_below_low, close_outside_high_low,
expired_contract, negative_volume_or_oi, duplicate_identical, duplicate_conflicting_values,
conflicts_with_existing_observation}.

## Manifest (`manifest.json`)
One entry per ingested file: file name, SHA-256 of the bytes, order-independent content SHA-256,
rows read / accepted / rejected / out of scope / duplicates_existing, rejection reasons, origin
(`file` or `upload`), UTC timestamp, data mode. Invariant: read = accepted + rejected + out of scope +
duplicates_existing.

## Experiment log (`experiments.jsonl`)
One JSON line per Backtesting Lab run (including infeasible and unfavourable runs): configuration ID,
pair, period, windows, grid, fill rule, folds, OOS trades, net results, protocol conclusion.

## Pair analysis table (in memory)

| Column | Origin | Meaning |
|---|---|---|
| a_* / b_* | EX/CALC | Leg fields; `a_dte` = calendar days to expiry |
| a_norm, b_norm | CALC | Rs per 10 g 995-equivalent (purity-adjusted) |
| a_norm_naive, b_norm_naive | CALC | Units-only conversion (for comparison) |
| a_multiplier | CFG | Rs per lot per Rs 1 price move = lot_grams / quote_unit_grams |
| a_fine_grams_per_lot | CFG | lot_grams × purity/1000 |
| spread_abs | CALC | b_norm − a_norm |
| spread_pct / spread_pct_naive | CALC | % spread, adjusted / naive |
| implied_carry_pct | CALC | Calendar only: annualised carry, % p.a. |
| carry_used_pct, spread_pct_carry_adj, carry_adjusted | CALC | Cross only: same-day carry of leg A's product and the spread after removing carry over the expiry gap |
| metric / baseline_metric | CALC | AURUM measure / naive baseline measure |
| mean, std, z, percentile, n_hist | CALC | Rolling stats over prior window, excluding t |
| roll, gap_days | CALC | Contract change marker; days since previous row |
| comparability, pnl_verified | CALC/CFG | Per-row VALID / ADJUSTED / UNVERIFIED / INVALID; lot & tick verified |
| metric_raw | CALC | Measure on every row (exploratory); `metric` is blanked on non-comparable rows |
| a_ret_pct, b_ret_pct | CALC | Daily move vs exchange previous close (fallback: previous stored close) |
| a_source_file, a_source_row (and b_) | META | Raw file and line of each leg's observation |
| a_stale, b_stale, a_unchanged, b_unchanged, low_volume, near_expiry, roll, data_gap, expiry_gap_flag, extreme_move, oi_missing | CALC | Flags; see `analytics.FLAGS` for rule, effect and confirming data |

## Trade ledger (backtest)
direction, signal/entry/exit dates, exit reason (mean_reverted, time_stop, delivery_exit,
forced_end), contracts, whole lots per leg, entry/exit prices, pnl_a / pnl_b, fees_a / fees_b,
slippage and slip_unit (Rs per tick), gross / net P&L, turnover, participation (lots ÷ day volume),
exit deadline before the tender period, signal-day flags, notes (delayed exits, capacity).
