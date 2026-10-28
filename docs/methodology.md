# AURUM methodology

## 1. The question AURUM answers

"Should I trust this apparent price dislocation between two MCX gold contracts, and what evidence
supports or weakens that conclusion?" AURUM does not predict gold prices and never claims arbitrage.

Pipeline: raw observations → validation → contract matching → comparability → normalisation →
spread / carry → past-only statistics → quality flags → classification → (optional) protocol-based
validation of a strategy.

## 2. Data

- Source: MCX bhavcopy CSV files downloaded by the user (`BhavCopyDateWise_DDMMYYYY.csv`). Only CSV
  is accepted; other formats fail with an explicit message.
- Every input row ends as **accepted**, **rejected** (with a reason), **out of scope** (other
  commodities, options) or **duplicate of a stored observation**. The invariant
  `rows_read = accepted + rejected + out_of_scope + duplicates_existing` is checked for every file.
- A row for a (date, contract) already stored with **different** values is rejected as
  `conflicts_with_existing_observation`; the first-ingested row is kept.
- Re-downloads are skipped by file hash and by an order-independent content hash.
- Data modes: `HISTORICAL` (files placed locally), `USER_UPLOAD` (dashboard uploads), `SYNTHETIC`
  (generated; never MCX data). Real and synthetic data are stored separately.
- **Finding on real data:** after a zero-volume day the exchange's next-day "Previous Close" differs
  from the repeated close 99% of the time; after traded days it matches 99.5%. Zero-volume closes are
  therefore not prices. The close itself is MCX's 30-minute weighted average (per the contract PDFs).

## 3. Contract specifications and comparability

Specifications are versioned **per contract series** (MCX publishes them "for contracts expiring
<month> onwards"): a contract uses the latest version whose `applies_from_expiry` is on or before
its expiry. Every field has a status: VERIFIED (read in the official PDF), SECONDARY, ASSUMED.

| Status | Meaning |
|---|---|
| VALID | same quotation unit and purity on both legs, no conversion needed |
| ADJUSTED | different units/purity, converted using **VERIFIED** quotation unit and purity |
| UNVERIFIED | a conversion would rely on unverified fields: shown as exploratory only |
| INVALID | a specification is missing, or the spec changed between the two series |

Rules: calendar pairs (same product) need no conversion and are VALID unless the specification
changes between the two series. Cross-product pairs require VERIFIED quotation unit **and** purity on
both legs. Lot size and tick (P&L fields) do not affect comparability but must be VERIFIED for a
RESEARCH CANDIDATE. Each row of a pair series carries its own status; only VALID/ADJUSTED rows feed
the validated measures (`metric`), statistics, signals and backtests. `metric_raw` keeps every value
for clearly labelled exploration.

## 4. Normalisation and measures

`normalised = close ÷ quote_unit_g ÷ (purity/1000) × 10 × 0.995` (Rs per 10 g of 995-equivalent gold).
For 999 contracts this is the 995/999 factor that MCX itself uses in final settlement.

- **Calendar:** implied carry `c = 100 · ln(F_far/F_near) / ((T_far − T_near)/365)`, not computed when
  the expiries are fewer than 7 days apart. Baseline measure: raw spread `F_far − F_near`.
- **Cross:** `S = 100 · (B_norm/A_norm − 1)`; when expiries differ, carry over the gap is removed with
  A's same-day carry: `S_adj = 100 · ((B_norm/A_norm) · exp(−c · gap/365) − 1)`. Baseline measure:
  units-only spread ignoring purity.

## 5. Statistics without look-ahead

`μ_t, σ_t` = mean and sample std (ddof 1) of the **previous** W valid observations (x_t excluded);
`z_t = (x_t − μ_t)/σ_t`; percentile = share of the previous W values ≤ x_t. Fewer than
`min_periods` prior values → no z. A std below `max(1e-9, 1e-6·|μ|)` counts as zero dispersion → no z.
Tests check (i) future values never change past statistics and (ii) analysing data up to T gives
exactly the rows up to T of the full analysis (end-to-end prefix invariance).

## 6. Flags (evidence, not verdicts)

| Flag | Trigger | Gates entries |
|---|---|---|
| stale price | a leg traded 0 lots | yes |
| unchanged close | close equals the previous stored close | yes |
| low volume | a leg < `min_volume_lots` | yes |
| near expiry | nearer leg ≤ `near_expiry_days` | yes |
| roll | a leg changed contract | yes |
| data gap | > `max_gap_days` since previous row | yes |
| unadjusted expiry gap | cross legs > 7 days apart without carry estimate | yes |
| limit-size move | a leg moved ≥ 3% (MCX base price-limit slab) | yes |
| OI missing | blank open interest | no (informational) |

Each flag records what triggered it, the row's evidence, its effect and what data would confirm it.

## 7. Classification (one module, `aurum/signals.py`)

Three separate statuses: **comparability**, **alert** (ALERT / NO ALERT / INSUFFICIENT HISTORY / NOT
COMPUTABLE) and **validation** (PASS / FAIL / INCONCLUSIVE / NOT RUN). Overall status, first match wins:
INVALID COMPARISON (invalid or unverified) → INCONCLUSIVE (no measure / history) → NORMAL (no alert) →
EXPLAINABLE / RISK FLAG (alert + risk flag) → REJECTED (protocol FAIL, including no improvement over
the baseline) → RESEARCH CANDIDATE (protocol PASS and P&L specs VERIFIED) → WATCH.

## 8. Backtest execution model

- Signal from z at close t; **fill on day t+1** at its close (or open, configurable). Both legs must
  have traded that day: otherwise the entry is skipped (logged) and an exit waits for the next
  traded day. Never a same-bar fill.
- **Delivery:** MCX's staggered tender period is the last 3 trading days. No entry unless the
  position can be closed before it; forced exit by the last trading day before it (exchange calendar).
- Whole lots; leg B lots = round(lots_A × fine_g_A / fine_g_B); residual exposure recorded. Size
  above 10% of a leg's daily volume is flagged.
- Per leg and side: brokerage = min(flat per order, % cap × turnover); exchange fee and SEBI fee on
  turnover; CTT on sell turnover; stamp duty on buy turnover; GST on brokerage + exchange + SEBI;
  slippage = ticks × tick × multiplier × lots. Every rate carries a status and source
  (`cost_sources` in `config/research.yaml`).
- Candidate: adjusted measure + all risk-flag gates. Baseline: naive measure, no gates. Both use the
  same rows, costs, fills, lots and parameter-selection procedure.

## 9. Walk-forward validation and the protocol (`aurum/validation.py`)

1. **Feasibility first.** `fold_plan` splits rows into [development | holdout]. Folds: train
   `[s−train, s)`, test `[s, s+test)`. If no fold fits, nothing runs. Windows below 60/20 are only
   allowed in **exploratory mode**, whose conclusion is always "EXPLORATORY ONLY — NOT EVIDENCE".
2. Parameters (entry, exit, lookback) are chosen on each **training** window only, by net P&L.
3. A **final holdout** is evaluated once, with parameters chosen on the last training window before it.
4. Every OOS trade is re-priced under cost scenarios (0, 1, 3 ticks; 1.5× fees + 5 ticks).
5. The protocol (thresholds fixed in config before results):
   - sufficiency: ≥ 3 folds, ≥ 10 OOS trades (else INSUFFICIENT EVIDENCE);
   - integrity: no same-bar fills, no positions in the tender period, no zero-volume fills, folds and
     holdout correctly ordered;
   - performance: net > 0, ≥ 50% of traded folds positive, top-3 trades ≤ 80% of profit, survives the
     stress scenario, ≥ 50% of the (development-only) sensitivity grid positive, holdout net > 0;
   - baseline: better in > 50% of folds and higher total net (else NO ROBUST IMPROVEMENT).

The same protocol result is shown on Signal Evidence and in the Backtesting Lab, and every Lab run is
appended to an experiment log, so unfavourable runs cannot quietly disappear.
