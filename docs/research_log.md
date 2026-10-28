# Research log

## 2026-10-23 - Phase 0 market and source research

Sources accessed 2026-10-23. "Secondary" = not the official current MCX document.

| Source | Supports | Type |
|---|---|---|
| [MCX Gold product page](https://www.mcxindia.com/products/bullion/gold) | Official specs/PDF links | Official - **HTTP 403 to automated fetch; not read** |
| [Zerodha Varsity: Gold part 1](https://zerodha.com/varsity/chapter/gold-part-1/) | Quote units (Rs/10 g, /8 g, /1 g), lot sizes, expiry days (2016) | Secondary, dated |
| [ICICI Direct gold guide](https://www.icicidirect.com/ilearn/commodity/articles/how-to-trade-gold-in-commodity-derivatives-market-in-india-1) | GOLD/GOLDM 995; TEN/GUINEA/PETAL 999 | Secondary |
| [Gold Petal spec Oct 2019 (broker copy)](https://www.achiieversequitiesltd.com/wp-content/uploads/2019/04/gold-petal-october-2019-contract-onwards.pdf) | 999 coins, DDR = Mumbai 995 spot × 999/995 | Secondary copy of MCX doc |
| [MCX/TRD/456/2025 (broker copy)](https://www.steelcitynettrade.com/Circulars/Modification%20in%20Launch%20Calendar%20of%20Gold%20Mini,%20Gold%20Ten,%20Gold%20Guinea.pdf) | 999/995 purity adjustment; Ahmedabad delivery centre | Secondary copy of MCX doc |
| [MCX data feed policy](https://www.mcxindia.com/docs/default-source/datafeedlink/mcx-data-feed-policy.pdf?sfvrsn=8fc1b490_10) | All rights reserved; redistribution needs agreement | Official (via search summary) |
| [MCX datafeed page](https://www.mcxindia.com/technology/datafeed) | Limited free historical data | Official (via search summary) |
| [TradingView spread charts](https://www.tradingview.com/support/solutions/43000502298-spread-charts-explained/) | Formula spread charts | Official |
| [NexusFi on TradingView spreads](https://nexusfi.com/a/platforms/tradingview-spread-charts) | Spreads need Plus+; Pine for hedge ratios | Secondary |
| [TT Autospreader](https://library.tradingtechnologies.com/trade/spread-trading/autospreader/description-autospreader/introduction-to-autospreader/) | Synthetic spread execution | Official |
| [CQG Spreader](https://www.cqg.com/products/cqg-spreader) | Spread execution, legging | Official |
| [himanshugit786/MCX-quantitative-arbitrage-](https://github.com/himanshugit786/MCX-quantitative-arbitrage-) | Prior art: units-only normalisation, no purity, no backtest | Public repo |

### Competitor summary

| Product | Users | Spread features | Backtest/execution | AURUM difference | Unverified |
|---|---|---|---|---|---|
| TradingView | Retail/pro | Formula spreads, Pine | Pine strategy tester | Spec/purity-aware comparability, rejection workflow | MCX per-contract history coverage |
| TT / CQG | Professional desks | Execution spreaders | Execution | None - not competing on execution | MCX connectivity |
| Indian brokers | Retail | Charts, spec explainers | Execution | Investigation workflow | Their spread tools - not checked hands-on |
| GitHub prior art | Hobbyist | Units-only normalisation | None | Purity correction, validation | - |

### Findings
1. Units-only normalisation of 999 contracts against 995 contracts creates a permanent ≈0.40%
   fake premium. This is AURUM's clearest, demonstrable value.
2. Calendar spreads within one product are the most economically clean comparison; implied
   carry gives them an interpretable unit (% p.a.).
3. Data access is the main project risk: MCX blocks automated access and restricts redistribution.

## 2026-10-23 - First real data (user-downloaded MCX bhavcopy)

**Data:** 84 `BhavCopyDateWise_DDMMYYYY.csv` files → 76 unique trading days (8 Jul – 22 Oct 2026);
8 re-downloads skipped (identical rows, different order); 2,265 gold-futures rows; 0 rejected.
**Specs:** official MCX PDFs for GOLD (Feb-2027 onwards), GOLDM (Aug-2026), GOLDTEN, GOLDGUINEA,
GOLDPETAL (Jul-2026) stored in `docs/specs/`. All calculation fields VERIFIED for those series.

**Bugs found by real data**
- Futures rows use option type `"-"`; the parser treated them as out of scope. Fixed + tested.
- Specs were keyed by trading date; MCX keys them by contract series. Fixed (lookup by expiry).
- Cross pairs were 100% flagged for expiry gap; added same-day carry adjustment.

**Descriptive findings (not trading results)**

| Pair | Naive | Purity-adj. | Purity+carry-adj. | p10–p90 |
|---|---|---|---|---|
| GOLD vs GOLDM (same expiry) | −0.012% | −0.012% | −0.012% | −0.10 … 0.10 |
| GOLDM vs GOLDTEN | 0.90% | 0.50% | −0.06% | −0.22 … 0.03 |
| GOLDM vs GOLDPETAL | 1.41% | 1.00% | 0.40% | 0.16 … 0.57 |
| GOLDM vs GOLDGUINEA | 1.39% | 0.98% | 0.41% | 0.18 … 0.62 |
| GOLDTEN vs GOLDPETAL (same expiry) | 0.40% | 0.40% | 0.40% | 0.23 … 0.55 |

Implied carry: GOLDM near/next median 9.2% p.a. (p10 6.0, p90 12.4); GOLD 6.7% p.a.

Interpretation (hypotheses, not causal claims):
1. GOLD and GOLDM are priced identically for the same expiry → normalisation and data consistent.
2. GOLDTEN is fully explained by purity + carry.
3. Petal and Guinea carry a persistent ~0.40% premium, measured two independent ways. Guinea is
   Ex-Ahmedabad like Ten, so location alone does not explain it; a small-denomination / retail
   premium is more consistent with the data. A persistent premium is not a dislocation.

**Validation:** all walk-forward verdicts INCONCLUSIVE — 76 days < 180-day train window.
Need ≥ ~12 months of daily files before any strategy conclusion.

## 2026-10-23 - Quantitative audit (engine v0.1.0 → current)

**Data:** 197 files → 189 trading days (29 Jan – 22 Oct 2026), 5,632 gold-futures rows, 0 rejected.

**Defects confirmed by probes and fixed (each has a regression test)**
| # | Defect | Evidence before | After |
|---|---|---|---|
| D1 | Unverified purity multipliers applied (cross pairs labelled ADJUSTED) | 106/189 real rows of GOLDM vs GOLDPETAL; GOLD vs GOLDM all rows | per-row comparability; UNVERIFIED rows excluded from validated measures |
| D2 | Two acceptance protocols disagreed | GOLD_CAL_1_2: PASS on Signal Evidence, NO ROBUST IMPROVEMENT in the Lab | one protocol module used everywhere |
| D3 | Infeasible walk-forward allowed; quality "OK" | 0 folds possible at 189 rows | feasibility check before running; exploratory mode labelled |
| D4 | Positions held into the delivery tender period | 5 trade legs | calendar-based exit deadline; integrity check = 0 |
| D5 | Near-zero std produced z = 63,278,118 | probe | std floor; z = N/A |
| D6 | Disk-persisted dashboard cache not invalidated by engine changes | `.memo` files keyed on arguments only | engine fingerprint in every cache key |
| D7 | Conflicting cross-file duplicate silently discarded | probe | rejected as `conflicts_with_existing_observation` |
| D8 | `strategy.execution` config ignored | grep | next_close / next_open implemented |
| D9 | No guard against fills on zero-volume days | 0 observed impact on tested pairs | both legs must have traded on the fill day |
| D10/11 | Unverified single-scenario costs; no holdout, stress test, per-leg P&L, capacity check | — | cost_sources with status; cost scenarios; holdout; per-leg ledger; participation |
| D12 | Limit-size moves not flagged | 30 Jan 2026: GOLDM −12%, closed at the low | `extreme_move` flag (≥ 3%) |

**Correction to the earlier entry:** the "GOLD vs GOLDM −0.012%" result used the GOLD Dec-2026
contract series, whose specification is not verified from an official PDF. Under the comparability
rule it is now UNVERIFIED (exploratory only). The other cross-pair figures in that entry used
Jul-2026+ series and remain on verified specifications.

**New data findings**
- Exchange "Previous Close" equals our stored previous close 99.5% of the time after traded days and
  differs 99.2% of the time after zero-volume days → zero-volume closes are not prices.
- Mismatches after traded days cluster on 2 Feb 2026, after the 30 Jan limit-size fall.

**First protocol runs on real data** (shortest protocol windows: 60 train / 20 test, no holdout,
7 folds; every pair): all **INSUFFICIENT EVIDENCE** — 0–6 out-of-sample trades per pair. Candidate
net results were negative on every pair; the naive baseline was positive on several (GOLD_CAL_1_2
+Rs 171,406 over 6 trades). Inspection: the baseline's signal days had real volume (my stale-price
hypothesis was wrong); its result came from 2 of 6 trades, the largest entered days before the near
contract's expiry, a situation the candidate's near-expiry gate refuses. Interpretation: a 6-trade
sample is dominated by one or two events; no evidence of an edge for either method.

## 2026-10-23 - Build decisions
- Brokerage modelled per order, not per lot (per-lot would charge 100 Petal lots Rs 2,000/side).
- Statuses capped at WATCH while specs are unverified.
- Synthetic-data results are not evidence (generator contains mean reversion by construction).
