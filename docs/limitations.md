# Known limitations

**Data**
- Real history held: 189 trading days (29 Jan – 22 Oct 2026). The default walk-forward needs about
  420 (180 train + 3 × 60 test + 60 holdout), so every protocol verdict is INSUFFICIENT EVIDENCE.
  With the shortest protocol windows (60/20, no holdout) 7 folds fit, but each pair produced only
  1–6 out-of-sample trades: still insufficient.
- End-of-day data only: no bid/ask, depth or intraday timing. MCX's close is a 30-minute weighted
  average; zero-volume closes repeat old prices and are flagged, never used as fills.
- AURUM never redistributes exchange data; raw and processed real data are git-ignored.
- SYNTHETIC mode is a structural toy model with mean reversion built in; its results are not evidence.

**Specifications**
- Official PDFs are held only for GOLD (Feb-2027 series onward), GOLDM (Aug-2026), GOLDTEN,
  GOLDGUINEA, GOLDPETAL (Jul-2026). Earlier series use secondary/assumed values, so cross-product
  comparisons before late June 2026 — and GOLD vs GOLDM until Feb-2027 contracts — are UNVERIFIED and
  excluded from validated analysis. Adding the older PDFs would unlock that history.

**Execution and costs**
- Fills at the next day's close/open plus assumed slippage; real fills, especially for 100 lots of
  GOLDPETAL or 1 kg GOLD in thin months, may be much worse. Re-pricing under cost scenarios is a
  sensitivity test, not a measurement.
- No cost rate is VERIFIED from an exchange circular or statute (see cost_sources). Stamp duty is an
  assumption; sources conflict on whether CTT applies to both sides.
- Margin, financing, delivery and making charges are not modelled; drawdown is on realised trades,
  not daily mark-to-market.

**Statistics**
- A rolling z-score assumes a locally stable mean and dispersion; regime changes (e.g. interest-rate
  moves for carry) produce long runs of "unusual" readings.
- With few folds and trades, results have wide uncertainty; no confidence intervals or multiple-
  testing corrections across pairs and configurations are computed yet (the experiment log records
  how many configurations were tried).
- The carry adjustment assumes leg A's near/next carry applies across the expiry gap.
