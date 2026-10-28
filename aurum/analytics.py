"""Pair construction, relative-value metrics, leakage-free rolling statistics, quality flags.

Timing convention (important)
- Row t holds end-of-day values for date t.
- Rolling mean / std / percentile for row t use ONLY rows t-W ... t-1 (shift(1)), so the
  statistic at t is fully known at the close of t and never contains x_t or anything later.
- z_t = (x_t - mean_{t-W..t-1}) / std_{t-W..t-1}
- A signal formed at close t can only be traded at t+1 (enforced in backtest.py).

Comparability gate
- Every row carries `comparability` (VALID / ADJUSTED / UNVERIFIED / INVALID) for the two specific
  contracts on that row. `metric_raw` is computed for every row (exploratory display only);
  `metric` and `baseline_metric` are blanked (NaN) on rows that are not VALID/ADJUSTED, so the
  validated statistics never contain values produced with unverified conversions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import SpecBook, assess_comparability, normalised_price

LEG_COLS = ["contract_id", "symbol", "expiry", "open", "close", "prev_close", "prev_obs_close",
            "volume_lots", "oi_lots", "dte", "source_file", "source_row"]

# Numerical guards (not research assumptions): a rolling std below this floor is treated as zero
# dispersion, because dividing by it would turn rounding noise into an enormous z-score.
MIN_STD_ABS = 1e-9
MIN_STD_REL = 1e-6


def add_days_to_expiry(obs: pd.DataFrame) -> pd.DataFrame:
    out = obs.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["expiry"] = pd.to_datetime(out["expiry"])
    out["dte"] = (out["expiry"] - out["date"]).dt.days
    out = out.sort_values(["contract_id", "date"])
    # previous stored close of the SAME contract (known at t; used for the unchanged-close flag)
    out["prev_obs_close"] = out.groupby("contract_id")["close"].shift(1)
    for c in ("open", "source_file", "source_row", "prev_close", "oi_lots"):
        if c not in out.columns:
            out[c] = np.nan
    return out.sort_values(["date", "contract_id"])


def _ranked_eligible(obs: pd.DataFrame, product: str, roll_days: int) -> pd.DataFrame:
    sub = obs[(obs["symbol"] == product) & (obs["dte"] > roll_days)].copy()
    sub["rank"] = sub.groupby("date")["expiry"].rank(method="first").astype(int)
    return sub


def _prefix(df: pd.DataFrame, p: str) -> pd.DataFrame:
    return df[["date"] + LEG_COLS].rename(columns={c: f"{p}_{c}" for c in LEG_COLS})


def build_pair_series(obs: pd.DataFrame, pair_cfg: dict, book: SpecBook, roll_days: int) -> pd.DataFrame:
    """One row per date with the two specific contracts the pair rule selects.

    Contract identity is preserved on every row (a_contract_id / b_contract_id), and
    `roll` marks rows where either leg changed contract versus the previous row.
    Rows whose contracts have no specification at all are removed and reported in
    attrs["blocked"] (they cannot be normalised).
    """
    obs = add_days_to_expiry(obs)
    if "fixed" in pair_cfg:
        # Two explicitly chosen contracts (no rolling). All their common dates are kept; expiry
        # proximity is reported through the near_expiry flag instead of being excluded.
        a_id, b_id = pair_cfg["fixed"]
        a = _prefix(obs[obs["contract_id"] == a_id], "a")
        b = _prefix(obs[obs["contract_id"] == b_id], "b")
        pair = a.merge(b, on="date", how="inner")
    elif pair_cfg["kind"] == "calendar":
        ranked = _ranked_eligible(obs, pair_cfg["product"], roll_days)
        a = _prefix(ranked[ranked["rank"] == pair_cfg["near_rank"]], "a")
        b = _prefix(ranked[ranked["rank"] == pair_cfg["far_rank"]], "b")
        pair = a.merge(b, on="date", how="inner")
    else:
        a_r = _ranked_eligible(obs, pair_cfg["leg_a"], roll_days)
        a = _prefix(a_r[a_r["rank"] == 1], "a")
        b_all = obs[(obs["symbol"] == pair_cfg["leg_b"]) & (obs["dte"] > roll_days)]
        b = _prefix(b_all, "b")
        a["ym"] = a["a_expiry"].dt.to_period("M")
        b["ym"] = b["b_expiry"].dt.to_period("M")
        pair = a.merge(b, on=["date", "ym"], how="inner")
        # If several B contracts share the month (should not happen), keep the nearest expiry.
        pair = pair.sort_values(["date", "b_expiry"]).drop_duplicates("date", keep="first")
        pair = pair.drop(columns="ym")

    pair = pair.sort_values("date").reset_index(drop=True)
    pair.attrs["blocked"] = {}
    if pair.empty:
        return pair

    # normalised prices, using the spec of each leg's contract series (keyed by its expiry)
    keep = np.ones(len(pair), dtype=bool)
    for leg in ("a", "b"):
        norm, naive, mult, fine = [], [], [], []
        for i, (sym, exp, px) in enumerate(zip(pair[f"{leg}_symbol"], pair[f"{leg}_expiry"], pair[f"{leg}_close"])):
            try:
                spec = book.spec(sym, exp)
            except KeyError as exc:  # no specification covers this contract: block, do not guess
                keep[i] = False
                pair.attrs["blocked"][str(exc)] = pair.attrs["blocked"].get(str(exc), 0) + 1
                norm.append(np.nan); naive.append(np.nan); mult.append(np.nan); fine.append(np.nan)
                continue
            norm.append(float(normalised_price(px, spec, True)))
            naive.append(float(normalised_price(px, spec, False)))
            mult.append(spec.multiplier)
            fine.append(spec.fine_grams_per_lot)
        pair[f"{leg}_norm"] = norm
        pair[f"{leg}_norm_naive"] = naive
        pair[f"{leg}_multiplier"] = mult
        pair[f"{leg}_fine_grams_per_lot"] = fine
    blocked = pair.attrs["blocked"]
    pair = pair[keep].reset_index(drop=True)
    pair.attrs["blocked"] = blocked
    if pair.empty:
        return pair

    # comparability of the two specific contracts on each row (specs are per contract series)
    comp_cache = {}
    statuses, pnl_ok = [], []
    for ea, eb in zip(pair["a_expiry"], pair["b_expiry"]):
        if (ea, eb) not in comp_cache:
            comp_cache[(ea, eb)] = assess_comparability(pair_cfg, book, ea, eb)
        c = comp_cache[(ea, eb)]
        statuses.append(c.status)
        pnl_ok.append(c.pnl_verified)
    pair["comparability"] = statuses
    pair["pnl_verified"] = pnl_ok

    for leg in ("a", "b"):
        ref = pair[f"{leg}_prev_close"].where(pair[f"{leg}_prev_close"] > 0, pair[f"{leg}_prev_obs_close"])
        pair[f"{leg}_ret_pct"] = 100.0 * (pair[f"{leg}_close"] / ref - 1.0)

    pair["a_prev_contract"] = pair["a_contract_id"].shift(1)
    pair["b_prev_contract"] = pair["b_contract_id"].shift(1)
    pair["roll"] = (pair["a_contract_id"] != pair["a_prev_contract"]) | (pair["b_contract_id"] != pair["b_prev_contract"])
    pair.loc[0, "roll"] = False
    pair["gap_days"] = pair["date"].diff().dt.days
    return pair


def add_metrics(pair: pd.DataFrame, kind: str, min_carry_gap_days: int = 7) -> pd.DataFrame:
    """Relative-value measures. `metric` is AURUM's measure; `baseline_metric` the naive one.

    calendar: metric   = annualised implied carry (%) = 100 * ln(F_far/F_near) / years between expiries
              baseline = raw price difference F_far - F_near (Rs). Its level mechanically shrinks
                         as the gap in time changes, which the carry rate corrects for.
              Carry is not computed when the expiries are fewer than `min_carry_gap_days` apart:
              dividing by a tiny time gap turns a one-tick price difference into a huge rate.
    cross:    metric   = % spread of B over A on the purity-adjusted fine-gold basis
              baseline = % spread on the naive units-only basis (no purity adjustment)
    """
    out = pair.copy()
    if out.empty:
        return out
    out["spread_abs"] = out["b_norm"] - out["a_norm"]  # Rs per 10 g, 995-equivalent
    out["spread_pct"] = 100.0 * (out["b_norm"] / out["a_norm"] - 1.0)
    out["spread_pct_naive"] = 100.0 * (out["b_norm_naive"] / out["a_norm_naive"] - 1.0)
    gap = (out["b_expiry"] - out["a_expiry"]).dt.days
    out["expiry_gap_days"] = gap
    if kind == "calendar":
        years = gap / 365.0
        with np.errstate(divide="ignore", invalid="ignore"):
            out["implied_carry_pct"] = np.where(
                gap >= min_carry_gap_days, 100.0 * np.log(out["b_close"] / out["a_close"]) / years, np.nan)
        out["metric"] = out["implied_carry_pct"]
        out["baseline_metric"] = out["b_close"] - out["a_close"]
        out.attrs["metric_label"] = "Annualised implied carry (% p.a.)"
        out.attrs["baseline_label"] = "Raw calendar spread (Rs)"
    else:
        out["metric"] = out["spread_pct"]
        out["baseline_metric"] = out["spread_pct_naive"]
        out.attrs["metric_label"] = "Purity-adjusted spread, B over A (%)"
        out.attrs["baseline_label"] = "Naive units-only spread (%)"
    return out


def rolling_stats(x: pd.Series, window: int, min_periods: int) -> pd.DataFrame:
    """Leakage-free rolling statistics: every value at t uses x[t-window : t] (excludes x_t).

    NaN values inside the window are skipped; `n_hist` counts the valid prior values used.
    A standard deviation below max(MIN_STD_ABS, MIN_STD_REL * |mean|) is treated as zero dispersion
    and gives no z-score (classified INCONCLUSIVE downstream).
    """
    past = x.shift(1)
    mean = past.rolling(window, min_periods=min_periods).mean()
    std = past.rolling(window, min_periods=min_periods).std(ddof=1)
    n = past.rolling(window, min_periods=1).count()
    floor = np.maximum(MIN_STD_ABS, MIN_STD_REL * mean.abs())
    usable_std = std.where(std > floor)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (x - mean) / usable_std

    values = x.to_numpy(dtype=float)
    pct = np.full(len(values), np.nan)
    for t in range(len(values)):
        hist = values[max(0, t - window):t]
        hist = hist[~np.isnan(hist)]
        if len(hist) >= min_periods and not np.isnan(values[t]):
            pct[t] = 100.0 * np.mean(hist <= values[t])
    return pd.DataFrame({"mean": mean, "std": std, "z": z, "percentile": pct, "n_hist": n},
                        index=x.index)


# ----------------------------------------------------------------------------- flag registry
# gate=True: a risk flag. It makes an alert EXPLAINABLE / RISK FLAG and blocks candidate entries.
# gate=False: informational data-quality indicator.
FLAGS: dict[str, dict] = {
    "a_stale": dict(label="Stale price (leg A)", gate=True,
                    trigger="leg A traded zero lots on this date",
                    effect="the close repeats an older price; on real MCX data the next day's 'Previous Close' "
                           "differs from such closes 99% of the time, so the spread may be fictitious",
                    confirm="intraday trades or bid/ask for leg A on this date"),
    "b_stale": dict(label="Stale price (leg B)", gate=True,
                    trigger="leg B traded zero lots on this date",
                    effect="the close repeats an older price, so the spread may be fictitious",
                    confirm="intraday trades or bid/ask for leg B on this date"),
    "a_unchanged": dict(label="Unchanged close (leg A)", gate=True,
                        trigger="leg A closed exactly at its previous stored close",
                        effect="possible stale or administratively carried price",
                        confirm="traded volume and intraday price path"),
    "b_unchanged": dict(label="Unchanged close (leg B)", gate=True,
                        trigger="leg B closed exactly at its previous stored close",
                        effect="possible stale or administratively carried price",
                        confirm="traded volume and intraday price path"),
    "low_volume": dict(label="Low traded volume", gate=True,
                       trigger="a leg traded fewer than {min_volume_lots} lots",
                       effect="the close is less reliable and a real position may not be executable at it",
                       confirm="order-book depth / bid-ask spread"),
    "near_expiry": dict(label="Near expiry", gate=True,
                        trigger="the nearer leg is {near_expiry_days} or fewer calendar days from expiry",
                        effect="delivery intentions, tender-period margins and rolling distort spreads",
                        confirm="open-interest changes and delivery intention data"),
    "roll": dict(label="Contract roll", gate=True,
                 trigger="a leg switched to a different contract on this date",
                 effect="the measure now refers to a different contract pair than its history",
                 confirm="the same measure computed on fixed contracts across the roll"),
    "data_gap": dict(label="Gap in observations", gate=True,
                     trigger="more than {max_gap_days} calendar days since the previous observation",
                     effect="rolling statistics span a period with missing data",
                     confirm="the missing bhavcopy files for the gap"),
    "expiry_gap_flag": dict(label="Unadjusted expiry gap", gate=True,
                            trigger="cross legs expire more than {flag_expiry_gap_days} days apart and no carry "
                                    "estimate was available",
                            effect="the spread contains cost of carry over the gap, not only relative value",
                            confirm="a same-day carry estimate for leg A's product"),
    "extreme_move": dict(label="Price-limit-size move", gate=True,
                         trigger="a leg moved {extreme_move_pct}% or more versus the previous close "
                                 "(MCX's base daily price-limit slab is 3%)",
                         effect="closes on limit-size days can be locked at a limit and differ from the "
                                "exchange's settlement reference (seen on 30 Jan 2026)",
                         confirm="the exchange settlement price and whether a circuit was hit"),
    "oi_missing": dict(label="Open interest missing", gate=False,
                       trigger="open interest is blank for a leg",
                       effect="liquidity context is incomplete",
                       confirm="exchange open-interest report"),
}
GATE_FLAGS = [k for k, v in FLAGS.items() if v["gate"]]
FLAG_TEXT = {k: v["label"] for k, v in FLAGS.items()}  # backward-compatible short names


def flag_trigger(key: str, research: dict, pairs_settings: dict | None = None) -> str:
    params = dict(research["signal"])
    params.update(pairs_settings or {})
    return FLAGS[key]["trigger"].format(**params)


def add_flags(df: pd.DataFrame, signal_cfg: dict, flag_expiry_gap_days: int, kind: str) -> pd.DataFrame:
    """Row-level data-quality and risk flags. Flags are evidence, not verdicts."""
    out = df.copy()
    for leg in ("a", "b"):
        vol = out[f"{leg}_volume_lots"]
        out[f"{leg}_stale"] = vol.isna() | (vol <= 0)
        out[f"{leg}_unchanged"] = (out[f"{leg}_close"] == out[f"{leg}_prev_obs_close"]) & ~out[f"{leg}_stale"]
    out["low_volume"] = out[["a_volume_lots", "b_volume_lots"]].min(axis=1) < signal_cfg["min_volume_lots"]
    out["near_expiry"] = out[["a_dte", "b_dte"]].min(axis=1) <= signal_cfg["near_expiry_days"]
    out["data_gap"] = out["gap_days"] > signal_cfg["max_gap_days"]
    if kind == "cross":
        out["expiry_gap_flag"] = out["expiry_gap_days"].abs() > flag_expiry_gap_days
    else:
        out["expiry_gap_flag"] = False
    lim = signal_cfg.get("extreme_move_pct", 3.0)
    out["extreme_move"] = (out[["a_ret_pct", "b_ret_pct"]].abs() >= lim).any(axis=1)
    out["oi_missing"] = out[["a_oi_lots", "b_oi_lots"]].isna().any(axis=1)
    return out


def flag_evidence(row: pd.Series, key: str) -> str:
    """The concrete values on this row that raised the flag."""
    d = f"{pd.Timestamp(row['date']):%d %b %Y}"
    if key in ("a_stale", "b_stale"):
        leg = key[0]
        return f"{row[f'{leg}_contract_id']}: {row[f'{leg}_volume_lots']:.0f} lots traded on {d}"
    if key in ("a_unchanged", "b_unchanged"):
        leg = key[0]
        return f"{row[f'{leg}_contract_id']}: close {row[f'{leg}_close']:,.0f} = previous close {row[f'{leg}_prev_obs_close']:,.0f}"
    if key == "low_volume":
        return f"volumes A/B: {row['a_volume_lots']:,.0f} / {row['b_volume_lots']:,.0f} lots"
    if key == "near_expiry":
        return f"days to expiry A/B: {int(row['a_dte'])} / {int(row['b_dte'])}"
    if key == "roll":
        return (f"A {row.get('a_prev_contract')} → {row['a_contract_id']}; "
                f"B {row.get('b_prev_contract')} → {row['b_contract_id']}")
    if key == "data_gap":
        return f"{row['gap_days']:.0f} calendar days since previous observation"
    if key == "expiry_gap_flag":
        return f"expiries {pd.Timestamp(row['a_expiry']):%d %b} / {pd.Timestamp(row['b_expiry']):%d %b} ({int(row['expiry_gap_days'])} days)"
    if key == "extreme_move":
        return f"daily move A/B: {row['a_ret_pct']:+.2f}% / {row['b_ret_pct']:+.2f}%"
    if key == "oi_missing":
        return f"OI A/B: {row['a_oi_lots']} / {row['b_oi_lots']}"
    return ""


def active_flags(row: pd.Series, gate_only: bool = True) -> list[str]:
    keys = GATE_FLAGS if gate_only else list(FLAGS)
    return [k for k in keys if k in row.index and bool(row[k])]


def same_day_carry(obs: pd.DataFrame, product: str, book: SpecBook, roll_days: int,
                   min_carry_gap_days: int = 7) -> pd.Series:
    """Annualised implied carry (% p.a.) of `product` from its own near/next contracts, per date.
    Uses only that date's closes, so it introduces no look-ahead."""
    cal = build_pair_series(obs, {"kind": "calendar", "product": product, "near_rank": 1, "far_rank": 2},
                            book, roll_days)
    if cal.empty:
        return pd.Series(dtype=float)
    return add_metrics(cal, "calendar", min_carry_gap_days).set_index("date")["implied_carry_pct"]


def add_carry_adjustment(df: pd.DataFrame, carry: pd.Series) -> pd.DataFrame:
    """Remove the cost of carry between two different expiry dates from a cross spread:

        carry_adj_pct = 100 * ( (B_norm / A_norm) * exp(-c * gap_years) - 1 )

    c = leg A product's same-day implied carry, gap_years = (expiry_B - expiry_A) / 365.
    Where no carry estimate exists for a date, the value is NaN and the raw spread is used.
    """
    out = df.copy()
    c = out["date"].map(carry) / 100.0
    gap_years = out["expiry_gap_days"] / 365.0
    out["carry_used_pct"] = c * 100.0
    out["spread_pct_carry_adj"] = 100.0 * ((out["b_norm"] / out["a_norm"]) * np.exp(-c * gap_years) - 1.0)
    has = out["spread_pct_carry_adj"].notna()
    out["metric"] = np.where(has, out["spread_pct_carry_adj"], out["spread_pct"])
    out["carry_adjusted"] = has
    return out


def analyse_pair(obs: pd.DataFrame, pair_cfg: dict, book: SpecBook, pairs_settings: dict,
                 research: dict, window: int | None = None) -> pd.DataFrame:
    """Full analysis table for one pair: legs, metrics, rolling stats (both measures), flags."""
    roll_days = pairs_settings["roll_days_before_expiry"]
    min_gap = research["signal"].get("min_carry_gap_days", 7)
    pair = build_pair_series(obs, pair_cfg, book, roll_days)
    if pair.empty:
        return pair
    blocked = pair.attrs.get("blocked", {})
    df = add_metrics(pair, pair_cfg["kind"], min_gap)
    attrs = dict(df.attrs)
    if pair_cfg["kind"] == "cross" and (df["expiry_gap_days"].abs() > 0).any():
        df = add_carry_adjustment(df, same_day_carry(obs, pair_cfg["leg_a"], book, roll_days, min_gap))
        attrs["metric_label"] = "Purity- and carry-adjusted spread, B over A (%)"
    else:
        df["carry_adjusted"] = False

    # comparability gate: keep every value for exploratory display, but only usable rows feed
    # the validated measures, their statistics, the signals and the backtest
    usable = df["comparability"].isin(["VALID", "ADJUSTED"])
    df["metric_raw"] = df["metric"]
    df["metric"] = df["metric"].where(usable)
    df["baseline_metric"] = df["baseline_metric"].where(usable)

    st = research["statistics"]
    w = window or st["window"]
    mp = min(st["min_periods"], w)
    stats = rolling_stats(df["metric"], w, mp)
    df[["mean", "std", "z", "percentile", "n_hist"]] = stats
    bstats = rolling_stats(df["baseline_metric"], w, mp)
    df["baseline_z"] = bstats["z"]
    df = add_flags(df, research["signal"], pairs_settings["flag_expiry_gap_days"], pair_cfg["kind"])
    # An expiry gap is only a risk flag when we could NOT adjust for carry on that date.
    df["expiry_gap_flag"] = df["expiry_gap_flag"] & ~df["carry_adjusted"]
    attrs["blocked"] = blocked
    attrs["window"] = w
    df.attrs.update(attrs)
    return df
