"""SYNTHETIC bhavcopy-style data for software testing and demos.

!!! NOT MCX MARKET DATA. Nothing computed from these files is evidence about real markets. !!!

The generator is a simple structural model so the pipeline has something realistic to chew on:
  spot (Rs / 10 g, 995)    : geometric random walk
  carry rate               : ~6.5% p.a. + mean-reverting noise
  futures price            : spot * exp(carry * T) * purity factor * unit factor
                             * (1 + product premium) * (1 + contract-specific AR(1) dislocation)
  liquidity                : Poisson volumes, thinner for far expiries and small products;
                             a zero-volume day repeats the previous close (stale price), which
                             is exactly the trap the quality flags are meant to catch.
It also writes a handful of deliberately bad rows (H<L, duplicates, bad date, out-of-scope
symbol) so ingestion's rejection log has something to show.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HEADER = ["Date", "Instrument Name", "Symbol", "Expiry Date", "Option Type", "Strike Price", "Open",
          "High", "Low", "Close", "Previous Close", "Volume(Lots)", "Open Interest(Lots)"]

PRODUCTS = {
    # quote_unit_g, purity, expiry rule, listing horizon (days), base volume, premium
    "GOLD": (10, 995, "5th_even", 200, 300, 0.0000),
    "GOLDM": (10, 995, "5th", 130, 900, 0.0000),
    "GOLDGUINEA": (8, 999, "last", 130, 120, 0.0020),
    "GOLDPETAL": (1, 999, "last", 130, 60, 0.0030),
}


def _prev_bday(d: pd.Timestamp) -> pd.Timestamp:
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


def _expiries(rule: str, start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    out = []
    for m in pd.period_range(start.to_period("M"), (end + pd.Timedelta(days=250)).to_period("M"), freq="M"):
        if rule == "5th_even" and m.month % 2:
            continue
        if rule in ("5th", "5th_even"):
            out.append(_prev_bday(pd.Timestamp(m.year, m.month, 5)))
        else:
            out.append(_prev_bday(m.to_timestamp(how="end").normalize()))
    return out


def generate(start: str = "2022-01-03", end: str = "2024-12-31", seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    n = len(days)
    spot = 52000 * np.exp(np.cumsum(rng.normal(0.10 / 252, 0.14 / np.sqrt(252), n)))
    r = np.empty(n)
    r[0] = 0.065
    for t in range(1, n):
        r[t] = r[t - 1] + 0.05 * (0.065 - r[t - 1]) + rng.normal(0, 0.0015)

    rows = []
    for prod, (qu, pur, rule, horizon, base_vol, prem) in PRODUCTS.items():
        premium = prem + np.cumsum(rng.normal(0, 0.00005, n))  # slowly drifting product premium
        for exp in _expiries(rule, days[0], days[-1]):
            live = (days <= exp) & ((exp - days).days <= horizon)
            if not live.any():
                continue
            disloc, prev_close, oi = 0.0, None, 0
            for t in np.flatnonzero(live):
                d = days[t]
                T = (exp - d).days / 365.0
                disloc = 0.88 * disloc + rng.normal(0, 0.0005)
                if rng.random() < 0.01:
                    disloc += rng.choice([-1, 1]) * rng.uniform(0.002, 0.004)  # rare shock
                fair = spot[t] * np.exp(r[t] * T) * (pur / 995.0) * (qu / 10.0)
                close = round(fair * (1 + premium[t]) * (1 + disloc))
                rank_penalty = np.exp(-T * 6)  # near contracts trade more
                vol = int(rng.poisson(base_vol * rank_penalty + 0.3))
                if vol == 0 and prev_close is not None:
                    close = prev_close  # stale: no trade, close carried forward
                oi = max(0, oi + vol // 2 - int(rng.poisson(vol / 3 + 0.1)))
                o = prev_close if prev_close else close
                o = round(o * (1 + rng.normal(0, 0.002)))
                hi = max(o, close) + abs(round(rng.normal(0, 0.002) * close))
                lo = min(o, close) - abs(round(rng.normal(0, 0.002) * close))
                rows.append([d, "FUTCOM", prod, exp, "", 0, o, hi, lo, close,
                             prev_close if prev_close else close, vol, oi])
                prev_close = close
    df = pd.DataFrame(rows, columns=HEADER)
    # out-of-scope rows (another commodity) - ingestion must count, not silently drop
    silver = pd.DataFrame([[d, "FUTCOM", "SILVERM", days[-1], "", 0, 70000, 70100, 69900, 70000, 70000, 10, 50]
                           for d in days[::20]], columns=HEADER)
    return pd.concat([df, silver], ignore_index=True).sort_values(["Date", "Symbol", "Expiry Date"])


def _inject_bad_rows(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    gold = df[df["Symbol"] == "GOLDM"]
    picks = gold.sample(4, random_state=int(rng.integers(1e6)))
    bad = []
    r = picks.iloc[0].copy(); r["High"], r["Low"] = r["Low"] - 10, r["High"] + 10; bad.append(("swap", r))
    r = picks.iloc[1].copy(); bad.append(("dup_identical", r))
    r = picks.iloc[2].copy(); r["Close"] = r["Close"] + 50; r["High"] = max(r["High"], r["Close"]); bad.append(("dup_conflict", r))
    r = picks.iloc[3].copy(); r["Date"] = "31/02/2023"; bad.append(("bad_date", r))
    out = df.copy()
    idx = picks.index[0]
    out.loc[idx, ["High", "Low"]] = [bad[0][1]["High"], bad[0][1]["Low"]]
    extra = pd.DataFrame([bad[1][1], bad[2][1], bad[3][1]])
    return pd.concat([out, extra], ignore_index=True)


def write_sample_files(out_dir: Path, seed: int = 7) -> list[Path]:
    """Write one SYNTHETIC CSV per calendar year into out_dir. Returns the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + 1)
    df = _inject_bad_rows(generate(seed=seed), rng)
    paths = []
    years = pd.to_datetime(df["Date"], errors="coerce").dt.year.fillna(2023).astype(int)
    for y, part in df.groupby(years):
        part = part.copy()
        for c in ("Date", "Expiry Date"):
            part[c] = part[c].map(lambda v: v.strftime("%d %b %Y") if isinstance(v, pd.Timestamp) else v)
        p = out_dir / f"SYNTHETIC_bhavcopy_gold_{y}.csv"
        part.to_csv(p, index=False)
        paths.append(p)
    return paths
