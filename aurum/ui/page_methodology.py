"""Page 4 - Research & Methodology."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from aurum.analytics import FLAGS, analyse_pair, flag_trigger
from aurum.config import ROOT
from aurum.contracts import CALC_FIELDS, COMPARABILITY_STATUSES, assess_comparability, multiplier_check
from aurum.ingestion import Store, map_columns
from aurum.pipeline import read_experiments
from aurum.validation import CONCLUSIONS
from . import components as C
from .state import get_ctx, pair_df

FORMULAS = r"""
**Normalised price** — inputs: close, quotation unit (g), purity (per mille).
`P_norm = close ÷ quote_unit_g ÷ (purity/1000) × 10 × 0.995` → Rs per 10 g of 995-equivalent gold.
Assumes gold is valued by fine content; delivery form, location and liquidity are *not* removed.

**Absolute / relative spread** — `S_abs = B_norm − A_norm`; `S_% = 100 · (B_norm / A_norm − 1)`.

**Carry-adjusted cross spread** — inputs: both normalised prices, expiry gap, leg-A product's same-day implied carry `c`.
`S_adj = 100 · ((B_norm / A_norm) · exp(−c · gap_days/365) − 1)`. Assumes A's near/next carry applies over the gap.

**Implied carry (calendar)** — `c = 100 · ln(F_far / F_near) / ((T_far − T_near)/365)`, % p.a. Same product on both legs.

**Rolling mean, standard deviation, z-score** — window `W` of *prior* observations:
`μ_t = mean(x_{t−W} … x_{t−1})`, `σ_t = std(x_{t−W} … x_{t−1})` (ddof 1), `z_t = (x_t − μ_t)/σ_t`.
Fewer than `min_periods` prior values → no z.

**Percentile** — share of the previous `W` values ≤ `x_t`.

**Position sizing** — leg A: `n_A` whole lots; leg B: `n_B = round(n_A · fine_g_A / fine_g_B)` where
`fine_g = lot_grams × purity/1000`. Residual fine-gold exposure is recorded per trade.

**Profit** — per leg `side × lots × (lot_g / quote_unit_g) × (P_exit − P_entry)`; spread P&L is the sum of legs.

**Costs** (per leg, per side) — turnover `= price × multiplier × lots`; brokerage per order; exchange fee and SEBI fee on
turnover; CTT on sell turnover; stamp duty on buy turnover; GST on brokerage + exchange + SEBI; slippage
`= ticks × tick × multiplier × lots`. All rates are configurable estimates.

**Backtest metrics** — net = gross − costs; hit rate = share of trades with net > 0; max drawdown on cumulative realised
net P&L by exit date; top-3 share = sum of the three largest positive trades ÷ total net; turnover = Σ notional traded.
"""


def _run_pytest() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        xml = Path(tmp) / "junit.xml"
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "--ignore=tests/test_app.py",
                               f"--junitxml={xml}"], cwd=ROOT, capture_output=True, text=True, timeout=600)
        rows = []
        if xml.exists():
            for tc in ET.parse(xml).getroot().iter("testcase"):
                status = "failed" if tc.find("failure") is not None or tc.find("error") is not None else (
                    "skipped" if tc.find("skipped") is not None else "passed")
                rows.append({"file": tc.get("classname", "").split(".")[-1], "test": tc.get("name"),
                             "status": status, "seconds": float(tc.get("time", 0))})
    return {"ran_at": datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC"), "returncode": proc.returncode,
            "tests": pd.DataFrame(rows), "tail": proc.stdout[-1500:]}


def render(mode: str, key: str, obs: pd.DataFrame) -> None:
    ctx = get_ctx()
    store = Store(mode)
    tabs = st.tabs(["Data provenance", "Specification registry", "Pair registry", "Formula reference",
                    "Methodology & limitations", "Validation results", "Research log", "Rules & protocol"])

    # ---------------------------------------------------------------- rules: flags, protocol, cost sources
    with tabs[7]:
        st.markdown("**Risk and data-quality flags** — the exact rule behind every flag")
        st.dataframe(pd.DataFrame([{"flag": f["label"], "type": "risk flag (gates entries)" if f["gate"] else "informational",
                                    "triggered when": flag_trigger(k, ctx.research, ctx.settings), "effect": f["effect"],
                                    "would help confirm": f["confirm"]} for k, f in FLAGS.items()]),
                     hide_index=True, width="stretch")
        st.markdown("**Acceptance protocol** — thresholds fixed before results (config/research.yaml)")
        st.dataframe(pd.DataFrame([{"rule": k, "value": v} for k, v in ctx.research["validation_pass"].items()]
                                  + [{"rule": "cost stress scenario that must stay net-positive",
                                      "value": ctx.research["cost_stress"]["must_survive"]}]),
                     hide_index=True, width="stretch")
        for k_, v_ in CONCLUSIONS.items():
            C.status_line(k_, v_)
        st.markdown("**Transaction-cost rates and their sources**")
        srcs = ctx.research.get("cost_sources", {})
        st.dataframe(pd.DataFrame([{"item": k, "value": v, "status": srcs.get(k, {}).get("status", "ASSUMED"),
                                    "source": srcs.get(k, {}).get("source", "—")}
                                   for k, v in ctx.research["costs"].items()]).style.map(
            lambda v: {"VERIFIED": "color:#2F6B45;font-weight:600", "SECONDARY": "color:#A86A12;font-weight:600",
                       "ASSUMED": "color:#9B3B3B;font-weight:600"}.get(v, ""), subset=["status"]),
            hide_index=True, width="stretch")
        st.caption("No rate is VERIFIED from an exchange circular or statute yet. GST applies to brokerage + exchange "
                   "charges + SEBI fee only; CTT to sell turnover; stamp duty to buy turnover.")

    # ---------------------------------------------------------------- B. data provenance
    with tabs[0]:
        man = pd.DataFrame(store.manifest())
        c = st.columns(4)
        src = "MCX bhavcopy CSV (downloaded by the user)" if mode == "REAL" else "AURUM synthetic generator"
        c[0].metric("Source", "MCX bhavcopy" if mode == "REAL" else "Synthetic")
        c[1].metric("Observation period", "N/A" if obs.empty else
                    f"{pd.to_datetime(obs['date']).min():%d %b} – {pd.to_datetime(obs['date']).max():%d %b %Y}",
                    help="First and last trading date in the dataset.")
        c[2].metric("Frequency", "Daily, end of day")
        c[3].metric("Files ingested", len(man))
        c = st.columns(4)
        c[0].metric("Raw rows read", f"{int(man['rows_read'].sum()):,}" if len(man) else "N/A")
        c[1].metric("Accepted (gold futures)", f"{len(obs):,}")
        c[2].metric("Rejected", f"{int(man['rejected'].sum()):,}" if len(man) else "N/A")
        c[3].metric("Out of scope", f"{int(man['out_of_scope'].sum()):,}" if len(man) else "N/A",
                    help="Other commodities and options: counted, not analysed.")
        st.caption(f"Source: {src}. Data mode: {mode}.")
        if mode == "REAL":
            C.warning("Data-use restriction", "MCX market data is the exchange's property; redistribution requires an "
                      "agreement. AURUM keeps raw files local (git-ignored) and never republishes them.",
                      "sharing reports or deployments", level="info")
        else:
            C.warning("Synthetic data", "Generated for demonstration. Not MCX observations.", "every result", level="red")
        if len(man):
            if "duplicates_existing" not in man.columns:
                man["duplicates_existing"] = 0
            man["duplicates_existing"] = man["duplicates_existing"].fillna(0).astype(int)
            cols = [c for c in ["source_file", "ingested_at_utc", "origin", "rows_read", "accepted", "rejected",
                                "out_of_scope", "duplicates_existing", "sha256"] if c in man.columns]
            show = man[cols].copy()
            show["sha256"] = show["sha256"].str[:12]
            st.dataframe(show, hide_index=True, width="stretch", height=260)
            ok = (man["accepted"] + man["rejected"] + man["out_of_scope"] + man["duplicates_existing"] == man["rows_read"])
            st.caption(f"Row accounting (read = accepted + rejected + out of scope + duplicates): "
                       f"{'holds for every file' if ok.all() else f'FAILS for {int((~ok).sum())} file(s)'}.")
            st.markdown("**Cross-file integrity: exchange 'Previous Close' vs our stored close of the previous day**")
            o = obs.sort_values(["contract_id", "date"]).copy()
            o["our_prev"] = o.groupby("contract_id")["close"].shift(1)
            o["prev_vol"] = o.groupby("contract_id")["volume_lots"].shift(1)
            cmp_ = o.dropna(subset=["our_prev", "prev_close"])
            if len(cmp_):
                traded = cmp_[cmp_["prev_vol"] > 0]
                stale = cmp_[~(cmp_["prev_vol"] > 0)]
                c = st.columns(3)
                c[0].metric("Match after a traded day", f"{(traded['prev_close'] == traded['our_prev']).mean():.1%}",
                            help=f"{len(traded):,} comparisons")
                c[1].metric("Match after a zero-volume day",
                            "N/A" if stale.empty else f"{(stale['prev_close'] == stale['our_prev']).mean():.1%}",
                            help=f"{len(stale):,} comparisons")
                worst = traded[traded["prev_close"] != traded["our_prev"]].groupby("date").size().sort_values(ascending=False)
                c[2].metric("Day with most mismatches (traded)", "none" if worst.empty else f"{worst.index[0]:%d %b %Y}",
                            help="Mismatches cluster after limit-size moves, when the exchange reference price "
                                 "differs from the close.")
                st.caption("A zero-volume close repeats an old price; the exchange's own reference price for that day "
                           "usually differs. AURUM therefore flags such closes as stale and never fills at them.")
            reasons = {}
            for r in man.get("rejection_reasons", []):
                for k, v in (r or {}).items():
                    reasons[k] = reasons.get(k, 0) + v
            st.markdown("**Rejection reasons**")
            st.dataframe(pd.DataFrame([{"reason": k, "rows": v} for k, v in reasons.items()]) if reasons
                         else pd.DataFrame([{"reason": "none", "rows": 0}]), hide_index=True)
        with st.expander("Upload bhavcopy CSV files (REAL dataset)"):
            st.caption("Files are schema-checked first. Accepted files are stored locally under data/raw.")
            up = st.file_uploader("CSV files", type="csv", accept_multiple_files=True, key="meth_upload")
            if up and st.button("Validate and ingest", key="meth_ingest"):
                raw = ROOT / "data" / "raw"
                raw.mkdir(parents=True, exist_ok=True)
                real = Store("REAL")
                for f in up:
                    try:
                        header = f.getvalue().decode("utf-8", errors="replace").splitlines()[0]
                        map_columns([h.strip().strip('"') for h in header.split(",")])
                    except ValueError as exc:
                        C.warning(f"{f.name}: schema not recognised", str(exc), "this file only", level="red")
                        continue
                    p = raw / Path(f.name).name
                    if not p.exists():
                        p.write_bytes(f.getvalue())
                    res = real.ingest(p, ctx.book.products, origin="upload").summary()
                    st.success(f"{f.name}: {res['accepted']} accepted, {res['rejected']} rejected "
                               f"({res['rejection_reasons'] or 'no reasons'}), {res['out_of_scope']} out of scope, "
                               f"{res['duplicates_existing']} already stored"
                               + (" — file already ingested" if res["skipped_already_ingested"] else ""))
                st.cache_data.clear()

    # ---------------------------------------------------------------- C. specification registry
    with tabs[1]:
        t = ctx.book.table()
        idx = ["product", "applies_from_expiry"]
        val = t.pivot_table(index=idx, columns="field", values="value", aggfunc="first")
        stat = t.pivot_table(index=idx, columns="field", values="status", aggfunc="first")
        srcs = t[t["field"] == "purity_per_mille"].set_index(idx)["source"].map(
            lambda s: ctx.book.sources.get(s or "", {}).get("title", "no source"))
        reg = val.copy()
        reg["verification (calc fields)"] = stat[CALC_FIELDS].apply(
            lambda r: "VERIFIED" if (r == "VERIFIED").all() else ("ASSUMED" if (r == "ASSUMED").any() else "SECONDARY"),
            axis=1)
        reg["source document"] = srcs
        st.dataframe(reg.reset_index().style.map(
            lambda v: {"VERIFIED": "color:#2F6B45;font-weight:600", "SECONDARY": "color:#A86A12;font-weight:600",
                       "ASSUMED": "color:#9B3B3B;font-weight:600"}.get(v, ""),
            subset=["verification (calc fields)"]), hide_index=True, width="stretch")
        st.caption("A version applies to contracts expiring on/after `applies_from_expiry` (MCX publishes specs per "
                   "contract series). VERIFIED = read in the official MCX PDF; SECONDARY = broker/older source; "
                   "ASSUMED = no source.")
        with st.expander("Per-field verification status"):
            st.dataframe(stat.reset_index(), hide_index=True, width="stretch")
        st.markdown("**Cross-check against exchange data**")
        chk = multiplier_check(obs, ctx.book)
        if chk.empty:
            st.caption("No traded-value column in this dataset — multipliers cannot be cross-checked.")
        else:
            st.dataframe(chk.round(4), hide_index=True, width="stretch")
            st.caption("ratio = (traded value ÷ lots) ÷ (close × spec multiplier) — ≈1 means the spec matches the exchange.")

    # ---------------------------------------------------------------- D. pair registry
    with tabs[2]:
        for k_, v_ in COMPARABILITY_STATUSES.items():
            C.status_line(k_, v_)
        for name, p in ctx.pairs.items():
            df = pair_df(mode, key, name)
            with st.container(border=True):
                h1, h2 = st.columns([3, 1.4])
                h1.markdown(f"**{name}** · {p['kind']}")
                if df.empty:
                    h2.markdown(C.badge("INVALID"), unsafe_allow_html=True)
                    st.markdown("- No overlapping observations in this dataset"
                                + (f" (blocked: {df.attrs.get('blocked')})" if df.attrs.get("blocked") else ""))
                    continue
                counts = df["comparability"].value_counts()
                latest = df["comparability"].iloc[-1]
                h2.markdown(f"latest {C.badge(latest)}", unsafe_allow_html=True)
                ok = df[df["comparability"].isin(["VALID", "ADJUSTED"])]
                comp = assess_comparability(p, ctx.book, df["a_expiry"].iloc[-1], df["b_expiry"].iloc[-1])
                use = ("calendar carry: is the cost of carry between two expiries unusual?" if p["kind"] == "calendar"
                       else "cross-product premium: is one contract unusually rich or cheap on a fine-gold basis?")
                st.markdown(
                    f"- **Rows by status:** " + ", ".join(f"{k}: {v}" for k, v in counts.items())
                    + (f" — validated analysis from {ok['date'].min():%d %b %Y}" if len(ok) else " — no usable rows") + "\n"
                    f"- **Intended use:** {use}\n- **Rationale:** {p['rationale'].strip()}\n"
                    f"- **Latest status reason:** {'; '.join(comp.reasons)}\n"
                    f"- **Normalisation:** {'; '.join(comp.adjustments) or 'identical units and purity (no conversion)'}")
                if comp.confounders:
                    st.markdown("- **Confounders (not removed):** " + "; ".join(comp.confounders))
                if comp.unverified:
                    st.markdown("- **Unverified fields (latest contracts):** " + ", ".join(comp.unverified))
                if comp.missing:
                    st.markdown("- **Needed to verify:** " + "; ".join(comp.missing))
                rule = ("INVALID if the specification changes between the two contract series" if p["kind"] == "calendar"
                        else "no pair on dates without a B contract expiring in A's month; UNVERIFIED unless quotation "
                             "unit and purity are VERIFIED for both contract series")
                st.markdown(f"- **Cannot be compared when:** {rule}")

    # ---------------------------------------------------------------- E. formulas
    with tabs[3]:
        st.markdown(FORMULAS)

    # ---------------------------------------------------------------- F. methodology & limitations
    with tabs[4]:
        a, b = st.tabs(["Methodology", "Limitations"])
        a.markdown((ROOT / "docs" / "methodology.md").read_text(encoding="utf-8"))
        b.markdown((ROOT / "docs" / "limitations.md").read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- G. validation results
    with tabs[5]:
        st.markdown("**Live checks on the current dataset**")
        checks = []
        if not obs.empty:
            chk = multiplier_check(obs, ctx.book)
            checks.append(("Contract validation: lot multipliers vs traded value",
                           "PASS" if len(chk) and chk["consistent"].all() else ("n/a" if chk.empty else "FAIL"),
                           f"{len(chk)} products checked"))
            any_pair = next((n for n in ctx.pairs if not pair_df(mode, key, n).empty), None)
            if any_pair:
                full = pair_df(mode, key, any_pair)
                T = full["date"].iloc[len(full) // 2]
                cut = analyse_pair(obs[pd.to_datetime(obs["date"]) <= T], ctx.pairs[any_pair], ctx.book, ctx.settings,
                                   ctx.research)
                cols = ["metric", "mean", "std", "z", "percentile", "comparability"]
                same = full[full["date"] <= T][cols].reset_index(drop=True).equals(cut[cols].reset_index(drop=True))
                checks.append(("No look-ahead: analysis of data up to T equals the full analysis up to T",
                               "PASS" if same else "FAIL", f"{any_pair}, T = {T:%d %b %Y}"))
            man = store.manifest()
            checks.append(("Data quality: every input row accounted for",
                           "PASS" if all(m["accepted"] + m["rejected"] + m["out_of_scope"] + m.get("duplicates_existing", 0)
                                         == m["rows_read"] for m in man if not m.get("skipped_already_ingested"))
                           else "FAIL", f"{len(man)} files"))
        st.dataframe(pd.DataFrame(checks, columns=["check", "result", "detail"]).style.map(
            lambda v: {"PASS": "color:#2F6B45;font-weight:700", "FAIL": "color:#9B3B3B;font-weight:700"}.get(v, ""),
            subset=["result"]), hide_index=True, width="stretch")
        st.markdown("**Automated test suite**")
        if st.button("Run test suite now", key="meth_pytest"):
            with st.spinner("Running pytest…"):
                st.session_state["pytest_result"] = _run_pytest()
        res = st.session_state.get("pytest_result")
        if res is None:
            C.empty_state("Test suite not run in this session", "Press 'Run test suite now'. Results are shown exactly "
                          "as reported by pytest (UI tests excluded here because they launch the app).")
        else:
            t = res["tests"]
            c = st.columns(4)
            c[0].metric("Passed", int((t["status"] == "passed").sum()) if len(t) else 0)
            c[1].metric("Failed", int((t["status"] == "failed").sum()) if len(t) else 0)
            c[2].metric("Skipped", int((t["status"] == "skipped").sum()) if len(t) else 0)
            c[3].metric("Run at", res["ran_at"])
            if len(t):
                summ = t.groupby("file")["status"].value_counts().unstack(fill_value=0).reset_index()
                st.dataframe(summ, hide_index=True, width="stretch")
                with st.expander("Every test"):
                    st.dataframe(t, hide_index=True, width="stretch")
            else:
                st.code(res["tail"])
        st.markdown("**Reproduce**")
        st.code(".\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt\n"
                ".\\.venv\\Scripts\\python.exe -m pytest -q\n"
                ".\\.venv\\Scripts\\python.exe -m aurum.cli ingest \"data/raw/*.csv\"\n"
                ".\\.venv\\Scripts\\python.exe -m aurum.cli analyse --mode REAL --report reports/real.md",
                language="powershell")

    # ---------------------------------------------------------------- H. research log
    with tabs[6]:
        st.markdown("**Experiment log** — every Backtesting Lab run in this dataset, appended automatically "
                    "(including unfavourable ones)")
        ex = read_experiments(store)
        if ex.empty:
            C.empty_state("No experiments logged yet", "Run a backtest in the Backtesting Lab.")
        else:
            st.dataframe(ex.iloc[::-1], hide_index=True, width="stretch")
        st.markdown("**Research notes**")
        st.markdown((ROOT / "docs" / "research_log.md").read_text(encoding="utf-8"))
