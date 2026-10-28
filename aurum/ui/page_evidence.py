"""Page 2 - Signal Evidence."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

from aurum.analytics import FLAGS, active_flags, flag_trigger
from aurum.config import ROOT
from aurum.ingestion import Store
from aurum.pipeline import analysis_id, evidence_card, markdown_report
from aurum.signals import ALERT_INFO, STATUS_INFO, VALIDATION_INFO, classify
from aurum.contracts import assess_comparability
from . import components as C
from .state import default_validation, fp, get_ctx, go_to, pair_df


@st.cache_data(show_spinner=False)
def build_queue(mode: str, key: str, window: int, fingerprint: str) -> pd.DataFrame:
    """Every comparable observation where |z| reached the watch threshold, classified by the backend."""
    ctx = get_ctx()
    wz = ctx.research["signal"]["watch_z"]
    rows = []
    for name, pcfg in ctx.pairs.items():
        df = pair_df(mode, key, name, window)
        if df.empty:
            continue
        hits = df[df["z"].abs() >= wz]
        if hits.empty:
            continue
        val = default_validation(mode, key, name)
        protocol = val["protocol"] if val else None
        for _, r in hits.iterrows():
            comp = assess_comparability(pcfg, ctx.book, r["a_expiry"], r["b_expiry"])
            cls = classify(r, comp, ctx.research, protocol)
            flags = active_flags(r)
            rows.append({"Date": r["date"].date(), "Pair": name, "Z-score": float(r["z"]),
                         "Measure": float(r["metric"]), "Alert": cls.alert_status,
                         "Data quality": "Clean" if not flags else "Flagged: " + ", ".join(FLAGS[f]["label"] for f in flags),
                         "Comparability": comp.status, "Validation": cls.validation_status,
                         "Conclusion": cls.status})
    q = pd.DataFrame(rows)
    return q.sort_values(["Date", "Pair"], ascending=[False, True]).reset_index(drop=True) if len(q) else q


def _status_cell(v: str) -> str:
    colours = {"WATCH": "#1F3A5F", "EXPLAINABLE / RISK FLAG": "#A86A12", "REJECTED": "#9B3B3B",
               "RESEARCH CANDIDATE": "#9C7A2B", "INCONCLUSIVE": "#6B7280", "PASS": "#2F6B45", "FAIL": "#9B3B3B",
               "ALERT": "#1F3A5F"}
    return f"color:{colours.get(v, '#16213A')};font-weight:600"


def _inspect_calculation(df: pd.DataFrame, row: pd.Series) -> None:
    """Recompute mean, std and z for this row from the exact prior window values."""
    w = int(df.attrs.get("window", 40))
    i = int(df.index[df["date"] == row["date"]][0])
    prior = df["metric"].iloc[max(0, i - w):i]
    used = prior.dropna()
    if len(used) < 2:
        st.caption("Not enough prior valid values to recompute.")
        return
    mean, std = used.mean(), used.std(ddof=1)
    z = (row["metric"] - mean) / std if std > 0 else np.nan
    st.markdown(C.kv([
        ("Window", f"previous {w} rows ({prior.index.min()}–{prior.index.max()}), of which {len(used)} valid"),
        ("Mean of window", f"{mean:.6f}"), ("Std of window (ddof 1)", f"{std:.6f}"),
        ("x_t (this date)", f"{row['metric']:.6f}"),
        ("z = (x_t − mean) / std", f"<b>{z:.4f}</b> (stored: {row['z']:.4f})" if pd.notna(row["z"]) else f"{z:.4f}"),
    ]), unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({"date": df["date"].iloc[max(0, i - w):i].dt.date.values, "value": prior.values}),
                 hide_index=True, height=200)


def render(mode: str, key: str, obs: pd.DataFrame) -> None:
    ctx = get_ctx()
    s = st.session_state
    window = int(s["window"])
    wz = ctx.research["signal"]["watch_z"]

    # ---------------------------------------------------------------- B. investigation queue
    with st.container(border=True):
        st.markdown(f"**Investigation queue** <span style='color:#6B7280;font-size:.82rem'>— comparable observations "
                    f"with |z| ≥ {wz:g} (lookback {window}). Alert and validation are reported separately.</span>",
                    unsafe_allow_html=True)
        with st.spinner("Building queue and running the validation protocol (cached after the first run)…"):
            q = build_queue(mode, key, window, fp())
        if q.empty:
            C.empty_state("No statistical alerts", f"No comparable observation reached |z| ≥ {wz:g}. Try a different "
                          "lookback on the Explorer, add more history, or open any pair and date below.")
        else:
            f1, f2, f3, f4 = st.columns([1.6, 1.6, 1, 1])
            pf = f1.multiselect("Pair", sorted(q["Pair"].unique()), key="ev_fpair", placeholder="All pairs")
            sf = f2.multiselect("Conclusion", sorted(q["Conclusion"].unique()), key="ev_fstat", placeholder="All")
            zmin = f3.number_input("Min |z|", float(wz), 10.0, float(wz), 0.25, key="ev_zmin")
            qf = f4.selectbox("Data quality", ["All", "Clean only", "Flagged only"], key="ev_qual")
            g1, g2 = st.columns([2, 1])
            dmin, dmax = q["Date"].min(), q["Date"].max()
            dr = g1.slider("Date range", dmin, dmax, (dmin, dmax), format="DD MMM YYYY", key="ev_dates") \
                if dmin < dmax else (dmin, dmax)
            sort = g2.selectbox("Sort", ["Newest first", "Largest |z| first"], key="ev_sort")
            mask = (q["Z-score"].abs() >= zmin) & (q["Date"] >= dr[0]) & (q["Date"] <= dr[1])
            if pf:
                mask &= q["Pair"].isin(pf)
            if sf:
                mask &= q["Conclusion"].isin(sf)
            if qf != "All":
                mask &= (q["Data quality"] == "Clean") if qf == "Clean only" else (q["Data quality"] != "Clean")
            shown = q[mask]
            if sort.startswith("Largest"):
                shown = shown.assign(_a=shown["Z-score"].abs()).sort_values("_a", ascending=False).drop(columns="_a")
            shown = shown.reset_index(drop=True)
            ev = st.dataframe(shown.style.map(_status_cell, subset=["Alert", "Validation", "Conclusion"])
                              .format({"Z-score": "{:+.2f}", "Measure": "{:.3f}"}),
                              column_config={"Date": st.column_config.DateColumn(format="DD MMM YYYY")},
                              hide_index=True, width="stretch", height=min(38 + 35 * max(len(shown), 1), 300),
                              on_select="rerun", selection_mode="single-row", key="ev_queue")
            sel = ev.selection.rows if ev is not None else []
            if sel and sel != s.get("ev_last_sel") and sel[0] < len(shown):
                r = shown.iloc[sel[0]]
                s["pair"], s["as_of"] = r["Pair"], r["Date"]
            s["ev_last_sel"] = sel
            st.caption(f"{len(shown)} of {len(q)} alerts shown · select a row to open its evidence card.")

    # ---------------------------------------------------------------- selected investigation
    pairs = list(ctx.pairs)
    c1, c2 = st.columns([2, 1])
    if s.get("pair") not in pairs:
        s["pair"] = pairs[0]
    pair = c1.selectbox("Contract pair", pairs, index=pairs.index(s["pair"]), key=f"ev_pair_{s['pair']}")
    s["pair"] = pair
    df = pair_df(mode, key, pair, window)
    if df.empty:
        C.empty_state("No observations for this pair", "Both legs never trade on the same date under its rule.")
        return
    dates = list(df["date"].dt.date[::-1])
    as_of = s.get("as_of") if s.get("as_of") in dates else dates[0]
    as_of = c2.selectbox("Observation date", dates, index=dates.index(as_of), key=f"ev_date_{pair}_{as_of}",
                         format_func=lambda d: d.strftime("%d %b %Y"))
    s["as_of"] = as_of

    with st.spinner("Loading the validation protocol result…"):
        val = default_validation(mode, key, pair)
    card = evidence_card(ctx, mode, pair, obs, df, val, pd.Timestamp(as_of))
    row, comp, cls = card["row"], card["comparability"], card["classification"]
    hist = df[df["date"] <= pd.Timestamp(as_of)]
    aid = analysis_id({"mode": mode, "data": key, "pair": pair, "date": as_of, "window": window,
                       "research": ctx.research, "settings": ctx.settings, "engine": fp()})
    proto = val["protocol"] if val else None

    # ---------------------------------------------------------------- A. header metadata + D. status
    st.markdown(
        "<div class='aur-meta'>"
        f"<span>Pair <b>{C.esc(row['a_contract_id'])} vs {C.esc(row['b_contract_id'])}</b></span>"
        f"<span>Observation <b>{row['date']:%d %b %Y}</b> (end of day)</span>"
        f"<span>Analysis ID <b>{aid}</b></span></div>", unsafe_allow_html=True)
    with st.container(border=True):
        h1, h2 = st.columns([3, 2])
        h1.markdown(f"### Signal Evidence Card · {C.esc(pair)}")
        with h2:
            C.status_line(cls["status"])
        st.markdown(f"Alert {C.badge(cls['alert_status'])} &nbsp; Validation {C.badge(cls['validation_status'])} "
                    f"&nbsp; Comparability {C.badge(cls['comparability_status'])}", unsafe_allow_html=True)

        # ------------------------------------------------------------ C. evidence card
        r1 = st.columns(3)
        dev = row["metric"] - row["mean"] if pd.notna(row["metric"]) and pd.notna(row["mean"]) else np.nan
        with r1[0]:
            C.card("1 · Observation", C.kv([
                ("Measure", C.esc(card["metric_label"])),
                ("Value", C.fmt_num(row["metric"], 4) if pd.notna(row["metric"])
                 else f"N/A (exploratory {C.fmt_num(row.get('metric_raw'), 4)})"),
                ("Change vs rolling mean", C.fmt_num(dev, 4)),
                ("Z-score", f"<b>{C.fmt_num(row['z'], 2)}</b>"),
                ("Percentile", C.fmt_num(row["percentile"], 0)),
                ("Rolling window", f"previous {int(df.attrs.get('window', window))} rows; "
                                   f"{int(row['n_hist'])} valid values used"),
                ("Signal timestamp", f"{row['date']:%d %b %Y}, EOD close"),
            ]), gold=True)
        with r1[1]:
            yes = lambda b: "<span style='color:#2F6B45;font-weight:600'>Yes</span>" if b else "<span style='color:#9B3B3B;font-weight:600'>No</span>"
            C.card("2 · Comparability", C.kv([
                ("Status", C.badge(comp["status"])),
                ("Usable in validated analysis", yes(comp["usable"])),
                ("Conversion fields verified", yes(not any(f.split(".")[1] in ("quote_unit_grams", "purity_per_mille")
                                                           for f in comp["unverified"]))),
                ("Lot size / tick verified", yes(comp["pnl_verified"])),
                ("Expiries", f"{row['a_expiry']:%d %b %Y} / {row['b_expiry']:%d %b %Y} ({int(row['expiry_gap_days'])} d)"),
                ("Carry-adjusted", yes(bool(row.get("carry_adjusted", False))) if card["kind"] == "cross" else "n/a"),
                ("Rules applied", C.esc("; ".join(comp["adjustments"]) or "none needed")),
            ]))
        with r1[2]:
            std_med = hist["std"].median()
            disp = row["std"] / std_med if pd.notna(row["std"]) and std_med else np.nan
            facts = "".join(f"<li>{C.esc(f['label'])}: {C.esc(f['evidence'])}</li>" for f in card["flags"]) \
                or "<li>No data or risk flags on this date</li>"
            hyp = "".join(f"<li>{C.esc(h)}</li>" for h in comp["confounders"][:4]) or "<li>none documented</li>"
            C.card("3 · Explanations and risk flags", (
                "<div style='font-size:.78rem;color:#6B7280'>OBSERVED FACTS</div>"
                f"<ul style='margin:2px 0 6px 18px;font-size:.86rem'>{facts}"
                f"<li>Volume A/B {C.fmt_num(row['a_volume_lots'], 0)} / {C.fmt_num(row['b_volume_lots'], 0)} lots; "
                f"OI {C.fmt_num(row['a_oi_lots'], 0)} / {C.fmt_num(row['b_oi_lots'], 0)}</li>"
                f"<li>Rolling dispersion {C.fmt_num(disp, 2)}× its median to date</li></ul>"
                "<div style='font-size:.78rem;color:#6B7280'>HYPOTHESES (not established by the data)</div>"
                f"<ul style='margin:2px 0 0 18px;font-size:.86rem'>{hyp}</ul>"))

        r2 = st.columns([2, 1])
        with r2[0]:
            if val:
                feas, folds = val["feasibility"], val["candidate"]["folds"]
                periods = (f"{len(folds)} folds · train {ctx.research['walk_forward']['train_days']} / test "
                           f"{ctx.research['walk_forward']['test_days']} rows"
                           + (f" · tests {folds['test_start'].iloc[0]:%b %Y} → {folds['test_end'].iloc[-1]:%b %Y}"
                              if len(folds) else ""))
                C.card("4 · Strategy validation", C.kv([
                    ("Test run", "Yes — chronological walk-forward" if val["ran"] else
                     "No — " + C.esc("; ".join(feas["messages"]))),
                    ("Candidate", "z-score on the adjusted measure + risk-flag gates"),
                    ("Baseline", "z-score on the naive measure, no gates"),
                    ("Periods", C.esc(periods)),
                    ("Holdout", C.esc(f"{val['candidate']['holdout']['start']:%d %b %Y} → "
                                      f"{val['candidate']['holdout']['end']:%d %b %Y}") if val["candidate"]["holdout"]
                     else "none (insufficient data)"),
                    ("Costs", "Included — estimates, see Research & Methodology"),
                    ("Protocol conclusion", C.badge(proto["conclusion"])),
                    ("OOS evidence", f"candidate {val['candidate']['summary']['trades']} trades, net Rs "
                                     f"{val['candidate']['summary']['net_pnl']:,.0f}; baseline "
                                     f"{val['baseline']['summary']['trades']} trades, net Rs "
                                     f"{val['baseline']['summary']['net_pnl']:,.0f}"),
                ]))
            else:
                C.card("4 · Strategy validation", "No validation could be run for this pair.")
        with r2[1]:
            against = [f["label"] for f in card["flags"] if f["gate"]]
            if proto and proto["verdict"] != "PASS":
                against.insert(0, f"protocol: {proto['conclusion']}")
            against += cls["untested"][:2]
            nxt = []
            if val and not val["feasibility"]["sufficient"]:
                nxt.append(f"collect about {val['feasibility']['rows_needed']} trading days of history")
            if comp["missing"]:
                nxt.append("obtain: " + "; ".join(comp["missing"]))
            if card["flags"]:
                nxt.append("re-check on a date without risk flags")
            nxt.append("inspect intraday prices of both legs (outside AURUM)")
            C.card("5 · Conclusion", (
                f"{C.badge(cls['status'])}<div style='font-size:.86rem;margin-top:6px'><b>Supporting:</b> "
                + C.esc("; ".join(cls["reasons"])) + "</div>"
                "<div style='font-size:.86rem;margin-top:4px'><b>Against treating it as an opportunity:</b> "
                + C.esc("; ".join(against[:5]) or "none recorded") + "</div>"
                "<div style='font-size:.86rem;margin-top:4px'><b>Next:</b> " + C.esc("; ".join(nxt)) + "</div>"),
                gold=True)

    # ---------------------------------------------------------------- D. classification legend
    with st.expander("Classification system — what each status means"):
        st.markdown("**Overall status** (rules applied in order)")
        for label in STATUS_INFO:
            C.status_line(label)
        st.markdown("**Alert status** (statistics only)")
        for label in ALERT_INFO:
            C.status_line(label)
        st.markdown("**Validation status** (the predefined protocol)")
        for label in VALIDATION_INFO:
            C.status_line(label)
        st.caption("AURUM never outputs BUY, SELL, 'arbitrage confirmed' or 'profitable'.")

    # ---------------------------------------------------------------- E. timeline + G. export
    left, right = st.columns([3, 1])
    now = datetime.now(timezone.utc).strftime("%d %b %H:%M UTC")
    man = Store(mode).manifest()
    loaded = max((m.get("ingested_at_utc", "") for m in man), default="")
    gate_flags = [f for f in card["flags"] if f["gate"]]
    v = proto["verdict"] if proto else None
    with left, st.container(border=True):
        st.markdown("**Evidence timeline**")
        C.timeline([
            {"stage": "Data loaded", "result": "PASS", "detail": f"{len(man)} files, {len(obs):,} observations",
             "time": loaded[:16].replace("T", " ") + " UTC" if loaded else ""},
            {"stage": "Contract specifications checked", "result": "PASS" if comp["usable"] and comp["pnl_verified"]
             else ("WARN" if comp["usable"] else "FAIL"), "detail": f"comparability {comp['status']}"
             + ("" if comp["pnl_verified"] else "; lot/tick unverified"), "time": now},
            {"stage": "Price relationship calculated", "result": "PASS" if pd.notna(row["metric"]) else "FAIL",
             "detail": f"{int(hist['metric'].notna().sum())} comparable observations to this date", "time": now},
            {"stage": "Deviation identified", "result": "PASS" if cls["alert_status"] == "ALERT" else "SKIP",
             "detail": f"alert {cls['alert_status']}: z = {C.fmt_num(row['z'], 2)} vs threshold {wz:g}", "time": now},
            {"stage": "Risk and expiry checks", "result": "WARN" if gate_flags else "PASS",
             "detail": ", ".join(f["label"] for f in gate_flags) or "no risk flags", "time": now},
            {"stage": "Backtest / protocol", "result": {"PASS": "PASS", "FAIL": "FAIL"}.get(v, "WARN" if v else "SKIP"),
             "detail": (proto["conclusion"] if proto else "not run"), "time": now},
            {"stage": "Classification produced", "result": "PASS", "detail": cls["status"], "time": now},
        ])
    with right, st.container(border=True):
        st.markdown("**Export report**")
        md = f"<!-- analysis id: {aid} -->\n" + markdown_report(ctx, card, Store(mode))
        st.download_button("Download Markdown report", md, file_name=f"aurum_{aid}_{pair}_{as_of}.md",
                           type="primary", width="stretch")
        st.caption("Provenance, methodology, values, checks, classification, protocol criteria and assumptions.")
        st.button("Test in Backtesting Lab →", on_click=go_to, args=("Backtesting Lab",), kwargs={"pair": pair},
                  width="stretch")

    # ---------------------------------------------------------------- F. supporting evidence
    st.markdown("#### Supporting evidence")
    with st.expander("Inspect the calculation (recomputed from the window values)"):
        if pd.notna(row["metric"]):
            _inspect_calculation(df, row)
        else:
            st.caption("No validated value on this date.")
    with st.expander("Flags on this date — trigger, evidence, effect, what would confirm"):
        if card["flags"]:
            C.flag_details([{**f, "trigger": flag_trigger(f["key"], ctx.research, ctx.settings)} for f in card["flags"]])
        else:
            st.caption("No flags raised on this date.")
    with st.expander("Source observations (traceability)"):
        st.dataframe(pd.DataFrame(card["sources"]), hide_index=True, width="stretch")
        st.caption("File name and 1-based line number of each leg's row in the ingested bhavcopy.")
    with st.expander("Validation protocol criteria"):
        if proto:
            st.dataframe(proto["criteria"], hide_index=True, width="stretch")
        else:
            st.caption("Not available.")
    with st.expander("Contract specification sources"):
        st.dataframe(pd.DataFrame(ctx.book.sources).T[["title", "accessed"]], width="stretch")
    with st.expander("Historical statistics of the measure (to this date)"):
        st.dataframe(hist["metric"].describe().to_frame("AURUM measure").T, width="stretch")
    with st.expander("Backtest configuration and cost assumptions"):
        st.code(json.dumps({k: ctx.research[k] for k in ("strategy", "walk_forward", "validation_pass", "costs",
                                                          "cost_stress")}, indent=2), language="json")
    with st.expander("Known limitations"):
        st.markdown((ROOT / "docs" / "limitations.md").read_text(encoding="utf-8"))
