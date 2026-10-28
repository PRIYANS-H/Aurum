"""Page 3 - Backtesting Lab."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from aurum import __version__
from aurum.ingestion import Store
from aurum.pipeline import analysis_id, log_experiment, run_validation
from aurum.validation import CONCLUSIONS, feasibility
from . import components as C
from .state import fp, get_ctx, go_to, load_obs, load_pair, research_with, settings_with
from .theme import GOLD, INK, RED, style_fig


@st.cache_data(show_spinner=False, persist="disk")
def run_backtest(mode: str, key: str, pair: str, research_json: str, settings_json: str,
                 start: str, end: str, exploratory: bool, fingerprint: str):
    """Walk-forward on the chosen period. Rolling statistics are computed on the full history (causal),
    then rows are restricted to the period, so the first fold still has a warm-up."""
    ctx = get_ctx()
    research = json.loads(research_json)
    df = load_pair(mode, key, pair, research["statistics"]["window"], research_json, settings_json,
                   fingerprint=fingerprint)
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))].reset_index(drop=True)
    if df.empty:
        return None, 0
    return run_validation(ctx, load_obs(mode, key), df, ctx.pairs[pair], research, exploratory), len(df)


def _drawdown(trades: pd.DataFrame) -> pd.DataFrame:
    t = trades.sort_values("exit_date")
    eq = t["net_pnl"].cumsum()
    return pd.DataFrame({"date": t["exit_date"], "equity": eq, "dd": eq - eq.cummax().clip(lower=0)})


def _result_style(v):
    return {"PASS": "color:#2F6B45;font-weight:700", "FAIL": "color:#9B3B3B;font-weight:700",
            "n/a": "color:#6B7280"}.get(v, "")


def render(mode: str, key: str, obs: pd.DataFrame) -> None:
    ctx = get_ctx()
    s = st.session_state
    R = ctx.research
    pairs = list(ctx.pairs)

    # ---------------------------------------------------------------- B. configuration
    with st.form("bt_form"):
        st.markdown("**Strategy configuration** <span style='color:#6B7280;font-size:.82rem'>— only settings the "
                    "engine uses are shown; feasibility is checked before anything runs</span>", unsafe_allow_html=True)
        c = st.columns(4)
        pair = c[0].selectbox("Contract pair", pairs, index=pairs.index(s["pair"]) if s.get("pair") in pairs else 0)
        base_df = load_pair(mode, key, pair, R["statistics"]["window"], fingerprint=fp())
        if base_df.empty:
            st.form_submit_button("Run walk-forward", disabled=True)
            C.empty_state("No observations for this pair", "Choose another pair or load more data.")
            return
        dmin, dmax = base_df["date"].min().date(), base_df["date"].max().date()
        period = c[1].date_input("Analysis period", (dmin, dmax), min_value=dmin, max_value=dmax, format="DD/MM/YYYY")
        exploratory = c[2].toggle("Exploratory mode", False,
                                  help="Allows windows shorter than the protocol minimum (60 train / 20 test). "
                                       "Results are labelled EXPLORATORY and can never pass the protocol.")
        execution = c[3].selectbox("Fill rule", ["next_close", "next_open"],
                                   index=["next_close", "next_open"].index(R["strategy"]["execution"]),
                                   format_func=lambda x: {"next_close": "Next day's close",
                                                          "next_open": "Next day's open"}[x],
                                   help="Signals use the close of day t. Fills happen on day t+1; both legs must have "
                                        "traded that day.")
        c = st.columns(4)
        train = c[0].number_input("Training window (rows)", 20, 750, R["walk_forward"]["train_days"], 10)
        test = c[1].number_input("Test window (rows)", 10, 250, R["walk_forward"]["test_days"], 5)
        holdout = c[2].number_input("Final holdout (rows)", 0, 250, R["walk_forward"]["holdout_days"], 10,
                                    help="Kept untouched until every fold is done; 0 = none.")
        hold = c[3].number_input("Maximum holding (rows)", 1, 60, R["strategy"]["max_hold_days"])
        c = st.columns(4)
        entry = c[0].multiselect("Entry thresholds |z| (grid)", [1.0, 1.5, 2.0, 2.5, 3.0],
                                 R["walk_forward"]["grid"]["entry_z"], help="Chosen per fold on TRAINING rows only.")
        exit_ = c[1].multiselect("Exit thresholds |z| (grid)", [0.0, 0.25, 0.5, 1.0], R["walk_forward"]["grid"]["exit_z"])
        windows = c[2].multiselect("Lookback windows (grid)", [20, 40, 60, 90], R["walk_forward"]["grid"]["window"])
        lots = c[3].number_input("Leg A size (whole lots)", 1, 100, int(R["strategy"].get("lots_leg_a", 1)),
                                 help="Leg B lots = nearest integer giving equal fine-gold grams (never fractional).")
        c = st.columns(4)
        roll = c[0].number_input("Expiry exclusion (calendar days)", 0, 20, ctx.settings["roll_days_before_expiry"],
                                 help="Contracts this close to expiry leave the pair series. Positions are always closed "
                                      f"before the {R['strategy']['tender_days']}-day delivery tender period.")
        minvol = c[1].number_input("Min. volume per leg (lots)", 0, 10_000, R["signal"]["min_volume_lots"],
                                   help="Candidate does not enter on days below this (liquidity gate).")
        near = c[2].number_input("Near-expiry flag (days)", 0, 30, R["signal"]["near_expiry_days"])
        c[3].text_input("Delivery tender period", f"last {R['strategy']['tender_days']} trading days (MCX spec)",
                        disabled=True)
        st.markdown("<span style='color:#6B7280;font-size:.82rem'>TRANSACTION COSTS — estimates (status and sources on "
                    "Research & Methodology)</span>", unsafe_allow_html=True)
        co = R["costs"]
        c = st.columns(7)
        brk = c[0].number_input("Brokerage/order (Rs)", 0.0, 500.0, float(co["brokerage_per_order_rs"]))
        cap = c[1].number_input("…or % cap", 0.0, 1.0, float(co.get("brokerage_pct_cap", 0.0)), 0.01, format="%.2f")
        exch = c[2].number_input("Exchange fee (%)", 0.0, 0.1, float(co["exchange_txn_pct"]), 0.0001, format="%.4f")
        ctt = c[3].number_input("CTT sell (%)", 0.0, 0.1, float(co["ctt_sell_pct"]), 0.001, format="%.3f")
        stamp = c[4].number_input("Stamp buy (%)", 0.0, 0.1, float(co["stamp_duty_buy_pct"]), 0.001, format="%.3f")
        gst = c[5].number_input("GST (%)", 0.0, 30.0, float(co["gst_pct"]))
        slip = c[6].number_input("Slippage (ticks)", 0.0, 20.0, float(co["slippage_ticks_per_side"]), 0.5)
        submitted = st.form_submit_button("Check and run walk-forward", type="primary")

    # ---------------------------------------------------------------- C. strategy definitions
    with st.expander("Strategy definitions and execution model"):
        t1, t2, t3 = st.tabs(["Baseline strategy", "Candidate strategy (AURUM)", "Execution model"])
        t1.markdown("- Measure: **naive** spread (raw rupee calendar spread / units-only cross spread)\n"
                    "- Signal: rolling z-score on prior observations only\n"
                    "- Entry: |z| ≥ entry threshold at close t → fill on day t+1\n"
                    "- Exit: |z| back within the exit threshold, max holding reached, or delivery deadline\n"
                    "- No data-quality gates")
        t2.markdown("- Measure: **purity- and carry-adjusted** spread / annualised implied carry\n"
                    "- Same signal, entry, exit and fill rules as the baseline\n"
                    "- Plus gates: no entry on a date with any risk flag (stale or unchanged close, low volume, near "
                    "expiry, roll, data gap, unadjusted expiry gap, limit-size move)\n"
                    "- Same costs, lots, rows and parameter-selection procedure")
        t3.markdown("- Fill on the next trading day at its close (MCX close = weighted average of the last 30 "
                    "minutes) or open; **both legs must have traded** that day, otherwise the entry is skipped and an "
                    "exit is delayed to the next traded day.\n"
                    "- Never enter unless the position can be closed before the delivery tender period; always exit "
                    "by the last trading day before it (exchange calendar).\n"
                    "- Whole lots only; leg B sized to equal fine-gold grams. Size above "
                    f"{R['strategy']['max_participation']:.0%} of a leg's daily volume is flagged.\n"
                    "- Bid/ask, depth and intraday timing are **not observed**: real slippage cannot be measured, so "
                    "results are re-priced under several slippage/fee scenarios.")

    # ---------------------------------------------------------------- input validation + feasibility
    errors = []
    if not isinstance(period, tuple) or len(period) != 2:
        errors.append("Choose a start and an end date for the analysis period.")
    if not entry or not exit_ or not windows:
        errors.append("Each grid (entry, exit, lookback) needs at least one value.")
    elif min(entry) <= max(exit_):
        errors.append("Every entry threshold must be larger than every exit threshold.")
    if windows and train <= max(windows):
        errors.append("The training window must be longer than the largest lookback window.")
    for e in errors:
        C.warning("Invalid configuration", e, "the backtest run", level="red")
    if errors:
        return
    start, end = period
    research = research_with({
        "walk_forward": {"train_days": int(train), "test_days": int(test), "holdout_days": int(holdout),
                         "grid": {"entry_z": sorted(entry), "exit_z": sorted(exit_), "window": sorted(windows)}},
        "strategy": {"max_hold_days": int(hold), "lots_leg_a": int(lots), "execution": execution,
                     "entry_z": sorted(entry)[len(entry) // 2], "exit_z": sorted(exit_)[0]},
        "statistics": {"window": int(sorted(windows)[len(windows) // 2])},
        "signal": {"min_volume_lots": int(minvol), "near_expiry_days": int(near)},
        "costs": {"brokerage_per_order_rs": brk, "brokerage_pct_cap": cap, "exchange_txn_pct": exch,
                  "ctt_sell_pct": ctt, "stamp_duty_buy_pct": stamp, "gst_pct": gst, "slippage_ticks_per_side": slip},
    })
    settings = settings_with(roll)
    n_rows = int(((base_df["date"].dt.date >= start) & (base_df["date"].dt.date <= end)).sum())
    feas = feasibility(n_rows, research, exploratory)
    with st.container(border=True):
        st.markdown("**Feasibility check** <span style='color:#6B7280;font-size:.82rem'>— computed before running</span>",
                    unsafe_allow_html=True)
        m = st.columns(4)
        m[0].metric("Rows in period", n_rows, help="Pair observations in the analysis period, before any signal filter.")
        m[1].metric("Folds that fit", feas["folds"], help=f"Protocol requires {R['validation_pass']['min_folds']}.")
        m[2].metric("Final holdout", "yes" if feas["holdout"] else "none")
        m[3].metric("Rows needed for a verdict", feas["rows_needed"],
                    help="train + min_folds × test + holdout")
        for msg in feas["messages"]:
            C.warning("Feasibility", msg, "the verdict", level="red" if not feas["runnable"] else "amber")
        if feas["runnable"] and feas["sufficient"]:
            st.caption("Configuration can reach a protocol verdict.")
        elif exploratory and feas["runnable"]:
            C.warning("Exploratory run", "Shorter windows describe how the strategy behaves on this data but are "
                      "statistically limited. The conclusion will be EXPLORATORY ONLY and never PASS.",
                      "the conclusion", level="amber")

    cfg_id = analysis_id({"mode": mode, "data": key, "pair": pair, "research": research, "settings": settings,
                          "period": [str(start), str(end)], "exploratory": exploratory, "version": __version__,
                          "engine": fp()})
    st.markdown(f"<div class='aur-meta'><span>Pair <b>{pair}</b></span><span>Data <b>{dmin:%d %b %Y} – {dmax:%d %b %Y}"
                f"</b></span><span>Period <b>{start:%d %b %Y} – {end:%d %b %Y}</b> ({n_rows} rows)</span>"
                f"<span>Configuration <b>{cfg_id}</b></span><span>Engine <b>v{__version__} · {fp()}</b></span></div>",
                unsafe_allow_html=True)

    if submitted:
        if not feas["runnable"]:
            C.warning("Not run", "This configuration cannot produce a single out-of-sample fold. Nothing was "
                      "computed and no metrics are shown.", "the backtest",
                      "use more history, shorter windows (exploratory mode), or a smaller holdout", level="red")
            log_experiment(Store(mode), {"logged_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                         "config_id": cfg_id, "mode": mode, "pair": pair, "period": f"{start} – {end}",
                                         "train/test/holdout": f"{train}/{test}/{holdout}", "exploratory": exploratory,
                                         "conclusion": "NOT RUN — infeasible"})
            return
        s["bt_cfg"] = cfg_id
    if s.get("bt_cfg") != cfg_id:
        C.empty_state("No backtest run for this configuration",
                      "Set the configuration and press 'Check and run walk-forward'. Every run is recorded in the "
                      "research log, whatever its result.")
        return
    with st.spinner("Running chronological walk-forward for candidate and baseline…"):
        val, n_used = run_backtest(mode, key, pair, json.dumps(research, sort_keys=True),
                                   json.dumps(settings, sort_keys=True), str(start), str(end), exploratory, fp())
    if val is None:
        C.empty_state("No data in the selected period", "Widen the analysis period.")
        return
    proto = val["protocol"]
    concl = proto["conclusion"]
    if submitted:
        log_experiment(Store(mode), {
            "logged_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "config_id": cfg_id,
            "mode": mode, "pair": pair, "period": f"{start} – {end}", "train/test/holdout": f"{train}/{test}/{holdout}",
            "exploratory": exploratory, "execution": execution,
            "grid": f"entry {sorted(entry)} exit {sorted(exit_)} lookback {sorted(windows)}",
            "folds": len(val["candidate"]["folds"]), "oos_trades": val["candidate"]["summary"]["trades"],
            "net_candidate": round(val["candidate"]["summary"]["net_pnl"], 0),
            "net_baseline": round(val["baseline"]["summary"]["net_pnl"], 0), "conclusion": concl})

    cv, bv = val["candidate"], val["baseline"]
    if exploratory:
        C.warning("EXPLORATORY ONLY — NOT EVIDENCE", CONCLUSIONS[concl], "every number below", level="amber")

    # ---------------------------------------------------------------- D. walk-forward evaluation
    with st.container(border=True):
        C.chart_header("Walk-forward evaluation", "Parameters chosen on each grey training block, applied unchanged to "
                       "the gold test block. The navy block is the final holdout, used once at the end.",
                       None, f"{len(cv['folds'])} folds")
        folds = cv["folds"]
        if folds.empty:
            C.empty_state("No complete fold", f"{n_used} rows cannot hold one {train}-row training window plus a test window.")
        else:
            fig = go.Figure()
            blocks = [(f"Fold {int(f['fold'])}", f["train_start"], f["train_end"], f["test_start"], f["test_end"],
                       f"net Rs {f['oos_net_pnl']:,.0f}", GOLD) for _, f in folds.iterrows()]
            h = cv["holdout"]
            if h:
                blocks.append(("Holdout", h["train_start"], h["train_end"], h["start"], h["end"],
                               f"net Rs {h['summary']['net_pnl']:,.0f}", INK))
            for k, (lab, trs, tre, tes, tee, txt, colr) in enumerate(blocks):
                for kind, a, b, cl, nm in (("train", trs, tre, "#C9CED6", "Train (in-sample)"),
                                           ("test", tes, tee, colr, "Test (out-of-sample)" if colr == GOLD else "Holdout")):
                    fig.add_trace(go.Scatter(x=[pd.Timestamp(a), pd.Timestamp(b)], y=[lab, lab], mode="lines",
                                             line=dict(color=cl, width=14), name=nm, legendgroup=nm,
                                             showlegend=k == 0 or (nm == "Holdout"),
                                             hovertemplate=f"{kind} {a} → {b}" + (f": {txt}" if kind == "test" else "")
                                                           + "<extra></extra>"))
            fig.update_layout(yaxis=dict(autorange="reversed"), hovermode="closest")
            st.plotly_chart(style_fig(fig, 90 + 28 * len(blocks)), width="stretch")
            with st.expander("Fold details (parameters chosen on training rows, results out-of-sample)"):
                st.dataframe(folds, hide_index=True, width="stretch")
            if val["candidate"]["skipped"]:
                with st.expander(f"Entry signals that could not be filled ({len(val['candidate']['skipped'])})"):
                    st.dataframe(pd.DataFrame(val["candidate"]["skipped"]), hide_index=True, width="stretch")

    # ---------------------------------------------------------------- E. performance summary
    with st.container(border=True):
        st.markdown("**Performance summary** <span style='color:#6B7280;font-size:.82rem'>— out-of-sample test folds "
                    "only (holdout reported separately), Rs</span>", unsafe_allow_html=True)
        cs, bs = cv["summary"], bv["summary"]
        if cs["trades"] == 0 and bs["trades"] == 0:
            C.empty_state("No out-of-sample trades", "No metrics can be calculated from zero trades.")
        else:
            m = st.columns(5)
            C.metric(m[0], "Net result", cs["net_pnl"], 0, "Rs", ref="AURUM, after costs")
            C.metric(m[1], "Trades", cs["trades"], ref="out-of-sample")
            m[2].metric("Hit rate", "N/A" if pd.isna(cs["win_rate"]) else f"{cs['win_rate']:.0%}")
            C.metric(m[3], "Max drawdown", cs["max_drawdown"], 0, "Rs", ref="realised, not mark-to-market")
            C.metric(m[4], "Gross result", cs["gross_pnl"], 0, "Rs", ref="before costs")
            rows = [("Net result (after costs)", "net_pnl", 0), ("Gross result", "gross_pnl", 0),
                    ("Estimated costs", "costs", 0), ("Max drawdown (realised)", "max_drawdown", 0),
                    ("Number of trades", "trades", 0), ("Hit rate", "win_rate", None),
                    ("Average net per trade", "avg_net", 0), ("Turnover (notional)", "turnover", 0),
                    ("Top-3 trades' share of net profit", "top3_share", None)]
            tbl = []
            for label, k, nd in rows:
                fmt = (lambda v: "N/A" if pd.isna(v) else f"{v:.0%}") if nd is None else (lambda v: C.fmt_num(v, 0))
                tbl.append({"Metric": label, "AURUM candidate": fmt(cs[k]), "Baseline": fmt(bs[k])})
            st.dataframe(pd.DataFrame(tbl), hide_index=True, width="stretch")
            st.caption("No Sharpe or annualised figures: trades are few and irregular, so annualisation would add "
                       "assumptions the data cannot support.")
        h = cv["holdout"]
        if h:
            hs = h["summary"]
            st.markdown(f"**Final holdout** {h['start']:%d %b %Y} – {h['end']:%d %b %Y}: parameters {h['params']} "
                        f"(chosen on {h['train_start']:%d %b %Y} – {h['train_end']:%d %b %Y}) → {hs['trades']} trades, "
                        f"net Rs {hs['net_pnl']:,.0f}")
        else:
            st.caption("No final holdout: the data does not allow one with this configuration.")

    # ---------------------------------------------------------------- F. charts
    with st.container(border=True):
        tabs = st.tabs(["Cumulative result", "Drawdown", "Baseline vs candidate by fold", "Cost scenarios",
                        "Parameter sensitivity"])
        ct, bt = cv["trades"], bv["trades"]
        with tabs[0]:
            if ct.empty and bt.empty:
                C.empty_state("Not calculable", "No out-of-sample trades.")
            else:
                f = go.Figure()
                for t, name, colr, dash in ((ct, "Candidate net", GOLD, None), (bt, "Baseline net", INK, None),
                                            (ct, "Candidate gross", GOLD, "dot")):
                    if len(t):
                        t = t.sort_values("exit_date")
                        y = t["gross_pnl" if "gross" in name else "net_pnl"].cumsum()
                        f.add_trace(go.Scatter(x=t["exit_date"], y=y, name=name, mode="lines+markers",
                                               line=dict(color=colr, dash=dash, width=1.6), marker=dict(size=4)))
                C.chart_header("Cumulative realised result", "Out-of-sample trades, by exit date", "Rs")
                st.plotly_chart(style_fig(f, 320, "Rs"), width="stretch")
        with tabs[1]:
            if ct.empty:
                C.empty_state("Not calculable", "No candidate trades.")
            else:
                d = _drawdown(ct)
                f = go.Figure(go.Scatter(x=d["date"], y=d["dd"], fill="tozeroy", line=dict(color=RED, width=1),
                                         name="Drawdown"))
                C.chart_header("Drawdown (candidate)", "Realised equity below its running peak", "Rs")
                st.plotly_chart(style_fig(f, 280, "Rs"), width="stretch")
        with tabs[2]:
            if cv["folds"].empty:
                C.empty_state("Not calculable", "No folds.")
            else:
                j = cv["folds"][["fold", "oos_net_pnl"]].merge(bv["folds"][["fold", "oos_net_pnl"]], on="fold",
                                                                suffixes=("_c", "_b"))
                f = go.Figure([go.Bar(x=j["fold"], y=j["oos_net_pnl_c"], name="Candidate", marker_color=GOLD),
                               go.Bar(x=j["fold"], y=j["oos_net_pnl_b"], name="Baseline", marker_color=INK)])
                f.update_layout(barmode="group", hovermode="x")
                C.chart_header("Net result per test fold", "Same folds, costs and timing for both", "Rs")
                st.plotly_chart(style_fig(f, 300, "Rs"), width="stretch")
        with tabs[3]:
            stress = val["cost_stress"]
            if stress.empty or ct.empty:
                C.empty_state("Not calculable", "No candidate trades to re-price.")
            else:
                C.chart_header("Same trades, different cost assumptions",
                               "net = gross − fees × multiplier − slippage ticks × Rs per tick", "Rs")
                st.dataframe(stress.style.format({"net_pnl": "{:,.0f}"}), hide_index=True, width="stretch")
                st.caption(f"Protocol: the '{R['cost_stress']['must_survive']}' scenario must stay net-positive.")
        with tabs[4]:
            sens = val["sensitivity"]
            if sens.empty or sens["trades"].sum() == 0:
                C.empty_state("Not enough data", "No trades anywhere in the grid.")
            else:
                pv = sens.pivot(index="entry_z", columns="exit_z", values="net_pnl")
                f = go.Figure(go.Heatmap(z=pv.values, x=[f"exit {x}" for x in pv.columns],
                                         y=[f"entry {y}" for y in pv.index], colorscale="RdBu", zmid=0,
                                         text=np.round(pv.values, 0), texttemplate="%{text:,}"))
                C.chart_header("Entry/exit sensitivity — IN-SAMPLE", "Development rows only (the holdout is never "
                               "used). Descriptive, not out-of-sample evidence.", "Rs")
                st.plotly_chart(style_fig(f, 320), width="stretch")

    # ---------------------------------------------------------------- G. trade ledger
    with st.container(border=True):
        st.markdown("**Trade ledger** <span style='color:#6B7280;font-size:.82rem'>— fills on the next traded day "
                    "for the specific contracts held; slippage assumed</span>", unsafe_allow_html=True)
        lt = st.tabs(["Candidate", "Baseline"])
        for tab, t in zip(lt, (cv["trades"], bv["trades"])):
            with tab:
                if t.empty:
                    C.empty_state("No trades", "The strategy never entered in the test windows.")
                    continue
                led = pd.DataFrame({
                    "Fold": t["fold"], "Direction": t["direction"],
                    "Signal": pd.to_datetime(t["signal_date"]).dt.date, "Entry": pd.to_datetime(t["entry_date"]).dt.date,
                    "Exit": pd.to_datetime(t["exit_date"]).dt.date, "Exit reason": t["exit_reason"],
                    "Leg A": t["a_contract"], "A lots": t["lots_a"], "A entry→exit": t["a_entry"].map("{:,.0f}".format)
                    + " → " + t["a_exit"].map("{:,.0f}".format), "A P&L": t["pnl_a"].round(0),
                    "Leg B": t["b_contract"], "B lots": t["lots_b"], "B entry→exit": t["b_entry"].map("{:,.0f}".format)
                    + " → " + t["b_exit"].map("{:,.0f}".format), "B P&L": t["pnl_b"].round(0),
                    "Gross": t["gross_pnl"].round(0), "Fees": t["fees"].round(0), "Slippage": t["slippage"].round(0),
                    "Net": t["net_pnl"].round(0), "Size vs volume": t["participation"].map("{:.1%}".format),
                    "Signal-day flags": t["signal_flags"].replace("", "—"), "Note": t["note"]})
                st.dataframe(led, hide_index=True, width="stretch", height=min(40 + 35 * len(led), 380))

    # ---------------------------------------------------------------- H + I. protocol and conclusion
    left, right = st.columns([2.2, 1])
    with left, st.container(border=True):
        st.markdown("**Predefined acceptance protocol** <span style='color:#6B7280;font-size:.82rem'>(thresholds fixed in "
                    "config/research.yaml before results; same rules as Signal Evidence)</span>", unsafe_allow_html=True)
        st.dataframe(proto["criteria"].style.map(_result_style, subset=["result"]), hide_index=True, width="stretch",
                     height=min(40 + 35 * len(proto["criteria"]), 560))
    with right, st.container(border=True):
        st.markdown("**Conclusion**")
        C.status_line(concl, CONCLUSIONS[concl])
        st.caption("A research conclusion, not an investment recommendation.")
        for r in proto["reasons"]:
            st.caption(f"• {r}")
        cfg = {"config_id": cfg_id, "engine_version": __version__, "engine_fingerprint": fp(), "dataset_mode": mode,
               "dataset_files_sha256": json.loads(key), "pair": pair, "period": [str(start), str(end)],
               "exploratory": exploratory, "research": research, "settings": settings, "conclusion": concl}
        st.download_button("Download configuration (JSON)", json.dumps(cfg, indent=2, default=str),
                           file_name=f"aurum_backtest_{cfg_id}.json", width="stretch")
        st.button("Open Research & Methodology →", on_click=go_to, args=("Research & Methodology",), width="stretch")
