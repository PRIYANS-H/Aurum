"""Page 1 - Relative Value Explorer."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from aurum.analytics import FLAGS, active_flags, flag_evidence, flag_trigger
from aurum.contracts import CALC_FIELDS, assess_comparability, check_fixed_contracts
from . import components as C
from .state import get_ctx, go_to, pair_df
from .theme import GOLD, INK, RED, style_fig

ROLLING = "Rolling (rule)"
WINDOWS = {"All history": None, "Last 3 months": 63, "Last 6 months": 126, "Last 1 year": 252,
           "Last 2 years": 504, "Custom": "custom"}


def _swap() -> None:
    s = st.session_state
    s["exp_prodA"], s["exp_prodB"] = s.get("exp_prodB"), s.get("exp_prodA")
    s["exp_expA"], s["exp_expB"] = s.get("exp_expB", ROLLING), s.get("exp_expA", ROLLING)


def _expiry_options(obs: pd.DataFrame, product: str) -> list[str]:
    ids = obs.loc[obs["symbol"] == product, ["contract_id", "expiry"]].drop_duplicates()
    ids = ids.sort_values("expiry")
    return [ROLLING] + [f"{pd.Timestamp(e):%d %b %Y}  ·  {c}" for c, e in zip(ids["contract_id"], ids["expiry"])]


def _leg_spec_rows(ctx, row) -> pd.DataFrame:
    out = []
    for leg in ("a", "b"):
        spec = ctx.book.spec(row[f"{leg}_symbol"], row[f"{leg}_expiry"])
        statuses = {getattr(spec, f).status for f in CALC_FIELDS}
        worst = "ASSUMED" if "ASSUMED" in statuses else ("SECONDARY" if "SECONDARY" in statuses else "VERIFIED")
        src = ctx.book.sources.get(spec.purity_per_mille.source or "", {}).get("title", "—")
        out.append({
            "Leg": leg.upper(), "Contract": row[f"{leg}_contract_id"], "Expiry": row[f"{leg}_expiry"].date(),
            "Days to expiry": int(row[f"{leg}_dte"]), "Lot size": f"{spec.lot_grams.value:,} g",
            "Quotation unit": f"Rs per {spec.quote_unit_grams.value} g", "Tick": f"Rs {spec.tick_rs.value:g}",
            "Purity": spec.purity_per_mille.value, "Price basis": spec.price_basis.value,
            "Spec status": worst, "Spec source": src,
        })
    return pd.DataFrame(out)


def render(mode: str, key: str, obs: pd.DataFrame) -> None:
    ctx = get_ctx()
    products = sorted(set(obs["symbol"]) & ctx.book.products)
    s = st.session_state

    # keep selectors in sync when another page changed the shared pair
    if s.get("exp_synced_pair") != s.get("pair"):
        cur = ctx.pairs.get(s.get("pair"), {})
        a0, b0 = cur.get("product", cur.get("leg_a")), cur.get("product", cur.get("leg_b"))
        if a0 in products and b0 in products:
            s["exp_prodA"], s["exp_prodB"] = a0, b0
            s["exp_expA"] = s["exp_expB"] = ROLLING
        s["exp_synced_pair"] = s.get("pair")
    s.setdefault("exp_prodA", products[0])
    s.setdefault("exp_prodB", products[0])

    # ---------------------------------------------------------------- B. analysis controls
    with st.container(border=True):
        c = st.columns([1.1, 1.5, 0.5, 1.1, 1.5, 1.1, 0.9], vertical_alignment="bottom")
        a = c[0].selectbox("Contract A", products, key="exp_prodA",
                           help="Only products present in the dataset and in the contract registry.")
        if s.get("exp_expA") not in _expiry_options(obs, a):
            s["exp_expA"] = ROLLING
        ea = c[1].selectbox("Expiry", _expiry_options(obs, a), key="exp_expA",
                            help="'Rolling' follows the pair's selection rule; or pick one specific contract.")
        c[2].button("⇄", on_click=_swap, help="Swap contract A and B", key="exp_swap")
        b = c[3].selectbox("Contract B", products, key="exp_prodB")
        if s.get("exp_expB") not in _expiry_options(obs, b):
            s["exp_expB"] = ROLLING
        eb = c[4].selectbox("Expiry ", _expiry_options(obs, b), key="exp_expB")
        win_label = c[5].selectbox("Analysis window", list(WINDOWS), key="exp_win",
                                   help="Which dates are shown. Statistics always use only data before each date.")
        lookback = c[6].number_input("Lookback (obs)", 20, 120, int(s["window"]), 5, key="exp_lookback",
                                     help="Prior observations used for rolling mean, std and z (date t excluded).")
        s["window"] = int(lookback)
        with st.expander("Advanced options"):
            st.selectbox("Price field", ["Close (end of day)"], disabled=True,
                         help="The engine uses the bhavcopy close; settlement price is not in the files.")
            st.caption(f"Rolling rule: calendar = near vs next expiry, rolling when the near contract is within "
                       f"{ctx.settings['roll_days_before_expiry']} days of expiry; cross = nearest eligible A vs the "
                       "B contract expiring in the same month.")

    pair, note = C.resolve_pair(ctx, a, b)
    if pair is None:
        C.warning("Comparison not available", note, "all calculations on this page",
                  "pick one of the registered comparisons", level="red")
        C.empty_state("No validated comparison selected",
                      "AURUM only calculates spreads for pairs with a documented comparability rule.")
        return
    s["pair"] = s["exp_synced_pair"] = pair
    pcfg = ctx.pairs[pair]
    is_cal = pcfg["kind"] == "calendar"

    # fixed contracts (both legs must be explicit), oriented as the registered rule
    fixed = None
    if (ea == ROLLING) != (eb == ROLLING):
        C.warning("Choose both expiries or neither", "One leg is rolling and the other is fixed, which would mix a "
                  "moving contract with a static one.", "leg selection", "set both to Rolling or pick both contracts",
                  level="red")
        return
    if ea != ROLLING:
        id_a, id_b = ea.split("·")[-1].strip(), eb.split("·")[-1].strip()
        if not is_cal and (a, b) != (pcfg["leg_a"], pcfg["leg_b"]):
            id_a, id_b = id_b, id_a
        ex = obs.drop_duplicates("contract_id").set_index("contract_id")["expiry"]
        reason = check_fixed_contracts(pcfg, id_a, id_b, ex[id_a], ex[id_b])
        if reason:
            C.warning("Invalid comparison", reason, "all calculations on this page", level="red")
            C.status_line("INVALID COMPARISON")
            return
        fixed = (id_a, id_b)
    if note and note != "calendar":
        st.caption(note)

    df = pair_df(mode, key, pair, int(lookback), fixed=fixed)
    if df.empty:
        blocked = "; ".join(f"{k} ({v} rows)" for k, v in (df.attrs.get("blocked") or {}).items())
        C.warning("No comparable observations", "There is no date on which both legs have prices and a "
                  "specification." + (f" Blocked: {blocked}" if blocked else ""), "every chart and statistic",
                  level="red")
        return

    dmin, dmax = df["date"].min().date(), df["date"].max().date()
    w = WINDOWS[win_label]
    if w == "custom" and dmin < dmax:
        rng = st.slider("Custom date range", dmin, dmax, (dmin, dmax), format="DD MMM YYYY", key="exp_range")
    elif isinstance(w, int):
        rng = (df["date"].iloc[max(0, len(df) - w)].date(), dmax)
    else:
        rng = (dmin, dmax)
    s["date_range"] = rng
    view = df[(df["date"].dt.date >= rng[0]) & (df["date"].dt.date <= rng[1])]
    if view.empty:
        C.empty_state("No observations in the selected range", "Widen the analysis window.")
        return
    last = view.iloc[-1]
    comp = assess_comparability(pcfg, ctx.book, last["a_expiry"], last["b_expiry"])
    valid_n = int(view["z"].notna().sum())
    flags_now = active_flags(last, gate_only=False)
    unusable = ~view["comparability"].isin(["VALID", "ADJUSTED"])
    if unusable.any():
        statuses = view.loc[unusable, "comparability"].value_counts().to_dict()
        first_ok = view.loc[~unusable, "date"].min()
        C.warning(f"{int(unusable.sum())} of {len(view)} rows are not comparable ({', '.join(f'{k}: {v}' for k, v in statuses.items())})",
                  "Those rows rely on specification fields that are not VERIFIED (or missing) for their contract "
                  "series. They are drawn grey and dashed as exploratory only and are excluded from every statistic, "
                  "signal and backtest. " + (f"Validated analysis starts {first_ok:%d %b %Y}." if pd.notna(first_ok)
                                             else "No row in range is usable.") +
                  (" Missing evidence: " + "; ".join(comp.missing) if comp.missing else ""),
                  "spread values, z-scores and classifications", "add the official MCX PDFs for those contract series",
                  level="red" if unusable.all() else "amber")

    # ---------------------------------------------------------------- A. page metadata
    st.markdown(
        "<div class='aur-meta'>"
        f"<span>Selection <b>{C.esc(last['a_contract_id'])}</b> vs <b>{C.esc(last['b_contract_id'])}</b>"
        f" ({'fixed contracts' if fixed else 'rule ' + C.esc(pair)})</span>"
        f"<span>Pair history <b>{dmin:%d %b %Y} – {dmax:%d %b %Y}</b></span>"
        f"<span>Shown <b>{rng[0]:%d %b %Y} – {rng[1]:%d %b %Y}</b></span>"
        f"<span>Overlapping obs <b>{len(view):,}</b></span>"
        f"<span>Comparability {C.badge(comp.status)}</span></div>", unsafe_allow_html=True)
    st.write("")

    # ---------------------------------------------------------------- C. summary metrics
    unit = "Rs/10 g, 995-eq."
    m = st.columns(6)
    C.metric(m[0], "A price", last["a_norm"], 0, help=f"{last['a_contract_id']} close, comparable basis ({unit}), "
             f"{last['date']:%d %b %Y}.", ref="per 10 g, 995-eq.")
    C.metric(m[1], "B price", last["b_norm"], 0, help=f"{last['b_contract_id']} close, comparable basis ({unit}), "
             f"{last['date']:%d %b %Y}.", ref="per 10 g, 995-eq.")
    C.metric(m[2], "Carry" if is_cal else "Spread", last["metric"], 3,
             "%" if not is_cal else "% p.a.", help=df.attrs.get("metric_label"),
             ref="implied, annualised" if is_cal else ("purity + carry adj." if last.get("carry_adjusted") else "purity adj."))
    C.metric(m[3], "Z-score", last["z"], 2, help="(x_t − rolling mean) / rolling std, prior observations only.",
             ref=f"lookback {int(lookback)}")
    C.metric(m[4], "Valid obs", valid_n, help="Observations in range with enough history for a z-score.",
             ref=f"of {len(view):,} in range")
    m[5].metric("Quality", "Good" if not flags_now else f"{len(flags_now)} flag(s)",
                help="Flags on the latest observation: " + (", ".join(flags_now) or "none"))
    m[5].caption("latest observation")

    # ---------------------------------------------------------------- D + E. charts side by side
    left, right = st.columns(2)
    with left, st.container(border=True):
        C.chart_header("Normalised price comparison", "Click a legend item to hide it", unit,
                       f"{rng[0]:%d %b %Y} – {rng[1]:%d %b %Y}",
                       formula=("`normalised = close ÷ quote_unit_grams ÷ (purity ÷ 1000) × 10 × 0.995`\n\n"
                                "Rs per 10 g of 995-equivalent gold (GOLD/GOLDM basis). For 999 contracts this is "
                                "the 995/999 factor used in MCX's own settlement rules. Specs are those of each "
                                "leg's contract series."))
        fig = go.Figure()
        for leg, colr in (("a", INK), ("b", GOLD)):
            fig.add_trace(go.Scatter(x=view["date"], y=view[f"{leg}_norm"], name=f"{leg.upper()} · {last[f'{leg}_symbol']}",
                                     line=dict(color=colr, width=1.5), customdata=view[f"{leg}_contract_id"],
                                     hovertemplate="%{y:,.0f}  %{customdata}<extra></extra>"))
        for d in view.loc[view["roll"], "date"]:
            fig.add_vline(x=d, line=dict(color="#C9CED6", width=1, dash="dot"))
        st.plotly_chart(style_fig(fig, 310, unit), width="stretch")
        st.caption("Dotted lines: contract rolls." if view["roll"].any() else "No contract rolls in range.")

    with right, st.container(border=True):
        measures = ({"Carry": "metric", "Raw spread": "baseline_metric", "Z-score": "z", "Percentile": "percentile"}
                    if is_cal else
                    {"Spread": "metric", "Purity only": "spread_pct", "Naive": "spread_pct_naive",
                     "Absolute": "spread_abs", "Z-score": "z", "Percentile": "percentile"})
        choice = st.segmented_control("Measure", list(measures), default=list(measures)[0], key="exp_measure",
                                      label_visibility="collapsed") or list(measures)[0]
        col = measures[choice]
        units = {"metric": "% p.a." if is_cal else "%", "baseline_metric": "Rs", "spread_pct": "%",
                 "spread_pct_naive": "%", "spread_abs": unit, "z": "std devs", "percentile": "percentile"}
        formulas = {
            "metric": ("Implied carry = 100 · ln(F_far / F_near) / ((T_far − T_near)/365)" if is_cal else
                       "Spread = 100 · ((B_norm/A_norm) · exp(−c · gap_years) − 1); c = A's same-day implied "
                       "carry, gap = expiry difference. Without a carry estimate: purity-adjusted spread."),
            "baseline_metric": "Raw calendar spread = F_far − F_near (Rs per quote unit).",
            "spread_pct": "100 · (B_norm / A_norm − 1) — purity and units adjusted, no carry adjustment.",
            "spread_pct_naive": "Units-only conversion that ignores purity. Shown to expose its bias.",
            "spread_abs": "B_norm − A_norm.",
            "z": f"z_t = (x_t − mean of previous {int(lookback)}) / std of previous {int(lookback)}; x_t excluded.",
            "percentile": f"Share of the previous {int(lookback)} values ≤ x_t.",
        }
        C.chart_header(f"Spread analysis · {choice}", "Unusual under the rolling model ≠ arbitrage",
                       units[col], f"lookback {int(lookback)} obs", formulas[col])
        wz = ctx.research["signal"]["watch_z"]
        f2 = go.Figure()
        if col == "metric" and unusable.any():
            f2.add_trace(go.Scatter(x=view["date"], y=view["metric_raw"].where(unusable), name="UNVERIFIED (exploratory)",
                                    line=dict(color="#A8B0BC", width=1.2, dash="dash")))
        f2.add_trace(go.Scatter(x=view["date"], y=view[col], name=choice, line=dict(color=INK, width=1.3)))
        if col == "metric":
            f2.add_trace(go.Scatter(x=view["date"], y=view["mean"], name="Rolling mean",
                                    line=dict(color=GOLD, width=1.2, dash="dot")))
            for sgn in (1, -1):
                f2.add_trace(go.Scatter(x=view["date"], y=view["mean"] + sgn * wz * view["std"], showlegend=sgn == 1,
                                        name=f"±{wz:g} std", line=dict(color="#B9C2CF", width=1, dash="dash")))
        if col == "z":
            for h in (wz, -wz):
                f2.add_hline(y=h, line=dict(color=RED, width=1, dash="dash"))
            hits = view[view["z"].abs() >= wz]
            f2.add_trace(go.Scatter(x=hits["date"], y=hits["z"], mode="markers", name=f"|z| ≥ {wz:g}",
                                    marker=dict(color=RED, size=6)))
        st.plotly_chart(style_fig(f2, 310, units[col]), width="stretch")
        sm = view["metric"].dropna()
        st.caption(f"AURUM measure in range: min {sm.min():.3f} · median {sm.median():.3f} · max {sm.max():.3f}"
                   if len(sm) else "No valid values in range.")

    # ---------------------------------------------------------------- F + G
    left, right = st.columns([1.55, 1])
    with left, st.container(border=True):
        st.markdown("**Contract and expiry context**")
        tbl = _leg_spec_rows(ctx, last)
        st.dataframe(tbl.style.map(lambda v: {"VERIFIED": "color:#2F6B45;font-weight:600",
                                              "SECONDARY": "color:#A86A12;font-weight:600",
                                              "ASSUMED": "color:#9B3B3B;font-weight:600"}.get(v, ""),
                                   subset=["Spec status"]), hide_index=True, width="stretch")
        gap = int(last["expiry_gap_days"])
        if is_cal:
            ok, txt = gap > 0, f"Calendar rule: two different expiries required — {gap} days apart."
        else:
            ok = last["a_expiry"].to_period("M") == last["b_expiry"].to_period("M")
            txt = (f"Cross rule: same expiry month required — {last['a_expiry']:%b %Y} / {last['b_expiry']:%b %Y}; "
                   f"gap {gap} days, carry-adjusted: {'yes' if bool(last.get('carry_adjusted', False)) else 'no'}.")
        C.status_line("VALID" if ok else "INVALID", txt)
        C.status_line(comp.status, "Latest row: " + "; ".join(comp.reasons))
        if not comp.pnl_verified:
            st.caption("Lot size / tick for these contract series are not VERIFIED: rupee P&L is indicative and "
                       "RESEARCH CANDIDATE is blocked.")
        with st.expander("Adjustments applied and known confounders"):
            st.markdown("**Adjustments**\n" + "\n".join(f"- {x}" for x in
                        (comp.adjustments or ["none needed — identical quotation unit and purity"])))
            st.markdown("**Confounders (not removed)**\n" + "\n".join(f"- {x}" for x in comp.confounders))

    with right, st.container(border=True):
        st.markdown("**Data quality and risk flags** <span style='color:#6B7280;font-size:.8rem'>(selected range)</span>",
                    unsafe_allow_html=True)
        n = len(view)
        items, raised = [], []
        for k, f in FLAGS.items():
            cnt = int(view[k].sum())
            if cnt == 0:
                items.append((True, f"No '{f['label'].lower()}' days", ""))
            else:
                items.append((False, f"{f['label']}: {cnt} of {n} days", f"Effect: {f['effect']}"))
                ex = view[view[k]].iloc[-1]
                raised.append({**f, "trigger": flag_trigger(k, ctx.research, ctx.settings),
                               "evidence": f"{cnt} days; latest {flag_evidence(ex, k)}"})
        nz = int(view["metric"].notna().sum() - view["z"].notna().sum())
        items.append((nz == 0, f"Enough history (≥ {ctx.research['statistics']['min_periods']} prior obs) on every "
                      "comparable row" if nz == 0 else f"Insufficient history for z: {nz} comparable rows", ""))
        items.append((False, "Closes are end-of-day (MCX close = 30-min weighted average); no bid/ask",
                      "Effect: executable prices and intraday timing are unobserved"))
        C.checklist(items)
        if raised:
            st.markdown("<span style='color:#6B7280;font-size:.8rem'>Flag details</span>", unsafe_allow_html=True)
            C.flag_details(raised)

    # ---------------------------------------------------------------- H. next action
    st.button("Investigate in Signal Evidence →", type="primary", on_click=go_to, args=("Signal Evidence",),
              kwargs={"pair": pair, "as_of": last["date"].date(), "window": int(lookback)})
    st.caption("Opens the evidence card for this pair on the latest date shown, with the same lookback."
               + (" Signal Evidence and the Backtesting Lab use the rolling rule, not fixed contracts." if fixed else ""))
