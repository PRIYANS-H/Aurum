"""Reusable UI components. Every page uses these instead of ad-hoc markup."""

from __future__ import annotations

import html
import json
import math

import pandas as pd
import streamlit as st

from aurum.contracts import COMPARABILITY_STATUSES
from aurum.signals import ALERT_INFO, STATUS_INFO, VALIDATION_INFO
from aurum.validation import CONCLUSIONS
from .theme import GREEN, AMBER, RED, GREY, STATUS_STYLE

# All status explanations come from the backend; the UI defines none of its own.
STATUS_EXPLAIN = {**COMPARABILITY_STATUSES, **ALERT_INFO, **VALIDATION_INFO, **CONCLUSIONS, **STATUS_INFO}


def esc(x) -> str:
    return html.escape(str(x))


def badge(label: str) -> str:
    fg, bg = STATUS_STYLE.get(label, (GREY, "#EEF0F3"))
    return f"<span class='aur-badge' style='color:{fg};background:{bg};border:1px solid {fg}33'>{esc(label)}</span>"


def status_line(label: str, explain: str | None = None) -> None:
    text = explain if explain is not None else STATUS_EXPLAIN.get(label, "")
    st.markdown(f"{badge(label)} <span style='color:{GREY};font-size:.88rem'>&nbsp;{esc(text)}</span>",
                unsafe_allow_html=True)


def fmt_num(v, nd: int = 2, unit: str = "", missing: str = "N/A") -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) or (
            not isinstance(v, str) and pd.isna(v)):
        return missing
    if isinstance(v, (int,)) and not isinstance(v, bool):
        s = f"{v:,}"
    else:
        s = f"{float(v):,.{nd}f}"
    return f"{s} {unit}".strip()


def metric(col, label: str, value, nd: int = 2, unit: str = "", help: str | None = None,
           ref: str | None = None) -> None:
    """Metric card: label, value with unit, time reference and definition. Missing -> N/A."""
    col.metric(label, fmt_num(value, nd, unit), help=help)
    if ref:
        col.caption(ref)


def chart_header(title: str, desc: str, unit: str | None = None, dates: str | None = None,
                 formula: str | None = None) -> None:
    meta = " · ".join(x for x in [desc, f"Unit: {unit}" if unit else None, dates] if x)
    st.markdown(f"<div class='aur-chart-h'><div class='t'>{esc(title)}</div><div class='d'>{esc(meta)}</div></div>",
                unsafe_allow_html=True)
    if formula:
        with st.popover("How is this calculated?"):
            st.markdown(formula)


def empty_state(title: str, body: str) -> None:
    st.markdown(f"<div class='aur-empty'><b>{esc(title)}</b><br>{esc(body)}</div>", unsafe_allow_html=True)


def warning(title: str, reason: str, affects: str, next_step: str | None = None, level: str = "amber") -> None:
    cls = {"amber": "", "red": " red", "info": " info"}[level]
    nxt = f" · Next step: {esc(next_step)}" if next_step else ""
    st.markdown(
        f"<div class='aur-warn{cls}'><div class='w-title'>{esc(title)}</div>{esc(reason)}"
        f"<div class='w-meta'>Affects: {esc(affects)}{nxt}</div></div>", unsafe_allow_html=True)


def kv(rows: list[tuple[str, str]]) -> str:
    return "<div class='aur-kv'>" + "".join(
        f"<div class='k'>{esc(k)}</div><div class='v'>{v}</div>" for k, v in rows) + "</div>"


def card(title: str, body_html: str, gold: bool = False) -> None:
    st.markdown(f"<div class='aur-card{' gold' if gold else ''}'><h4>{esc(title)}</h4>{body_html}</div>",
                unsafe_allow_html=True)


def timeline(stages: list[dict]) -> None:
    """stages: [{'stage','result': PASS|FAIL|WARN|SKIP, 'detail', 'time'}]"""
    colours = {"PASS": GREEN, "FAIL": RED, "WARN": AMBER, "SKIP": GREY}
    items = "".join(
        f"<div class='aur-tl-item' style='--dot:{colours.get(s['result'], GREY)}'>"
        f"<span class='s'>{i}. {esc(s['stage'])}</span> {badge(s['result'] if s['result'] in ('PASS','FAIL') else ('CHECK' if s['result']=='WARN' else 'n/a'))}"
        f"<div class='r'>{esc(s['detail'])}{(' · ' + esc(s['time'])) if s.get('time') else ''}</div></div>"
        for i, s in enumerate(stages, 1))
    st.markdown(f"<div class='aur-tl'>{items}</div>", unsafe_allow_html=True)


def repro_metadata(meta: dict) -> None:
    with st.expander("Reproducibility metadata"):
        st.code(json.dumps(meta, indent=2, default=str), language="json")


def flag_details(flags: list[dict]) -> None:
    """flags: [{'label','trigger','evidence','effect','confirm','gate'}] — one expandable row each."""
    for f in flags:
        kind = "risk flag" if f.get("gate", True) else "data-quality indicator"
        with st.expander(f"{f['label']} — {kind}"):
            st.markdown(kv([("Triggered by", esc(f.get("trigger", ""))), ("Evidence", esc(f.get("evidence", ""))),
                            ("Effect on interpretation", esc(f.get("effect", ""))),
                            ("Would help confirm", esc(f.get("confirm", "")))]), unsafe_allow_html=True)


def checklist(items: list[tuple[bool, str, str]]) -> None:
    """items: (ok, text, detail). Green check for passed checks, amber marker for issues."""
    rows = "".join(
        f"<div style='padding:3px 0'><span style='color:{GREEN if ok else AMBER};font-weight:700'>"
        f"{'✓' if ok else '!'}</span>&nbsp; {esc(text)}"
        + (f"<div style='color:{GREY};font-size:.8rem;margin-left:18px'>{esc(detail)}</div>" if detail else "")
        + "</div>" for ok, text, detail in items)
    st.markdown(f"<div style='font-size:.9rem'>{rows}</div>", unsafe_allow_html=True)


def data_status_banner(mode_label: str, quality: dict, unverified: int = 0) -> None:
    if mode_label == "SYNTHETIC":
        warning("SYNTHETIC DATA — not MCX market data",
                "Generated by aurum/sample_data.py to demonstrate the software. It contains mean-reverting "
                "dislocations by construction, so strategies look better than they would on real data.",
                "every number on every page", "switch the dataset to REAL to analyse MCX files", level="red")
    for issue in quality.get("issues", []):
        warning("Data quality", issue, "conclusions that need long or gap-free history", level="amber")
    if unverified:
        warning("Unverified specifications", f"{unverified} calculation fields in use are not VERIFIED from an "
                "official MCX document.", "status cap: nothing above WATCH",
                "add the contract-series PDF and mark fields VERIFIED", level="info")


def resolve_pair(ctx, a: str, b: str) -> tuple[str | None, str]:
    """Map two products to a registered comparison rule, or explain why none exists."""
    for name, p in ctx.pairs.items():
        if p["kind"] == "calendar" and a == b == p["product"]:
            return name, "calendar"
        if p["kind"] == "cross" and a != b and {a, b} == {p["leg_a"], p["leg_b"]}:
            return name, ("" if (a, b) == (p["leg_a"], p["leg_b"])
                          else f"Oriented as registered: {p['leg_a']} vs {p['leg_b']}.")
    related = [n for n, p in ctx.pairs.items() if a in (p.get("product"), p.get("leg_a"), p.get("leg_b"))]
    return None, (f"No validated comparison rule exists for {a} vs {b}. Registered comparisons involving {a}: "
                  + (", ".join(related) if related else "none") + ".")


def pair_selector(ctx, products_available: list[str], key_prefix: str = "") -> tuple[str | None, str]:
    """Pick two products; resolve them to a registered comparison rule or explain why none exists."""
    opts = sorted(products_available)
    ka, kb, ks = f"{key_prefix}prodA", f"{key_prefix}prodB", f"{key_prefix}synced_pair"
    # If the shared pair changed elsewhere (another page / a button), move the selectors to it.
    if st.session_state.get(ks) != st.session_state.get("pair"):
        current = ctx.pairs.get(st.session_state.get("pair"), {})
        a0 = current.get("product", current.get("leg_a"))
        b0 = current.get("product", current.get("leg_b"))
        if a0 in opts and b0 in opts:
            st.session_state[ka], st.session_state[kb] = a0, b0
        st.session_state[ks] = st.session_state.get("pair")
    st.session_state.setdefault(ka, opts[0] if opts else None)
    st.session_state.setdefault(kb, opts[0] if opts else None)
    c1, c2 = st.columns(2)
    a = c1.selectbox("First contract", opts, key=ka,
                     help="Only products present in the dataset AND in the contract registry are listed.")
    b = c2.selectbox("Second contract", opts, key=kb,
                     help="Choose the same product for a calendar (near vs next expiry) comparison.")
    name, note = resolve_pair(ctx, a, b)
    if name:
        st.session_state["pair"] = st.session_state[ks] = name
    return name, note
