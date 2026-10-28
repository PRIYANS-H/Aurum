"""AURUM dashboard shell.   Run:  streamlit run app.py

Four pages share one sidebar, one top data bar and one session context (pair, dates, lookback).
Every number shown is computed by the aurum package from ingested data.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

st.set_page_config(page_title="AURUM · Gold Relative-Value Intelligence", page_icon="◆", layout="wide",
                   initial_sidebar_state="expanded")

from aurum.config import ROOT  # noqa: E402
from aurum.ingestion import Store  # noqa: E402
from aurum.sample_data import write_sample_files  # noqa: E402
from aurum.ui import components as C  # noqa: E402
from aurum.ui import page_backtest, page_evidence, page_explorer, page_methodology  # noqa: E402
from aurum.ui.state import (DATASETS, PAGES, data_mode_label, fp, get_ctx, init_state, load_obs,  # noqa: E402
                            load_quality, manifest_key)
from aurum.ui.theme import inject_css  # noqa: E402

SIDEBAR_CSS = """
<style>
section[data-testid="stSidebar"] { background:#131B2E !important; border-right:none; }
section[data-testid="stSidebar"] * { color:#D5DAE3; }
section[data-testid="stSidebar"] .aur-name { color:#E8D9B0; }
section[data-testid="stSidebar"] .aur-sub { color:#8E99AE; }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] { padding:9px 12px; border-radius:5px;
    margin:1px 0; width:100%; cursor:pointer; }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover { background:#1E2840; }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child { display:none; }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] { background:#9C7A2B; }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] * { color:#FFFFFF;
    font-weight:600; }
div[data-testid="stDecoration"] { display:none; }
section[data-testid="stSidebar"] .aur-side-foot { border-top:1px solid #26314A; color:#8E99AE; }
section[data-testid="stSidebar"] .aur-side-foot b { color:#E6EAF1; }
section[data-testid="stSidebar"] div[data-testid="stExpander"] { background:#182238; border-color:#26314A; }
</style>
"""

PAGE_MODULES = {"Relative Value Explorer": page_explorer, "Signal Evidence": page_evidence,
                "Backtesting Lab": page_backtest, "Research & Methodology": page_methodology}

inject_css()
st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)
init_state()
ctx = get_ctx()
mode = st.session_state["dataset"]
key = manifest_key(mode)
obs = load_obs(mode, key)
quality = load_quality(mode, key, fp())
mode_label = data_mode_label(mode)
latest = pd.to_datetime(obs["date"]).max() if not obs.empty else None

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("<div class='aur-brand'><div class='aur-logo'></div><div class='aur-name'>AURUM</div></div>"
                "<div class='aur-sub'>Gold Relative-Value Intelligence</div>", unsafe_allow_html=True)
    st.radio("Navigation", list(PAGES), key="page", label_visibility="collapsed")
    latest_txt = f"{latest:%d %b %Y}" if latest is not None else "no data"
    st.markdown(
        "<div class='aur-side-foot'>"
        f"Data source<br><b>{C.esc(DATASETS[mode])}</b><br><br>"
        f"Data status<br>{C.badge(mode_label)} {C.badge(quality['status'])}<br><br>"
        f"Latest data date<br><b>{latest_txt}</b></div>", unsafe_allow_html=True)
    with st.expander("About"):
        st.caption("AURUM is a quantitative research tool for MCX gold futures. Historical end-of-day data only — "
                   "no live prices, no orders, no investment advice.")

# --------------------------------------------------------------------------- top bar
page = st.session_state["page"]
st.markdown(f"<div class='aur-title'>{C.esc(page)}</div><div class='aur-subtitle'>{C.esc(PAGES[page])}</div>",
            unsafe_allow_html=True)
with st.container(border=True):
    t = st.columns([1.6, 2.2, 1.4, 1.3, 1.3, 0.9], vertical_alignment="center")
    t[0].selectbox("Dataset", list(DATASETS), key="dataset", format_func=lambda m: DATASETS[m],
                   label_visibility="collapsed")
    if latest is not None:
        first = pd.to_datetime(obs["date"]).min()
        t[1].markdown(f"<div class='aur-meta'><span>Date range <b>{first:%d %b %Y} – {latest:%d %b %Y}</b></span></div>",
                      unsafe_allow_html=True)
        t[2].markdown(f"<div class='aur-meta'><span>Latest <b>{latest:%d %b %Y}</b></span></div>", unsafe_allow_html=True)
    t[3].markdown(f"<span class='aur-meta'>Mode</span> {C.badge(mode_label)}", unsafe_allow_html=True)
    t[4].markdown(f"<span class='aur-meta'>Quality</span> {C.badge(quality['status'])}", unsafe_allow_html=True,
                  help="; ".join(quality.get("issues", [])) or "no dataset-level issues")
    if t[5].button("Reload", help="Re-read processed data from disk (e.g. after ingesting new files)."):
        st.cache_data.clear()
        st.rerun()

C.data_status_banner(mode_label, quality)

# --------------------------------------------------------------------------- empty dataset
if obs.empty:
    if mode == "SYNTHETIC":
        C.empty_state("No synthetic data yet", "Generate the clearly labelled demo dataset to explore the application.")
        if st.button("Generate synthetic demo data", type="primary"):
            store = Store("SYNTHETIC")
            store.clear()
            for p in write_sample_files(ROOT / "data" / "sample"):
                store.ingest(p, ctx.book.products)
            st.cache_data.clear()
            st.rerun()
    else:
        C.empty_state("No MCX data ingested", "Download bhavcopy CSV files from mcxindia.com in your browser, then "
                      "upload them on Research & Methodology → Data provenance, or run "
                      "`python -m aurum.cli ingest \"data/raw/*.csv\"`.")
        if page == "Research & Methodology":
            page_methodology.render(mode, key, obs)
    st.stop()

PAGE_MODULES[page].render(mode, key, obs)
