"""Design system: colour tokens, global CSS and the Plotly chart style."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

NAVY = "#16213A"
INK = "#1F3A5F"
GOLD = "#9C7A2B"
GREY = "#6B7280"
LINE = "#E3E6EB"
GREEN = "#2F6B45"
AMBER = "#A86A12"
RED = "#9B3B3B"
SLATE = "#5A6B85"
SERIES = [INK, GOLD, "#7A8CA8", "#4E7D6A", "#8A6F9E"]

STATUS_STYLE = {  # label -> (text colour, background)
    "INVALID COMPARISON": (GREY, "#EEF0F3"),
    "INCONCLUSIVE": (GREY, "#EEF0F3"),
    "INSUFFICIENT EVIDENCE": (GREY, "#EEF0F3"),
    "EXPLORATORY ONLY — NOT EVIDENCE": (AMBER, "#FBF1E1"),
    "NORMAL": (GREEN, "#E7F1EA"),
    "EXPLAINABLE / RISK FLAG": (AMBER, "#FBF1E1"),
    "ALERT": (INK, "#E6EDF6"),
    "NO ALERT": (GREEN, "#E7F1EA"),
    "INSUFFICIENT HISTORY": (GREY, "#EEF0F3"),
    "NOT COMPUTABLE": (GREY, "#EEF0F3"),
    "NOT RUN": (GREY, "#EEF0F3"),
    "USER_UPLOAD": (INK, "#E6EDF6"),
    "HISTORICAL + USER_UPLOAD": (INK, "#E6EDF6"),
    "WATCH": (INK, "#E6EDF6"),
    "RESEARCH CANDIDATE": (GOLD, "#F6F0E1"),
    "CANDIDATE PASSED DEFINED TESTS": (GOLD, "#F6F0E1"),
    "REJECTED": (RED, "#F6E6E6"),
    "FAILED VALIDATION": (RED, "#F6E6E6"),
    "NO ROBUST IMPROVEMENT": (AMBER, "#FBF1E1"),
    "PASS": (GREEN, "#E7F1EA"),
    "FAIL": (RED, "#F6E6E6"),
    "VALID": (GREEN, "#E7F1EA"),
    "ADJUSTED": (INK, "#E6EDF6"),
    "INVALID": (RED, "#F6E6E6"),
    "UNVERIFIED": (AMBER, "#FBF1E1"),
    "VERIFIED": (GREEN, "#E7F1EA"),
    "SECONDARY": (AMBER, "#FBF1E1"),
    "ASSUMED": (RED, "#F6E6E6"),
    "OK": (GREEN, "#E7F1EA"),
    "CHECK": (AMBER, "#FBF1E1"),
    "NO DATA": (GREY, "#EEF0F3"),
    "SYNTHETIC": (AMBER, "#FBF1E1"),
    "HISTORICAL": (INK, "#E6EDF6"),
    "USER UPLOAD": (INK, "#E6EDF6"),
    "n/a": (GREY, "#EEF0F3"),
}

CSS = f"""
<style>
:root {{ --gold:{GOLD}; --ink:{INK}; --line:{LINE}; --grey:{GREY}; }}
html, body, [class*="css"] {{ font-feature-settings: "tnum" 1, "lnum" 1; }}
.block-container {{ max-width: 1320px; padding-top: 1.2rem; padding-bottom: 3rem; }}
h1, h2, h3, h4 {{ color: {NAVY}; letter-spacing: -0.01em; }}
section[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {LINE}; }}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1rem; }}
div[data-testid="stMetric"] {{ background:#fff; border:1px solid {LINE}; border-radius:6px; padding:10px 14px; }}
div[data-testid="stMetricValue"] {{ font-size: 1.35rem; font-variant-numeric: tabular-nums; color:{NAVY}; }}
div[data-testid="stMetricLabel"] p {{ color:{GREY}; font-size:0.78rem; text-transform:uppercase; letter-spacing:.04em; }}
div[data-testid="stExpander"] {{ background:#fff; border:1px solid {LINE}; border-radius:6px; }}
div[data-testid="stDataFrame"] {{ border:1px solid {LINE}; border-radius:6px; }}
.aur-brand {{ display:flex; align-items:center; gap:10px; margin-bottom:2px; }}
.aur-logo {{ width:26px; height:26px; border-radius:50%; background:{GOLD}; box-shadow: inset 0 0 0 4px #F6F0E1; }}
.aur-name {{ font-weight:700; font-size:1.25rem; color:{NAVY}; letter-spacing:.08em; }}
.aur-sub {{ color:{GREY}; font-size:.8rem; margin:0 0 14px 36px; }}
.aur-topbar {{ display:flex; flex-wrap:wrap; justify-content:space-between; align-items:flex-end; gap:12px;
              border-bottom:1px solid {LINE}; padding-bottom:10px; margin-bottom:14px; }}
.aur-title {{ font-size:1.6rem; font-weight:700; color:{NAVY}; margin:0; line-height:1.2; }}
.aur-subtitle {{ color:{GREY}; font-size:.92rem; margin-top:2px; }}
.aur-meta {{ display:flex; flex-wrap:wrap; gap:6px 14px; color:{GREY}; font-size:.82rem; }}
.aur-meta b {{ color:{NAVY}; font-weight:600; font-variant-numeric: tabular-nums; }}
.aur-badge {{ display:inline-block; padding:2px 9px; border-radius:4px; font-size:.75rem; font-weight:700;
             letter-spacing:.03em; white-space:nowrap; }}
.aur-card {{ background:#fff; border:1px solid {LINE}; border-radius:6px; padding:14px 16px; margin-bottom:12px; }}
.aur-card h4 {{ margin:0 0 6px 0; font-size:.95rem; text-transform:uppercase; letter-spacing:.05em; color:{SLATE}; }}
.aur-card.gold {{ border-left:3px solid {GOLD}; }}
.aur-chart-h {{ margin:6px 0 2px 0; }}
.aur-chart-h .t {{ font-weight:650; color:{NAVY}; font-size:1rem; }}
.aur-chart-h .d {{ color:{GREY}; font-size:.82rem; }}
.aur-warn {{ border:1px solid #EBD3A8; background:#FDF8EE; border-radius:6px; padding:9px 12px; margin:6px 0; font-size:.88rem; }}
.aur-warn.red {{ border-color:#E4C3C3; background:#FBF3F3; }}
.aur-warn.info {{ border-color:#CFDBEA; background:#F4F7FB; }}
.aur-warn .w-title {{ font-weight:650; color:{NAVY}; }}
.aur-warn .w-meta {{ color:{GREY}; font-size:.8rem; }}
.aur-empty {{ border:1px dashed #C9CED6; border-radius:6px; padding:22px; text-align:center; color:{GREY}; background:#fff; }}
.aur-tl {{ border-left:2px solid {LINE}; margin-left:6px; padding-left:16px; }}
.aur-tl-item {{ position:relative; padding:4px 0 10px 0; }}
.aur-tl-item:before {{ content:""; position:absolute; left:-23px; top:9px; width:10px; height:10px; border-radius:50%;
                      background:var(--dot); }}
.aur-tl-item .s {{ font-weight:650; color:{NAVY}; }}
.aur-tl-item .r {{ color:{GREY}; font-size:.85rem; }}
.aur-kv {{ display:grid; grid-template-columns: max-content 1fr; gap:3px 14px; font-size:.9rem; }}
.aur-kv .k {{ color:{GREY}; }}
.aur-kv .v {{ color:{NAVY}; font-variant-numeric: tabular-nums; }}
.aur-side-foot {{ font-size:.8rem; color:{GREY}; border-top:1px solid {LINE}; padding-top:10px; margin-top:14px; }}
.aur-side-foot b {{ color:{NAVY}; }}
@media (max-width: 700px) {{ .aur-title {{ font-size:1.3rem; }} .aur-sub {{ margin-left:0; }} }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 360, y_title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height, template="none", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=12, color=NAVY),
        margin=dict(l=56, r=18, t=28, b=40), colorway=SERIES, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=11)),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EEF0F3", gridwidth=1, linecolor=LINE, ticks="outside",
                     tickcolor=LINE, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#EEF0F3", gridwidth=1, linecolor=LINE, zeroline=False,
                     tickformat=",", title_text=y_title, title_font=dict(size=11, color=GREY))
    return fig
