"""Session state, navigation and cached access to the analysis pipeline.

Every cached function takes `fp` = config.engine_fingerprint(): a hash of the engine code and all
config files. Any change to formulas, rules or assumptions therefore invalidates cached results
(including Streamlit's disk-persisted cache), so the dashboard can never show stale numbers.
"""

from __future__ import annotations

import copy
import json

import pandas as pd
import streamlit as st

from aurum.analytics import analyse_pair
from aurum.config import engine_fingerprint
from aurum.ingestion import Store
from aurum.pipeline import Context, data_quality, run_validation

PAGES = {
    "Relative Value Explorer": "Explore contract relationships, historical deviations and data quality.",
    "Signal Evidence": "Investigate unusual contract relationships before drawing conclusions.",
    "Backtesting Lab": "Evaluate a strategy against explicit assumptions and out-of-sample evidence.",
    "Research & Methodology": "Sources, assumptions, formulas and validation standards behind AURUM.",
}
DATASETS = {"REAL": "MCX bhavcopy", "SYNTHETIC": "Synthetic demo"}


@st.cache_resource
def get_ctx() -> Context:
    return Context.load()


def fp() -> str:
    return engine_fingerprint()


def init_state() -> None:
    ctx = get_ctx()
    defaults = {
        "page": "Relative Value Explorer",
        "dataset": "REAL" if not Store("REAL").observations().empty else "SYNTHETIC",
        "pair": next(iter(ctx.pairs)),
        "window": ctx.research["statistics"]["window"],
        "as_of": None,
        "date_range": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def go_to(page: str, **updates) -> None:
    """Button callback: change page and carry context (pair, dates, window) across pages."""
    st.session_state["page"] = page
    for k, v in updates.items():
        st.session_state[k] = v


def manifest_key(mode: str) -> str:
    return json.dumps([m["sha256"] for m in Store(mode).manifest()])


@st.cache_data(show_spinner=False)
def load_obs(mode: str, key: str) -> pd.DataFrame:
    return Store(mode).observations()


@st.cache_data(show_spinner=False)
def load_quality(mode: str, key: str, fingerprint: str) -> dict:
    return data_quality(Store(mode), load_obs(mode, key), get_ctx().research)


def research_with(overrides: dict | None = None) -> dict:
    r = copy.deepcopy(get_ctx().research)
    for section, vals in (overrides or {}).items():
        r[section].update(vals)
    return r


def settings_with(roll_days: int | None = None) -> dict:
    s = dict(get_ctx().settings)
    if roll_days is not None:
        s["roll_days_before_expiry"] = int(roll_days)
    return s


@st.cache_data(show_spinner=False)
def load_pair(mode: str, key: str, pair: str, window: int, research_json: str | None = None,
              settings_json: str | None = None, fixed: tuple[str, str] | None = None,
              fingerprint: str = "") -> pd.DataFrame:
    """Analysis table for a registered pair rule, or for two fixed contracts under that rule."""
    ctx = get_ctx()
    research = json.loads(research_json) if research_json else ctx.research
    settings = json.loads(settings_json) if settings_json else ctx.settings
    cfg = dict(ctx.pairs[pair], **({"fixed": list(fixed)} if fixed else {}))
    return analyse_pair(load_obs(mode, key), cfg, ctx.book, settings, research, window)


def pair_df(mode: str, key: str, pair: str, window: int | None = None, fixed=None) -> pd.DataFrame:
    """Convenience wrapper with the default configuration and the engine fingerprint."""
    ctx = get_ctx()
    return load_pair(mode, key, pair, int(window or ctx.research["statistics"]["window"]), fixed=fixed,
                     fingerprint=fp())


@st.cache_data(show_spinner=False, persist="disk")
def load_validation(mode: str, key: str, pair: str, research_json: str, settings_json: str,
                    fingerprint: str, exploratory: bool = False):
    ctx = get_ctx()
    research = json.loads(research_json)
    df = load_pair(mode, key, pair, research["statistics"]["window"], research_json, settings_json,
                   fingerprint=fingerprint)
    if df.empty:
        return None
    return run_validation(ctx, load_obs(mode, key), df, ctx.pairs[pair], research, exploratory)


def default_validation(mode: str, key: str, pair: str):
    ctx = get_ctx()
    return load_validation(mode, key, pair, json.dumps(ctx.research, sort_keys=True),
                           json.dumps(ctx.settings, sort_keys=True), fp())


def data_mode_label(mode: str) -> str:
    """SYNTHETIC, HISTORICAL (files placed locally by the user), USER_UPLOAD (dashboard uploads)."""
    if mode == "SYNTHETIC":
        return "SYNTHETIC"
    origins = {m.get("origin", "file") for m in Store(mode).manifest()}
    if origins == {"upload"}:
        return "USER_UPLOAD"
    return "HISTORICAL + USER_UPLOAD" if "upload" in origins else "HISTORICAL"
