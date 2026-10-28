"""Render the dashboard headlessly: every page in both datasets, plus behaviour checks that compare
what the UI shows with what the backend computes."""

import html

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from aurum.config import ROOT
from aurum.ingestion import Store

PAGES = ["Relative Value Explorer", "Signal Evidence", "Backtesting Lab", "Research & Methodology"]
HAVE = {m: not Store(m).observations().empty for m in ("SYNTHETIC", "REAL")}
need_syn = pytest.mark.skipif(not HAVE["SYNTHETIC"], reason="run `python -m aurum.cli sample` first")
need_real = pytest.mark.skipif(not HAVE["REAL"], reason="no REAL data ingested locally")


def _app(dataset: str, page: str) -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    at.session_state["dataset"] = dataset
    at.session_state["page"] = page
    at.run()
    return at


def _ok(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]


def _text(at: AppTest) -> str:
    return html.unescape(" ".join(m.value for m in at.markdown))


@pytest.mark.parametrize("dataset", ["SYNTHETIC", "REAL"])
@pytest.mark.parametrize("page", PAGES)
def test_page_renders(dataset, page):
    if not HAVE[dataset]:
        pytest.skip(f"no {dataset} data ingested locally")
    at = _app(dataset, page)
    _ok(at)
    text = _text(at)
    assert page in text
    assert ("SYNTHETIC DATA" in text) == (dataset == "SYNTHETIC")  # banner only on synthetic


@need_syn
def test_explorer_metrics_equal_backend_values():
    from aurum.analytics import analyse_pair
    from aurum.pipeline import Context
    at = _app("SYNTHETIC", "Relative Value Explorer")
    _ok(at)
    ctx = Context.load()
    df = analyse_pair(Store("SYNTHETIC").observations(), ctx.pairs[at.session_state["pair"]], ctx.book,
                      ctx.settings, ctx.research)
    last = df.iloc[-1]
    shown = {m.label: m.value for m in at.metric}
    assert shown["A price"] == f"{last['a_norm']:,.0f}"
    assert shown["B price"] == f"{last['b_norm']:,.0f}"
    assert shown["Z-score"] == ("N/A" if pd.isna(last["z"]) else f"{last['z']:,.2f}")


@need_syn
def test_explorer_to_evidence_keeps_context():
    at = _app("SYNTHETIC", "Relative Value Explorer")
    at.selectbox(key="exp_prodA").set_value("GOLDM")
    at.selectbox(key="exp_prodB").set_value("GOLDPETAL")
    at.run()
    _ok(at)
    assert at.session_state["pair"] == "GOLDM_vs_GOLDPETAL"
    # synthetic 2022-24 Petal series have no verified purity -> shown as exploratory, not validated
    assert "not comparable" in _text(at)
    next(b for b in at.button if "Signal Evidence" in b.label).click()
    at.run()
    _ok(at)
    assert at.session_state["page"] == "Signal Evidence"
    assert at.session_state["pair"] == "GOLDM_vs_GOLDPETAL"


@need_syn
def test_unregistered_pair_is_refused():
    at = _app("SYNTHETIC", "Relative Value Explorer")
    at.selectbox(key="exp_prodA").set_value("GOLD")
    at.selectbox(key="exp_prodB").set_value("GOLDPETAL")
    at.run()
    _ok(at)
    assert "No validated comparison rule" in _text(at)


@need_syn
def test_queue_filter_changes_rows():
    at = _app("SYNTHETIC", "Signal Evidence")
    _ok(at)
    q = at.dataframe[0].value
    q = getattr(q, "data", q)
    pairs = sorted(q["Pair"].unique())
    if len(pairs) < 2:
        pytest.skip("queue has fewer than two pairs")
    at.multiselect(key="ev_fpair").set_value([pairs[0]])
    at.run()
    _ok(at)
    f = at.dataframe[0].value
    f = getattr(f, "data", f)
    assert set(f["Pair"]) == {pairs[0]} and len(f) < len(q)
    assert {"Alert", "Validation", "Conclusion"} <= set(f.columns)


@need_syn
def test_backtest_runs_and_concludes():
    at = _app("SYNTHETIC", "Backtesting Lab")
    next(b for b in at.button if "run walk-forward" in b.label).click()
    at.run()
    _ok(at)
    text = _text(at)
    assert any(c in text for c in ("CANDIDATE PASSED DEFINED TESTS", "NO ROBUST IMPROVEMENT",
                                   "FAILED VALIDATION", "INSUFFICIENT EVIDENCE"))
    assert "Predefined acceptance protocol" in text


@need_real
def test_infeasible_backtest_on_real_history_is_not_run():
    at = _app("REAL", "Backtesting Lab")
    _ok(at)
    assert "cannot hold one 180-row training window" in _text(at)
    next(b for b in at.button if "run walk-forward" in b.label).click()
    at.run()
    _ok(at)
    text = _text(at)
    assert "Not run" in text and "Nothing was computed" in text
    assert "Performance summary" not in text                     # no metrics invented


@need_syn
def test_invalid_backtest_configuration_is_explained():
    at = _app("SYNTHETIC", "Backtesting Lab")
    at.multiselect[0].set_value([1.0])          # entry thresholds (options 1.0-3.0)
    at.multiselect[1].set_value([1.0])          # exit thresholds -> entry is not strictly above exit
    next(b for b in at.button if "run walk-forward" in b.label).click()
    at.run()
    _ok(at)
    assert "Every entry threshold must be larger than every exit threshold" in _text(at)
