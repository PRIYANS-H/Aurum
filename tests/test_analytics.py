import copy

import numpy as np
import pandas as pd
import pytest

from aurum.analytics import (FLAGS, GATE_FLAGS, add_metrics, analyse_pair, build_pair_series, flag_evidence,
                             rolling_stats)
from aurum.contracts import SpecBook
from tests.conftest import make_obs


def test_rolling_stats_exclude_current_value():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    st = rolling_stats(x, window=3, min_periods=3)
    # at t=4 the window is [2,3,4]; x_4=100 must not be in it
    assert st["mean"].iloc[4] == pytest.approx(3.0)
    assert st["std"].iloc[4] == pytest.approx(1.0)
    assert st["z"].iloc[4] == pytest.approx(97.0)
    assert np.isnan(st["z"].iloc[2])  # only 2 past values
    assert st["percentile"].iloc[4] == pytest.approx(100.0)


def test_zero_and_near_zero_variance_give_no_z():
    flat = pd.Series([5.0] * 40 + [5.0001])
    assert np.isnan(rolling_stats(flat, 40, 30)["z"].iloc[-1])
    near = pd.Series([1.0] * 40 + [1.0 + 1e-13] + [1.0] * 5 + [1.000001])   # audit case: z was 63 million
    assert np.isnan(rolling_stats(near, 40, 30)["z"].iloc[-1])
    normal = pd.Series(np.random.default_rng(0).normal(size=60))
    assert rolling_stats(normal, 40, 30)["z"].iloc[-1] == pytest.approx(
        (normal.iloc[-1] - normal.iloc[19:59].mean()) / normal.iloc[19:59].std())


def test_no_lookahead_future_changes_do_not_alter_past():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=200))
    base = rolling_stats(x, 20, 10)
    y = x.copy()
    y.iloc[150:] = 1e6  # wildly change the future
    changed = rolling_stats(y, 20, 10)
    pd.testing.assert_frame_equal(base.iloc[:150], changed.iloc[:150])


def _calendar_obs():
    days = pd.bdate_range("2024-01-01", periods=30)
    rows = []
    for i, d in enumerate(days):
        rows.append((d, "GOLDM", "2024-03-05", 70000 + i, 50))
        rows.append((d, "GOLDM", "2024-04-05", 70400 + i, 20))
        rows.append((d, "GOLDM", "2024-02-05", 69800 + i, 80))
    return make_obs(rows)


def test_calendar_pair_keeps_contract_identity_and_rolls(book):
    obs = _calendar_obs()
    pair = build_pair_series(obs, {"kind": "calendar", "product": "GOLDM", "near_rank": 1, "far_rank": 2},
                             book, roll_days=5)
    first, last = pair.iloc[0], pair.iloc[-1]
    assert first["a_contract_id"] == "GOLDM_20240205" and first["b_contract_id"] == "GOLDM_20240305"
    assert last["a_contract_id"] == "GOLDM_20240305" and last["b_contract_id"] == "GOLDM_20240405"
    assert pair["roll"].sum() == 1
    assert (pair["a_dte"] > 5).all()
    assert (pair["comparability"] == "VALID").all()


def test_implied_carry_formula_and_min_gap_guard(book):
    obs = _calendar_obs()
    pair = build_pair_series(obs, {"kind": "calendar", "product": "GOLDM", "near_rank": 1, "far_rank": 2},
                             book, roll_days=5)
    m = add_metrics(pair, "calendar").iloc[0]
    years = (pd.Timestamp("2024-03-05") - pd.Timestamp("2024-02-05")).days / 365
    assert m["implied_carry_pct"] == pytest.approx(100 * np.log(70000 / 69800) / years)
    assert m["baseline_metric"] == pytest.approx(200)
    assert np.isnan(add_metrics(pair, "calendar", min_carry_gap_days=60)["implied_carry_pct"]).all()


def test_cross_pair_requires_same_expiry_month(book, pairs_cfg, research):
    days = pd.bdate_range("2026-08-03", periods=10)
    rows = []
    for d in days:
        rows.append((d, "GOLDM", "2026-09-04", 140000, 50))
        rows.append((d, "GOLDPETAL", "2026-10-30", 14070, 50))  # different month -> no pair
    pair = build_pair_series(make_obs(rows), pairs_cfg["pairs"]["GOLDM_vs_GOLDPETAL"], book, 5)
    assert pair.empty


def test_unverified_rows_are_kept_out_of_validated_measures(book, pairs_cfg, research):
    """2024 Petal series: purity is not VERIFIED, so the purity-adjusted spread is exploratory only."""
    days = pd.bdate_range("2024-01-01", periods=10)
    rows = [(d, s, e, c, v) for d in days for s, e, c, v in
            [("GOLDM", "2024-02-05", 70000, 50), ("GOLDPETAL", "2024-02-29", 7028, 0)]]
    df = analyse_pair(make_obs(rows), pairs_cfg["pairs"]["GOLDM_vs_GOLDPETAL"], book, pairs_cfg["settings"], research)
    assert len(df) == 10 and (df["comparability"] == "UNVERIFIED").all()
    assert df["metric"].isna().all() and df["baseline_metric"].isna().all() and df["z"].isna().all()
    assert df["metric_raw"].notna().all()          # still available for clearly labelled exploration
    assert df["b_stale"].all() and df["expiry_gap_flag"].all()
    # raw columns: ignoring purity overstates the spread by ~0.40 points
    assert df["spread_pct_naive"].iloc[0] - df["spread_pct"].iloc[0] == pytest.approx(0.40, abs=0.01)


def test_missing_specification_blocks_rows_instead_of_crashing(specs_cfg, pairs_cfg, research):
    cfg = copy.deepcopy(specs_cfg)
    cfg["products"]["GOLDM"]["versions"] = [v for v in cfg["products"]["GOLDM"]["versions"]
                                            if v["applies_from_expiry"] != "2000-01-01"]
    book = SpecBook(cfg)                               # no GOLDM spec before Aug 2026
    df = analyse_pair(_calendar_obs(), pairs_cfg["pairs"]["GOLDM_CAL_1_2"], book, pairs_cfg["settings"], research)
    assert df.empty


def test_carry_adjustment_removes_expiry_gap(book, pairs_cfg, research):
    """GOLDM curve implies 8% p.a. carry; a Petal contract 26 days later priced exactly at fine-gold
    parity plus that carry should show ~0 carry-adjusted spread."""
    days = pd.bdate_range("2026-08-03", periods=10)
    c = 0.08
    near, nxt, petal_exp = pd.Timestamp("2026-09-04"), pd.Timestamp("2026-10-05"), pd.Timestamp("2026-09-30")
    f_near = 140000.0
    f_next = f_near * np.exp(c * (nxt - near).days / 365)
    petal_10g_995 = f_near * np.exp(c * (petal_exp - near).days / 365)
    petal = petal_10g_995 / 10 * 999 / 995
    rows = [(d, s, e, px, 50) for d in days for s, e, px in
            [("GOLDM", near, f_near), ("GOLDM", nxt, f_next), ("GOLDPETAL", petal_exp, petal)]]
    df = analyse_pair(make_obs(rows), pairs_cfg["pairs"]["GOLDM_vs_GOLDPETAL"], book,
                      pairs_cfg["settings"], research)
    assert (df["comparability"] == "ADJUSTED").all()
    assert df["carry_adjusted"].all() and not df["expiry_gap_flag"].any()
    assert df["carry_used_pct"].iloc[0] == pytest.approx(8.0, rel=1e-3)
    assert df["metric"].abs().max() < 0.01          # carry + purity fully explained
    assert df["spread_pct"].iloc[0] > 0.4           # raw purity-adjusted spread still shows the carry


def test_new_flags_unchanged_close_and_extreme_move(book, pairs_cfg, research):
    days = pd.bdate_range("2026-08-03", periods=4)
    near = [140000, 140000, 125000, 125500]          # day 2 unchanged, day 3 -10.7% (limit-size move)
    rows = [(d, "GOLDM", "2026-09-04", near[i], 50) for i, d in enumerate(days)]
    rows += [(d, "GOLDM", "2026-10-05", 141000 + i, 40) for i, d in enumerate(days)]
    df = analyse_pair(make_obs(rows), pairs_cfg["pairs"]["GOLDM_CAL_1_2"], book, pairs_cfg["settings"], research)
    assert df["a_unchanged"].tolist() == [False, True, False, False]
    assert df["extreme_move"].tolist() == [False, False, True, False]
    assert "−" in flag_evidence(df.iloc[2], "extreme_move") or "-10" in flag_evidence(df.iloc[2], "extreme_move")


def test_flag_registry_is_complete(book, pairs_cfg, research):
    for k, v in FLAGS.items():
        assert {"label", "gate", "trigger", "effect", "confirm"} <= set(v), k
    df = analyse_pair(_calendar_obs(), pairs_cfg["pairs"]["GOLDM_CAL_1_2"], book, pairs_cfg["settings"], research)
    assert set(FLAGS) <= set(df.columns)
    assert "oi_missing" not in GATE_FLAGS   # informational only


def test_prefix_invariance_end_to_end(book, pairs_cfg, research):
    """No look-ahead anywhere in the pipeline: analysing data up to T gives exactly the rows up to T
    of the full analysis."""
    rng = np.random.default_rng(3)
    days = pd.bdate_range("2026-08-03", periods=80)
    rows = []
    for i, d in enumerate(days):
        base = 140000 + 300 * np.sin(i / 7) + rng.normal(0, 50)
        rows += [(d, "GOLDM", "2026-12-04", round(base), 60), (d, "GOLDM", "2027-01-05", round(base * 1.008 + rng.normal(0, 20)), 30),
                 (d, "GOLDPETAL", "2026-12-31", round(base / 10 * 999 / 995 * 1.004 + rng.normal(0, 2)), 90)]
    obs = make_obs(rows)
    for pair in ("GOLDM_CAL_1_2", "GOLDM_vs_GOLDPETAL"):
        full = analyse_pair(obs, pairs_cfg["pairs"][pair], book, pairs_cfg["settings"], research)
        T = full["date"].iloc[50]
        cut = analyse_pair(obs[obs["date"] <= T], pairs_cfg["pairs"][pair], book, pairs_cfg["settings"], research)
        pd.testing.assert_frame_equal(full[full["date"] <= T].reset_index(drop=True), cut.reset_index(drop=True),
                                      check_like=True)
