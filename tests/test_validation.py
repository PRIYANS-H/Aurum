"""The single acceptance protocol: feasibility, cost stress, integrity and verdict mapping."""

import copy

import pandas as pd
import pytest

from aurum.backtest import trade_metrics
from aurum.validation import cost_stress, evaluate_protocol, feasibility


def test_feasibility_on_short_real_history(research):
    f = feasibility(189, research)                    # 189 real trading days, default 180/60 + 60 holdout
    assert not f["runnable"] and f["folds"] == 0
    assert f["rows_needed"] == 180 + 3 * 60 + 60
    assert any("cannot hold one 180-row training window" in m for m in f["messages"])


def test_feasibility_sufficient_with_enough_history(research):
    f = feasibility(500, research)
    assert f["runnable"] and f["sufficient"] and f["holdout"] and f["folds"] >= 3


def test_short_windows_only_in_exploratory_mode(research):
    r = copy.deepcopy(research)
    r["walk_forward"].update(train_days=40, test_days=20, holdout_days=0)
    assert not feasibility(189, r)["runnable"]                        # protocol mode refuses short windows
    f = feasibility(189, r, exploratory=True)
    assert f["runnable"] and not f["sufficient"]                      # runs, but can never be sufficient


def _trades(nets, gross=None, fold=1):
    gross = gross or nets
    return pd.DataFrame({"exit_date": pd.bdate_range("2024-01-01", periods=len(nets)), "gross_pnl": gross,
                         "fees": [10.0] * len(nets), "slippage": [2.0] * len(nets), "slip_unit": [2.0] * len(nets),
                         "net_pnl": nets, "turnover": [1e6] * len(nets), "fold": fold,
                         "signal_date": pd.bdate_range("2023-12-01", periods=len(nets)),
                         "entry_date": pd.bdate_range("2023-12-04", periods=len(nets))})


def test_cost_stress_reprices_fixed_trades(research):
    t = _trades([100.0, -20.0], gross=[112.0, -8.0])
    s = cost_stress(t, research).set_index("scenario")
    # stress: gross - fees*1.5 - slip_unit*5 = (112-15-10) + (-8-15-10) = 87 - 33 = 54
    assert s.loc["stress", "net_pnl"] == pytest.approx(54.0)
    assert s.loc["no slippage", "net_pnl"] == pytest.approx((112 - 10) + (-8 - 10))


def _val(cand_fold_nets, base_fold_nets, trades_per_fold=4, sens_positive=True, holdout_net=50.0,
         integ_ok=True, stress_net=100.0):
    def run(fold_nets):
        trades = pd.concat([_trades([n / trades_per_fold] * trades_per_fold, fold=i + 1)
                            for i, n in enumerate(fold_nets)], ignore_index=True)
        folds = pd.DataFrame({"fold": range(1, len(fold_nets) + 1), "oos_net_pnl": fold_nets,
                              "oos_trades": [trades_per_fold] * len(fold_nets),
                              "train_end": pd.bdate_range("2024-01-01", periods=len(fold_nets), freq="60B"),
                              "test_start": pd.bdate_range("2024-01-02", periods=len(fold_nets), freq="60B"),
                              "test_end": pd.bdate_range("2024-03-01", periods=len(fold_nets), freq="60B")})
        return {"folds": folds, "trades": trades, "summary": trade_metrics(trades), "holdout": None}
    cand, base = run(cand_fold_nets), run(base_fold_nets)
    if holdout_net is not None:
        ht = _trades([holdout_net])
        cand["holdout"] = {"start": pd.Timestamp("2026-01-01"), "summary": trade_metrics(ht), "trades": ht}
    return {"candidate": cand, "baseline": base,
            "sensitivity": pd.DataFrame({"net_pnl": [10.0, 5.0] if sens_positive else [-1.0, -2.0], "trades": [3, 3]}),
            "cost_stress": pd.DataFrame({"scenario": ["stress"], "net_pnl": [stress_net]}),
            "integrity": {"candidate": {"same_bar_fills": 0, "tender_violations": 0 if integ_ok else 2,
                                        "stale_fills": 0}, "fold_order_ok": True, "holdout_after_folds": True}}


def test_protocol_pass(research):
    p = evaluate_protocol(_val([100, 80, 120], [10, 20, 30]), research)
    assert p["verdict"] == "PASS" and p["conclusion"] == "CANDIDATE PASSED DEFINED TESTS"


def test_protocol_no_robust_improvement_when_baseline_is_better(research):
    p = evaluate_protocol(_val([100, 80, 120], [200, 200, 200]), research)
    assert p["verdict"] == "FAIL" and p["conclusion"] == "NO ROBUST IMPROVEMENT"


def test_protocol_fails_on_performance_cost_stress_or_integrity(research):
    assert evaluate_protocol(_val([-100, -80, 20], [-200, -200, -200]), research)["conclusion"] == "FAILED VALIDATION"
    assert evaluate_protocol(_val([100, 80, 120], [1, 1, 1], stress_net=-5), research)["conclusion"] == "FAILED VALIDATION"
    assert evaluate_protocol(_val([100, 80, 120], [1, 1, 1], integ_ok=False), research)["conclusion"] == "FAILED VALIDATION"
    assert evaluate_protocol(_val([100, 80, 120], [1, 1, 1], holdout_net=-30), research)["conclusion"] == "FAILED VALIDATION"


def test_protocol_insufficient_and_exploratory(research):
    p = evaluate_protocol(_val([100, 80], [1, 1]), research)           # 2 folds < 3
    assert p["verdict"] == "INCONCLUSIVE" and p["conclusion"] == "INSUFFICIENT EVIDENCE"
    p2 = evaluate_protocol(_val([100, 80, 120], [1, 1, 1]), research, exploratory=True)
    assert p2["verdict"] == "INCONCLUSIVE" and p2["conclusion"].startswith("EXPLORATORY")


def test_protocol_thresholds_come_from_config(research):
    r = copy.deepcopy(research)
    r["validation_pass"]["min_oos_trades"] = 1000
    assert evaluate_protocol(_val([100, 80, 120], [1, 1, 1]), r)["verdict"] == "INCONCLUSIVE"
