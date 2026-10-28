"""Backtest engine: timing, fills, delivery rules, lots, per-leg accounting, costs, walk-forward."""

import numpy as np
import pandas as pd
import pytest

from aurum.backtest import (PriceBook, StrategyParams, fold_plan, leg_side_costs, simulate, trade_metrics,
                            walk_forward)

ZERO_COSTS = {"brokerage_per_order_rs": 0, "exchange_txn_pct": 0, "ctt_sell_pct": 0, "stamp_duty_buy_pct": 0,
              "sebi_fee_per_crore_rs": 0, "gst_pct": 0, "slippage_ticks_per_side": 0}
FAR = pd.Timestamp("2030-12-31")


def _setup(a_prices, b_prices, z, a_vol=None, b_vol=None, a_open=None, b_open=None, expiry=FAR,
           start="2024-01-01", a_fine=99.5, b_fine=99.5, a_mult=10.0, b_mult=10.0):
    dates = pd.bdate_range(start, periods=len(z))
    n = len(z)
    df = pd.DataFrame({
        "date": dates, "a_contract_id": "A", "b_contract_id": "B",
        "a_expiry": expiry, "b_expiry": expiry,
        "a_multiplier": a_mult, "b_multiplier": b_mult,
        "a_fine_grams_per_lot": a_fine, "b_fine_grams_per_lot": b_fine,
    })
    obs = pd.DataFrame({"date": list(dates) * 2, "contract_id": ["A"] * n + ["B"] * n,
                        "close": list(a_prices) + list(b_prices),
                        "open": list(a_open or a_prices) + list(b_open or b_prices),
                        "volume_lots": list(a_vol or [100] * n) + list(b_vol or [100] * n),
                        "expiry": [expiry] * (2 * n)})
    return df, PriceBook(obs), pd.Series(z, dtype=float)


def test_signal_at_t_fills_at_t_plus_1():
    # z spikes on day 1; day-1 prices are very favourable but must NOT be used
    df, pb, zs = _setup([100] * 5, [100, 150, 110, 100, 100], [0, 3.0, 2.5, 0.0, 0.0])
    t = simulate(df, zs, StrategyParams(2.0, 0.5, 15, 20), pb, ZERO_COSTS, 1, 1, use_gates=False)
    assert len(t) == 1
    tr = t.iloc[0]
    assert tr["direction"] == "short"
    assert tr["entry_date"] == df["date"][2] and tr["b_entry"] == 110   # filled day 2, not day 1 @150
    assert tr["exit_signal_date"] == df["date"][3] and tr["exit_date"] == df["date"][4]
    # hand calculation, short spread = short B (110 -> 100) and long A (flat), 1 lot each, multiplier 10:
    #   pnl_B = -1 * 1 * 10 * (100 - 110) = +100 ; pnl_A = +1 * 1 * 10 * (100 - 100) = 0
    assert tr["pnl_b"] == pytest.approx(100.0) and tr["pnl_a"] == pytest.approx(0.0)
    assert tr["gross_pnl"] == pytest.approx(tr["pnl_a"] + tr["pnl_b"])
    assert tr["net_pnl"] == pytest.approx(100.0)


def test_next_open_execution_uses_open_prices():
    df, pb, zs = _setup([100] * 5, [100, 150, 110, 100, 100], [0, 3.0, 2.5, 0.0, 0.0],
                        b_open=[100, 150, 108, 101, 99])
    t = simulate(df, zs, StrategyParams(2.0, 0.5, 15, 20), pb, ZERO_COSTS, 1, 1, use_gates=False,
                 execution="next_open")
    tr = t.iloc[0]
    assert tr["b_entry"] == 108 and tr["b_exit"] == 99
    assert tr["gross_pnl"] == pytest.approx(-1 * 10 * (99 - 108))


def test_entry_skipped_when_a_leg_does_not_trade_on_fill_day():
    df, pb, zs = _setup([100] * 6, [100, 150, 110, 100, 100, 100], [0, 3.0, 0, 0, 0, 0],
                        b_vol=[100, 100, 0, 100, 100, 100])
    t = simulate(df, zs, StrategyParams(2.0, 0.5, 15, 20), pb, ZERO_COSTS, 1, 1, use_gates=False)
    assert t.empty
    assert t.attrs["skipped"][0]["reason"] == "a leg did not trade on the fill day"


def test_exit_delayed_to_next_traded_day():
    df, pb, zs = _setup([100] * 7, [100, 150, 110, 105, 103, 100, 100], [0, 3.0, 2.5, 0.0, 0, 0, 0],
                        b_vol=[100, 100, 100, 100, 0, 100, 100])
    t = simulate(df, zs, StrategyParams(2.0, 0.5, 15, 20), pb, ZERO_COSTS, 1, 1, use_gates=False)
    tr = t.iloc[0]
    # exit signal on day 3, intended fill day 4 has zero volume on B -> filled day 5
    assert tr["exit_date"] == df["date"][5] and tr["b_exit"] == 100
    assert "exit delayed" in tr["note"]


def test_delivery_tender_period_rules():
    # 12 business days; contracts expire on the last one; tender period = last 3 trading days,
    # so the last day a position may be held/closed is the 4th-last date.
    z = [0, 0, 0, 3.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]
    dates = pd.bdate_range("2024-01-01", periods=len(z))
    df, pb, zs = _setup([100] * 12, [100] * 12, z, expiry=dates[-1])
    t = simulate(df, zs, StrategyParams(2.0, 0.5, 30, 20), pb, ZERO_COSTS, 1, 1, use_gates=False)
    deadline = dates[-4]
    assert pb.exit_deadline("A", dates[-1], 3) == deadline
    assert len(t) == 1 and t.iloc[0]["exit_reason"] == "delivery_exit"
    assert pd.Timestamp(t.iloc[0]["exit_date"]) <= deadline
    # a signal too close to the tender period is not entered at all
    z2 = [0] * 7 + [3.0] + [0] * 4
    df2, pb2, zs2 = _setup([100] * 12, [100] * 12, z2, expiry=dates[-1])
    t2 = simulate(df2, zs2, StrategyParams(2.0, 0.5, 30, 20), pb2, ZERO_COSTS, 1, 1, use_gates=False)
    assert t2.empty and "tender" in t2.attrs["skipped"][0]["reason"]


def test_gates_block_flagged_entries():
    df, pb, zs = _setup([100] * 5, [100, 150, 110, 100, 100], [0, 3.0, 2.5, 0, 0])
    df["b_stale"] = [False, True, True, False, False]
    t = simulate(df, zs, StrategyParams(2.0, 0.5, 15, 20), pb, ZERO_COSTS, 1, 1, use_gates=True)
    assert t.empty


def test_unusable_comparability_blocks_entries_even_for_baseline():
    df, pb, zs = _setup([100] * 5, [100, 150, 110, 100, 100], [0, 3.0, 2.5, 0, 0])
    df["comparability"] = "UNVERIFIED"
    assert simulate(df, zs, StrategyParams(2.0, 0.5, 15, 20), pb, ZERO_COSTS, 1, 1, use_gates=False).empty


def test_time_stop_and_forced_end():
    df, pb, zs = _setup([100] * 6, [100, 101, 102, 103, 104, 105], [0, 3, 3, 3, 3, 3])
    t = simulate(df, zs, StrategyParams(2.0, 0.5, 2, 20), pb, ZERO_COSTS, 1, 1, use_gates=False)
    assert t.iloc[0]["exit_reason"] == "time_stop"
    t2 = simulate(df, zs, StrategyParams(2.0, 0.5, 99, 20), pb, ZERO_COSTS, 1, 1, use_gates=False)
    assert t2.iloc[-1]["exit_reason"] == "forced_end"


def test_integer_lots_and_residual_exposure():
    # GOLDM (100 g x 995 = 99.5 g fine) vs GOLDGUINEA (8 g x 999 = 7.992 g fine)
    df, pb, zs = _setup([100] * 5, [100, 150, 110, 100, 100], [0, 3.0, 2.5, 0, 0], a_fine=99.5, b_fine=7.992,
                        b_mult=1.0)
    tr = simulate(df, zs, StrategyParams(2.0, 0.5, 15, 20), pb, ZERO_COSTS, 1, 1, use_gates=False).iloc[0]
    assert tr["lots_b"] == 12 and isinstance(tr["lots_b"], (int, np.integer))   # round(12.45) = 12, never 12.45
    assert tr["residual_fine_g"] == pytest.approx(12 * 7.992 - 99.5, abs=1e-3)


def test_costs_components_and_bases():
    cfg = {"brokerage_per_order_rs": 20, "exchange_txn_pct": 0.01, "ctt_sell_pct": 0.01, "stamp_duty_buy_pct": 0.002,
           "sebi_fee_per_crore_rs": 10, "gst_pct": 18, "slippage_ticks_per_side": 1}
    buy = leg_side_costs(70000, 1, 10, 1, "buy", cfg)     # turnover 7,00,000
    sell = leg_side_costs(70000, 1, 10, 1, "sell", cfg)
    exch, sebi = 70.0, 0.7
    assert buy["turnover"] == 700000
    # stamp duty only on the buy side, CTT only on the sell side, GST only on brokerage+exchange+SEBI
    assert buy["stamp"] == pytest.approx(14.0) and buy["ctt"] == 0
    assert sell["ctt"] == pytest.approx(70.0) and sell["stamp"] == 0
    assert buy["gst"] == pytest.approx(0.18 * (20 + exch + sebi))
    assert buy["fees"] == pytest.approx(20 + exch + sebi + 14.0 + 0.18 * (20 + exch + sebi))
    assert buy["slippage"] == 10 and buy["slip_unit"] == 10


def test_brokerage_percentage_cap_for_small_orders():
    cfg = {"brokerage_per_order_rs": 20, "brokerage_pct_cap": 0.03, "exchange_txn_pct": 0, "ctt_sell_pct": 0,
           "stamp_duty_buy_pct": 0, "sebi_fee_per_crore_rs": 0, "gst_pct": 0, "slippage_ticks_per_side": 0}
    small = leg_side_costs(14000, 1, 1, 1, "buy", cfg)      # 1 Gold Petal lot: 0.03% of 14,000 = 4.2
    big = leg_side_costs(140000, 1, 10, 1, "buy", cfg)      # 1 Gold Mini lot: 0.03% of 14 lakh = 420 -> 20
    assert small["brokerage"] == pytest.approx(4.2) and big["brokerage"] == 20


def test_costs_turn_gross_profit_into_net_loss():
    df, pb, zs = _setup([70000] * 5, [70000, 70010, 70010, 70005, 70005], [0, 3, 3, 0, 0])
    cfg = {"brokerage_per_order_rs": 20, "exchange_txn_pct": 0.0021, "ctt_sell_pct": 0.01, "stamp_duty_buy_pct": 0.002,
           "sebi_fee_per_crore_rs": 10, "gst_pct": 18, "slippage_ticks_per_side": 1}
    tr = simulate(df, zs, StrategyParams(2.0, 0.5, 15, 20), pb, cfg, 1, 1, use_gates=False).iloc[0]
    assert tr["gross_pnl"] > 0 and tr["net_pnl"] < 0
    assert tr["fees"] == pytest.approx(tr["fees_a"] + tr["fees_b"])
    assert tr["net_pnl"] == pytest.approx(tr["gross_pnl"] - tr["fees"] - tr["slippage"])


def test_capacity_warning_when_size_is_large_vs_volume():
    df, pb, zs = _setup([100] * 5, [100, 150, 110, 100, 100], [0, 3.0, 2.5, 0, 0], b_vol=[2] * 5)
    tr = simulate(df, zs, StrategyParams(2.0, 0.5, 15, 20), pb, ZERO_COSTS, 1, 1, use_gates=False).iloc[0]
    assert tr["participation"] == pytest.approx(0.5) and "capacity risk" in tr["note"]


def test_metrics_match_hand_calculation():
    trades = pd.DataFrame({"exit_date": pd.bdate_range("2024-01-01", periods=4),
                           "gross_pnl": [100, -50, 200, -10], "fees": [5] * 4, "slippage": [5] * 4,
                           "net_pnl": [90, -60, 190, -20], "turnover": [1000] * 4})
    m = trade_metrics(trades)
    # equity 90, 30, 220, 200 ; peak (floored at 0) 90, 90, 220, 220 -> drawdowns 0, -60, 0, -20
    assert m["trades"] == 4 and m["net_pnl"] == 200 and m["max_drawdown"] == -60
    assert m["win_rate"] == 0.5 and m["costs"] == 40 and m["turnover"] == 4000
    assert m["top3_share"] == pytest.approx((190 + 90 + 0) / 200)
    assert m["profit_factor"] == pytest.approx(280 / 80)
    assert trade_metrics(pd.DataFrame())["trades"] == 0


def test_fold_plan_is_chronological_and_holdout_untouched():
    p = fold_plan(500, 180, 60, 60)
    assert p["dev_n"] == 440 and p["holdout"] == (260, 440, 440, 500)
    prev_end = 0
    for tr_s, tr_e, te_s, te_e in p["folds"]:
        assert tr_e == te_s and tr_e - tr_s == 180      # train immediately precedes test
        assert te_s >= prev_end and te_e <= 440         # no overlap, never touches the holdout
        prev_end = te_e
    assert [f[3] - f[2] for f in p["folds"]] == [60, 60, 60, 60, 20]  # last partial fold kept (>= 20 rows)
    assert fold_plan(189, 180, 60, 60)["folds"] == []                 # infeasible: no fold fits
    assert fold_plan(189, 180, 60, 60)["holdout"] is None


def test_walk_forward_parameters_ignore_future_data():
    """Changing everything after fold 2's training window must not change fold 1-2 parameters."""
    rng = np.random.default_rng(1)
    n = 300
    dates = pd.bdate_range("2024-01-01", periods=n)
    metric = pd.Series(np.cumsum(rng.normal(size=n)) * 0.1 + rng.normal(size=n))
    df = pd.DataFrame({"date": dates, "a_contract_id": "A", "b_contract_id": "B", "a_expiry": FAR, "b_expiry": FAR,
                       "a_multiplier": 10.0, "b_multiplier": 10.0, "a_fine_grams_per_lot": 99.5,
                       "b_fine_grams_per_lot": 99.5, "metric": metric})
    a = 100000 + np.zeros(n)
    b = a * (1 + metric.to_numpy() / 1000)
    obs = pd.DataFrame({"date": list(dates) * 2, "contract_id": ["A"] * n + ["B"] * n,
                        "close": list(a) + list(b), "volume_lots": 100, "expiry": FAR})
    research = {"walk_forward": {"train_days": 100, "test_days": 50, "holdout_days": 0, "min_train_trades": 1,
                                 "grid": {"entry_z": [1.0, 2.0], "exit_z": [0.0, 0.5], "window": [20, 40]}},
                "strategy": {"entry_z": 2.0, "exit_z": 0.5, "max_hold_days": 10},
                "statistics": {"window": 20, "min_periods": 15}, "costs": ZERO_COSTS}
    r1 = walk_forward(df, obs, research, 1, 1)
    df2 = df.copy()
    df2.loc[200:, "metric"] = df2.loc[200:, "metric"] * -5 + 3     # rewrite everything after row 200
    r2 = walk_forward(df2, obs, research, 1, 1)
    cols = ["param_entry_z", "param_exit_z", "param_window"]
    # fold 1 trains on rows 0-99, fold 2 on rows 50-149: both before row 200
    pd.testing.assert_frame_equal(r1["folds"][cols].iloc[:2], r2["folds"][cols].iloc[:2])
