"""Spread backtester: explicit timing, integer lots, per-leg accounting, costs, walk-forward.

Timing (no same-bar fills)
- Signals are evaluated with the z-score known at the close of row t.
- Entries and exits are FILLED on the next trading row (t+1) at that day's close or open
  (`strategy.execution`), using the prices of the specific contracts held.
- Both legs must have TRADED (volume > 0) on the fill day. On real MCX data a zero-volume close
  just repeats an old price, so it is not a fill. Entries without a tradable fill day are skipped
  (recorded in attrs["skipped"]); exits are delayed to the next day both legs traded.
- At the end of an evaluation window open positions are marked at that window's last close
  ("forced_end"); this is an accounting convention and is reported as such.

Delivery
- MCX marks every open position for compulsory delivery and adds tender-period margin during the
  last `tender_days` trading days of a contract. Positions are never entered unless they can be
  closed before that period, and are always exited by the last trading day before it. The
  trading calendar used for this is the exchange calendar (published in advance), not price data.

Position
- Spread = B - A (normalised). "short" = sell B / buy A, "long" = buy B / sell A.
- Leg A trades `lots_a`; leg B lots are the integer closest to equal fine-gold grams (never fractional).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import product as iproduct

import numpy as np
import pandas as pd

from .analytics import GATE_FLAGS, rolling_stats


@dataclass(frozen=True)
class StrategyParams:
    entry_z: float = 2.0
    exit_z: float = 0.5
    max_hold_days: int = 15
    window: int = 40


# ---------------------------------------------------------------------------- costs
def leg_side_costs(price: float, lots: int, multiplier: float, tick: float, side: str, cfg: dict) -> dict:
    """Estimated costs (Rs) for one leg on one side ('buy' or 'sell').

    turnover  = price × multiplier × lots
    brokerage = min(flat per order, brokerage_pct_cap % × turnover)   (cap optional)
    exchange  = exchange_txn_pct % × turnover         SEBI = turnover / 1 crore × fee
    CTT       = ctt_sell_pct % × turnover on SELL      stamp = stamp_duty_buy_pct % × turnover on BUY
    GST       = gst_pct % × (brokerage + exchange + SEBI)   — not on CTT or stamp duty
    slippage  = ticks × tick × multiplier × lots (assumed; reported separately)
    """
    turnover = price * multiplier * lots
    brokerage = cfg["brokerage_per_order_rs"]
    if cfg.get("brokerage_pct_cap"):
        brokerage = min(brokerage, turnover * cfg["brokerage_pct_cap"] / 100.0)
    exchange = turnover * cfg["exchange_txn_pct"] / 100.0
    sebi = turnover / 1e7 * cfg["sebi_fee_per_crore_rs"]
    ctt = turnover * cfg["ctt_sell_pct"] / 100.0 if side == "sell" else 0.0
    stamp = turnover * cfg["stamp_duty_buy_pct"] / 100.0 if side == "buy" else 0.0
    gst = (brokerage + exchange + sebi) * cfg["gst_pct"] / 100.0
    slip_unit = tick * multiplier * lots           # Rs per tick of slippage
    slippage = cfg["slippage_ticks_per_side"] * slip_unit
    fees = brokerage + exchange + sebi + ctt + stamp + gst
    return {"fees": fees, "slippage": slippage, "slip_unit": slip_unit, "turnover": turnover,
            "brokerage": brokerage, "exchange": exchange, "sebi": sebi, "ctt": ctt, "stamp": stamp, "gst": gst}


# ---------------------------------------------------------------------------- market data access
class PriceBook:
    """Prices, volumes and the trading calendar from the full observation panel."""

    def __init__(self, obs: pd.DataFrame):
        o = obs.copy()
        o["date"] = pd.to_datetime(o["date"])
        self.has_volume = "volume_lots" in o.columns
        opens = o["open"] if "open" in o.columns else pd.Series(np.nan, index=o.index)
        vols = o["volume_lots"] if self.has_volume else pd.Series(np.nan, index=o.index)
        self._row = {(d, c): (px, op, v) for d, c, px, op, v in
                     zip(o["date"], o["contract_id"], o["close"], opens, vols)}
        self._dates = {c: sorted(g["date"]) for c, g in o.groupby("contract_id")}
        self.calendar = sorted(o["date"].unique())
        self._expiry = ({c: pd.Timestamp(e) for c, e in zip(o["contract_id"], o["expiry"])}
                        if "expiry" in o.columns else {})
        self._deadline: dict = {}

    def price(self, date, contract_id, field: str = "close"):
        r = self._row.get((pd.Timestamp(date), contract_id))
        if r is None:
            return None
        v = r[0] if field == "close" else r[1]
        return None if v is None or pd.isna(v) or v <= 0 else float(v)

    def volume(self, date, contract_id):
        r = self._row.get((pd.Timestamp(date), contract_id))
        return None if r is None or pd.isna(r[2]) else float(r[2])

    def tradable(self, date, contract_id, field: str = "close") -> bool:
        if self.price(date, contract_id, field) is None:
            return False
        if not self.has_volume:
            return True
        v = self.volume(date, contract_id)
        return v is not None and v > 0

    def exit_deadline(self, contract_id, expiry=None, tender_days: int = 3) -> pd.Timestamp:
        """Last trading day on which a position in this contract may still be held / closed:
        the trading day before the delivery tender period (last `tender_days` trading days)."""
        key = (contract_id, tender_days)
        if key in self._deadline:
            return self._deadline[key]
        exp = pd.Timestamp(expiry) if expiry is not None else self._expiry.get(contract_id)
        if exp is None or pd.isna(exp):
            self._deadline[key] = pd.Timestamp.max
            return pd.Timestamp.max
        cal = [d for d in self.calendar if d <= exp]
        if self.calendar and self.calendar[-1] >= exp and len(cal) > tender_days:
            deadline = cal[-tender_days - 1]
        else:  # calendar does not reach expiry: approximate with business days
            deadline = pd.bdate_range(end=exp, periods=tender_days + 1)[0]
        self._deadline[key] = pd.Timestamp(deadline)
        return self._deadline[key]

    def first_tradable(self, start_date, a_id, b_id, field="close", until=None):
        """First date >= start_date (and <= until) on which BOTH contracts traded."""
        for d in self.calendar:
            if d < pd.Timestamp(start_date):
                continue
            if until is not None and d > pd.Timestamp(until):
                break
            if self.tradable(d, a_id, field) and self.tradable(d, b_id, field):
                return d
        return None


# ---------------------------------------------------------------------------- simulation
def simulate(df: pd.DataFrame, z: pd.Series, params: StrategyParams, prices: PriceBook,
             cost_cfg: dict, tick_a: float, tick_b: float, lots_a: int = 1,
             start: int = 0, end: int | None = None, use_gates: bool = True,
             execution: str = "next_close", tender_days: int = 3,
             max_participation: float = 0.10) -> pd.DataFrame:
    """Simulate trades whose entry SIGNAL falls in rows [start, end). One row per trade.

    attrs["skipped"] lists entry signals that could not be filled (and why)."""
    end = len(df) if end is None else end
    field = "open" if execution == "next_open" else "close"
    dates = [pd.Timestamp(d) for d in df["date"]]
    zarr = z.to_numpy(dtype=float)
    gate_cols = [g for g in GATE_FLAGS if g in df.columns]
    gated = df[gate_cols].fillna(False).astype(bool).any(axis=1).to_numpy() if gate_cols else np.zeros(len(df), bool)
    usable = (df["comparability"].isin(["VALID", "ADJUSTED"]).to_numpy() if "comparability" in df.columns
              else np.ones(len(df), bool))
    trades, skipped, pos = [], [], None
    last = end - 1

    def nth(i):
        return dates[i] if i < len(dates) else None

    for i in range(start, end):
        zt, today = zarr[i], dates[i]

        if pos is not None:
            held = i - pos["entry_row"]
            deadline = pos["deadline"]
            nxt, after = nth(i + 1), nth(i + 2)
            reason = None
            if i == last:
                reason = "forced_end"
            elif nxt is not None and (nxt > deadline or (after is not None and after > deadline)):
                reason = "delivery_exit"   # last fill opportunity before the tender period
            elif pos["dir"] == "short" and not np.isnan(zt) and zt <= params.exit_z:
                reason = "mean_reverted"
            elif pos["dir"] == "long" and not np.isnan(zt) and zt >= -params.exit_z:
                reason = "mean_reverted"
            elif held >= params.max_hold_days:
                reason = "time_stop"
            if reason:
                trades.append(_close(pos, reason, today, prices, cost_cfg, tick_a, tick_b,
                                     dates[i] if reason == "forced_end" else nxt, field, max_participation))
                pos = None
                continue

        if pos is None and i + 1 < last and not np.isnan(zt) and abs(zt) >= params.entry_z:
            if not usable[i]:
                continue
            if use_gates and gated[i]:
                continue
            row = df.iloc[i]
            a_id, b_id = row["a_contract_id"], row["b_contract_id"]
            deadline = min(prices.exit_deadline(a_id, row["a_expiry"], tender_days),
                           prices.exit_deadline(b_id, row["b_expiry"], tender_days))
            fill_day, after = nth(i + 1), nth(i + 2)
            if after is None or after > deadline:
                skipped.append({"signal_date": today, "reason": "too close to the delivery tender period"})
                continue
            if not (prices.tradable(fill_day, a_id, field) and prices.tradable(fill_day, b_id, field)):
                skipped.append({"signal_date": today, "reason": "a leg did not trade on the fill day"})
                continue
            lots_b = max(1, int(round(lots_a * row["a_fine_grams_per_lot"] / row["b_fine_grams_per_lot"])))
            pos = {"dir": "short" if zt > 0 else "long", "entry_row": i + 1, "signal_date": today,
                   "entry_z": zt, "a_id": a_id, "b_id": b_id, "deadline": deadline,
                   "lots_a": int(lots_a), "lots_b": lots_b,
                   "a_mult": row["a_multiplier"], "b_mult": row["b_multiplier"],
                   "residual_fine_g": lots_b * row["b_fine_grams_per_lot"] - lots_a * row["a_fine_grams_per_lot"],
                   "entry_date": fill_day, "a_entry": prices.price(fill_day, a_id, field),
                   "b_entry": prices.price(fill_day, b_id, field),
                   "signal_flags": ", ".join(g for g in gate_cols if bool(row.get(g, False)))}

    out = pd.DataFrame(trades)
    out.attrs["skipped"] = skipped
    return out


def _close(pos: dict, reason: str, signal_date, prices: PriceBook, cost_cfg, tick_a, tick_b,
           intended, field: str, max_participation: float) -> dict:
    notes = []
    a_id, b_id = pos["a_id"], pos["b_id"]
    if reason == "forced_end":
        exit_date = pd.Timestamp(intended)
        a_exit = prices.price(exit_date, a_id, "close") or prices.price(exit_date, a_id, field)
        b_exit = prices.price(exit_date, b_id, "close") or prices.price(exit_date, b_id, field)
        notes.append("marked at window-end close (accounting)")
    else:
        exit_date = prices.first_tradable(intended, a_id, b_id, field)
        if exit_date is None:   # never traded again: last available price, clearly noted
            exit_date = pd.Timestamp(intended)
            notes.append("exit at stale price: legs did not trade again")
        elif exit_date != pd.Timestamp(intended):
            notes.append(f"exit delayed to {exit_date:%Y-%m-%d}: a leg did not trade on the intended day")
        a_exit = prices.price(exit_date, a_id, field) or prices.price(exit_date, a_id, "close")
        b_exit = prices.price(exit_date, b_id, field) or prices.price(exit_date, b_id, "close")
    if a_exit is None or b_exit is None:   # last resort: most recent stored price before exit
        a_exit = a_exit or pos["a_entry"]
        b_exit = b_exit or pos["b_entry"]
        notes.append("missing exit price: carried entry price")

    # long spread = long B, short A ; short spread = short B, long A
    sb = 1 if pos["dir"] == "long" else -1
    sa = -sb
    pnl_a = sa * pos["lots_a"] * pos["a_mult"] * (a_exit - pos["a_entry"])
    pnl_b = sb * pos["lots_b"] * pos["b_mult"] * (b_exit - pos["b_entry"])

    def side(sign, opening):  # buy when opening a long leg or closing a short leg
        return "buy" if (sign > 0) == opening else "sell"

    fees = {"a": 0.0, "b": 0.0}
    slip_unit = turnover = 0.0
    for leg, px, lots, mult, tick, sign, opening in [
        ("a", pos["a_entry"], pos["lots_a"], pos["a_mult"], tick_a, sa, True),
        ("b", pos["b_entry"], pos["lots_b"], pos["b_mult"], tick_b, sb, True),
        ("a", a_exit, pos["lots_a"], pos["a_mult"], tick_a, sa, False),
        ("b", b_exit, pos["lots_b"], pos["b_mult"], tick_b, sb, False),
    ]:
        c = leg_side_costs(px, lots, mult, tick, side(sign, opening), cost_cfg)
        fees[leg] += c["fees"]
        slip_unit += c["slip_unit"]
        turnover += c["turnover"]
    slippage = cost_cfg["slippage_ticks_per_side"] * slip_unit

    def part(d, cid, lots):
        v = prices.volume(d, cid)
        return lots / v if v and v > 0 else np.nan
    participation = np.nanmax([part(pos["entry_date"], a_id, pos["lots_a"]), part(pos["entry_date"], b_id, pos["lots_b"]),
                               part(exit_date, a_id, pos["lots_a"]), part(exit_date, b_id, pos["lots_b"]), 0.0])
    if participation > max_participation:
        notes.append(f"size is {participation:.0%} of a leg's daily volume (capacity risk)")
    gross = pnl_a + pnl_b
    return {
        "direction": pos["dir"], "signal_date": pos["signal_date"], "entry_date": pos["entry_date"],
        "exit_signal_date": signal_date, "exit_date": exit_date, "exit_reason": reason,
        "entry_z": round(float(pos["entry_z"]), 3),
        "a_contract": a_id, "b_contract": b_id, "lots_a": pos["lots_a"], "lots_b": pos["lots_b"],
        "a_entry": pos["a_entry"], "b_entry": pos["b_entry"], "a_exit": a_exit, "b_exit": b_exit,
        "pnl_a": pnl_a, "pnl_b": pnl_b, "fees_a": fees["a"], "fees_b": fees["b"],
        "residual_fine_g": round(float(pos["residual_fine_g"]), 3),
        "gross_pnl": gross, "fees": fees["a"] + fees["b"], "slippage": slippage, "slip_unit": slip_unit,
        "net_pnl": gross - fees["a"] - fees["b"] - slippage,
        "turnover": turnover, "participation": float(participation), "exit_deadline": pos["deadline"],
        "signal_flags": pos.get("signal_flags", ""), "note": "; ".join(notes),
    }


# ---------------------------------------------------------------------------- metrics
def trade_metrics(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {"trades": 0, "gross_pnl": 0.0, "costs": 0.0, "net_pnl": 0.0, "win_rate": np.nan,
                "avg_net": np.nan, "max_drawdown": 0.0, "top3_share": np.nan, "profit_factor": np.nan,
                "turnover": 0.0}
    t = trades.sort_values("exit_date")
    net = t["net_pnl"]
    equity = net.cumsum()
    dd = (equity - equity.cummax().clip(lower=0)).min()
    total = float(net.sum())
    wins, losses = net[net > 0].sum(), -net[net < 0].sum()
    top3 = float(net.nlargest(3).clip(lower=0).sum() / total) if total > 0 else np.nan
    return {
        "trades": int(len(t)),
        "gross_pnl": float(t["gross_pnl"].sum()),
        "costs": float((t["fees"] + t["slippage"]).sum()),
        "net_pnl": total,
        "win_rate": float((net > 0).mean()),
        "avg_net": float(net.mean()),
        "max_drawdown": float(min(dd, 0.0)),  # on realised (exit-date) equity, not mark-to-market
        "top3_share": top3,
        "profit_factor": float(wins / losses) if losses > 0 else np.nan,
        "turnover": float(t["turnover"].sum()) if "turnover" in t else np.nan,
    }


# ---------------------------------------------------------------------------- walk-forward
def fold_plan(n: int, train: int, test: int, holdout: int = 0) -> dict:
    """Chronological plan over n rows: [development | holdout].

    Folds: train [s - train, s), test [s, s + test), s advancing by `test` inside the development
    rows. A final partial test window is kept only if it has at least max(5, test // 3) rows.
    Holdout: parameters chosen on the last `train` development rows, tested once on the last
    `holdout` rows, which no fold ever sees.
    """
    dev_n = n - holdout if holdout and n - holdout >= train else n
    hold = None
    if holdout and n - holdout >= train:
        hold = (dev_n - train, dev_n, dev_n, n)
    folds, s = [], train
    while s < dev_n:
        e = min(s + test, dev_n)
        if e - s < max(5, test // 3):
            break
        folds.append((s - train, s, s, e))
        s = e
    return {"n": n, "dev_n": dev_n, "folds": folds, "holdout": hold}


class ZCache:
    def __init__(self, series: pd.Series, min_periods: int):
        self.s, self.mp, self._c = series, min_periods, {}

    def z(self, window: int) -> pd.Series:
        if window not in self._c:
            self._c[window] = rolling_stats(self.s, window, min(self.mp, window))["z"]
        return self._c[window]


def _sim_kwargs(research: dict, cost_cfg: dict, use_gates: bool) -> dict:
    s = research["strategy"]
    return dict(cost_cfg=cost_cfg, lots_a=int(s.get("lots_leg_a", 1)), use_gates=use_gates,
                execution=s.get("execution", "next_close"), tender_days=int(s.get("tender_days", 3)),
                max_participation=float(s.get("max_participation", 0.10)))


def _choose_params(df, zc, grid, default, prices, tick_a, tick_b, start, end, min_trades, kw):
    best, best_net, chosen_by = default, -np.inf, "default (too few train trades)"
    for p in grid:
        t = simulate(df, zc.z(p.window), p, prices, tick_a=tick_a, tick_b=tick_b, start=start, end=end, **kw)
        m = trade_metrics(t)
        if m["trades"] >= min_trades and m["net_pnl"] > best_net:
            best, best_net, chosen_by = p, m["net_pnl"], "best train net P&L"
    return best, chosen_by


def walk_forward(df: pd.DataFrame, obs: pd.DataFrame, research: dict, tick_a: float, tick_b: float,
                 metric_col: str = "metric", use_gates: bool = True, cost_cfg: dict | None = None,
                 prices: PriceBook | None = None) -> dict:
    """Chronological walk-forward: choose parameters on each train window, test on the next.
    Then, if configured and possible, choose once more on the last train window before the
    holdout and test on the untouched holdout."""
    wf, strat = research["walk_forward"], research["strategy"]
    cost_cfg = cost_cfg or research["costs"]
    prices = prices or PriceBook(obs)
    kw = _sim_kwargs(research, cost_cfg, use_gates)
    zc = ZCache(df[metric_col], research["statistics"]["min_periods"])
    default = StrategyParams(strat["entry_z"], strat["exit_z"], strat["max_hold_days"], research["statistics"]["window"])
    grid = [StrategyParams(e, x, strat["max_hold_days"], w)
            for e, x, w in iproduct(wf["grid"]["entry_z"], wf["grid"]["exit_z"], wf["grid"]["window"])]
    plan = fold_plan(len(df), wf["train_days"], wf["test_days"], wf.get("holdout_days", 0))
    folds, all_trades, skipped = [], [], []
    for k, (tr_s, tr_e, te_s, te_e) in enumerate(plan["folds"], 1):
        best, chosen_by = _choose_params(df, zc, grid, default, prices, tick_a, tick_b, tr_s, tr_e,
                                         wf["min_train_trades"], kw)
        t = simulate(df, zc.z(best.window), best, prices, tick_a=tick_a, tick_b=tick_b, start=te_s, end=te_e, **kw)
        skipped += [dict(s, fold=k) for s in t.attrs.get("skipped", [])]
        m = trade_metrics(t)
        if not t.empty:
            all_trades.append(t.assign(fold=k))
        folds.append({"fold": k, "train_start": df["date"].iloc[tr_s].date(), "train_end": df["date"].iloc[tr_e - 1].date(),
                      "test_start": df["date"].iloc[te_s].date(), "test_end": df["date"].iloc[te_e - 1].date(),
                      **{f"param_{a}": v for a, v in asdict(best).items()}, "chosen_by": chosen_by,
                      **{f"oos_{a}": v for a, v in m.items()}})
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    holdout = None
    if plan["holdout"]:
        tr_s, tr_e, h_s, h_e = plan["holdout"]
        best, chosen_by = _choose_params(df, zc, grid, default, prices, tick_a, tick_b, tr_s, tr_e,
                                         wf["min_train_trades"], kw)
        ht = simulate(df, zc.z(best.window), best, prices, tick_a=tick_a, tick_b=tick_b, start=h_s, end=h_e, **kw)
        holdout = {"train_start": df["date"].iloc[tr_s].date(), "train_end": df["date"].iloc[tr_e - 1].date(),
                   "start": df["date"].iloc[h_s].date(), "end": df["date"].iloc[h_e - 1].date(),
                   "params": asdict(best), "chosen_by": chosen_by, "trades": ht, "summary": trade_metrics(ht)}
    return {"folds": pd.DataFrame(folds), "trades": trades, "summary": trade_metrics(trades),
            "holdout": holdout, "plan": plan, "skipped": skipped}


def sensitivity_grid(df: pd.DataFrame, obs: pd.DataFrame, research: dict, tick_a: float, tick_b: float,
                     metric_col: str = "metric", use_gates: bool = True, cost_cfg: dict | None = None,
                     end: int | None = None, prices: PriceBook | None = None) -> pd.DataFrame:
    """IN-SAMPLE net P&L over entry/exit thresholds on the development rows [0, end) only (the
    holdout is never used). Descriptive robustness view, not evidence."""
    cost_cfg = cost_cfg or research["costs"]
    prices = prices or PriceBook(obs)
    kw = _sim_kwargs(research, cost_cfg, use_gates)
    w = research["statistics"]["window"]
    z = rolling_stats(df[metric_col], w, min(research["statistics"]["min_periods"], w))["z"]
    rows = []
    for e in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for x in [0.0, 0.5, 1.0]:
            p = StrategyParams(e, x, research["strategy"]["max_hold_days"], w)
            m = trade_metrics(simulate(df, z, p, prices, tick_a=tick_a, tick_b=tick_b, end=end, **kw))
            rows.append({"entry_z": e, "exit_z": x, "net_pnl": m["net_pnl"], "trades": m["trades"]})
    return pd.DataFrame(rows)
