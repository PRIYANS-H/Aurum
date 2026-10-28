"""The single validation protocol: feasibility, integrity checks, cost stress, acceptance verdict.

Signal Evidence, the Backtesting Lab and the classifier all use `evaluate_protocol`, so one
configuration always produces one verdict everywhere. All thresholds live in config/research.yaml
(validation_pass, cost_stress) and are fixed before results are seen.

Verdicts
  PASS          every criterion met                       -> "CANDIDATE PASSED DEFINED TESTS"
  FAIL          a performance or integrity criterion fails -> "FAILED VALIDATION"
  FAIL          only the baseline comparison fails         -> "NO ROBUST IMPROVEMENT"
  INCONCLUSIVE  too few folds or trades                    -> "INSUFFICIENT EVIDENCE"
  INCONCLUSIVE  exploratory (short-window) run             -> "EXPLORATORY ONLY — NOT EVIDENCE"
"""

from __future__ import annotations

import pandas as pd

from .backtest import PriceBook, fold_plan

CONCLUSIONS = {
    "CANDIDATE PASSED DEFINED TESTS": "Every predefined criterion was met. This is not a guarantee of future profit.",
    "NO ROBUST IMPROVEMENT": "The candidate passes on its own but does not beat the naive baseline under the "
                             "predefined criteria, so AURUM's adjustments add no demonstrated value here.",
    "FAILED VALIDATION": "The candidate failed at least one predefined out-of-sample criterion.",
    "INSUFFICIENT EVIDENCE": "Too few folds or trades to reach any conclusion.",
    "EXPLORATORY ONLY — NOT EVIDENCE": "Shorter-than-protocol windows were used. Results describe the software's "
                                       "behaviour on this data and cannot pass or fail the protocol.",
}


def feasibility(n_rows: int, research: dict, exploratory: bool = False) -> dict:
    """Can the configured walk-forward run on n_rows observations, and can it reach a verdict?"""
    wf, rules = research["walk_forward"], research["validation_pass"]
    train, test, hold = int(wf["train_days"]), int(wf["test_days"]), int(wf.get("holdout_days", 0))
    plan = fold_plan(n_rows, train, test, hold)
    folds = len(plan["folds"])
    need_min = train + max(5, test // 3)
    need_full = train + rules["min_folds"] * test + hold
    msgs = []
    lo_tr, lo_te = ((wf.get("exploratory_min_train", 30), wf.get("exploratory_min_test", 10)) if exploratory
                    else (60, 20))
    window_ok = train >= lo_tr and test >= lo_te
    if not window_ok:
        msgs.append(f"windows must be at least {lo_tr} train / {lo_te} test rows"
                    + ("" if exploratory else " (shorter windows only in exploratory mode)"))
    if folds == 0:
        msgs.append(f"{n_rows} observations cannot hold one {train}-row training window plus a test window "
                    f"(needs at least {need_min})")
    elif folds < rules["min_folds"] and not exploratory:
        msgs.append(f"only {folds} fold(s) fit; the protocol needs {rules['min_folds']} — the verdict will be "
                    "INSUFFICIENT EVIDENCE")
    if hold and plan["holdout"] is None:
        msgs.append(f"no {hold}-row final holdout fits; results will have no untouched holdout")
    runnable = folds >= 1 and window_ok
    return {"n_rows": n_rows, "folds": folds, "holdout": plan["holdout"] is not None, "runnable": runnable,
            "sufficient": folds >= rules["min_folds"] and not exploratory, "rows_needed": need_full,
            "messages": msgs, "plan": plan}


def cost_stress(trades: pd.DataFrame, research: dict) -> pd.DataFrame:
    """Re-price the same out-of-sample trades under each cost scenario (trades held fixed).

    net = gross − fees × fee_multiplier − slip_unit × slippage_ticks"""
    rows = []
    for sc in research.get("cost_stress", {}).get("scenarios", []):
        if trades is None or trades.empty:
            net = 0.0
        else:
            net = float((trades["gross_pnl"] - trades["fees"] * sc["fee_multiplier"]
                         - trades["slip_unit"] * sc["slippage_ticks"]).sum())
        rows.append({"scenario": sc["name"], "fee_multiplier": sc["fee_multiplier"],
                     "slippage_ticks": sc["slippage_ticks"], "net_pnl": net,
                     "trades": 0 if trades is None else len(trades)})
    return pd.DataFrame(rows)


def tender_violations(trades: pd.DataFrame, prices: PriceBook, tender_days: int) -> int:
    """Number of trade legs still held on or after the first day of the delivery tender period."""
    if trades is None or trades.empty:
        return 0
    n = 0
    for _, t in trades.iterrows():  # window-end accounting marks are included: they must also be early
        for cid in (t["a_contract"], t["b_contract"]):
            if pd.Timestamp(t["exit_date"]) > prices.exit_deadline(cid, None, tender_days):
                n += 1
    return n


def stale_fills(trades: pd.DataFrame, prices: PriceBook) -> int:
    """Entry/exit fills (other than window-end accounting marks) on days a leg did not trade."""
    if trades is None or trades.empty or not prices.has_volume:
        return 0
    n = 0
    for _, t in trades.iterrows():
        checks = [(t["entry_date"], t["a_contract"]), (t["entry_date"], t["b_contract"])]
        if t["exit_reason"] != "forced_end" and "stale price" not in str(t.get("note", "")):
            checks += [(t["exit_date"], t["a_contract"]), (t["exit_date"], t["b_contract"])]
        n += sum(1 for d, c in checks if not ((prices.volume(d, c) or 0) > 0))
    return n


def integrity(val: dict, prices: PriceBook, research: dict) -> dict:
    """Structural checks that must always hold. A failure means an engine bug, not a market result."""
    tender = int(research["strategy"].get("tender_days", 3))
    out = {}
    for name in ("candidate", "baseline"):
        t = val[name]["trades"]
        h = val[name].get("holdout")
        allt = pd.concat([t, h["trades"]], ignore_index=True) if h and len(h["trades"]) else t
        out[name] = {
            "same_bar_fills": int((pd.to_datetime(allt["entry_date"]) <= pd.to_datetime(allt["signal_date"])).sum())
            if len(allt) else 0,
            "tender_violations": tender_violations(allt, prices, tender),
            "stale_fills": stale_fills(allt, prices),
        }
    f = val["candidate"]["folds"]
    out["fold_order_ok"] = bool((pd.to_datetime(f["test_start"]) > pd.to_datetime(f["train_end"])).all()) if len(f) else True
    if len(f) > 1:
        out["fold_order_ok"] &= bool((pd.to_datetime(f["test_start"]).iloc[1:].values
                                      > pd.to_datetime(f["test_end"]).iloc[:-1].values).all())
    h = val["candidate"].get("holdout")
    out["holdout_after_folds"] = (h is None or not len(f) or
                                  pd.Timestamp(h["start"]) > pd.to_datetime(f["test_end"]).max())
    return out


def evaluate_protocol(val: dict, research: dict, exploratory: bool = False) -> dict:
    """Apply the predefined criteria. Returns criteria rows, verdict, conclusion and reasons."""
    rules = research["validation_pass"]
    c, b = val["candidate"], val["baseline"]
    cs, bs = c["summary"], b["summary"]
    cf, bf = c["folds"], b["folds"]
    rows = []

    def add(group, name, threshold, observed, passed, note=""):
        rows.append({"group": group, "criterion": name, "threshold": threshold, "observed": observed,
                     "result": "n/a" if passed is None else ("PASS" if passed else "FAIL"), "note": note})

    # -- sufficiency
    add("sufficiency", "Walk-forward folds", f">= {rules['min_folds']}", len(cf), len(cf) >= rules["min_folds"])
    add("sufficiency", "Out-of-sample trades", f">= {rules['min_oos_trades']}", cs["trades"],
        cs["trades"] >= rules["min_oos_trades"])
    # -- integrity (engine guarantees; a FAIL means a bug)
    integ = val.get("integrity", {})
    ci = integ.get("candidate", {})
    add("integrity", "No same-bar fills (entry strictly after signal)", "0", ci.get("same_bar_fills", "n/a"),
        None if "same_bar_fills" not in ci else ci["same_bar_fills"] == 0)
    add("integrity", "No position held into the delivery tender period", "0", ci.get("tender_violations", "n/a"),
        None if "tender_violations" not in ci else ci["tender_violations"] == 0)
    add("integrity", "No fills on zero-volume days", "0", ci.get("stale_fills", "n/a"),
        None if "stale_fills" not in ci else ci["stale_fills"] == 0)
    add("integrity", "Test windows strictly after training windows", "all folds",
        "ok" if integ.get("fold_order_ok", True) else "violated", integ.get("fold_order_ok", True))
    add("integrity", "Holdout untouched until after all folds", "true",
        "ok" if integ.get("holdout_after_folds", True) else "violated", integ.get("holdout_after_folds", True))
    # -- performance
    add("performance", "Net out-of-sample result", "> 0", f"Rs {cs['net_pnl']:,.0f}",
        cs["net_pnl"] > 0 if cs["trades"] else None)
    traded = cf[cf["oos_trades"] > 0] if len(cf) else cf
    pos = float((traded["oos_net_pnl"] > 0).mean()) if len(traded) else None
    add("performance", "Traded folds with positive net result", f">= {rules['min_positive_fold_share']:.0%}",
        "n/a" if pos is None else f"{pos:.0%}", None if pos is None else pos >= rules["min_positive_fold_share"])
    top3 = cs["top3_share"]
    add("performance", "Not dependent on a few exceptional trades",
        f"top-3 share <= {rules['max_top3_profit_share']:.0%}", "n/a" if pd.isna(top3) else f"{top3:.0%}",
        None if pd.isna(top3) else top3 <= rules["max_top3_profit_share"])
    stress = val.get("cost_stress")
    must = research.get("cost_stress", {}).get("must_survive")
    if stress is not None and len(stress) and must in set(stress["scenario"]):
        sv = float(stress.loc[stress["scenario"] == must, "net_pnl"].iloc[0])
        add("performance", f"Survives cost stress ('{must}' scenario)", "net > 0", f"Rs {sv:,.0f}",
            sv > 0 if cs["trades"] else None)
    sens = val.get("sensitivity")
    sshare = float((sens["net_pnl"] > 0).mean()) if sens is not None and len(sens) and sens["trades"].sum() else None
    add("performance", "Parameter sensitivity: grid cells net-positive (in-sample, development rows)",
        f">= {rules['min_positive_sensitivity_share']:.0%}", "n/a" if sshare is None else f"{sshare:.0%}",
        None if sshare is None else sshare >= rules["min_positive_sensitivity_share"])
    h = c.get("holdout")
    if h is not None and rules.get("require_holdout_positive", True):
        hs = h["summary"]
        add("performance", "Final holdout net result", "> 0", f"Rs {hs['net_pnl']:,.0f} ({hs['trades']} trades)",
            None if hs["trades"] == 0 else hs["net_pnl"] > 0,
            "no trades in holdout" if hs["trades"] == 0 else "")
    else:
        add("performance", "Final holdout net result", "> 0", "no holdout", None,
            "data too short for a holdout" if h is None else "")
    # -- baseline comparison
    joined = (cf[["fold", "oos_net_pnl"]].merge(bf[["fold", "oos_net_pnl"]], on="fold", suffixes=("_c", "_b"))
              if len(cf) and len(bf) else pd.DataFrame())
    better = float((joined["oos_net_pnl_c"] > joined["oos_net_pnl_b"]).mean()) if len(joined) else None
    add("baseline", "Better net result than baseline in most folds", f"> {rules['min_better_fold_share']:.0%}",
        "n/a" if better is None else f"{better:.0%}", None if better is None else better > rules["min_better_fold_share"])
    add("baseline", "Higher total net result than baseline", "candidate > baseline",
        f"Rs {cs['net_pnl']:,.0f} vs Rs {bs['net_pnl']:,.0f}", cs["net_pnl"] > bs["net_pnl"] if len(cf) else None)

    crit = pd.DataFrame(rows)
    failed = lambda g: bool(((crit["group"] == g) & (crit["result"] == "FAIL")).any())
    if exploratory:
        verdict, concl = "INCONCLUSIVE", "EXPLORATORY ONLY — NOT EVIDENCE"
    elif failed("sufficiency"):
        verdict, concl = "INCONCLUSIVE", "INSUFFICIENT EVIDENCE"
    elif failed("integrity") or failed("performance"):
        verdict, concl = "FAIL", "FAILED VALIDATION"
    elif failed("baseline"):
        verdict, concl = "FAIL", "NO ROBUST IMPROVEMENT"
    else:
        verdict, concl = "PASS", "CANDIDATE PASSED DEFINED TESTS"
    reasons = [f"{r.criterion}: {r.observed} (needs {r.threshold})" for r in crit.itertuples() if r.result == "FAIL"]
    if verdict == "PASS":
        reasons = [f"all {int((crit['result'] == 'PASS').sum())} applicable criteria passed"]
    return {"criteria": crit, "verdict": verdict, "conclusion": concl, "reasons": reasons,
            "explanation": CONCLUSIONS[concl]}
