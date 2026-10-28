import pandas as pd

from aurum.contracts import Comparability
from aurum.signals import ALERT_INFO, STATUS_INFO, STATUSES, VALIDATION_INFO, classify

OK = Comparability("VALID", [], [], [], [])
PNL_UNVERIFIED = Comparability("VALID", [], [], [], ["GOLDM.lot_grams"], pnl_verified=False)
UNVERIFIED = Comparability("UNVERIFIED", ["purity not verified"], [], [], ["GOLDPETAL.purity_per_mille"],
                           missing=["official MCX PDF for GOLDPETAL contracts expiring from Jan 2000"],
                           pnl_verified=False)
PASS = {"verdict": "PASS", "conclusion": "CANDIDATE PASSED DEFINED TESTS", "reasons": ["all passed"]}
FAIL = {"verdict": "FAIL", "conclusion": "FAILED VALIDATION", "reasons": ["net <= 0"]}
NRI = {"verdict": "FAIL", "conclusion": "NO ROBUST IMPROVEMENT", "reasons": ["baseline better"]}
INC = {"verdict": "INCONCLUSIVE", "conclusion": "INSUFFICIENT EVIDENCE", "reasons": ["0 folds"]}


def row(z=2.5, n=40, std=0.1, metric=1.0, **flags):
    base = {"metric": metric, "z": z, "n_hist": n, "std": std}
    base.update(flags)
    return pd.Series(base)


def test_order_of_rules(research):
    assert classify(row(), Comparability("INVALID", ["x"], [], [], []), research).status == "INVALID COMPARISON"
    assert classify(None, OK, research).status == "INVALID COMPARISON"
    assert classify(row(metric=float("nan")), OK, research).status == "INCONCLUSIVE"
    assert classify(row(n=5), OK, research).status == "INCONCLUSIVE"
    assert classify(row(std=float("nan"), z=float("nan")), OK, research).status == "INCONCLUSIVE"
    assert classify(row(z=1.0), OK, research).status == "NORMAL"
    c = classify(row(b_stale=True), OK, research, PASS)
    assert c.status == "EXPLAINABLE / RISK FLAG" and c.flags == ["b_stale"]
    assert classify(row(), OK, research, FAIL).status == "REJECTED"
    assert classify(row(), OK, research, NRI).status == "REJECTED"        # not beating the baseline = rejected
    assert classify(row(), OK, research, PASS).status == "RESEARCH CANDIDATE"
    assert classify(row(), OK, research, INC).status == "WATCH"
    assert classify(row(), OK, research).status == "WATCH"


def test_unverified_comparison_never_classified_beyond_invalid(research):
    c = classify(row(z=5.0), UNVERIFIED, research, PASS)
    assert c.status == "INVALID COMPARISON" and c.alert_status == "NOT COMPUTABLE"
    assert any("official MCX PDF" in u for u in c.untested)


def test_unverified_pnl_fields_cap_at_watch(research):
    c = classify(row(), PNL_UNVERIFIED, research, PASS)
    assert c.status == "WATCH" and any("capped at WATCH" in r for r in c.reasons)


def test_alert_and_validation_are_separate(research):
    c = classify(row(z=3.0), OK, research, INC)
    assert (c.status, c.alert_status, c.validation_status) == ("WATCH", "ALERT", "INCONCLUSIVE")
    n = classify(row(z=0.5), OK, research, FAIL)
    assert (n.status, n.alert_status, n.validation_status) == ("NORMAL", "NO ALERT", "FAIL")


def test_negative_z_also_unusual(research):
    c = classify(row(z=-3.0), OK, research)
    assert c.status == "WATCH" and any("below" in r for r in c.reasons)


def test_vocabulary_is_defined_once():
    assert STATUSES == list(STATUS_INFO) and len(STATUSES) == 7
    assert set(ALERT_INFO) == {"ALERT", "NO ALERT", "INSUFFICIENT HISTORY", "NOT COMPUTABLE"}
    assert set(VALIDATION_INFO) == {"PASS", "FAIL", "INCONCLUSIVE", "NOT RUN"}
    assert not any(w in " ".join(STATUS_INFO) for w in ("BUY", "SELL", "ARBITRAGE", "PROFITABLE"))
