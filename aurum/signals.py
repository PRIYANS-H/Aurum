"""Central signal classification. The UI only displays what this module decides.

Three separate questions, three separate statuses:
  comparability   can the two contracts be compared at all?      (contracts.assess_comparability)
  alert           is the latest value statistically unusual?     ALERT / NO ALERT / INSUFFICIENT HISTORY / NOT COMPUTABLE
  validation      did the strategy pass the predefined protocol? PASS / FAIL / INCONCLUSIVE / NOT RUN

The overall STATUS combines them. Rules are applied IN ORDER; the first match decides:
1. INVALID COMPARISON     comparability is INVALID or UNVERIFIED (missing / unverified specifications).
2. INCONCLUSIVE           measure not computable, too little history, or (near-)zero dispersion.
3. NORMAL                 no alert: |z| below the watch threshold.
4. EXPLAINABLE / RISK FLAG alert, and at least one risk flag on this row (see analytics.FLAGS).
5. REJECTED               alert, no flags, and the strategy FAILED the protocol (incl. no robust
                          improvement over the baseline).
6. RESEARCH CANDIDATE     alert, no flags, protocol PASSED, and lot size / tick (P&L fields) VERIFIED.
7. WATCH                  any other alert: validation INCONCLUSIVE / not run, or P&L specs unverified.
No status is an execution recommendation. AURUM never outputs BUY / SELL / "arbitrage".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .analytics import FLAGS, active_flags
from .contracts import Comparability

STATUS_INFO = {
    "INVALID COMPARISON": "The pair is invalid here, or essential specifications are missing or unverified.",
    "INCONCLUSIVE": "Data coverage or sample size is insufficient to judge this observation.",
    "NORMAL": "No statistical alert: the measure is within its usual range under the rolling model.",
    "EXPLAINABLE / RISK FLAG": "An alert exists, but a documented data, expiry or liquidity issue is present. "
                               "No cause is claimed.",
    "WATCH": "A statistical alert that merits investigation; not validated as a trading opportunity.",
    "RESEARCH CANDIDATE": "Valid comparison and the predefined out-of-sample protocol passed. "
                          "Not a guarantee of profitability.",
    "REJECTED": "A tested strategy for this pair failed the predefined validation criteria.",
}
STATUSES = list(STATUS_INFO)
ALERT_INFO = {
    "ALERT": "|z| at or above the watch threshold: unusual relative to the rolling model.",
    "NO ALERT": "|z| below the watch threshold.",
    "INSUFFICIENT HISTORY": "Too few prior valid observations, or zero dispersion, for a z-score.",
    "NOT COMPUTABLE": "The measure cannot be computed for this row (comparability or data).",
}
VALIDATION_INFO = {
    "PASS": "The strategy met every predefined criterion.",
    "FAIL": "The strategy failed a predefined criterion.",
    "INCONCLUSIVE": "Too little out-of-sample evidence (or an exploratory run).",
    "NOT RUN": "No validation result is available for this pair.",
}


@dataclass
class Classification:
    status: str
    alert_status: str = "NOT COMPUTABLE"
    validation_status: str = "NOT RUN"
    comparability_status: str = ""
    reasons: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    untested: list[str] = field(default_factory=list)


UNTESTED = [
    "true executable prices (bid/ask, depth): end-of-day data only",
    "intraday timing of each leg's prices (MCX close = weighted average of the last 30 minutes)",
    "margin requirements and financing cost of holding the spread",
]


def classify(row: pd.Series | None, comp: Comparability, research: dict, protocol: dict | None = None) -> Classification:
    """Classify one observation. `protocol` is validation.evaluate_protocol(...) output or None."""
    sig = research["signal"]
    min_p = research["statistics"]["min_periods"]
    vstat = protocol["verdict"] if protocol else "NOT RUN"
    untested = list(UNTESTED)
    if comp.missing:
        untested.insert(0, "specification evidence: " + "; ".join(comp.missing))

    base = dict(validation_status=vstat, comparability_status=comp.status, untested=untested)
    if comp.status in ("INVALID", "UNVERIFIED"):
        return Classification("INVALID COMPARISON", "NOT COMPUTABLE", reasons=list(comp.reasons), **base)
    if row is None:
        return Classification("INVALID COMPARISON", "NOT COMPUTABLE",
                              reasons=["no observation with both legs available"], **base)
    if pd.isna(row.get("metric")):
        return Classification("INCONCLUSIVE", "NOT COMPUTABLE",
                              reasons=["the measure is not computable on this row (e.g. expiries too close for "
                                       "a carry rate, or a missing price)"], **base)
    if row["n_hist"] < min_p or pd.isna(row["std"]) or pd.isna(row["z"]):
        return Classification("INCONCLUSIVE", "INSUFFICIENT HISTORY",
                              reasons=[f"{int(row['n_hist'])} prior valid observations (need {min_p}) or "
                                       "zero / near-zero dispersion"], **base)

    z = float(row["z"])
    if abs(z) < sig["watch_z"]:
        return Classification("NORMAL", "NO ALERT",
                              reasons=[f"|z| = {abs(z):.2f} is below the {sig['watch_z']} watch threshold"], **base)

    head = f"|z| = {abs(z):.2f} >= {sig['watch_z']} (measure {'above' if z > 0 else 'below'} its rolling mean)"
    flags = active_flags(row)
    if flags:
        return Classification("EXPLAINABLE / RISK FLAG", "ALERT", reasons=[head] + [FLAGS[f]["label"] for f in flags],
                              flags=flags, **base)
    if vstat == "FAIL":
        return Classification("REJECTED", "ALERT", reasons=[head, f"protocol conclusion: {protocol['conclusion']}"]
                              + protocol["reasons"], **base)
    if vstat == "PASS":
        if not comp.pnl_verified:
            return Classification("WATCH", "ALERT", reasons=[head, "protocol passed, but status is capped at WATCH "
                                                             "because lot size / tick are not VERIFIED"], **base)
        return Classification("RESEARCH CANDIDATE", "ALERT", reasons=[head, "no risk flags", "protocol passed"]
                              + protocol["reasons"], **base)
    why = "validation inconclusive: " + "; ".join(protocol["reasons"]) if protocol else "validation not run"
    return Classification("WATCH", "ALERT", reasons=[head, "no risk flags", why], **base)
