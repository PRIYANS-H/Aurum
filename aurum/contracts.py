"""Contract specifications, comparability checks and price normalisation.

Normalisation basis: "Rs per 10 g of 995-equivalent gold" (the GOLD/GOLDM quotation basis).

    price_per_gram        = quoted_price / quote_unit_grams
    fine_price_per_gram   = price_per_gram / (purity_per_mille / 1000)
    normalised            = fine_price_per_gram * 10 * 0.995

For a 995 contract quoted per 10 g this is the identity. For a 999 contract it scales the
price down by 995/999 (~0.40%). This mirrors the 999/995 conversion that (secondary copies
of) MCX delivery documents use. It values gold by fine content only; it does NOT remove
differences in delivery form (coin vs bar), centre, expiry day or liquidity - those remain
in the spread and are reported as confounders, not "mispricing".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

REFERENCE_GRAMS = 10.0
REFERENCE_PURITY = 995.0
SPEC_FIELDS = ["quote_unit_grams", "lot_grams", "tick_rs", "purity_per_mille", "delivery_form",
               "price_basis", "final_settlement", "making_charge_rs_per_lot"]
# Fields needed to put two DIFFERENT products on one price basis (cross pairs). Must be VERIFIED.
COMPARISON_FIELDS = ["quote_unit_grams", "purity_per_mille"]
# Fields needed for rupee P&L and costs. Unverified -> analysis allowed, RESEARCH CANDIDATE blocked.
PNL_FIELDS = ["lot_grams", "tick_rs"]
CALC_FIELDS = COMPARISON_FIELDS + PNL_FIELDS
COMPARABILITY_STATUSES = {
    "VALID": "Same quotation unit and purity on both legs; no conversion needed.",
    "ADJUSTED": "Different units and/or purity, converted with VERIFIED specifications.",
    "UNVERIFIED": "A conversion would rely on specification fields not verified from an official "
                  "document. Shown only as exploratory; excluded from validated analysis.",
    "INVALID": "Cannot be compared: a specification is missing or the contracts are not comparable.",
}


@dataclass(frozen=True)
class SpecField:
    value: Any
    status: str  # VERIFIED | SECONDARY | ASSUMED
    source: str | None


@dataclass(frozen=True)
class ContractSpec:
    product: str
    applies_from_expiry: pd.Timestamp
    quote_unit_grams: SpecField
    lot_grams: SpecField
    tick_rs: SpecField
    purity_per_mille: SpecField
    delivery_form: SpecField
    price_basis: SpecField
    final_settlement: SpecField
    making_charge_rs_per_lot: SpecField

    @property
    def multiplier(self) -> float:
        """Rupees of P&L per lot for a 1-rupee move in the quoted price."""
        return float(self.lot_grams.value) / float(self.quote_unit_grams.value)

    @property
    def fine_grams_per_lot(self) -> float:
        return float(self.lot_grams.value) * float(self.purity_per_mille.value) / 1000.0

    def unverified_fields(self, fields: list[str] | None = None) -> list[str]:
        return [f for f in (fields or CALC_FIELDS) if getattr(self, f).status != "VERIFIED"]


class SpecBook:
    """All product specifications, versioned by the contract series they apply to.

    MCX publishes specifications "for contracts expiring <month> onwards", so a version is
    selected by the CONTRACT'S EXPIRY DATE: the latest version whose applies_from_expiry is on
    or before that expiry.
    """

    def __init__(self, cfg: dict):
        self.sources: dict = cfg.get("sources", {})
        self._versions: dict[str, list[ContractSpec]] = {}
        for product, pcfg in cfg["products"].items():
            versions = []
            for v in pcfg["versions"]:
                fields = {}
                for f in SPEC_FIELDS:
                    if f not in v:
                        raise ValueError(f"{product}: spec field {f!r} missing")
                    fields[f] = SpecField(v[f]["value"], v[f]["status"], v[f].get("source"))
                versions.append(ContractSpec(product, pd.Timestamp(v["applies_from_expiry"]), **fields))
            self._versions[product] = sorted(versions, key=lambda s: s.applies_from_expiry)
        self.descriptions = {p: c.get("description", p) for p, c in cfg["products"].items()}

    @property
    def products(self) -> set[str]:
        return set(self._versions)

    def spec(self, product: str, expiry: pd.Timestamp) -> ContractSpec:
        """Specification for the contract of `product` expiring on `expiry`."""
        if product not in self._versions:
            raise KeyError(f"No specification for product {product!r}")
        valid = [s for s in self._versions[product] if s.applies_from_expiry <= pd.Timestamp(expiry)]
        if not valid:
            raise KeyError(f"No specification for {product} contracts expiring {pd.Timestamp(expiry):%Y-%m-%d}")
        return valid[-1]

    def table(self, expiry: pd.Timestamp | None = None) -> pd.DataFrame:
        """Long table of spec fields; all versions if expiry is None."""
        rows = []
        for p in sorted(self._versions):
            specs = self._versions[p] if expiry is None else [self.spec(p, expiry)]
            rows += [{"product": p, "applies_from_expiry": s.applies_from_expiry.date(), "field": f,
                      "value": getattr(s, f).value, "status": getattr(s, f).status, "source": getattr(s, f).source}
                     for s in specs for f in SPEC_FIELDS]
        return pd.DataFrame(rows)


def multiplier_check(obs: pd.DataFrame, book: SpecBook) -> pd.DataFrame:
    """Cross-check lot multipliers against the exchange's own traded value.

    Bhavcopy reports traded value (Rs lakh) and volume (lots). For each traded row,
        implied Rs per lot = value_lacs * 1e5 / volume_lots ≈ VWAP * multiplier.
    Dividing by (close * spec multiplier) should give ≈ 1 (VWAP and close differ intraday).
    A median far from 1 means the spec's lot/quote ratio does not match the exchange data.
    """
    if "value_lacs" not in obs.columns:
        return pd.DataFrame()
    o = obs[(obs["volume_lots"] > 0) & (obs["value_lacs"] > 0)].copy()
    if o.empty:
        return pd.DataFrame()
    o["spec_multiplier"] = [book.spec(s, pd.Timestamp(e)).multiplier for s, e in zip(o["symbol"], o["expiry"])]
    o["ratio"] = o["value_lacs"] * 1e5 / o["volume_lots"] / (o["close"] * o["spec_multiplier"])
    g = o.groupby("symbol").agg(rows=("ratio", "size"), spec_multiplier=("spec_multiplier", "first"),
                                median_ratio=("ratio", "median"),
                                p05=("ratio", lambda r: r.quantile(0.05)),
                                p95=("ratio", lambda r: r.quantile(0.95))).reset_index()
    g["consistent"] = g["median_ratio"].between(0.97, 1.03)
    return g


def normalised_price(price: float | np.ndarray, spec: ContractSpec, apply_purity: bool = True):
    """Convert a quoted price to Rs per 10 g of 995-equivalent gold (see module docstring).

    apply_purity=False gives the naive 'units only' conversion, kept so users can see the
    bias it creates between 995 and 999 contracts.
    """
    per_gram = np.asarray(price, dtype=float) / float(spec.quote_unit_grams.value)
    if not apply_purity:
        return per_gram * REFERENCE_GRAMS
    fine_per_gram = per_gram / (float(spec.purity_per_mille.value) / 1000.0)
    return fine_per_gram * REFERENCE_GRAMS * (REFERENCE_PURITY / 1000.0)


@dataclass
class Comparability:
    status: str          # VALID | ADJUSTED | UNVERIFIED | INVALID  (see COMPARABILITY_STATUSES)
    reasons: list[str]
    adjustments: list[str]
    confounders: list[str]
    unverified: list[str]                              # calc fields (comparison + P&L) not VERIFIED
    missing: list[str] = field(default_factory=list)   # what would be needed to verify / repair
    pnl_verified: bool = True                          # lot size and tick VERIFIED on both legs

    @property
    def usable(self) -> bool:
        """Whether the pair may enter the validated analysis pipeline."""
        return self.status in ("VALID", "ADJUSTED")


def check_fixed_contracts(pair_cfg: dict, a_id: str, b_id: str, a_expiry: pd.Timestamp,
                          b_expiry: pd.Timestamp) -> str | None:
    """Rules for an explicitly chosen contract pair. Returns a reason if invalid, else None."""
    if a_id == b_id:
        return "Both legs are the same contract; a spread with itself is always zero."
    if pair_cfg["kind"] == "calendar":
        if pd.Timestamp(b_expiry) <= pd.Timestamp(a_expiry):
            return "For a calendar comparison the second contract must expire after the first."
        return None
    if pd.Timestamp(a_expiry).to_period("M") != pd.Timestamp(b_expiry).to_period("M"):
        return ("Cross-product comparisons require both contracts to expire in the same month; otherwise "
                "the spread mixes delivery months and is dominated by carry.")
    return None


def assess_comparability(pair_cfg: dict, book: SpecBook, expiry_a: pd.Timestamp,
                         expiry_b: pd.Timestamp | None = None) -> Comparability:
    """Decide whether a configured pair can be compared, and how.

    expiry_a / expiry_b identify the contract series of each leg (specs are per series).
    """
    kind = pair_cfg["kind"]
    if kind == "calendar":
        products = [pair_cfg["product"], pair_cfg["product"]]
    elif kind == "cross":
        products = [pair_cfg["leg_a"], pair_cfg["leg_b"]]
    else:
        return Comparability("INVALID", [f"unknown pair kind {kind!r}"], [], [], [])

    try:
        sa = book.spec(products[0], expiry_a)
        sb = book.spec(products[1], expiry_b if expiry_b is not None else expiry_a)
    except KeyError as exc:
        return Comparability("INVALID", [str(exc)], [], [], [],
                             missing=["a specification version covering this contract series"], pnl_verified=False)

    unverified = sorted({f"{s.product}.{f}" for s in (sa, sb) for f in s.unverified_fields(CALC_FIELDS)})
    pnl_verified = not any(s.unverified_fields(PNL_FIELDS) for s in (sa, sb))
    series = {f"{s.product} contracts expiring from {s.applies_from_expiry:%b %Y}" for s in (sa, sb)
              if s.unverified_fields(CALC_FIELDS)}
    missing = [f"official MCX contract-specification PDF for {x}" for x in sorted(series)]

    if kind == "calendar":
        same = (sa.quote_unit_grams.value == sb.quote_unit_grams.value
                and sa.purity_per_mille.value == sb.purity_per_mille.value)
        if not same:
            return Comparability(
                "INVALID", ["the specification changed between the two contract series (quotation unit or "
                            "purity differs), so a calendar spread would mix two different products"],
                [], [], unverified, missing, pnl_verified)
        return Comparability(
            "VALID", ["same product: identical quotation unit and purity on both legs, no conversion needed"], [],
            ["cost of carry (interest rate) is the expected driver of the spread",
             "far contract usually less liquid than near"],
            unverified, missing, pnl_verified,
        )

    comp_unverified = sorted({f"{s.product}.{f}" for s in (sa, sb) for f in s.unverified_fields(COMPARISON_FIELDS)})
    if comp_unverified:
        return Comparability(
            "UNVERIFIED",
            ["putting two different products on one basis requires VERIFIED quotation unit and purity; not "
             "verified: " + ", ".join(comp_unverified)],
            [], [], unverified, missing, pnl_verified)

    adjustments, confounders = [], []
    if sa.quote_unit_grams.value != sb.quote_unit_grams.value:
        adjustments.append(
            f"quotation units: {sa.product} per {sa.quote_unit_grams.value} g, "
            f"{sb.product} per {sb.quote_unit_grams.value} g -> both converted to per 10 g")
    if sa.purity_per_mille.value != sb.purity_per_mille.value:
        adjustments.append(
            f"purity: {sa.product} {sa.purity_per_mille.value}, {sb.product} "
            f"{sb.purity_per_mille.value} -> fine-gold equivalent (factor "
            f"{sb.purity_per_mille.value}/{sa.purity_per_mille.value})")
    if sa.delivery_form.value != sb.delivery_form.value:
        confounders.append(
            f"delivery form differs ({sa.delivery_form.value} vs {sb.delivery_form.value}): "
            "a persistent premium is economically plausible")
    if sa.lot_grams.value != sb.lot_grams.value:
        confounders.append(
            f"lot sizes differ ({sa.lot_grams.value} g vs {sb.lot_grams.value} g): "
            "hedge ratio is integer-lot approximate")
    if sa.price_basis.value != sb.price_basis.value:
        confounders.append(
            f"price basis differs ({sa.price_basis.value} vs {sb.price_basis.value}): "
            "a location premium between spot markets can persist")
    if sa.final_settlement.value != sb.final_settlement.value:
        confounders.append(
            f"final settlement differs ({sa.final_settlement.value} vs {sb.final_settlement.value})")
    if sa.making_charge_rs_per_lot.value != sb.making_charge_rs_per_lot.value:
        confounders.append(
            f"buyer making charge on delivery differs (Rs {sa.making_charge_rs_per_lot.value}/lot vs "
            f"Rs {sb.making_charge_rs_per_lot.value}/lot): affects holding to delivery, not the futures price formula")
    confounders.append("expiry day may differ within the same month (carry over the gap)")
    status = "ADJUSTED" if adjustments else "VALID"
    reason = ("quotation unit and purity VERIFIED on both legs; " +
              ("converted to Rs per 10 g of 995-equivalent gold" if adjustments else "no conversion needed"))
    return Comparability(status, [reason, "same expiry month required on both legs"], adjustments,
                         confounders, unverified, missing, pnl_verified)
