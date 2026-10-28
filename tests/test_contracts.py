import copy

import pandas as pd
import pytest

from aurum.contracts import SpecBook, assess_comparability, normalised_price

D = pd.Timestamp("2024-01-02")


def test_identity_for_995_per_10g(book):
    assert normalised_price(70000.0, book.spec("GOLDM", D)) == pytest.approx(70000.0)
    assert normalised_price(70000.0, book.spec("GOLD", D)) == pytest.approx(70000.0)


def test_petal_purity_and_units(book):
    petal = book.spec("GOLDPETAL", D)
    # 1 g of 999 at Rs 7,000 -> Rs per 10 g of 995-equivalent = 7000*10*995/999
    assert normalised_price(7000.0, petal) == pytest.approx(7000 * 10 * 995 / 999)
    # naive conversion ignores purity: exactly 10x
    assert normalised_price(7000.0, petal, apply_purity=False) == pytest.approx(70000.0)


def test_guinea_units(book):
    g = book.spec("GOLDGUINEA", D)
    assert normalised_price(56000.0, g) == pytest.approx(56000 / 8 * 10 * 995 / 999)


def test_fair_999_price_normalises_to_995_price(book):
    """If a 999 contract is priced at exactly fine-gold parity, the adjusted spread is zero
    and the naive spread shows the ~0.40% purity bias."""
    p995 = 70000.0
    petal_fair = p995 / 10 * 999 / 995
    adj = normalised_price(petal_fair, book.spec("GOLDPETAL", D))
    naive = normalised_price(petal_fair, book.spec("GOLDPETAL", D), apply_purity=False)
    assert adj == pytest.approx(p995)
    assert (naive / p995 - 1) * 100 == pytest.approx((999 / 995 - 1) * 100)  # ~0.402%


def test_multiplier_and_fine_grams(book):
    assert book.spec("GOLD", D).multiplier == 100      # 1000 g lot / 10 g quote
    assert book.spec("GOLDM", D).multiplier == 10
    assert book.spec("GOLDPETAL", D).multiplier == 1
    assert book.spec("GOLDM", D).fine_grams_per_lot == pytest.approx(99.5)


def test_spec_versions_by_contract_expiry(specs_cfg):
    cfg = copy.deepcopy(specs_cfg)
    v2 = copy.deepcopy(cfg["products"]["GOLDM"]["versions"][0])
    v2["applies_from_expiry"] = "2030-01-01"
    v2["lot_grams"]["value"] = 50
    cfg["products"]["GOLDM"]["versions"].append(v2)
    b = SpecBook(cfg)
    assert b.spec("GOLDM", pd.Timestamp("2029-12-05")).lot_grams.value == 100
    assert b.spec("GOLDM", pd.Timestamp("2030-01-05")).lot_grams.value == 50
    with pytest.raises(KeyError):
        b.spec("GOLDM", pd.Timestamp("1999-01-01"))
    with pytest.raises(KeyError):
        b.spec("SILVER", D)


def test_official_pdf_versions_are_verified(book):
    """Contracts covered by the official MCX PDFs in docs/specs are VERIFIED; older series are not."""
    for sym, exp in [("GOLDPETAL", "2026-12-31"), ("GOLDGUINEA", "2026-12-31"), ("GOLDTEN", "2026-12-31"),
                     ("GOLDM", "2026-12-04"), ("GOLD", "2027-02-05")]:
        s = book.spec(sym, pd.Timestamp(exp))
        assert s.unverified_fields() == [], sym
    assert book.spec("GOLD", pd.Timestamp("2026-12-04")).unverified_fields()  # Dec-2026 GOLD: PDF not held
    petal = book.spec("GOLDPETAL", pd.Timestamp("2026-12-31"))
    assert petal.price_basis.value == "Ex-Mumbai" and petal.making_charge_rs_per_lot.value == 100


def test_multiplier_check_against_traded_value(book):
    from aurum.contracts import multiplier_check
    obs = pd.DataFrame({
        "symbol": ["GOLDM", "GOLDPETAL", "GOLD"],
        "expiry": pd.to_datetime(["2026-11-05", "2026-10-30", "2026-12-04"]),
        "close": [141802.0, 14287.0, 143309.0], "volume_lots": [45850, 189887, 7716],
        # value in Rs lakh; numbers shaped like a real bhavcopy row (VWAP slightly above close)
        "value_lacs": [654149.30, 27303.50, 1112628.16],
    })
    chk = multiplier_check(obs, book).set_index("symbol")
    assert chk["consistent"].all()
    assert chk.loc["GOLD", "spec_multiplier"] == 100


def test_comparability_with_verified_specs(book, pairs_cfg):
    pairs = pairs_cfg["pairs"]
    nov, dec, dec31 = pd.Timestamp("2026-11-05"), pd.Timestamp("2026-12-04"), pd.Timestamp("2026-12-31")
    cal = assess_comparability(pairs["GOLDM_CAL_1_2"], book, nov, dec)
    assert cal.status == "VALID" and cal.pnl_verified and cal.usable
    petal = assess_comparability(pairs["GOLDM_vs_GOLDPETAL"], book, dec, dec31)
    assert petal.status == "ADJUSTED" and petal.usable
    assert any("purity" in a for a in petal.adjustments)
    assert any("delivery form" in c for c in petal.confounders)
    assert any("price basis differs" in c for c in petal.confounders)   # Ex-Ahmedabad vs Ex-Mumbai
    same = assess_comparability(pairs["GOLD_vs_GOLDM"], book, pd.Timestamp("2027-02-05"), pd.Timestamp("2027-02-05"))
    assert same.status == "VALID" and not same.adjustments


def test_unverified_specs_block_cross_comparisons(book, pairs_cfg):
    """Rule: never apply a purity multiplier that is not VERIFIED from an official document."""
    pairs = pairs_cfg["pairs"]
    old = assess_comparability(pairs["GOLDM_vs_GOLDPETAL"], book, D, pd.Timestamp("2024-01-31"))
    assert old.status == "UNVERIFIED" and not old.usable and not old.adjustments
    assert "GOLDPETAL.purity_per_mille" in old.unverified
    assert old.missing and "official MCX contract-specification PDF" in old.missing[0]
    # GOLD Dec-2026 series: values from secondary sources only -> also UNVERIFIED vs GOLDM
    gold_dec = assess_comparability(pairs["GOLD_vs_GOLDM"], book, pd.Timestamp("2026-12-04"), pd.Timestamp("2026-12-04"))
    assert gold_dec.status == "UNVERIFIED"
    # a calendar spread needs no conversion: VALID, but rupee P&L relies on unverified lot/tick
    cal_old = assess_comparability(pairs["GOLDM_CAL_1_2"], book, D, pd.Timestamp("2024-02-05"))
    assert cal_old.status == "VALID" and not cal_old.pnl_verified


def test_missing_or_inconsistent_specs_are_invalid(book, specs_cfg):
    bad = assess_comparability({"kind": "cross", "leg_a": "GOLDM", "leg_b": "SILVER"}, book, D)
    assert bad.status == "INVALID" and not bad.usable
    cfg = copy.deepcopy(specs_cfg)
    v2 = copy.deepcopy(cfg["products"]["GOLDM"]["versions"][-1])
    v2["applies_from_expiry"] = "2030-01-01"
    v2["quote_unit_grams"]["value"] = 1          # quotation unit changes between series
    cfg["products"]["GOLDM"]["versions"].append(v2)
    changed = assess_comparability({"kind": "calendar", "product": "GOLDM"}, SpecBook(cfg),
                                   pd.Timestamp("2029-12-05"), pd.Timestamp("2030-01-05"))
    assert changed.status == "INVALID"
