import pandas as pd
import pytest

from aurum.ingestion import Store, map_columns, parse_and_validate, parse_date

HEADER = "Date,Instrument Name,Symbol,Expiry Date,Option Type,Open,High,Low,Close,Previous Close,Volume(Lots),Open Interest(Lots)\n"
PRODUCTS = {"GOLD", "GOLDM", "GOLDPETAL", "GOLDGUINEA"}


def write(tmp_path, body, name="b.csv"):
    p = tmp_path / name
    p.write_text(HEADER + body, encoding="utf-8")
    return p


def test_parse_date_formats():
    assert parse_date("05 Jan 2024") == pd.Timestamp(2024, 1, 5)
    assert parse_date("05-JAN-2024") == pd.Timestamp(2024, 1, 5)
    assert parse_date("05/01/2024") == pd.Timestamp(2024, 1, 5)  # day-first, never US
    assert parse_date("2024-01-05") == pd.Timestamp(2024, 1, 5)
    assert parse_date("31/02/2024") is None
    assert parse_date("") is None


def test_missing_required_column_fails_loudly():
    with pytest.raises(ValueError, match="missing"):
        map_columns(["Date", "Symbol", "Close"])


def test_rows_are_accepted_rejected_or_out_of_scope(tmp_path):
    body = (
        "02 Jan 2024,FUTCOM, goldm ,05 Feb 2024,,100,110,90,105,100,10,50\n"   # ok (symbol spaces/case)
        "02 Jan 2024,FUTCOM,GOLDM,05 Mar 2024,,100,90,110,105,100,10,50\n"     # high < low
        "02 Jan 2024,FUTCOM,GOLDM,05 Apr 2024,,100,110,90,0,100,10,50\n"       # non-positive close
        "02 Jan 2024,FUTCOM,GOLDM,05 Dec 2023,,100,110,90,105,100,10,50\n"     # expired
        "xx,FUTCOM,GOLDM,05 Feb 2024,,100,110,90,105,100,10,50\n"              # bad date
        "02 Jan 2024,FUTCOM,SILVERM,05 Feb 2024,,100,110,90,105,100,10,50\n"   # out of scope
        "02 Jan 2024,OPTFUT,GOLDM,05 Feb 2024,CE,100,110,90,105,100,10,50\n"   # option: out of scope
        "02 Jan 2024,FUTCOM,GOLDM,05 Feb 2024,,100,110,90,120,100,10,50\n"     # close outside H/L
    )
    r = parse_and_validate(write(tmp_path, body), "SYNTHETIC", PRODUCTS)
    assert r.rows_read == 8
    assert r.out_of_scope == 2
    assert len(r.accepted) + len(r.rejected) + r.out_of_scope == r.rows_read
    reasons = sorted(r.rejected["reason"])
    assert "high_below_low" in reasons and "non_positive_close" in reasons
    assert "expired_contract" in reasons and "unparseable_date" in reasons
    assert "close_outside_high_low" in reasons
    assert r.accepted.iloc[0]["symbol"] == "GOLDM"
    assert r.accepted.iloc[0]["source_row"] == 2


def test_duplicates_identical_vs_conflicting(tmp_path):
    body = (
        "02 Jan 2024,FUTCOM,GOLDM,05 Feb 2024,,100,110,90,105,100,10,50\n"
        "02 Jan 2024,FUTCOM,GOLDM,05 Feb 2024,,100,110,90,105,100,10,50\n"   # identical -> drop extra
        "02 Jan 2024,FUTCOM,GOLD,05 Feb 2024,,100,110,90,105,100,10,50\n"
        "02 Jan 2024,FUTCOM,GOLD,05 Feb 2024,,100,110,90,106,100,10,50\n"    # conflicting -> reject both
    )
    r = parse_and_validate(write(tmp_path, body), "SYNTHETIC", PRODUCTS)
    assert len(r.accepted) == 1 and r.accepted.iloc[0]["symbol"] == "GOLDM"
    assert (r.rejected["reason"] == "duplicate_conflicting_values").sum() == 2
    assert (r.rejected["reason"] == "duplicate_identical").sum() == 1


def test_reingest_is_idempotent_and_raw_untouched(tmp_path):
    body = "02 Jan 2024,FUTCOM,GOLDM,05 Feb 2024,,100,110,90,105,100,10,50\n"
    p = write(tmp_path, body)
    before = p.read_bytes()
    store = Store("SYNTHETIC", base=tmp_path / "store")
    store.ingest(p, PRODUCTS)
    second = store.ingest(p, PRODUCTS)
    assert second.skipped_already_ingested
    assert len(store.observations()) == 1
    assert len(store.manifest()) == 1
    assert p.read_bytes() == before

    # a re-download with the same rows in a different order is recognised and skipped
    lines = (HEADER + body + "03 Jan 2024,FUTCOM,GOLD,05 Feb 2024,,1,1,1,1,1,1,1\n").splitlines(keepends=True)
    pa, pb = tmp_path / "x.csv", tmp_path / "x (1).csv"
    pa.write_text("".join(lines), encoding="utf-8")
    pb.write_text("".join([lines[0], lines[2], lines[1]]), encoding="utf-8")
    store.ingest(pa, PRODUCTS)
    assert store.ingest(pb, PRODUCTS).skipped_already_ingested

    # a different file overlapping the same (date, contract) does not duplicate
    p2 = write(tmp_path, body + "03 Jan 2024,FUTCOM,GOLDM,05 Feb 2024,,100,110,90,106,105,10,50\n", "c.csv")
    store.ingest(p2, PRODUCTS)
    assert len(store.observations()) == 3  # GOLDM 2 Jan + GOLD 3 Jan (from x.csv) + GOLDM 3 Jan


def test_real_mcx_layout(tmp_path):
    """Layout observed in real BhavCopyDateWise files: quoted fields, padded symbols, '-' option
    type for futures, DDMONYYYY expiry, blank O/H/L on no-trade rows, extra text columns.
    (Numbers are made up; no exchange data is stored in the repo.)"""
    header = ('Date,Instrument Name,Symbol,Expiry Date,Option Type,Strike Price,Open,High,Low,Close,'
              "Previous Close,Volume(Lots),Volume(In 000's),Value(Lacs),Open Interest(Lots)\n")
    body = (
        '"22 Oct 2026","FUTCOM","GOLDM        ","05NOV2026","-","0","100.00","110.00","90.00","105.00","104.00","10","1.000 GRMS ","10.50","20"\n'
        '"22 Oct 2026","FUTCOM","GOLD         ","05AUG2027","-","0","","","","200.00","200.00","0","0.000 GRMS ","0.00","0"\n'
        '"22 Oct 2026","OPTFUT","GOLDM        ","05NOV2026","CE","150000","1.00","1.00","1.00","1.00","1.00","1","0.1 GRMS ","0.01","1"\n'
        '"22 Oct 2026","FUTCOM","ALUMINI      ","30OCT2026","-","0","1.00","1.00","1.00","1.00","1.00","1","1 KGS ","0.01","1"\n'
    )
    p = tmp_path / "BhavCopyDateWise_22102026.csv"
    p.write_text(header + body, encoding="utf-8")
    r = parse_and_validate(p, "REAL", PRODUCTS)
    assert len(r.accepted) == 2 and r.out_of_scope == 2 and r.rejected.empty
    a = r.accepted.set_index("symbol")
    assert a.loc["GOLDM", "expiry"] == pd.Timestamp(2026, 11, 5)
    assert a.loc["GOLDM", "value_lacs"] == 10.5
    assert pd.isna(a.loc["GOLD", "high"]) and a.loc["GOLD", "volume_lots"] == 0  # no-trade row kept


def test_cross_file_duplicates_identical_vs_conflicting(tmp_path):
    """Audit P7: a conflicting row in a later file used to be silently discarded."""
    store = Store("SYNTHETIC", base=tmp_path / "store")
    store.ingest(write(tmp_path, "02 Jan 2026,FUTCOM,GOLDM,05 Feb 2026,-,100,110,90,105,100,10,50\n", "a.csv"), PRODUCTS)
    same = store.ingest(write(tmp_path, "02 Jan 2026,FUTCOM,GOLDM,05 Feb 2026,-,100,110,90,105,100,10,50\n"
                              "05 Jan 2026,FUTCOM,GOLDM,05 Feb 2026,-,100,110,90,106,105,10,50\n", "b.csv"), PRODUCTS)
    s = same.summary()
    assert s["duplicates_existing"] == 1 and s["accepted"] == 1 and s["rejected"] == 0
    conflict = store.ingest(write(tmp_path, "02 Jan 2026,FUTCOM,GOLDM,05 Feb 2026,-,100,110,90,107,100,10,50\n",
                                  "c.csv"), PRODUCTS)
    c = conflict.summary()
    assert c["rejected"] == 1 and c["rejection_reasons"] == {"conflicts_with_existing_observation": 1}
    stored = store.observations()
    assert len(stored) == 2 and stored.loc[stored["date"] == "2026-01-02", "close"].item() == 105  # first kept
    for m in store.manifest():                      # every row of every file accounted for
        assert m["accepted"] + m["rejected"] + m["out_of_scope"] + m["duplicates_existing"] == m["rows_read"]


def test_gitignore_keeps_restricted_and_secret_files_out(tmp_path):
    import shutil
    import subprocess
    from aurum.config import ROOT
    rules = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("data/raw/", "data/processed/real/", ".venv/", ".env", "docs/specs/*.pdf"):
        assert pattern in rules, pattern
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    shutil.copy(ROOT / ".gitignore", repo / ".gitignore")
    paths = ["data/raw/BhavCopyDateWise_01102026.csv", "data/processed/real/observations.parquet", ".env",
             "docs/specs/gold-petal.pdf", ".venv/x", "aurum/analytics.py"]
    for p in paths:
        (repo / p).parent.mkdir(parents=True, exist_ok=True)
        (repo / p).write_text("x")
    out = subprocess.run(["git", "-C", str(repo), "check-ignore", *paths], capture_output=True, text=True).stdout
    ignored = set(out.split())
    assert ignored == set(paths) - {"aurum/analytics.py"}     # code is tracked, data/secrets are not


def test_unsupported_file_format_fails_clearly(tmp_path):
    p = tmp_path / "BhavCopy.xlsx"
    p.write_bytes(b"PK\x03\x04 not a csv")
    with pytest.raises(ValueError, match="Unsupported file format '.xlsx'"):
        parse_and_validate(p, "REAL", PRODUCTS)


def test_invalid_mode_rejected(tmp_path):
    with pytest.raises(ValueError):
        parse_and_validate(write(tmp_path, ""), "LIVE", PRODUCTS)
