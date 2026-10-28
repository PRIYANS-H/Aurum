"""Smoke test: synthetic files -> ingestion -> analysis -> walk-forward -> protocol -> card -> report."""

import copy

from aurum.analytics import analyse_pair
from aurum.ingestion import Store
from aurum.pipeline import Context, evidence_card, markdown_report, run_validation
from aurum.sample_data import write_sample_files
from aurum.signals import STATUSES


def test_pipeline_on_synthetic(tmp_path):
    ctx = Context.load()
    store = Store("SYNTHETIC", base=tmp_path / "store")
    for p in write_sample_files(tmp_path / "raw", seed=3):
        s = store.ingest(p, ctx.book.products).summary()
        assert s["accepted"] + s["rejected"] + s["out_of_scope"] + s["duplicates_existing"] == s["rows_read"]
    obs = store.observations()
    assert set(obs["data_mode"]) == {"SYNTHETIC"}

    research = copy.deepcopy(ctx.research)
    research["walk_forward"]["grid"] = {"entry_z": [2.0], "exit_z": [0.5], "window": [40]}  # keep test fast
    df = analyse_pair(obs, ctx.pairs["GOLDM_CAL_1_2"], ctx.book, ctx.settings, research)
    assert len(df) > 300
    assert df["a_contract_id"].nunique() > 5          # many contracts, identity preserved
    assert (df["b_expiry"] > df["a_expiry"]).all()

    val = run_validation(ctx, obs, df, ctx.pairs["GOLDM_CAL_1_2"], research)
    assert val["ran"] and val["feasibility"]["sufficient"]
    folds = val["candidate"]["folds"]
    assert len(folds) >= 3
    assert (folds["test_start"] > folds["train_end"]).all()
    assert (folds["test_start"].iloc[1:].values > folds["test_end"].iloc[:-1].values).all()
    hold = val["candidate"]["holdout"]
    assert hold is not None and hold["start"] > folds["test_end"].max()       # holdout after every fold
    for name in ("candidate", "baseline"):                                    # engine integrity guarantees
        assert val["integrity"][name] == {"same_bar_fills": 0, "tender_violations": 0, "stale_fills": 0}

    card = evidence_card(ctx, "SYNTHETIC", "GOLDM_CAL_1_2", obs, df, val)
    assert card["classification"]["status"] in STATUSES
    assert card["classification"]["validation_status"] == val["protocol"]["verdict"]   # one protocol everywhere
    assert all(s["file"] and s["row"] for s in card["sources"])                         # traceable to raw rows
    report = markdown_report(ctx, card, store)
    assert "SYNTHETIC DATA - NOT MCX MARKET DATA" in report
    assert val["protocol"]["conclusion"] in report


def test_infeasible_history_is_not_run(tmp_path):
    ctx = Context.load()
    store = Store("SYNTHETIC", base=tmp_path / "store")
    for p in write_sample_files(tmp_path / "raw", seed=3)[:1]:     # one year only
        store.ingest(p, ctx.book.products)
    obs = store.observations()
    df = analyse_pair(obs, ctx.pairs["GOLDM_CAL_1_2"], ctx.book, ctx.settings, ctx.research)
    df = df.iloc[:189]                                              # same length as the real history
    val = run_validation(ctx, obs, df, ctx.pairs["GOLDM_CAL_1_2"])
    assert not val["ran"] and val["protocol"]["conclusion"] == "INSUFFICIENT EVIDENCE"
    assert val["candidate"]["summary"]["trades"] == 0               # nothing fabricated


def test_empty_store_is_handled(tmp_path):
    ctx = Context.load()
    obs = Store("SYNTHETIC", base=tmp_path / "empty").observations()
    df = analyse_pair(obs, ctx.pairs["GOLDM_CAL_1_2"], ctx.book, ctx.settings, ctx.research)
    assert df.empty
    card = evidence_card(ctx, "SYNTHETIC", "GOLDM_CAL_1_2", obs, df, None)
    assert card["classification"]["status"] == "INVALID COMPARISON"
