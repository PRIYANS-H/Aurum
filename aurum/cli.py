"""Command line entry point.

    python -m aurum.cli sample                 # generate + ingest SYNTHETIC data
    python -m aurum.cli ingest data/raw/*.csv  # ingest REAL bhavcopy files you downloaded
    python -m aurum.cli analyse --mode SYNTHETIC --pair GOLDM_CAL_1_2 [--report out.md]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from .analytics import analyse_pair
from .config import DATA_DIR
from .ingestion import Store
from .pipeline import Context, evidence_card, markdown_report, run_validation
from .sample_data import write_sample_files


def cmd_sample(args) -> None:
    ctx = Context.load()
    store = Store("SYNTHETIC")
    store.clear()
    paths = write_sample_files(DATA_DIR / "sample", seed=args.seed)
    for p in paths:
        print(json.dumps(store.ingest(p, ctx.book.products).summary(), default=str))


def cmd_ingest(args) -> None:
    ctx = Context.load()
    store = Store("REAL")
    files = [Path(f) for pattern in args.files for f in glob.glob(pattern)]
    if not files:
        sys.exit("no files matched")
    for p in files:
        try:
            print(json.dumps(store.ingest(p, ctx.book.products).summary(), default=str))
        except ValueError as exc:   # unsupported format or unrecognised layout: report, continue with others
            print(json.dumps({"source_file": p.name, "error": str(exc)}))


def cmd_analyse(args) -> None:
    ctx = Context.load()
    store = Store(args.mode)
    obs = store.observations()
    if obs.empty:
        sys.exit(f"no {args.mode} observations ingested yet")
    pairs = [args.pair] if args.pair else list(ctx.pairs)
    for name in pairs:
        df = analyse_pair(obs, ctx.pairs[name], ctx.book, ctx.settings, ctx.research)
        val = run_validation(ctx, obs, df, ctx.pairs[name]) if len(df) else None
        card = evidence_card(ctx, args.mode, name, obs, df, val)
        c = card["classification"]
        print(f"\n=== {name} [{args.mode}] ===")
        print(f"status: {c['status']}  (alert {c['alert_status']}, validation {c['validation_status']}, "
              f"comparability {c['comparability_status']})")
        for r in c["reasons"]:
            print(f"  - {r}")
        if val:
            p = val["protocol"]
            print(f"  protocol: {p['conclusion']}")
            for m in val["feasibility"]["messages"]:
                print(f"    ! {m}")
            for k in ("candidate", "baseline"):
                s = val[k]["summary"]
                print(f"  {k:9s}: trades={s['trades']:3d} gross={s['gross_pnl']:>12,.0f} net={s['net_pnl']:>12,.0f}")
        if args.report:
            rp = Path(args.report)
            out = rp if len(pairs) == 1 else rp.with_name(f"{rp.stem}_{name}.md")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(markdown_report(ctx, card, store), encoding="utf-8")
            print(f"  report -> {out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="aurum")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="generate and ingest SYNTHETIC data")
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(func=cmd_sample)
    i = sub.add_parser("ingest", help="ingest REAL bhavcopy CSV files")
    i.add_argument("files", nargs="+")
    i.set_defaults(func=cmd_ingest)
    a = sub.add_parser("analyse", help="analyse pairs and run walk-forward validation")
    a.add_argument("--mode", choices=["REAL", "SYNTHETIC"], default="SYNTHETIC")
    a.add_argument("--pair")
    a.add_argument("--report")
    a.set_defaults(func=cmd_analyse)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
