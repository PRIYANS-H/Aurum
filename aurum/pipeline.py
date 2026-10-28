"""End-to-end orchestration shared by the CLI and the dashboard."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

from .analytics import FLAGS, active_flags, flag_evidence
from .backtest import PriceBook, sensitivity_grid, trade_metrics, walk_forward
from .config import load_yaml
from .contracts import SpecBook, assess_comparability
from .ingestion import Store
from .signals import Classification, classify
from .validation import cost_stress, evaluate_protocol, feasibility, integrity


@dataclass
class Context:
    specs_cfg: dict
    pairs_cfg: dict
    research: dict
    book: SpecBook

    @classmethod
    def load(cls) -> "Context":
        specs = load_yaml("contracts.yaml")
        return cls(specs, load_yaml("pairs.yaml"), load_yaml("research.yaml"), SpecBook(specs))

    @property
    def pairs(self) -> dict:
        return self.pairs_cfg["pairs"]

    @property
    def settings(self) -> dict:
        return self.pairs_cfg["settings"]


def pair_products(pair_cfg: dict) -> tuple[str, str]:
    if pair_cfg["kind"] == "calendar":
        return pair_cfg["product"], pair_cfg["product"]
    return pair_cfg["leg_a"], pair_cfg["leg_b"]


def _empty_run() -> dict:
    return {"folds": pd.DataFrame(), "trades": pd.DataFrame(), "summary": trade_metrics(pd.DataFrame()),
            "holdout": None, "plan": None, "skipped": []}


def run_validation(ctx: Context, obs: pd.DataFrame, df: pd.DataFrame, pair_cfg: dict,
                   research: dict | None = None, exploratory: bool = False) -> dict:
    """Feasibility check, then walk-forward for the AURUM candidate (adjusted measure + risk gates)
    and the naive baseline (no gates) on identical rows, costs and timing; then integrity checks,
    cost stress and the single acceptance protocol. (The expiry exclusion setting is already
    applied when `df` is built.)"""
    research = research or ctx.research
    feas = feasibility(len(df), research, exploratory)
    if not feas["runnable"]:
        val = {"candidate": _empty_run(), "baseline": _empty_run(), "sensitivity": pd.DataFrame(),
               "cost_stress": pd.DataFrame(), "integrity": {}, "feasibility": feas, "ran": False}
        val["protocol"] = evaluate_protocol(val, research, exploratory)
        return val
    pa, pb = pair_products(pair_cfg)
    last = df.iloc[-1]
    ta = ctx.book.spec(pa, last["a_expiry"]).tick_rs.value
    tb = ctx.book.spec(pb, last["b_expiry"]).tick_rs.value
    prices = PriceBook(obs)
    cand = walk_forward(df, obs, research, ta, tb, "metric", use_gates=True, prices=prices)
    base = walk_forward(df, obs, research, ta, tb, "baseline_metric", use_gates=False, prices=prices)
    dev_end = cand["plan"]["dev_n"]
    val = {"candidate": cand, "baseline": base, "feasibility": feas, "ran": True,
           "sensitivity": sensitivity_grid(df, obs, research, ta, tb, "metric", use_gates=True, end=dev_end,
                                           prices=prices),
           "cost_stress": cost_stress(cand["trades"], research)}
    val["integrity"] = integrity(val, prices, research)
    val["protocol"] = evaluate_protocol(val, research, exploratory)
    return val


def analysis_id(payload: dict) -> str:
    """Short reproducible identifier for an analysis configuration."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return "AUR-" + hashlib.sha256(blob).hexdigest()[:10].upper()


def data_quality(store: Store, obs: pd.DataFrame, research: dict | None = None) -> dict:
    """Dataset-level quality summary used by the status banner."""
    man = store.manifest()
    read = sum(m.get("rows_read", 0) for m in man)
    rejected = sum(m.get("rejected", 0) for m in man)
    accepted = sum(m.get("accepted", 0) for m in man)
    if obs.empty:
        return {"status": "NO DATA", "issues": ["no observations ingested"], "rejected": 0, "accepted": 0,
                "zero_volume_share": float("nan"), "max_gap_days": 0, "files": len(man), "trading_days": 0}
    dates = pd.Series(sorted(pd.to_datetime(obs["date"]).unique()))
    max_gap = int(dates.diff().dt.days.max()) if len(dates) > 1 else 0
    zero = float((obs["volume_lots"].fillna(0) <= 0).mean())
    issues = []
    rej_share = rejected / max(accepted + rejected, 1)
    if rej_share > 0.01:
        issues.append(f"{rej_share:.1%} of in-scope rows rejected")
    if max_gap > 5:
        issues.append(f"gap of {max_gap} calendar days between observations")
    if research is not None:
        f = feasibility(len(dates), research)
        if not f["sufficient"]:
            issues.append(f"{len(dates)} trading days: the configured walk-forward needs about {f['rows_needed']} "
                          f"({f['folds']} of {research['validation_pass']['min_folds']} required folds fit), so "
                          "strategy validation will be INCONCLUSIVE")
    return {"status": "OK" if not issues else "CHECK", "issues": issues, "rejected": rejected,
            "accepted": accepted, "rows_read": read, "zero_volume_share": zero, "max_gap_days": max_gap,
            "files": len(man), "trading_days": len(dates)}


def log_experiment(store: Store, record: dict) -> None:
    """Append every backtest run (not only good ones) to an experiment log."""
    path = store.dir / "experiments.jsonl"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def read_experiments(store: Store) -> pd.DataFrame:
    path = store.dir / "experiments.jsonl"
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()])


def evidence_card(ctx: Context, mode: str, pair_name: str, obs: pd.DataFrame, df: pd.DataFrame,
                  val: dict | None, as_of: pd.Timestamp | None = None) -> dict:
    """Everything the Signal Evidence Card shows, computed - never hand-entered."""
    pair_cfg = ctx.pairs[pair_name]
    protocol = val["protocol"] if val else None
    if df.empty:
        comp = assess_comparability(pair_cfg, ctx.book, pd.Timestamp.today())
        cls = classify(None, comp, ctx.research, protocol)
        return {"pair": pair_name, "mode": mode, "row": None, "comparability": asdict(comp) | {"usable": False},
                "classification": asdict(cls), "validation": val, "flags": [], "sources": []}
    sub = df if as_of is None else df[df["date"] <= as_of]
    row = sub.iloc[-1]
    comp = assess_comparability(pair_cfg, ctx.book, row["a_expiry"], row["b_expiry"])
    cls: Classification = classify(row, comp, ctx.research, protocol)
    hist = sub["metric"].dropna()
    flags = [{"key": k, "label": FLAGS[k]["label"], "evidence": flag_evidence(row, k), "effect": FLAGS[k]["effect"],
              "confirm": FLAGS[k]["confirm"], "gate": FLAGS[k]["gate"]} for k in active_flags(row, gate_only=False)]
    sources = [{"leg": leg.upper(), "contract": row[f"{leg}_contract_id"], "file": row.get(f"{leg}_source_file"),
                "row": None if pd.isna(row.get(f"{leg}_source_row")) else int(row[f"{leg}_source_row"])}
               for leg in ("a", "b")]
    return {
        "pair": pair_name, "mode": mode, "kind": pair_cfg["kind"], "rationale": pair_cfg["rationale"].strip(),
        "row": row, "comparability": asdict(comp) | {"usable": comp.usable}, "classification": asdict(cls),
        "metric_label": df.attrs.get("metric_label", "metric"), "flags": flags, "sources": sources,
        "history": {"n": int(len(hist)), "min": float(hist.min()) if len(hist) else np.nan,
                    "median": float(hist.median()) if len(hist) else np.nan,
                    "max": float(hist.max()) if len(hist) else np.nan,
                    "first": sub["date"].iloc[0], "last": row["date"]},
        "validation": val,
    }


def _fmt(v, nd=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    return str(v)


def markdown_report(ctx: Context, card: dict, store: Store) -> str:
    """Reproducible research report built only from computed values."""
    lines = []
    banner = ("> **SYNTHETIC DATA - NOT MCX MARKET DATA. Results below test the software only.**\n"
              if card["mode"] == "SYNTHETIC" else "")
    lines += [f"# AURUM research report - {card['pair']}", "", banner,
              f"Generated (UTC): {datetime.now(timezone.utc):%Y-%m-%d %H:%M}  ",
              f"Data mode: **{card['mode']}**  ", ""]
    lines += ["## Data provenance", "", "| file | sha256 (first 12) | rows | accepted | rejected | out of scope |",
              "|---|---|---|---|---|---|"]
    for m in store.manifest():
        lines.append(f"| {m['source_file']} | {m['sha256'][:12]} | {m['rows_read']} | {m['accepted']} | "
                     f"{m['rejected']} | {m['out_of_scope']} |")
    comp = card["comparability"]
    lines += ["", "## Comparability", "", f"Status: **{comp['status']}**", "", card.get("rationale", ""), ""]
    for k in ("reasons", "adjustments", "confounders", "unverified", "missing"):
        if comp.get(k):
            lines += [f"**{k.capitalize()}**", ""] + [f"- {x}" for x in comp[k]] + [""]
    row = card["row"]
    cls = card["classification"]
    if row is not None:
        lines += ["## Observation", "",
                  f"- Date: {row['date']:%Y-%m-%d} (end of day)",
                  f"- Leg A: {row['a_contract_id']} close {_fmt(row['a_close'])} (normalised {_fmt(row['a_norm'])})",
                  f"- Leg B: {row['b_contract_id']} close {_fmt(row['b_close'])} (normalised {_fmt(row['b_norm'])})",
                  f"- {card['metric_label']}: {_fmt(row['metric'], 4)}",
                  f"- Rolling mean / std ({int(row['n_hist'])} prior valid obs): {_fmt(row['mean'], 4)} / {_fmt(row['std'], 4)}",
                  f"- z-score: {_fmt(row['z'])}; percentile: {_fmt(row['percentile'], 0)}", "",
                  "**Source observations**", ""]
        lines += [f"- Leg {s['leg']}: {s['contract']} from {s['file']} row {s['row']}" for s in card["sources"]]
        lines += [""]
        if card["flags"]:
            lines += ["**Flags on this date**", ""]
            lines += [f"- {f['label']}: {f['evidence']}. Effect: {f['effect']}. Would confirm: {f['confirm']}."
                      for f in card["flags"]]
            lines += [""]
    lines += ["## Classification", "", f"**{cls['status']}** (alert: {cls['alert_status']}, validation: "
              f"{cls['validation_status']}, comparability: {cls['comparability_status']})", ""]
    lines += [f"- {r}" for r in cls["reasons"]]
    lines += ["", "**Untested / unavailable**", ""] + [f"- {u}" for u in cls["untested"]]
    val = card["validation"]
    if val:
        p = val["protocol"]
        lines += ["", "## Validation protocol", "", f"**{p['conclusion']}** — {p['explanation']}", ""]
        lines += [f"- {m}" for m in val["feasibility"]["messages"]] + [""]
        lines += [p["criteria"].to_markdown(index=False), ""]
        for name in ("candidate", "baseline"):
            s = val[name]["summary"]
            lines += [f"- {'AURUM' if name == 'candidate' else 'Baseline'}: {s['trades']} OOS trades, gross Rs "
                      f"{_fmt(s['gross_pnl'], 0)}, costs Rs {_fmt(s['costs'], 0)}, net Rs {_fmt(s['net_pnl'], 0)}"]
        if not val["candidate"]["folds"].empty:
            lines += ["", "### Folds (AURUM)", "", val["candidate"]["folds"].to_markdown(index=False), ""]
    lines += ["## Assumptions", "", "```yaml", yaml.safe_dump(ctx.research, sort_keys=False), "```", "",
              "## Specifications (all versions; selected per contract by expiry)", "",
              ctx.book.table().to_markdown(index=False), ""]
    return "\n".join(lines)
