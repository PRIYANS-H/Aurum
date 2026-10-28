"""Bhavcopy ingestion: parse -> validate -> record provenance -> store.

Design rules
- Raw files are read-only. We never rewrite them.
- Every input row ends up either ACCEPTED, REJECTED (with a reason) or OUT_OF_SCOPE
  (not a gold future); nothing disappears silently.
- Each accepted row keeps source_file / source_row so any number can be traced to its origin.
- Re-ingesting the same file (same SHA-256) is a no-op; overlapping files are de-duplicated
  on (date, contract_id).

The exact MCX bhavcopy column names could not be verified (the site blocks automated
access), so columns are matched through COLUMN_ALIASES. Unknown layouts fail loudly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_MODES, processed_dir

# canonical name -> accepted raw header spellings (compared case/space-insensitively)
COLUMN_ALIASES: dict[str, list[str]] = {
    "date": ["date", "trade date", "tradedate", "trading date"],
    "instrument": ["instrument name", "instrument", "instrumentname"],
    "symbol": ["symbol", "commodity", "symbol name"],
    "expiry": ["expiry date", "expiry", "expirydate", "contract expiry"],
    "option_type": ["option type", "optiontype", "opttype"],
    "open": ["open", "open price"],
    "high": ["high", "high price"],
    "low": ["low", "low price"],
    "close": ["close", "close price", "closing price"],
    "prev_close": ["previous close", "prev close", "previous close price", "prevclose"],
    "settle": ["settlement price", "settle price", "settle"],
    "volume_lots": ["volume(lots)", "volume (lots)", "volume", "traded quantity (lots)"],
    "oi_lots": ["open interest(lots)", "open interest (lots)", "open interest", "oi"],
    "value_lacs": ["value(lacs)", "value (lacs)", "value(lakhs)", "value (lakhs)", "value(lakh)"],
}
# Real MCX bhavcopy (verified 2026-10-23 on BhavCopyDateWise_*.csv) uses "-" as the option type
# of futures rows and "FUTCOM" as the instrument name.
FUTURE_OPTION_TYPES = {"", "-", "XX", "FUT", "NAN"}
SUPPORTED_FORMATS = (".csv",)
REQUIRED = ["date", "symbol", "expiry", "close"]
PRICE_COLS = ["open", "high", "low", "close", "prev_close", "settle"]
DATE_FORMATS = ["%d %b %Y", "%d-%b-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d%b%Y"]

STORE_COLUMNS = [
    "date", "symbol", "expiry", "contract_id", "open", "high", "low", "close", "prev_close",
    "settle", "volume_lots", "oi_lots", "value_lacs", "data_mode", "source_file", "source_row",
]


def _norm_header(h: str) -> str:
    return " ".join(str(h).strip().lower().split())


def map_columns(raw_columns: list[str]) -> dict[str, str]:
    """Return {raw_header: canonical_name}. Raises if a required column is missing."""
    lookup = {_norm_header(a): canon for canon, aliases in COLUMN_ALIASES.items() for a in aliases}
    mapping: dict[str, str] = {}
    for col in raw_columns:
        canon = lookup.get(_norm_header(col))
        if canon and canon not in mapping.values():
            mapping[col] = canon
    missing = [c for c in REQUIRED if c not in mapping.values()]
    if missing:
        raise ValueError(
            f"Unrecognised bhavcopy layout: missing {missing}. Columns found: {list(raw_columns)}. "
            "Add the header spelling to COLUMN_ALIASES in aurum/ingestion.py."
        )
    return mapping


def parse_date(value) -> pd.Timestamp | None:
    """Parse a date using an explicit list of formats (no ambiguous guessing)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = " ".join(str(value).strip().split())
    if not text:
        return None
    for fmt in DATE_FORMATS:
        try:
            return pd.Timestamp(datetime.strptime(text, fmt))
        except ValueError:
            continue
    return None


def _to_number(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    cleaned = cleaned.replace({"": np.nan, "-": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_sha256(path: Path) -> str:
    """Hash of the file's sorted lines: identical for re-downloads that only reorder rows
    (observed with MCX 'BhavCopyDateWise_... (1).csv' duplicates)."""
    with open(path, "rb") as fh:
        lines = sorted(line.strip() for line in fh if line.strip())
    return hashlib.sha256(b"\n".join(lines)).hexdigest()


@dataclass
class IngestResult:
    source_file: str
    sha256: str
    rows_read: int = 0
    accepted: pd.DataFrame = field(default_factory=pd.DataFrame)
    rejected: pd.DataFrame = field(default_factory=pd.DataFrame)
    out_of_scope: int = 0
    duplicates_existing: int = 0   # identical to an observation already in the store (not stored twice)
    skipped_already_ingested: bool = False
    # Invariant for every ingested file: rows_read == accepted + rejected + out_of_scope + duplicates_existing

    def summary(self) -> dict:
        reasons = (
            self.rejected["reason"].value_counts().to_dict() if len(self.rejected) else {}
        )
        return {
            "source_file": self.source_file,
            "sha256": self.sha256,
            "rows_read": self.rows_read,
            "accepted": len(self.accepted),
            "rejected": len(self.rejected),
            "out_of_scope": self.out_of_scope,
            "duplicates_existing": self.duplicates_existing,
            "rejection_reasons": reasons,
            "skipped_already_ingested": self.skipped_already_ingested,
        }


def parse_and_validate(path: Path, data_mode: str, products: set[str]) -> IngestResult:
    """Parse one raw bhavcopy CSV and split it into accepted / rejected / out-of-scope rows."""
    if data_mode not in DATA_MODES:
        raise ValueError(f"data_mode must be one of {DATA_MODES}")
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported file format {path.suffix!r} for {path.name}: AURUM ingests MCX bhavcopy "
                         f"{', '.join(SUPPORTED_FORMATS)} files (download the CSV, not PDF or Excel).")
    result = IngestResult(source_file=path.name, sha256=file_sha256(path))

    raw = pd.read_csv(path, dtype=str, keep_default_na=False, skipinitialspace=True)
    result.rows_read = len(raw)
    mapping = map_columns(list(raw.columns))
    df = raw.rename(columns=mapping)[list(mapping.values())].copy()
    df["source_row"] = np.arange(2, len(df) + 2)  # 1-based, header is row 1
    df["source_file"] = path.name
    df["data_mode"] = data_mode

    for col in COLUMN_ALIASES:
        if col not in df.columns:
            df[col] = np.nan if col not in ("instrument", "option_type") else ""

    # --- scope: gold futures only -------------------------------------------------------
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    opt = df["option_type"].astype(str).str.strip().str.upper()
    instr = df["instrument"].astype(str).str.strip().str.upper()
    is_future = opt.isin(FUTURE_OPTION_TYPES) & ~instr.str.startswith("OPT")
    in_scope = df["symbol"].isin(products) & is_future
    result.out_of_scope = int((~in_scope).sum())
    df = df[in_scope].copy()

    # --- types --------------------------------------------------------------------------
    df["date"] = pd.to_datetime(df["date"].map(parse_date))
    df["expiry"] = pd.to_datetime(df["expiry"].map(parse_date))
    for col in PRICE_COLS + ["volume_lots", "oi_lots", "value_lacs"]:
        df[col] = _to_number(df[col])

    # --- row-level validation: first failing rule is the recorded reason ---------------
    reason = pd.Series("", index=df.index, dtype=object)

    def flag(mask: pd.Series, text: str) -> None:
        reason.loc[mask & (reason == "")] = text

    flag(df["date"].isna(), "unparseable_date")
    flag(df["expiry"].isna(), "unparseable_expiry")
    flag(df["close"].isna(), "missing_close")
    flag(df["close"] <= 0, "non_positive_close")
    for col in ["open", "high", "low"]:
        flag(df[col] < 0, f"negative_{col}")
    hl = df["high"].notna() & df["low"].notna() & (df["high"] > 0) & (df["low"] > 0)
    flag(hl & (df["high"] < df["low"]), "high_below_low")
    flag(hl & ((df["close"] > df["high"]) | (df["close"] < df["low"])), "close_outside_high_low")
    flag(df["expiry"].notna() & df["date"].notna() & (df["expiry"] < df["date"]), "expired_contract")
    flag((df["volume_lots"] < 0) | (df["oi_lots"] < 0), "negative_volume_or_oi")

    df["contract_id"] = df["symbol"] + "_" + df["expiry"].map(
        lambda d: d.strftime("%Y%m%d") if isinstance(d, pd.Timestamp) else "NA"
    )

    # --- duplicates within the file -----------------------------------------------------
    ok = reason == ""
    key = ["date", "contract_id"]
    value_cols = ["open", "high", "low", "close", "volume_lots", "oi_lots"]
    dup_any = ok & df.duplicated(key, keep=False)
    if dup_any.any():
        grp = df[dup_any].groupby(key)[value_cols]
        conflicting = grp.nunique(dropna=False).max(axis=1) > 1
        conflict_keys = set(conflicting[conflicting].index)
        is_conflict = df[key].apply(tuple, axis=1).isin(conflict_keys) & dup_any
        flag(is_conflict, "duplicate_conflicting_values")
        identical_extra = dup_any & ~is_conflict & df.duplicated(key, keep="first")
        flag(identical_extra, "duplicate_identical")

    df["reason"] = reason
    result.rejected = df[df["reason"] != ""].copy()
    result.accepted = df[df["reason"] == ""][STORE_COLUMNS].copy()
    return result


class Store:
    """Processed observation store for one data mode (Parquet + JSON manifest)."""

    def __init__(self, mode: str, base: Path | None = None):
        self.mode = mode
        self.dir = processed_dir(mode) if base is None else Path(base)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.obs_path = self.dir / "observations.parquet"
        self.rej_path = self.dir / "rejections.csv"
        self.manifest_path = self.dir / "manifest.json"

    def manifest(self) -> list[dict]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return []

    def observations(self) -> pd.DataFrame:
        if not self.obs_path.exists():
            return pd.DataFrame(columns=STORE_COLUMNS)
        return pd.read_parquet(self.obs_path)

    def rejections(self) -> pd.DataFrame:
        if not self.rej_path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.rej_path)

    def ingest(self, path: Path, products: set[str], origin: str = "file") -> IngestResult:
        """origin: 'file' (placed in data/raw by the user, CLI) or 'upload' (dashboard upload)."""
        sha, csha = file_sha256(Path(path)), content_sha256(Path(path))
        if any(m["sha256"] == sha or m.get("content_sha256") == csha for m in self.manifest()):
            res = IngestResult(source_file=Path(path).name, sha256=sha, skipped_already_ingested=True)
            return res

        res = parse_and_validate(Path(path), self.mode, products)
        existing = self.observations()
        if len(existing) and len(res.accepted):
            # Rows for a (date, contract) already in the store: identical -> counted duplicate;
            # different values -> rejected with a reason (the stored, first-ingested row is kept).
            key = ["date", "contract_id"]
            vals = ["open", "high", "low", "close", "volume_lots", "oi_lots"]
            ex = existing[key + vals].copy()
            ex["date"] = pd.to_datetime(ex["date"])
            m = res.accepted.reset_index().merge(ex, on=key, how="inner", suffixes=("", "_old"))
            if len(m):
                same = np.ones(len(m), dtype=bool)
                for v in vals:
                    a, b = m[v].astype(float), m[f"{v}_old"].astype(float)
                    same &= ((a == b) | (a.isna() & b.isna())).to_numpy()
                dup_idx = m.loc[same, "index"]
                conf_idx = m.loc[~same, "index"]
                res.duplicates_existing = int(len(dup_idx))
                if len(conf_idx):
                    conflict = res.accepted.loc[conf_idx].copy()
                    conflict["reason"] = "conflicts_with_existing_observation"
                    res.rejected = pd.concat([res.rejected, conflict], ignore_index=True)
                res.accepted = res.accepted.drop(index=list(dup_idx) + list(conf_idx))
        combined = pd.concat([existing, res.accepted], ignore_index=True) if len(existing) else res.accepted
        combined = combined.drop_duplicates(["date", "contract_id"], keep="first")  # safety net; none expected
        combined = combined.sort_values(["date", "symbol", "expiry"]).reset_index(drop=True)
        combined.to_parquet(self.obs_path, index=False)

        if len(res.rejected):
            rej = res.rejected.copy()
            rej.to_csv(self.rej_path, mode="a", header=not self.rej_path.exists(), index=False)

        entry = res.summary() | {"content_sha256": csha, "origin": origin,
                                 "ingested_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 "data_mode": self.mode}
        manifest = self.manifest() + [entry]
        self.manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        return res

    def clear(self) -> None:
        for p in (self.obs_path, self.rej_path, self.manifest_path):
            if p.exists():
                p.unlink()
