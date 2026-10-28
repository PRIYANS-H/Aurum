import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aurum.config import load_yaml  # noqa: E402
from aurum.contracts import SpecBook  # noqa: E402


@pytest.fixture(scope="session")
def specs_cfg():
    return load_yaml("contracts.yaml")


@pytest.fixture(scope="session")
def book(specs_cfg):
    return SpecBook(specs_cfg)


@pytest.fixture(scope="session")
def research():
    return load_yaml("research.yaml")


@pytest.fixture(scope="session")
def pairs_cfg():
    return load_yaml("pairs.yaml")


def make_obs(rows):
    """rows: (date, symbol, expiry, close, volume) -> observation frame like the Store returns."""
    df = pd.DataFrame(rows, columns=["date", "symbol", "expiry", "close", "volume_lots"])
    df["date"] = pd.to_datetime(df["date"])
    df["expiry"] = pd.to_datetime(df["expiry"])
    df["contract_id"] = df["symbol"] + "_" + df["expiry"].dt.strftime("%Y%m%d")
    for c in ["open", "high", "low", "settle"]:
        df[c] = float("nan")
    df["prev_close"] = df.groupby("contract_id")["close"].shift(1)
    df["oi_lots"] = 100
    return df
