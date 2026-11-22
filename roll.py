"""Globex session dates and the sticky volume roll.

The replay keeps one outright at a time. A roll is decided from finished
sessions only: once the next quarter trades more than the current one, the
following session uses the next quarter and does not flip back.
"""

from __future__ import annotations

import pandas as pd

NY = "America/New_York"
MONTH = {"G": 2, "H": 3, "J": 4, "M": 6, "Q": 8, "U": 9, "Z": 12}


def globex_trade_date(ts: pd.Series) -> pd.Series:
    """CME Globex session date.

    Equity-index futures open at 18:00 ET. Bars from 18:00 onward belong to the
    next calendar date's session, so Sunday evening is Monday's trade_date.
    The day boundary is the 17:00–18:00 ET maintenance break.
    """
    local = ts.dt.tz_convert(NY).dt.tz_localize(None)
    return (local + pd.Timedelta(hours=6)).dt.normalize()


def _resolve_year(yy: str, bar_year: int) -> int:
    """Map a Databento year code (0 / 00 / 2024) onto a full year near the bar."""
    n = len(yy)
    v = int(yy)
    if n >= 4:
        return v
    if n == 2:
        return 2000 + v
    y = (bar_year // 10) * 10 + v
    if y < bar_year - 1:
        y += 10
    if y > bar_year + 8:
        y -= 10
    return y


def _next_benchmark_month(current: int) -> int:
    """Next NQ quarterly month. H, M, U, Z are three months apart."""
    return current + 3


def roll_schedule(volumes: pd.DataFrame, root: str) -> pd.DataFrame:
    """Sticky first volume cross. The roll takes effect the next session."""
    v = volumes[volumes["root"] == root].copy()
    if v.empty:
        raise ValueError(f"No volume rows for {root}")

    sym = (
        v.groupby("contract_month")["symbol"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )
    piv = (
        v.pivot_table(
            index="trade_date",
            columns="contract_month",
            values="volume",
            aggfunc="sum",
        )
        .fillna(0.0)
        .sort_index()
    )
    dates = list(piv.index)
    current = int(piv.iloc[0].idxmax())
    pending = None
    rows = []

    for date in dates:
        if pending is not None:
            current = pending
            pending = None

        nxt = _next_benchmark_month(current)
        vol_a = float(piv.at[date, current]) if current in piv.columns else 0.0
        vol_b = float(piv.at[date, nxt]) if nxt in piv.columns else 0.0
        rows.append(
            {
                "root": root,
                "trade_date": date,
                "active_symbol": sym.get(current, f"{root}?"),
                "active_month": current,
                "next_symbol": sym.get(nxt, ""),
                "next_month": nxt if nxt in piv.columns else pd.NA,
                "front_volume": vol_a,
                "next_volume": vol_b,
                "crossed": bool(vol_b > vol_a and nxt in piv.columns),
            }
        )
        if vol_b > vol_a and nxt in piv.columns:
            pending = nxt

    sched = pd.DataFrame(rows)
    prev = sched["active_symbol"].shift(1)
    sched["is_roll"] = sched["active_symbol"].ne(prev) & prev.notna()
    return sched
