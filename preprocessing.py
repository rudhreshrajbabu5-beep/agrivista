"""
preprocessing.py
------------------
Lightweight, reusable dataframe filtering utilities shared across
dashboard pages. All filters are additive (AND logic) and no-op when a
selection is empty, so a UI edge case (e.g. clearing a multiselect) can
never crash the app or silently show a blank dataset.
"""

from __future__ import annotations

import pandas as pd


def apply_filters(
    df: pd.DataFrame,
    crops=None,
    states=None,
    seasons=None,
    year_range=None,
) -> pd.DataFrame:
    """Apply a standard set of filters. Any filter that is None or an
    empty list/tuple is treated as 'no filter'."""
    out = df

    if crops and "Crop" in out.columns:
        out = out[out["Crop"].isin(crops)]

    if states and "State" in out.columns:
        out = out[out["State"].isin(states)]

    if seasons and "Season" in out.columns:
        out = out[out["Season"].isin(seasons)]

    if year_range and "Crop_Year" in out.columns:
        lo, hi = year_range
        out = out[(out["Crop_Year"] >= lo) & (out["Crop_Year"] <= hi)]

    return out


def safe_unique(df: pd.DataFrame, col: str) -> list:
    if col not in df.columns:
        return []
    return sorted(df[col].dropna().unique().tolist())


def safe_dataframe(df: pd.DataFrame) -> bool:
    """True if a dataframe has enough rows to safely chart/analyze."""
    return df is not None and len(df) > 0
