"""
analysis.py
------------
KPI computation and group-by helpers used across the dashboard. Every
number is computed live from whatever dataframe is passed in — nothing
here is hard-coded to a specific dataset value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import comparable_yield_df


def compute_kpis(df: pd.DataFrame) -> dict:
    """Headline KPIs for the Overview page. Yield- and Production-based
    KPIs use the comparable-unit subset of crops (Coconut excluded) so a
    single summed/averaged number never mixes nuts with tonnes."""
    kpis = {}
    comparable = comparable_yield_df(df)

    kpis["total_production"] = float(comparable["Production"].sum()) if "Production" in comparable else None
    kpis["avg_yield"] = float(comparable["Yield"].mean()) if "Yield" in comparable and len(comparable) else None
    kpis["num_crops"] = int(df["Crop"].nunique()) if "Crop" in df else None
    kpis["num_states"] = int(df["State"].nunique()) if "State" in df else None

    if "Crop" in comparable and "Yield" in comparable and len(comparable):
        top = comparable.groupby("Crop")["Yield"].mean().sort_values(ascending=False)
        if len(top):
            kpis["top_crop"] = top.index[0]
            kpis["top_crop_yield"] = float(top.iloc[0])
        else:
            kpis["top_crop"], kpis["top_crop_yield"] = None, None
    else:
        kpis["top_crop"], kpis["top_crop_yield"] = None, None

    if "Crop" in comparable and "Yield" in comparable and len(comparable) >= 1:
        from src.statistics import stability_ranking
        rank = stability_ranking(comparable)
        if len(rank):
            kpis["most_stable_crop"] = rank.iloc[0]["Crop"]
            kpis["most_stable_score"] = float(rank.iloc[0]["stability_score"])
        else:
            kpis["most_stable_crop"], kpis["most_stable_score"] = None, None
    else:
        kpis["most_stable_crop"], kpis["most_stable_score"] = None, None

    if "Crop_Year" in df.columns and len(df):
        kpis["year_min"] = int(df["Crop_Year"].min())
        kpis["year_max"] = int(df["Crop_Year"].max())

    return kpis


def production_by_crop(df: pd.DataFrame, top_n: int = None, comparable_only: bool = True) -> pd.DataFrame:
    data = comparable_yield_df(df) if comparable_only else df
    out = data.groupby("Crop", as_index=False)["Production"].sum().sort_values("Production", ascending=False)
    return out.head(top_n) if top_n else out


def avg_yield_by_crop(df: pd.DataFrame, top_n: int = None, comparable_only: bool = True) -> pd.DataFrame:
    data = comparable_yield_df(df) if comparable_only else df
    out = data.groupby("Crop", as_index=False)["Yield"].mean().sort_values("Yield", ascending=False)
    return out.head(top_n) if top_n else out


def yearly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Production and Yield are both aggregated on the comparable-unit
    subset (Coconut excluded) so the trend line never mixes nuts and
    tonnes within a single summed/averaged series."""
    comparable = comparable_yield_df(df)
    out = None
    if "Production" in comparable.columns:
        out = comparable.groupby("Crop_Year", as_index=False)["Production"].sum()
    if "Yield" in comparable.columns:
        yld = comparable.groupby("Crop_Year", as_index=False)["Yield"].mean()
        out = yld if out is None else out.merge(yld, on="Crop_Year", how="outer")
    return out.sort_values("Crop_Year") if out is not None else pd.DataFrame()


def statewise_summary(df: pd.DataFrame) -> pd.DataFrame:
    """State-level Production and Yield, both computed on the
    comparable-unit subset (Coconut excluded)."""
    comparable = comparable_yield_df(df)
    agg = {}
    if "Production" in comparable.columns:
        agg["Production"] = "sum"
    if "Area" in comparable.columns:
        agg["Area"] = "sum"
    out = comparable.groupby("State", as_index=False).agg(agg) if agg else comparable[["State"]].drop_duplicates()
    if "Yield" in comparable.columns:
        yld = comparable.groupby("State", as_index=False)["Yield"].mean()
        out = out.merge(yld, on="State", how="left")
    return out.sort_values("Production", ascending=False) if "Production" in out.columns else out


def crop_state_heatmap_data(df: pd.DataFrame, top_crops=None, top_states=None, comparable_only: bool = True) -> pd.DataFrame:
    """Pivot table: rows=Crop, cols=State, values=mean Yield."""
    data = comparable_yield_df(df) if comparable_only else df
    if top_crops:
        data = data[data["Crop"].isin(top_crops)]
    if top_states:
        data = data[data["State"].isin(top_states)]
    return data.pivot_table(index="Crop", columns="State", values="Yield", aggfunc="mean")


def top_movers(df: pd.DataFrame, value_col: str = "Yield") -> dict:
    """Largest year-over-year increase/decrease in the mean of value_col.
    Yield and Production both use the comparable-unit subset (Coconut
    excluded) to avoid a unit artifact dominating the trend."""
    comparable = comparable_yield_df(df) if value_col in ("Yield", "Production") else df
    yt = comparable.groupby("Crop_Year")[value_col].mean().sort_index()
    if len(yt) < 2:
        return {}
    delta = yt.diff().dropna()
    if not len(delta):
        return {}
    max_inc_year = delta.idxmax()
    max_dec_year = delta.idxmin()
    return {
        "max_increase": {"Crop_Year": int(max_inc_year), "delta": float(delta.loc[max_inc_year])},
        "max_decrease": {"Crop_Year": int(max_dec_year), "delta": float(delta.loc[max_dec_year])},
    }


def correlation_matrix(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    return df[cols].corr()


def yield_area_relationship_check(df: pd.DataFrame) -> dict:
    """Empirically verify the Yield ≈ Production / Area relationship that
    justifies excluding Production from the ML feature set. Computed live
    from the actual data every time — never hard-coded."""
    data = df[df["Area"] > 0].copy()
    computed = data["Production"] / data["Area"]
    corr = float(np.corrcoef(computed, data["Yield"])[0, 1])
    return {
        "correlation": corr,
        "n_rows_checked": int(len(data)),
        "n_rows_zero_area_excluded": int((df["Area"] == 0).sum()),
    }
