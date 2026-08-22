"""
statistics.py
==============
Descriptive statistics and the Crop Stability Score.

Crop Stability Score
---------------------
Defined purely from the Coefficient of Variation (CV) of a crop's yield
across all observed records:

    CV_crop = std(yield) / mean(yield)
    Stability Score = 1 / (1 + CV_crop)

Rationale:
    * CV is a scale-free measure of relative variability, so crops whose
      yields live on very different numeric scales can still be ranked
      on the same axis.
    * Lower CV -> more consistent yield -> higher stability score, bounded
      in (0, 1]. CV = 0 gives a score of 1 (perfectly stable); the score
      decreases monotonically as CV grows, asymptoting toward 0.
    * No arbitrary weights — the score is a direct, explainable transform
      of one standard variability statistic.

Coconut caveat: because Coconut's yield lives on an entirely different
unit (nuts/hectare vs tonnes/hectare), its CV and stability score are
still computed (CV is scale-free and technically valid within a single
crop), but it is excluded by default from any *cross-crop* stability
ranking table/chart to avoid implying its raw yield magnitude is
comparable to other crops elsewhere on the same page.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def crop_summary_stats(df: pd.DataFrame, group_col: str = "Crop", value_col: str = "Yield") -> pd.DataFrame:
    """Per-crop descriptive statistics: mean, median, std, min, max, quartiles, CV, stability score."""
    g = df.groupby(group_col)[value_col]
    stats = g.agg(
        count="count", mean="mean", median="median", std="std", min="min", max="max",
        q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75),
    ).reset_index()

    stats["std"] = stats["std"].fillna(0.0)
    stats["cv"] = np.where(stats["mean"] != 0, stats["std"] / stats["mean"], np.nan)
    stats["stability_score"] = 1.0 / (1.0 + stats["cv"].abs())
    return stats.sort_values("mean", ascending=False).reset_index(drop=True)


def production_summary_stats(df: pd.DataFrame, group_col: str = "Crop", value_col: str = "Production") -> pd.DataFrame:
    g = df.groupby(group_col)[value_col]
    stats = g.agg(total="sum", mean="mean", median="median", std="std").reset_index()
    stats["std"] = stats["std"].fillna(0.0)
    return stats.sort_values("total", ascending=False).reset_index(drop=True)


def stability_ranking(df: pd.DataFrame, group_col: str = "Crop", value_col: str = "Yield", min_obs: int = 5) -> pd.DataFrame:
    """Rank crops by stability score. Crops with too few observations are
    excluded since CV is unreliable on tiny samples."""
    stats = crop_summary_stats(df, group_col, value_col)
    stats = stats[stats["count"] >= min_obs].copy()
    stats = stats.sort_values("stability_score", ascending=False).reset_index(drop=True)
    stats["rank"] = np.arange(1, len(stats) + 1)
    return stats


def outlier_report(df: pd.DataFrame, group_col: str = "Crop", value_col: str = "Yield") -> dict:
    """IQR-based outlier detection, computed WITHIN each crop group (since
    different crops have very different natural yield scales — a global
    IQR would just flag every high-yield crop as an 'outlier').

    An observation is an outlier if value < Q1 - 1.5*IQR or value > Q3 + 1.5*IQR.
    """
    def flag(group):
        q1, q3 = group[value_col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return (group[value_col] < lower) | (group[value_col] > upper)

    mask = df.groupby(group_col, group_keys=False).apply(flag)
    mask = mask.reindex(df.index).fillna(False)
    outliers = df[mask].copy()

    return {
        "n_outliers": int(mask.sum()),
        "pct_outliers": float(mask.mean() * 100) if len(df) else 0.0,
        "outliers_df": outliers,
        "by_crop": outliers.groupby(group_col).size().sort_values(ascending=False) if len(outliers) else pd.Series(dtype=int),
    }


def correlation_matrix(df: pd.DataFrame, numeric_roles: list) -> pd.DataFrame:
    cols = [c for c in numeric_roles if c in df.columns]
    return df[cols].corr()
