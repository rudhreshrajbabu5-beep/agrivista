"""
insights.py
============
Rule-based automatic insight generator. Every sentence produced here is
derived directly from a calculation on the currently loaded (and possibly
filtered) dataframe — nothing is hard-coded. Only the sentence TEMPLATES
are fixed; every value inserted into them is computed live.

Cross-crop yield comparisons use the comparable-unit subset of the data
(Coconut excluded) so no insight ever implies nuts/hectare and
tonnes/hectare are on the same scale.
"""

from __future__ import annotations

import pandas as pd

from src import analysis
from src import statistics as stats_mod
from src.data_loader import comparable_yield_df


def generate_insights(df: pd.DataFrame, max_insights: int = 5) -> list:
    insights = []
    if df is None or len(df) == 0:
        return ["No data available for the current filter selection."]

    comparable = comparable_yield_df(df)

    # 1. Highest production crop (comparable units — Coconut's Production
    # is reported in nuts, not tonnes, so it is excluded from this ranking)
    if {"Crop", "Production"}.issubset(comparable.columns) and len(comparable):
        prod = comparable.groupby("Crop")["Production"].sum().sort_values(ascending=False)
        if len(prod):
            share = prod.iloc[0] / prod.sum() * 100 if prod.sum() else 0
            insights.append(
                f"🌾 **{prod.index[0]}** leads total production among comparable-unit crops, "
                f"contributing **{share:.1f}%** of that total output. (Coconut is tracked in nuts, "
                f"not tonnes, and is excluded from this comparison.)"
            )

    # 2. Highest comparable average yield
    if {"Crop", "Yield"}.issubset(comparable.columns) and len(comparable):
        yld = comparable.groupby("Crop")["Yield"].mean().sort_values(ascending=False)
        if len(yld):
            insights.append(
                f"📈 **{yld.index[0]}** achieves the highest average yield among "
                f"comparable-unit crops (**{yld.iloc[0]:,.2f} tonnes/hectare**). "
                f"Coconut (nuts/hectare) is excluded from this comparison — see Methodology."
            )

    # 3. Most stable / most variable crop (comparable units)
    if len(comparable):
        ranking = stats_mod.stability_ranking(comparable)
        if len(ranking) >= 2:
            most_stable, most_variable = ranking.iloc[0], ranking.iloc[-1]
            insights.append(
                f"🛡️ **{most_stable['Crop']}** is the most stable crop (stability score "
                f"**{most_stable['stability_score']:.2f}**, CV = {most_stable['cv']:.2f}), meaning its "
                f"yield varies the least across regions and years."
            )
            insights.append(
                f"⚠️ **{most_variable['Crop']}** shows the highest yield variability "
                f"(stability score **{most_variable['stability_score']:.2f}**, CV = {most_variable['cv']:.2f})."
            )

    # 4. Best-performing state
    if {"State", "Yield"}.issubset(comparable.columns) and len(comparable):
        state_yield = comparable.groupby("State")["Yield"].mean().sort_values(ascending=False)
        if len(state_yield):
            insights.append(
                f"🗺️ **{state_yield.index[0]}** records the highest average comparable yield "
                f"among states (**{state_yield.iloc[0]:,.2f} t/ha**)."
            )

    # 5. Strongest observed factor association
    factor_cols = [c for c in ["Annual_Rainfall", "Fertilizer", "Pesticide", "Area"] if c in comparable.columns]
    if factor_cols and "Yield" in comparable.columns and len(comparable) > 5:
        corrs = comparable[factor_cols + ["Yield"]].corr()["Yield"].drop("Yield").dropna()
        if len(corrs):
            top_factor = corrs.abs().sort_values(ascending=False).index[0]
            direction = "positive" if corrs[top_factor] > 0 else "negative"
            insights.append(
                f"🔬 **{top_factor.replace('_', ' ')}** shows the strongest observed association "
                f"with yield among measured inputs (r = **{corrs[top_factor]:.2f}**, {direction}). "
                f"This is a correlation, not proof of causation."
            )

    # 6. Yearly trend
    movers = analysis.top_movers(df, "Yield")
    if movers:
        up = movers.get("max_increase", {})
        if up:
            insights.append(
                f"📅 The largest year-over-year increase in average comparable yield occurred in "
                f"**{up['Crop_Year']}** (+{up['delta']:.2f} t/ha vs. the prior year)."
            )

    # 7. Best crop x state combination
    if {"Crop", "State", "Yield"}.issubset(comparable.columns) and len(comparable):
        cs = comparable.groupby(["Crop", "State"])["Yield"].mean()
        if len(cs):
            best = cs.idxmax()
            insights.append(
                f"🏆 The strongest single Crop × State combination is **{best[0]} in {best[1]}**, "
                f"averaging **{cs.max():,.2f} t/ha**."
            )

    return insights[:max_insights]
