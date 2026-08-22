"""
data_loader.py
================
Single source of truth for reading the raw agricultural dataset and
turning it into an analysis-ready dataframe.

Design goals
------------
* Never assume exact column names — detect them, and degrade gracefully
  (hide a feature) if an expected column is genuinely missing instead of
  crashing or inventing data.
* Never silently drop rows without recording *why* — every cleaning
  decision is written to a `CleaningReport` that the Methodology / Data
  Overview page shows verbatim to the user.
* Never impute or fabricate values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
import streamlit as st

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "crop_yield.csv")

# Candidate raw-column names (case/space-insensitive) mapped to the
# canonical role the rest of the app refers to. The first match wins.
# This lets the app survive minor naming/version differences in the
# source CSV without ever inventing a column that isn't there.
CANDIDATES = {
    "crop": ["crop", "crop_name", "cropname"],
    "year": ["crop_year", "year", "season_year"],
    "season": ["season"],
    "state": ["state", "state_name", "region"],
    "area": ["area", "area_ha", "area_hectare"],
    "production": ["production", "production_tonnes", "prod"],
    "yield": ["yield", "crop_yield", "yield_per_hectare"],
    "rainfall": ["annual_rainfall", "rainfall", "avg_rainfall"],
    "fertilizer": ["fertilizer", "fertiliser"],
    "pesticide": ["pesticide", "pesticides"],
}

# Canonical display names used everywhere downstream.
DISPLAY_NAMES = {
    "crop": "Crop", "year": "Crop_Year", "season": "Season", "state": "State",
    "area": "Area", "production": "Production", "yield": "Yield",
    "rainfall": "Annual_Rainfall", "fertilizer": "Fertilizer", "pesticide": "Pesticide",
}

# Coconut is reported in nuts/hectare (Yield) and nuts (Production); every
# other crop in this dataset is reported in tonnes/hectare and tonnes.
# Because Yield = Production / Area for every crop, a nuts-based Yield
# implies a nuts-based Production too — so BOTH the Yield and Production
# columns for Coconut are on an incomparable unit versus every other crop.
# This constant is referenced everywhere a cross-crop yield OR production
# comparison is made so the unit mismatch can never be silently baked into
# a ranking, KPI, or chart.
INCOMPARABLE_YIELD_CROPS = {"Coconut"}


@dataclass
class ColumnMap:
    """Semantic role -> actual column name present in the dataframe (or None)."""
    crop: Optional[str] = None
    year: Optional[str] = None
    season: Optional[str] = None
    state: Optional[str] = None
    area: Optional[str] = None
    production: Optional[str] = None
    yield_col: Optional[str] = None
    rainfall: Optional[str] = None
    fertilizer: Optional[str] = None
    pesticide: Optional[str] = None
    other_numeric_factors: List[str] = field(default_factory=list)

    def factor_columns(self) -> List[str]:
        """All numeric agricultural/environmental columns usable as EDA/ML factors."""
        cols = [c for c in [self.rainfall, self.fertilizer, self.pesticide] if c]
        cols.extend(self.other_numeric_factors)
        return cols

    def available(self) -> dict:
        return {
            "Crop": bool(self.crop), "Crop_Year": bool(self.year),
            "Season": bool(self.season), "State": bool(self.state),
            "Area": bool(self.area), "Production": bool(self.production),
            "Yield": bool(self.yield_col), "Annual_Rainfall": bool(self.rainfall),
            "Fertilizer": bool(self.fertilizer), "Pesticide": bool(self.pesticide),
        }


@dataclass
class CleaningReport:
    original_rows: int = 0
    cleaned_rows: int = 0
    rows_removed: int = 0
    duplicate_rows_removed: int = 0
    missing_value_rows_removed: int = 0
    invalid_numeric_rows_removed: int = 0
    text_columns_stripped: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def _match_column(columns: List[str], names: List[str]) -> Optional[str]:
    lower_map = {c.lower().strip(): c for c in columns}
    for name in names:
        if name in lower_map:
            return lower_map[name]
    return None


def build_column_map(df: pd.DataFrame) -> ColumnMap:
    cols = list(df.columns)
    cm = ColumnMap()
    cm.crop = _match_column(cols, CANDIDATES["crop"])
    cm.year = _match_column(cols, CANDIDATES["year"])
    cm.season = _match_column(cols, CANDIDATES["season"])
    cm.state = _match_column(cols, CANDIDATES["state"])
    cm.area = _match_column(cols, CANDIDATES["area"])
    cm.production = _match_column(cols, CANDIDATES["production"])
    cm.yield_col = _match_column(cols, CANDIDATES["yield"])
    cm.rainfall = _match_column(cols, CANDIDATES["rainfall"])
    cm.fertilizer = _match_column(cols, CANDIDATES["fertilizer"])
    cm.pesticide = _match_column(cols, CANDIDATES["pesticide"])

    claimed = {cm.crop, cm.year, cm.season, cm.state, cm.area,
               cm.production, cm.yield_col, cm.rainfall, cm.fertilizer, cm.pesticide}
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    for c in numeric_cols:
        if c not in claimed and c != cm.year:
            cm.other_numeric_factors.append(c)
    return cm


@st.cache_data(show_spinner="Loading agricultural dataset...")
def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the CSV exactly as provided on disk. No cleaning here."""
    return pd.read_csv(path)


@st.cache_data(show_spinner="Cleaning & preparing data...")
def load_clean_data(path: str = DATA_PATH):
    """
    Returns (clean_df, column_map, cleaning_report).

    Cleaning performed (all logged, nothing silent):
      1. Detect the semantic role of every relevant column.
      2. Rename to canonical display names for the rest of the app.
      3. Strip whitespace from text columns (source ships with trailing
         spaces, e.g. 'Kharif     ').
      4. Coerce numeric columns; rows that fail to parse are dropped.
      5. Drop exact duplicate rows.
      6. Drop rows missing any essential field (crop/year/state/yield/
         area/production).
      7. Drop rows with physically-impossible negative area/production/yield.
      8. Flag (but keep) Area == 0 rows; anything that divides by Area
         downstream must guard against this itself.
    """
    raw = load_raw_data(path)
    cm = build_column_map(raw)

    report = CleaningReport(original_rows=len(raw))
    work = raw.copy()

    rename_map = {}
    for role, col in [("crop", cm.crop), ("year", cm.year), ("season", cm.season),
                       ("state", cm.state), ("area", cm.area), ("production", cm.production),
                       ("yield", cm.yield_col), ("rainfall", cm.rainfall),
                       ("fertilizer", cm.fertilizer), ("pesticide", cm.pesticide)]:
        if col:
            rename_map[col] = DISPLAY_NAMES[role]
    work = work.rename(columns=rename_map)

    # Rebuild the column map against the renamed frame so downstream code
    # can rely on canonical names.
    cm2 = ColumnMap(
        crop="Crop" if cm.crop else None,
        year="Crop_Year" if cm.year else None,
        season="Season" if cm.season else None,
        state="State" if cm.state else None,
        area="Area" if cm.area else None,
        production="Production" if cm.production else None,
        yield_col="Yield" if cm.yield_col else None,
        rainfall="Annual_Rainfall" if cm.rainfall else None,
        fertilizer="Fertilizer" if cm.fertilizer else None,
        pesticide="Pesticide" if cm.pesticide else None,
        other_numeric_factors=list(cm.other_numeric_factors),
    )

    text_cols = [c for c in [cm2.crop, cm2.season, cm2.state] if c]
    for c in text_cols:
        work[c] = work[c].astype(str).str.strip()
    report.text_columns_stripped = text_cols

    numeric_targets = [c for c in [cm2.area, cm2.production, cm2.yield_col, cm2.rainfall,
                                    cm2.fertilizer, cm2.pesticide] + cm2.other_numeric_factors if c]
    before = len(work)
    for c in numeric_targets:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    if cm2.year:
        work[cm2.year] = pd.to_numeric(work[cm2.year], errors="coerce")

    before = len(work)
    work = work.drop_duplicates()
    report.duplicate_rows_removed = before - len(work)

    essential = [c for c in [cm2.crop, cm2.year, cm2.state, cm2.yield_col,
                              cm2.area, cm2.production] if c]
    before = len(work)
    work = work.dropna(subset=essential)
    report.missing_value_rows_removed = before - len(work)

    before = len(work)
    invalid_mask = pd.Series(False, index=work.index)
    if cm2.area:
        invalid_mask |= work[cm2.area] < 0
    if cm2.production:
        invalid_mask |= work[cm2.production] < 0
    if cm2.yield_col:
        invalid_mask |= work[cm2.yield_col] < 0
    work = work[~invalid_mask]
    report.invalid_numeric_rows_removed = before - len(work)
    if report.invalid_numeric_rows_removed > 0:
        report.notes.append(
            f"Removed {report.invalid_numeric_rows_removed} rows with a negative "
            f"area, production, or yield value (not physically possible)."
        )

    if cm2.area and (work[cm2.area] == 0).any():
        n_zero = int((work[cm2.area] == 0).sum())
        report.notes.append(
            f"{n_zero} row(s) have Area = 0. Kept in the dataset but excluded from "
            f"any calculation that divides by Area."
        )

    if cm2.crop and INCOMPARABLE_YIELD_CROPS & set(work[cm2.crop].unique()):
        report.notes.append(
            "Coconut is reported in nuts/hectare (Yield) and nuts (Production) "
            "while every other crop in this dataset is reported in tonnes/hectare "
            "and tonnes. Because Coconut's raw numbers are ~1000x larger than a "
            "typical tonnage figure, it is excluded by default from cross-crop "
            "Yield AND Production rankings/totals; it remains available for "
            "crop-specific analysis."
        )

    if cm2.year:
        work[cm2.year] = work[cm2.year].astype(int)

    report.cleaned_rows = len(work)
    report.rows_removed = report.original_rows - report.cleaned_rows

    work = work.reset_index(drop=True)
    return work, cm2, report


def comparable_yield_df(df: pd.DataFrame, crop_col: str = "Crop") -> pd.DataFrame:
    """Subset of df excluding crops whose yield unit is not comparable
    (currently: Coconut, nuts/hectare vs tonnes/hectare for everything else).
    Use this for any *cross-crop* yield ranking or aggregate statistic."""
    if crop_col not in df.columns:
        return df
    return df[~df[crop_col].isin(INCOMPARABLE_YIELD_CROPS)]
