"""
ml_model.py
============
Yield prediction pipeline.

LEAKAGE PREVENTION (mandatory, verified empirically — see
analysis.yield_area_relationship_check, which is re-run against the live
data rather than trusted as a historical claim)
------------------------------------------------------------------------
Yield in this dataset is (almost) exactly Production / Area
(Pearson r ≈ 0.997 between reported Yield and the computed ratio).
Because Yield can be trivially reconstructed from Production and Area,
`Production` is EXCLUDED from the feature set entirely — including it
would let a model "cheat" by reconstructing the target via near-exact
division instead of learning genuine agronomic relationships, and would
produce an artificially inflated R2 that could not generalize to a real
forecasting scenario (where the season's Production is not yet known at
prediction time). `Area` is kept: on its own it does not determine
Yield, and it is a genuine, independently-known agronomic input.

VALIDATION STRATEGY — RANDOM vs. TIME-AWARE
------------------------------------------------------------------------
This dataset spans 1997-2020. A random 80/20 split lets the model see
records from *future* years during training and get tested on *past*
years, which is unrealistic for a forecasting use case and can make the
model look better than it would in a genuine forecast. This module
therefore trains and evaluates BOTH:

  * Random split   — standard 80/20 random holdout. Shown for reference.
  * Time-aware split — train on earlier years, test on the most recent
    years, mirroring how the model would actually be used (predicting a
    future season from historical data). This is the PRIMARY evaluation:
    because AGRIVISTA's yield prediction is intended for forecasting a
    future season, the model type deployed to the interactive Predict
    and What-If tools is the one selected via time-aware validation, not
    whichever model happens to score highest on a random split.

Both are reported side by side in the UI; the app does not cherry-pick
whichever produces the higher score.

COCONUT UNIT EXCLUSION FROM THE GENERAL MODEL
------------------------------------------------------------------------
Coconut's Yield (and therefore Production, since Yield = Production /
Area) is reported in nuts, not tonnes — an entirely different unit from
every other crop. A single regression target that mixes nuts/hectare and
tonnes/hectare values is not scientifically meaningful, regardless of
validation strategy. Both `train_and_compare` and `train_time_aware`
therefore train and evaluate the GENERAL model on the comparable-unit
subset only (Coconut excluded) via `data_loader.comparable_yield_df`.
Coconut is never deleted from the dataset — it remains fully available
for crop-specific analysis elsewhere in the app (Crop Intelligence,
boxplots, stability ranking) — it is excluded only from this shared,
cross-crop model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.data_loader import comparable_yield_df, INCOMPARABLE_YIELD_CROPS

LEAKY_FEATURES = {"Production"}
CATEGORICAL_CANDIDATES = ["Crop", "State", "Season"]
NUMERIC_CANDIDATES = ["Crop_Year", "Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]
TARGET = "Yield"

def _model_zoo() -> Dict[str, object]:
    """Factory that returns FRESH, unfitted estimator instances every call.

    IMPORTANT: this must never be a module-level dict of pre-built
    instances. train_and_compare() and train_time_aware() each build a
    full model comparison, and if both reused the same estimator objects,
    fitting one would silently mutate the fitted state referenced by the
    other's already-returned TrainedModel (both Pipelines wrap the *same*
    estimator object) — corrupting whichever result was computed first.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=12, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=16, n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.08, random_state=42),
    }


@dataclass
class TrainedModel:
    name: str
    pipeline: Pipeline
    r2: float
    mae: float
    rmse: float
    y_test: np.ndarray
    y_pred: np.ndarray


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Only features that exist in the dataset AND are not leaky."""
    cat = [c for c in CATEGORICAL_CANDIDATES if c in df.columns]
    num = [c for c in NUMERIC_CANDIDATES if c in df.columns and c not in LEAKY_FEATURES]
    return cat, num


def _build_preprocessor(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric_pipe, num_cols), ("cat", categorical_pipe, cat_cols)])


def _fit_eval(X_train, X_test, y_train, y_test, cat_cols, num_cols) -> Tuple[pd.DataFrame, Dict[str, TrainedModel]]:
    rows, trained = [], {}
    for name, estimator in _model_zoo().items():
        pre = _build_preprocessor(cat_cols, num_cols)
        pipe = Pipeline([("preprocess", pre), ("model", estimator)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        rows.append({"Model": name, "R2": r2, "MAE": mae, "RMSE": rmse})
        trained[name] = TrainedModel(name, pipe, r2, mae, rmse, y_test.values if hasattr(y_test, "values") else y_test, y_pred)

    results = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    return results, trained


@st.cache_resource(show_spinner="Training yield-prediction models (runs once)...")
def train_and_compare(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Random-split evaluation of the GENERAL model. Shown for reference
    alongside the time-aware evaluation (see train_time_aware, which is
    the primary basis for model selection). Trained on the comparable-
    unit crop subset only — Coconut is excluded because its Yield/
    Production are in nuts, not tonnes, and a target mixing both units
    is not scientifically meaningful. Returns:
      results, trained_models (dict), feature_cols (cat, num), best_model_name
    """
    data_source = comparable_yield_df(df)
    cat_cols, num_cols = get_feature_columns(data_source)
    feature_cols = cat_cols + num_cols
    data = data_source.dropna(subset=feature_cols + [TARGET]).copy()

    X, y = data[feature_cols], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    results, trained = _fit_eval(X_train, X_test, y_train, y_test, cat_cols, num_cols)
    best_name = results.iloc[0]["Model"]
    return results, trained, (cat_cols, num_cols), best_name


@st.cache_resource(show_spinner="Running time-aware validation (runs once)...")
def train_time_aware(df: pd.DataFrame, test_fraction_years: float = 0.2):
    """
    PRIMARY evaluation for model selection. Chronological split: train on
    the earliest years, test on the most recent years. Mirrors real
    forecasting use (you never have future years' data at training time).
    The split point is chosen dynamically from the actual year
    distribution in the data (not a hard-coded year). Trained on the
    comparable-unit crop subset only — Coconut excluded, for the same
    reason as train_and_compare above.
    """
    if "Crop_Year" not in df.columns:
        return None

    data_source = comparable_yield_df(df)
    cat_cols, num_cols = get_feature_columns(data_source)
    feature_cols = cat_cols + num_cols
    data = data_source.dropna(subset=feature_cols + [TARGET]).copy()

    years = sorted(data["Crop_Year"].unique())
    if len(years) < 5:
        return None  # not enough distinct years for a meaningful chronological split

    split_idx = max(1, int(len(years) * (1 - test_fraction_years)))
    train_years = years[:split_idx]
    test_years = years[split_idx:]
    cutoff_year = train_years[-1]

    train_data = data[data["Crop_Year"].isin(train_years)]
    test_data = data[data["Crop_Year"].isin(test_years)]

    if len(train_data) < 50 or len(test_data) < 50:
        return None

    X_train, y_train = train_data[feature_cols], train_data[TARGET]
    X_test, y_test = test_data[feature_cols], test_data[TARGET]

    results, trained = _fit_eval(X_train, X_test, y_train, y_test, cat_cols, num_cols)
    return {
        "results": results,
        "trained": trained,
        "feature_cols": (cat_cols, num_cols),
        "cutoff_year": int(cutoff_year),
        "train_years": (int(train_years[0]), int(train_years[-1])),
        "test_years": (int(test_years[0]), int(test_years[-1])),
        "n_train": len(train_data),
        "n_test": len(test_data),
    }


def get_primary_model(df: pd.DataFrame):
    """
    Selects the PRIMARY deployed model — the one used by the interactive
    Predict Yield and What-If tools — based on TIME-AWARE validation
    (the realistic evaluation for a forecasting use case), NOT whichever
    model scores highest on the random split.

    Returns (trained_model, model_name, (cat_cols, num_cols), source) where
    source is "time_aware" normally, or "random_fallback" only in the
    edge case where the dataset has too few distinct years for a
    chronological split to be computed at all.
    """
    ta = train_time_aware(df)
    if ta is not None:
        best_name = ta["results"].iloc[0]["Model"]
        return ta["trained"][best_name], best_name, ta["feature_cols"], "time_aware"

    # Fallback: only reachable if the dataset has too few distinct years
    # for a meaningful chronological split (see train_time_aware).
    results, trained, feature_cols, best_name = train_and_compare(df)
    return trained[best_name], best_name, feature_cols, "random_fallback"


def get_feature_importance(trained_model: TrainedModel, cat_cols: List[str], num_cols: List[str]) -> pd.DataFrame:
    """Feature importance for tree models, rolled up from one-hot dummies
    back to the original column (all 'Crop_*' dummies summed into one
    'Crop' row) since users think in terms of the original variable."""
    model = trained_model.pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["Feature", "Importance"])

    pre = trained_model.pipeline.named_steps["preprocess"]
    ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = ohe.get_feature_names_out(cat_cols) if cat_cols else np.array([])
    all_names = list(num_cols) + list(cat_feature_names)
    imp_series = pd.Series(model.feature_importances_, index=all_names)

    rolled = {col: imp_series.get(col, 0.0) for col in num_cols}
    for col in cat_cols:
        mask = [f for f in cat_feature_names if f.startswith(col + "_")]
        rolled[col] = imp_series[mask].sum() if mask else 0.0

    out = pd.DataFrame({"Feature": list(rolled.keys()), "Importance": list(rolled.values())})
    return out.sort_values("Importance", ascending=False).reset_index(drop=True)


def is_excluded_from_general_model(crop: str) -> bool:
    """True if `crop` is not part of the comparable-unit general model
    (currently: Coconut, reported in nuts/hectare rather than
    tonnes/hectare). Used by the UI to show a clear message instead of
    silently predicting with a mismatched unit."""
    return crop in INCOMPARABLE_YIELD_CROPS


def predict_single(trained_model: TrainedModel, input_dict: dict, feature_cols: List[str]) -> float:
    row = pd.DataFrame([{c: input_dict.get(c) for c in feature_cols}])
    return float(trained_model.pipeline.predict(row)[0])


def what_if_scenario(trained_model: TrainedModel, base_input: dict, feature_cols: List[str],
                      vary_col: str, values: list) -> pd.DataFrame:
    """Model-based scenario analysis: sweep one feature across `values`,
    holding all others fixed at base_input, and record the predicted
    yield. Shows the model's learned association only — NOT a causal
    simulation of what would actually happen in a field."""
    rows = []
    for v in values:
        scenario = dict(base_input)
        scenario[vary_col] = v
        rows.append({vary_col: v, "Predicted Yield": predict_single(trained_model, scenario, feature_cols)})
    return pd.DataFrame(rows)
