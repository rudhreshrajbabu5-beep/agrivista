# 🌾 AGRIVISTA
### Smart Agricultural Crop Production, Yield Intelligence & Prediction Platform

A single, unified Streamlit application that turns a 19,689-record, 1997–2020 Indian agricultural dataset into interactive crop intelligence, regional insight, yield-stability analysis, and machine-learning yield prediction.

This project consolidates three independently-built prototype versions (AGRIVISTA / CropWise AI / AgriYield Intelligence) into one final, competition-ready platform — auditing each for its strongest implementation of every feature rather than merging all three wholesale.

---

## 1. Problem Statement

Raw agricultural records — crop, year, season, state, area, production, rainfall, fertilizer, pesticide, yield — are hard to reason about in tabular form. Judges and stakeholders need to see: which crops and regions perform best, how consistent that performance is over time, what environmental/input factors associate with yield, and whether yield can be predicted for a new season — all without misleading unit mixing or statistically invalid modeling shortcuts.

## 2. Objectives

- Transform raw records into KPIs, rankings, and trends judges can read in under 10 seconds.
- Surface regional and crop-level patterns (heatmap, stability ranking, bubble chart).
- Quantify factor–yield associations honestly (correlation, not causation).
- Build a genuinely non-leaking, validated ML yield predictor with an interactive prediction and what-if tool.
- Do all of the above without fabricating a single statistic — every number is computed live from the dataset currently loaded/filtered.

## 3. Dataset

- **19,689 records, 10 columns, 1997–2020, 30 Indian states, 55 crops.**
- Columns: `Crop, Crop_Year, Season, State, Area, Production, Annual_Rainfall, Fertilizer, Pesticide, Yield`.
- No missing values, no duplicate rows in the source file; text columns ship with trailing whitespace (e.g. `"Kharif     "`), which is stripped during cleaning.
- Column names are detected dynamically (`src/data_loader.py`) rather than hard-coded, so the app degrades gracefully (hides a feature) instead of crashing if a future version of the CSV renames or drops a column.

## 4. Data Cleaning

Every cleaning step is logged into a `CleaningReport` shown verbatim on the **Methodology** page — nothing is silently dropped:

1. Detect column roles semantically (`crop`, `year`, `state`, `yield`, …).
2. Strip whitespace from text columns.
3. Coerce numeric columns; unparseable values become `NaN`.
4. Drop exact duplicate rows.
5. Drop rows missing an essential field (crop, year, state, area, production, yield).
6. Drop rows with a physically-impossible negative area/production/yield.
7. Flag (but keep) `Area == 0` rows; anything that divides by Area guards against this explicitly.

On the current dataset, 0 rows are removed — the source file is already clean — but the pipeline is fully defensive for any future/edited version of the CSV.

## 5. ⚠️ Critical Data-Quality Issue: The Coconut Unit Problem

**Coconut yield is reported in nuts/hectare; every other crop in this dataset is reported in tonnes/hectare.** Because `Yield = Production / Area` for every row, Coconut's *Production* figures are correspondingly raw nut-counts, not tonnes — Coconut alone accounts for 95.4% of a naive "total production" sum, which would be meaningless.

This is **not treated as an error to silently fix or hide**. The platform:
- Clearly labels the unit difference in the Methodology page and in-app callouts.
- Excludes Coconut from every **cross-crop** Yield and Production ranking, KPI, and total (via `data_loader.comparable_yield_df()`), by default, throughout the app.
- Keeps Coconut fully available for crop-specific analysis (its own boxplot, its own stability score, its own ML prediction — labeled in nuts/hectare).

No arbitrary conversion factor is invented to force Coconut onto the tonnes scale.

## 6. Analyses (Mandatory + Additional)

**Mandatory (all present, clearly located):**
1. Crop production data overview (Overview KPIs)
2. Average yield by crop (lollipop chart, Overview)
3. Highest average-yielding crop (KPI card + insight)
4. Bar chart of crop production (Overview)
5. Yield boxplot comparison (Crop Intelligence)

**Additional (10 core + 3 optional, ~13 total — not dozens of throwaway charts):**

| # | Analysis | Page |
|---|---|---|
| 1 | Dataset overview / KPIs | Overview |
| 2 | Average yield by crop | Overview |
| 3 | Total production by crop | Overview |
| 4 | Yield distribution / boxplot | Crop Intelligence |
| 5 | Yearly production/yield trend | Overview |
| 6 | State-wise production/yield ranking | Regional Intelligence |
| 7 | Crop × State yield heatmap (signature viz) | Regional Intelligence |
| 8 | Agricultural factor vs yield (scatter + correlation matrix) | Agricultural Factors |
| 9 | Crop stability analysis (Coefficient of Variation) | Crop Intelligence |
| 10 | ML yield prediction (4-model comparison) | AI Yield Prediction |
| 11 | Interactive crop comparison | Crop Intelligence |
| 12 | What-if scenario analysis | What-If Analysis |
| 13 | Automatic insight engine | Overview |
| — | Productivity-vs-scale bubble chart | Regional Intelligence |

## 7. Crop Stability Analysis

```
CV (Coefficient of Variation) = Std Dev(Yield) / Mean(Yield)   [per crop]
Stability Score = 1 / (1 + CV)
```

CV is scale-free, so crops on different yield magnitudes can be ranked on one axis. Lower CV → higher score (bounded in (0, 1]) → more consistent yield historically. This is a direct statistical transform of one standard variability measure — not an invented weighted formula. Computed only on the comparable-unit crop subset when shown in a cross-crop ranking.

## 8. Agricultural Factor Analysis

Rainfall, Fertilizer, Pesticide, and Area are checked against Yield via a correlation matrix and per-factor scatter plots with an OLS trendline. Every mention of a relationship uses association language ("shows a positive association with yield"), **never causal language** ("causes higher yield") — enforced consistently across insights, captions, and the Methodology page.

## 9. Machine Learning

**Target:** `Yield`. **Candidate features:** Crop, State, Season, Crop_Year, Area, Annual_Rainfall, Fertilizer, Pesticide (only those actually present in the dataset are used).

**Models compared:** Linear Regression, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor — each in an sklearn `Pipeline` with median/most-frequent imputation, `StandardScaler` for numeric features, and `OneHotEncoder` for categorical features (fit only on the training fold).

**Metrics:** R² (goodness of fit, never called "accuracy"), MAE, RMSE.

### 9.1 Data Leakage Prevention

`Yield ≈ Production / Area` was **verified empirically on the live dataset** (not assumed from the source projects' claims): correlation between reported Yield and computed Production/Area = **0.9965** across all 19,689 rows. Because Yield can be reconstructed almost exactly from Production, **`Production` is excluded from every model's feature set** — including it would let a model "cheat" via near-exact division instead of learning genuine agronomic relationships, producing an inflated R² that would not generalize to real forecasting (where the season's Production is never known in advance). `Area` is kept: on its own it does not determine Yield and is a genuine, independently-known agronomic input.

### 9.2 Coconut Excluded from the General Model

Coconut's Yield (and Production) are reported in **nuts**, not tonnes — a different unit from every other crop. A single regression target mixing nuts/hectare and tonnes/hectare values is not scientifically meaningful, so **the general model is trained only on the comparable tonnes/hectare crop subset** (`data_loader.comparable_yield_df`), for both the random-split and time-aware evaluations below. Coconut is never removed from the dataset — it remains fully explorable in its native unit on the Crop Intelligence → *Single-Crop Deep Dive* tab — but it does not enter the shared cross-crop model. Selecting Coconut in the Predict Yield tab shows an explanatory message instead of a cross-unit prediction.

*(Removing Coconut changes the reported R² noticeably from an earlier build that trained across both units — see §9.3/9.4. That is expected and correct: Coconut's huge numeric scale was inflating apparent fit quality by dominating the target's variance, not by making per-crop predictions genuinely more accurate.)*

### 9.3 Random-Split Results (80/20, `random_state=42`, comparable-unit crops only) — Reference

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Gradient Boosting | 0.7997 | 1.50 | 5.62 |
| Linear Regression | 0.7373 | 3.00 | 6.44 |
| Random Forest | 0.5707 | 1.34 | 8.23 |
| Decision Tree | -0.7633 | 1.73 | 16.69 |

*(MAE/RMSE are in tonnes/hectare — comparable across the whole table now that Coconut is excluded. Shown for comparison only; not used to select the deployed model — see §9.4.)*

### 9.4 Time-Aware Validation (Primary Model Selection)

A random split lets the model train on records from *later* years and get tested on *earlier* years — unrealistic for forecasting, since future data is never available at training time. Because AGRIVISTA's yield prediction is intended for **future** use, this platform uses **chronological (time-aware) validation as the primary model-selection strategy**: train on the **earliest years only**, test on the **most recent years** (split point chosen dynamically from the actual year distribution — not hard-coded). The model type deployed to the interactive Predict Yield and What-If Analysis tools is whichever model wins this evaluation — **not** whichever wins the random split above.

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Decision Tree | 0.6800 | 1.59 | 7.74 |
| Gradient Boosting | 0.5897 | 1.77 | 8.77 |
| Random Forest | 0.5782 | 1.61 | 8.89 |
| Linear Regression | 0.5253 | 3.20 | 9.43 |

Trained on 1997–2015, tested on 2016–2020. **Decision Tree is the primary deployed model** (highest time-aware R²). Both tables are shown side by side in-app (never cherry-picked); the time-aware table is explicitly labeled "Primary" and the random-split table "Reference Only."

### 9.5 Prediction & What-If

The **AI Yield Prediction** page provides an interactive form (Crop, State, Season, Year, Area, Rainfall, Fertilizer, Pesticide) that returns a predicted yield from the **primary (time-aware selected) model**, the model used, and its evaluation metrics — no fake confidence percentages. Selecting Coconut shows an explanatory message instead of a cross-unit prediction (see §9.2).

The **What-If Analysis** page sweeps one input variable while holding the rest fixed at a base scenario and plots the primary model's predicted yield across that range, explicitly labeled "Model-based scenario analysis" with a caption clarifying this reflects the model's learned association, not a guaranteed real-world causal outcome. Its crop selector is limited to the comparable-unit subset for the same reason.

## 10. Automatic Insight Engine

3–7 natural-language insights are generated per page load, purely from live calculations on the currently filtered data (highest-production crop, highest comparable-yield crop, most/least stable crop, best-performing state, strongest factor association, largest year-over-year change, best Crop×State combination). Only the sentence *templates* are fixed in code — every number inserted is computed at render time.

## 11. Features & Technology Stack

- **Frontend:** Streamlit (multi-page via sidebar radio nav + custom CSS design system)
- **Data:** pandas, numpy
- **Visualization:** Plotly (Express + Graph Objects)
- **ML:** scikit-learn (Pipeline, ColumnTransformer, 4 regressors)
- **Caching:** `@st.cache_data` for cleaning, `@st.cache_resource` for model training (never retrained on every interaction)

## 12. Project Structure

```
agrivista/
├── app.py                    # Streamlit entry point — page routing & layout
├── requirements.txt
├── README.md
├── data/
│   └── crop_yield.csv
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # column detection, cleaning, CleaningReport
│   ├── preprocessing.py      # filter utilities
│   ├── statistics.py         # stability score, descriptive stats, outliers
│   ├── analysis.py           # KPIs, group-bys, leakage check
│   ├── insights.py           # automatic insight engine
│   ├── visualizations.py     # shared Plotly chart library + design tokens
│   └── ml_model.py           # leakage-guarded ML pipeline, time-aware validation
├── assets/
│   └── style.css
└── .streamlit/
    └── config.toml
```

## 13. Installation & Run

```bash
cd agrivista
pip install -r requirements.txt
streamlit run app.py
```

## 14. Limitations

- Correlations shown (rainfall, fertilizer, pesticide vs. yield) are associations, **not proof of causation**.
- The What-If tool shows model-based scenario predictions, not guaranteed real-world agronomic outcomes.
- Coconut's raw yield/production numbers are not comparable to other crops due to differing units.
- The dataset has no soil quality, irrigation infrastructure, crop variety, or farming-practice data — real factors that affect yield but aren't present here.
- ML models are trained on historical patterns; a genuinely novel Crop/State/Season combination outside the training distribution will be less reliable.

## 15. Future Improvements

- Geographic map visualization if reliable state-boundary geometry is sourced (deliberately omitted here rather than fabricated).
- District-level granularity if a more detailed dataset becomes available.
- Ensemble/stacked model combining the strongest individual models.
- User-uploadable dataset support with the same dynamic column-detection pipeline.

---

## 16. Competition Presentation

### 30-second explanation
AGRIVISTA turns 24 years of Indian crop data into an interactive dashboard: judges see production and yield rankings, a crop-stability score, a state×crop heatmap, and a machine-learning model that predicts yield for a new season — all validated with proper leakage prevention and both random and time-aware testing.

### 1-minute explanation
We took raw records of crop, region, year, and environmental inputs and built a five-stage pipeline: clean → analyze → visualize → model → predict. Along the way we caught and fixed two real data-quality issues most naive dashboards would miss — Coconut's yield/production being reported in nuts instead of tonnes, and the near-total leakage between Yield and Production/Area (r≈0.997) that would otherwise let an ML model "cheat." We compare four regression models, validate them two ways (random split and realistic time-aware split), and expose the result through an interactive prediction tool and a what-if scenario explorer — all wrapped in a clean, judge-readable UI.

### Technical explanation
Dynamic column detection makes the ingestion pipeline resilient to schema drift. Cleaning is fully logged (`CleaningReport`), not silent. Cross-crop aggregates route through a `comparable_yield_df()` filter to prevent the Coconut unit artifact from contaminating KPIs, rankings, and trends. ML feature selection explicitly excludes `Production` after empirically re-verifying the leakage correlation on the live data (not trusting a prior claim). Models run through identical sklearn `Pipeline`s (imputation → scaling/encoding → estimator) fit only on the training fold. Time-aware validation splits chronologically at a dynamically-computed cutoff rather than a hard-coded year.

### Why Production was excluded from ML
Yield is (almost) mathematically `Production / Area` (r = 0.9965, verified live). Including Production would let the model reconstruct the target via near-exact division rather than learn real agronomic relationships, producing an inflated score that would not hold up in genuine forecasting, where a season's Production is never known before the season happens.

### Why Coconut requires special unit handling
Coconut yield/production are recorded in nuts, not tonnes — roughly 1,000× the numeric scale of every other crop. Left unhandled, it would dominate every "total production" and "highest yield" ranking as a pure unit artifact rather than a genuine agronomic signal. This applies to the ML model too: the general yield predictor is trained only on the comparable tonnes/hectare crop subset, so Coconut never enters the shared regression target. It stays fully explorable on its own (Crop Intelligence → Single-Crop Deep Dive), just not blended with other crops anywhere.

### Why time-aware validation matters
A random split can train on future years and test on past years — the model gets an unfairly easy test. Time-aware validation (train on earliest years, test on most recent) mirrors the actual forecasting use case and gives a more honest estimate of real-world performance. That's why AGRIVISTA uses it as the **primary** basis for selecting which model gets deployed to the Predict Yield and What-If tools — the random-split table is kept visible for comparison, but a model is never chosen just because it scored higher on the easier, unrealistic split.

### Why correlation does not imply causation
A positive correlation between, say, Fertilizer and Yield could reflect fertilizer's genuine agronomic effect, or simply that better-resourced farms use more fertilizer *and* have other yield advantages (irrigation, soil quality) not captured in this dataset. The platform consistently uses association language and never causal language for these relationships.

### Possible judge questions
- *"Why did R² drop so much compared to an earlier version of this project?"* — An earlier iteration trained the general model across both Coconut (nuts/hectare) and tonnes/hectare crops together. Coconut's huge numeric scale inflated R² by dominating the target's variance, not by making predictions genuinely better. Excluding Coconut (this version) gives an honest, lower, but scientifically valid score.
- *"Why is Decision Tree your primary model instead of the higher-scoring Gradient Boosting?"* — Gradient Boosting scores higher only on the random 80/20 split, which lets it train on data from years *after* the years it's tested on — unrealistic for forecasting. Decision Tree scores highest on the time-aware split (train on early years, test on recent years), which is the realistic evaluation for a model meant to predict a future season, so it's the one deployed.
- *"Why not include Production as a feature if it improves accuracy?"* — Because it doesn't reflect a real predictive capability — see the leakage explanation above; it would be a data-quality mistake, not a genuine improvement.
- *"How would this generalize to another country's agricultural data?"* — The column-detection and comparable-unit logic are dataset-agnostic; the Coconut-specific unit exclusion is specific to this dataset and would need re-verification against a new source.
- *"What's the biggest limitation of your model?"* — It has no access to soil quality, irrigation, or crop variety — all materially affect real yield and aren't in this dataset. It's also trained only through 2015 (the time-aware cutoff), not refit on the full history for deployment — a deliberate simplicity tradeoff worth revisiting.
