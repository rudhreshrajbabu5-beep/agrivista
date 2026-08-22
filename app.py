"""
AGRIVISTA — Smart Agricultural Crop Production, Yield Intelligence &
Prediction Platform
=======================================================================
Main Streamlit entry point. Handles page routing, global filters, and UI
composition. All data cleaning, analysis, visualization, and ML logic
lives in the src/ package so this file stays focused on layout.
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src import data_loader, preprocessing, analysis, statistics as stats_mod
from src import insights as insights_mod, visualizations as viz, ml_model

# ------------------------------------------------------------------
# Page config & global styling
# ------------------------------------------------------------------

st.set_page_config(
    page_title="AGRIVISTA | Crop Yield Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS_PATH = os.path.join(os.path.dirname(__file__), "assets", "style.css")
with open(CSS_PATH) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Small UI helpers
# ------------------------------------------------------------------

def kpi_card(icon: str, label: str, value: str, sub: str = "") -> str:
    return f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """


def section_header(eyebrow: str, title: str):
    st.markdown(
        f'<div class="section-eyebrow">{eyebrow}</div><div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )


def fmt_num(n, decimals=0):
    if n is None or (isinstance(n, float) and pd.isna(n)):
        return "—"
    if abs(n) >= 1_000_000_000:
        return f"{n/1_000_000_000:,.2f}B"
    if abs(n) >= 1_000_000:
        return f"{n/1_000_000:,.2f}M"
    if abs(n) >= 1_000:
        return f"{n/1_000:,.1f}K"
    return f"{n:,.{decimals}f}"


def metric_chip(label, value):
    return f'<div class="metric-chip"><div class="val">{value}</div><div class="lab">{label}</div></div>'


def callout(msg: str, kind: str = "info"):
    st.markdown(f'<div class="callout-box {kind}">{msg}</div>', unsafe_allow_html=True)


def empty_state(msg="No data matches the current filter selection. Try widening your filters."):
    st.info(f"ℹ️ {msg}")


COCONUT_NOTE = (
    "⚠️ **Unit note:** Coconut is reported in *nuts/hectare* (and total nuts for "
    "Production), while every other crop here is reported in *tonnes/hectare* "
    "(and total tonnes). Coconut is therefore excluded from cross-crop Yield and "
    "Production comparisons by default, and shown separately where relevant. "
    "See the Methodology page for details."
)

# ------------------------------------------------------------------
# Load & clean data (once, cached)
# ------------------------------------------------------------------

df_full, colmap, cleaning_report = data_loader.load_clean_data()
available = colmap.available()

# ------------------------------------------------------------------
# Sidebar — navigation + global filters
# ------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="nav-brand">🌾 AGRIVISTA</div>'
        '<div class="nav-brand-sub">Crop Production &amp; Yield Intelligence</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "🌱 Crop Intelligence",
            "🗺️ Regional Intelligence",
            "🌦️ Agricultural Factors",
            "🤖 AI Yield Prediction",
            "🔬 What-If Analysis",
            "📚 Methodology",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div style="font-weight:700;font-size:0.9rem;margin-bottom:8px;">🔎 Global Filters</div>', unsafe_allow_html=True)
    st.caption("Applied to Overview, Crop Intelligence & Regional Intelligence")

    all_crops = preprocessing.safe_unique(df_full, "Crop")
    all_states = preprocessing.safe_unique(df_full, "State")
    all_seasons = preprocessing.safe_unique(df_full, "Season")

    sel_crops = st.multiselect("Crop", all_crops, default=[], help="Leave empty to include all crops")
    sel_states = st.multiselect("State", all_states, default=[], help="Leave empty to include all states")
    sel_seasons = st.multiselect("Season", all_seasons, default=[], help="Leave empty to include all seasons")

    if available.get("Crop_Year"):
        y_min, y_max = int(df_full["Crop_Year"].min()), int(df_full["Crop_Year"].max())
        sel_years = st.slider("Year Range", y_min, y_max, (y_min, y_max))
    else:
        sel_years = None

    if st.button("↺ Reset Filters", use_container_width=True):
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption(
        f"Dataset: {len(df_full):,} records · {df_full['Crop'].nunique()} crops · "
        f"{df_full['State'].nunique()} states · {int(df_full['Crop_Year'].min())}–{int(df_full['Crop_Year'].max())}"
    )

df = preprocessing.apply_filters(df_full, sel_crops, sel_states, sel_seasons, sel_years)


# ========================================================================
# PAGE 1 — OVERVIEW
# ========================================================================

def render_overview():
    st.markdown(
        """
        <div class="hero-wrap">
            <span class="hero-badge">Data-Driven Agriculture</span>
            <span class="hero-badge">Machine Learning</span>
            <div class="hero-title">🌾 AGRIVISTA</div>
            <div class="hero-subtitle">Smart Agricultural Crop Production, Yield Intelligence &amp; Prediction</div>
            <div class="hero-desc">Transforming historical crop, region, and environmental data into
            interactive intelligence — production trends, regional patterns, yield stability, and
            machine-learning-based yield prediction.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not preprocessing.safe_dataframe(df):
        empty_state()
        return

    kpis = analysis.compute_kpis(df)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(kpi_card("📦", "Total Production (tonnes)", fmt_num(kpis["total_production"]), "comparable-unit crops"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("📈", "Avg Yield (tonnes/hectare)", f"{kpis['avg_yield']:.2f} t/ha" if kpis["avg_yield"] is not None else "—", "comparable-unit crops"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("🌱", "Crops Tracked", str(kpis["num_crops"]), ""), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("🗺️", "States/Regions", str(kpis["num_states"]), ""), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi_card("🏆", "Top Comparable-Yield Crop", kpis["top_crop"] or "—", f"{kpis['top_crop_yield']:.2f} t/ha" if kpis.get("top_crop_yield") else ""), unsafe_allow_html=True)
    with c6:
        st.markdown(kpi_card("🛡️", "Most Stable Crop", kpis.get("most_stable_crop") or "—", f"score {kpis['most_stable_score']:.2f}" if kpis.get("most_stable_score") else ""), unsafe_allow_html=True)

    st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Mandatory Analysis", "Production by Crop")
        prod = analysis.production_by_crop(df, top_n=12)
        if len(prod):
            st.plotly_chart(viz.production_bar_chart(prod), use_container_width=True)
        else:
            empty_state()
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Mandatory Analysis", "Average Yield by Crop")
        yld = analysis.avg_yield_by_crop(df, top_n=12)
        if len(yld):
            st.plotly_chart(viz.yield_lollipop_chart(yld), use_container_width=True)
        else:
            empty_state()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("Trend", "Yearly Production & Yield Trend")
    yt = analysis.yearly_trend(df)
    if len(yt):
        st.plotly_chart(viz.yearly_trend_chart(yt), use_container_width=True)
    else:
        empty_state()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("Automatic Insight Engine", "Key Insights")
    for ins in insights_mod.generate_insights(df):
        st.markdown(f'<div class="insight-card">{ins}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ========================================================================
# PAGE 2 — CROP INTELLIGENCE
# ========================================================================

def render_crop_intelligence():
    st.markdown("## 🌱 Crop Intelligence")
    st.caption("Production, yield, distribution, and stability — per crop.")

    if not preprocessing.safe_dataframe(df):
        empty_state()
        return

    comparable = data_loader.comparable_yield_df(df)
    callout(COCONUT_NOTE, "warn")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Ranking & Distribution", "🛡️ Stability Analysis", "⚖️ Crop Comparison", "🔍 Single-Crop Deep Dive"])

    with tab1:
        stats = stats_mod.crop_summary_stats(comparable)
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Mandatory Analysis", "Yield Distribution (Boxplot)")
        default_crops = stats["Crop"].head(8).tolist()
        chosen = st.multiselect("Crops to compare", sorted(comparable["Crop"].unique()), default=default_crops, key="box_crops")
        if chosen:
            st.plotly_chart(viz.yield_boxplot(comparable, chosen), use_container_width=True)
        else:
            empty_state("Select at least one crop.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Descriptive Statistics", "Per-Crop Summary")
        st.caption("All yield figures in tonnes/hectare (comparable-unit crops only).")
        display_stats = stats[["Crop", "count", "mean", "median", "std", "cv", "stability_score"]].rename(
            columns={"count": "Records", "mean": "Mean Yield", "median": "Median Yield",
                     "std": "Std Dev", "cv": "Coeff. of Variation", "stability_score": "Stability Score"}
        )
        st.dataframe(display_stats.style.format({
            "Mean Yield": "{:.2f}", "Median Yield": "{:.2f}", "Std Dev": "{:.2f}",
            "Coeff. of Variation": "{:.2f}", "Stability Score": "{:.3f}",
        }), use_container_width=True, height=360)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Crop Stability Analysis", "Stability Ranking (Coefficient of Variation)")
        st.markdown(
            "Stability Score = 1 / (1 + CV), where **CV = Std Dev / Mean** yield for that crop "
            "across all records. Lower variation → higher score → more consistent yield "
            "historically. This is a direct statistical transform, not an arbitrary weighting."
        )
        ranking = stats_mod.stability_ranking(comparable)
        if len(ranking):
            st.plotly_chart(viz.stability_chart(ranking, top_n=20), use_container_width=True)
        else:
            empty_state()
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Interactive Comparison", "Compare Selected Crops")
        crops_avail = sorted(comparable["Crop"].unique())
        pick = st.multiselect("Choose crops", crops_avail, default=crops_avail[:3] if len(crops_avail) >= 3 else crops_avail, key="cmp_crops")
        if pick:
            cmp_stats = stats_mod.crop_summary_stats(comparable[comparable["Crop"].isin(pick)])
            st.dataframe(
                cmp_stats[["Crop", "count", "mean", "median", "std", "min", "max", "stability_score"]]
                .rename(columns={"count": "Records", "mean": "Mean", "median": "Median", "std": "Std Dev",
                                  "min": "Min", "max": "Max", "stability_score": "Stability"})
                .style.format({"Mean": "{:.2f}", "Median": "{:.2f}", "Std Dev": "{:.2f}",
                                "Min": "{:.2f}", "Max": "{:.2f}", "Stability": "{:.3f}"}),
                use_container_width=True,
            )
            st.plotly_chart(viz.yield_boxplot(comparable, pick), use_container_width=True)
        else:
            empty_state("Select at least one crop to compare.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Crop-Specific Exploration", "Any Crop — Including Coconut")
        st.caption(
            "Explores a single crop in its own native unit. Coconut is available here, reported "
            "in nuts/hectare, and is not mixed with tonnes/hectare crops elsewhere on this page."
        )
        all_crops_full = sorted(df["Crop"].unique())
        single_crop = st.selectbox("Choose a crop", all_crops_full, key="single_crop")
        crop_df = df[df["Crop"] == single_crop]
        unit = "nuts/hectare" if ml_model.is_excluded_from_general_model(single_crop) else "tonnes/hectare"
        if single_crop == "Coconut":
            callout(
                "🥥 Coconut yield is reported in <b>nuts/hectare</b>, not tonnes/hectare. The "
                "figures below describe Coconut on its own and are not comparable to any other "
                "crop's numbers shown elsewhere in this app.",
                "warn",
            )

        if len(crop_df):
            single_stats = stats_mod.crop_summary_stats(crop_df)
            row = single_stats.iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(metric_chip(f"Mean Yield ({unit})", f"{row['mean']:.2f}"), unsafe_allow_html=True)
            with m2:
                st.markdown(metric_chip(f"Median Yield ({unit})", f"{row['median']:.2f}"), unsafe_allow_html=True)
            with m3:
                st.markdown(metric_chip("Coeff. of Variation", f"{row['cv']:.2f}"), unsafe_allow_html=True)
            with m4:
                st.markdown(metric_chip("Stability Score", f"{row['stability_score']:.3f}"), unsafe_allow_html=True)
            st.plotly_chart(viz.yield_boxplot(crop_df, [single_crop], unit_label=unit), use_container_width=True)
        else:
            empty_state()
        st.markdown('</div>', unsafe_allow_html=True)


# ========================================================================
# PAGE 3 — REGIONAL INTELLIGENCE
# ========================================================================

def render_regional_intelligence():
    st.markdown("## 🗺️ Regional Intelligence")
    st.caption("State-level production and yield performance, and the Crop × State heatmap.")

    if not preprocessing.safe_dataframe(df):
        empty_state()
        return

    callout(COCONUT_NOTE, "warn")
    comparable = data_loader.comparable_yield_df(df)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Ranking", "State Production Ranking")
        state_sum = analysis.statewise_summary(df)
        if len(state_sum):
            st.plotly_chart(viz.state_ranking_chart(state_sum, "Production", top_n=15), use_container_width=True)
        else:
            empty_state()
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Ranking", "State Average Yield Ranking")
        if len(state_sum) and "Yield" in state_sum.columns:
            yield_sorted = state_sum.dropna(subset=["Yield"]).sort_values("Yield", ascending=False)
            st.plotly_chart(viz.state_ranking_chart(yield_sorted, "Yield", top_n=15), use_container_width=True)
        else:
            empty_state()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("Signature Visualization", "Crop × State Average Yield Heatmap")
    c1, c2 = st.columns(2)
    with c1:
        top_n_crops = st.slider("Top N crops (by production)", 5, 30, 15, key="heat_crops")
    with c2:
        top_n_states = st.slider("Top N states (by production)", 5, 30, 15, key="heat_states")

    top_crops_list = analysis.production_by_crop(comparable, top_n=top_n_crops)["Crop"].tolist()
    top_states_list = analysis.statewise_summary(comparable).head(top_n_states)["State"].tolist()
    pivot = analysis.crop_state_heatmap_data(comparable, top_crops=top_crops_list, top_states=top_states_list)
    if pivot.size:
        st.plotly_chart(viz.crop_state_heatmap(pivot), use_container_width=True)
    else:
        empty_state()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("Scale Overview", "Productivity vs Scale (Bubble Chart)")
    if len(comparable):
        st.plotly_chart(viz.yield_production_bubble(comparable), use_container_width=True)
    else:
        empty_state()
    st.markdown('</div>', unsafe_allow_html=True)


# ========================================================================
# PAGE 4 — AGRICULTURAL FACTORS
# ========================================================================

def render_agricultural_factors():
    st.markdown("## 🌦️ Agricultural Factors")
    st.caption("Association between environmental / input factors and yield. Correlation, not causation.")

    if not preprocessing.safe_dataframe(df):
        empty_state()
        return

    comparable = data_loader.comparable_yield_df(df)
    factor_cols = [c for c in ["Annual_Rainfall", "Fertilizer", "Pesticide", "Area"] if c in comparable.columns]
    factor_cols += [c for c in colmap.other_numeric_factors if c in comparable.columns]

    if not factor_cols:
        empty_state("No numeric agricultural factor columns were detected in this dataset.")
        return

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("Overview", "Correlation Matrix")
    corr = stats_mod.correlation_matrix(comparable, factor_cols + ["Yield"])
    st.plotly_chart(viz.correlation_heatmap(corr), use_container_width=True)
    st.caption("Correlation shows association strength and direction only. It does not prove causation.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("Detail", "Factor vs Yield")
    factor = st.selectbox("Choose a factor", factor_cols, key="factor_select")
    st.plotly_chart(viz.factor_scatter(comparable, factor), use_container_width=True)
    r = comparable[[factor, "Yield"]].corr().iloc[0, 1]
    direction = "positive" if r > 0 else "negative"
    st.markdown(
        f'<div class="insight-card">🔬 <b>{factor.replace("_", " ")}</b> shows a '
        f'<b>{direction}</b> association with yield (r = <b>{r:.2f}</b>) in the current selection. '
        f'This is an observed association, not evidence of a causal effect.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ========================================================================
# PAGE 5 — AI YIELD PREDICTION
# ========================================================================

def render_prediction():
    st.markdown("## 🤖 AI Yield Prediction")
    st.caption("Model comparison, evaluation, and interactive prediction. Uses the FULL dataset (unaffected by sidebar filters) for training.")

    required = {"Crop", "State", "Season", "Crop_Year", "Area", "Yield"}
    if not required.issubset(df_full.columns):
        empty_state("The dataset is missing columns required for yield prediction.")
        return

    leak_check = analysis.yield_area_relationship_check(df_full)
    callout(
        f"🚨 <b>Leakage check</b>: correlation between Yield and Production/Area = "
        f"<b>{leak_check['correlation']:.4f}</b> (computed live from the current dataset, "
        f"{leak_check['n_rows_checked']:,} rows checked). Because Yield can be almost exactly "
        f"reconstructed from Production ÷ Area, <b>Production is excluded</b> from every model's "
        f"feature set below.",
        "warn",
    )
    callout(
        "🥥 <b>Unit check</b>: Coconut's Yield (and Production) are reported in "
        "<b>nuts</b>, not tonnes — mixing that with tonnes/hectare crops in one regression "
        "target is not scientifically meaningful. The general model below is therefore trained "
        "only on the comparable tonnes/hectare crop subset; Coconut is excluded from training "
        "but remains selectable in the Predict tab, where it is handled with a clear message "
        "rather than a misleading prediction.",
        "warn",
    )

    with st.spinner("Training models..."):
        results, trained, (cat_cols, num_cols), random_best_name = ml_model.train_and_compare(df_full)
        ta = ml_model.train_time_aware(df_full)
        primary_model, primary_name, (p_cat, p_num), primary_source = ml_model.get_primary_model(df_full)

    callout(
        "Because yield prediction is intended for <b>future</b> use, AGRIVISTA uses chronological "
        "(time-aware) validation as the <b>primary</b> model-selection strategy. Random-split "
        "results are shown for comparison but are not used as the primary basis for model "
        "selection.",
        "info",
    )

    tab1, tab2, tab3 = st.tabs(["⏳ Time-Aware Validation (Primary)", "🔮 Predict Yield", "📊 Random Split (Reference)"])

    with tab1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Scientific Validity · Primary Evaluation", "Time-Aware (Chronological) Validation")
        st.markdown(
            "A random 80/20 split lets the model see records from *later* years during training "
            "and get tested on *earlier* years — unrealistic for a forecasting scenario, where "
            "future data is never available at training time. This evaluates the same models "
            "trained on the **earliest years only** and tested on the **most recent years**, "
            "which is the more realistic estimate of real-world forecasting performance — and is "
            "the basis on which the **primary deployed model** below is selected."
        )
        if ta is None:
            empty_state("Not enough distinct years in the current dataset for a meaningful chronological split.")
        else:
            st.caption(
                f"Trained on {ta['train_years'][0]}–{ta['train_years'][1]} ({ta['n_train']:,} rows) · "
                f"Tested on {ta['test_years'][0]}–{ta['test_years'][1]} ({ta['n_test']:,} rows) · "
                f"Split point: end of {ta['cutoff_year']} · Comparable-unit crops only (Coconut excluded)"
            )
            st.dataframe(
                ta["results"].style.format({"R2": "{:.4f}", "MAE": "{:.3f}", "RMSE": "{:.3f}"}),
                use_container_width=True,
            )
            st.caption("R² is a goodness-of-fit measure (0-1, higher is better) — not 'accuracy'. MAE and RMSE are in tonnes/hectare.")
            st.markdown(
                f'<div class="insight-card">🏆 <b>{primary_name}</b> is selected as the '
                f'<b>primary deployed model</b> (highest time-aware R²). It powers the Predict '
                f'Yield and What-If Analysis tools.</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        if ta is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                section_header(f"Primary Model: {primary_name}", "Actual vs Predicted Yield (Time-Aware Test Set)")
                st.plotly_chart(viz.actual_vs_predicted_chart(primary_model.y_test, primary_model.y_pred), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                section_header(f"Primary Model: {primary_name}", "Feature Importance")
                fi = ml_model.get_feature_importance(primary_model, p_cat, p_num)
                if len(fi):
                    st.plotly_chart(viz.feature_importance_chart(fi), use_container_width=True)
                    st.caption("Feature importance reflects influence on the model's predictions. It does not prove a causal relationship.")
                else:
                    st.info(f"{primary_name} does not expose feature importances (e.g. a linear model — see coefficients instead).")
                st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header(f"Interactive · Primary Model: {primary_name}", "Predict Yield for a New Scenario")

        in_crop = st.selectbox("Crop", sorted(df_full["Crop"].unique()), key="pred_crop")

        if ml_model.is_excluded_from_general_model(in_crop):
            callout(
                f"🥥 <b>{in_crop}</b> is measured in <b>nuts/hectare</b> in the source dataset and is "
                f"therefore excluded from the general comparable-unit prediction model. A prediction "
                f"here would mix units and would not be scientifically meaningful. "
                f"{in_crop} remains available for crop-specific exploration on the Crop Intelligence page.",
                "warn",
            )
        else:
            c1, c2 = st.columns(2)
            with c1:
                in_state = st.selectbox("State", sorted(df_full["State"].unique()), key="pred_state")
                in_season = st.selectbox("Season", sorted(df_full["Season"].unique()), key="pred_season")
            with c2:
                in_year = st.number_input("Year", min_value=1997, max_value=2035, value=2024, key="pred_year")
                in_area = st.number_input("Area (hectares)", min_value=0.1, value=1000.0, key="pred_area")

            c4, c5, c6 = st.columns(3)
            with c4:
                in_rain = st.number_input("Annual Rainfall (mm)", min_value=0.0, value=float(df_full["Annual_Rainfall"].median()), key="pred_rain") if "Annual_Rainfall" in p_num else None
            with c5:
                in_fert = st.number_input("Fertilizer (kg)", min_value=0.0, value=float(df_full["Fertilizer"].median()), key="pred_fert") if "Fertilizer" in p_num else None
            with c6:
                in_pest = st.number_input("Pesticide (kg)", min_value=0.0, value=float(df_full["Pesticide"].median()), key="pred_pest") if "Pesticide" in p_num else None

            input_dict = {"Crop": in_crop, "State": in_state, "Season": in_season, "Crop_Year": in_year, "Area": in_area,
                           "Annual_Rainfall": in_rain, "Fertilizer": in_fert, "Pesticide": in_pest}

            if st.button("🔮 Predict Yield", type="primary"):
                pred = ml_model.predict_single(primary_model, input_dict, p_cat + p_num)
                st.markdown(
                    f"""
                    <div class="predict-result">
                        <div class="predict-label">Predicted Yield (tonnes/hectare) · {primary_name} · time-aware selected</div>
                        <div class="predict-value">{pred:,.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.markdown(metric_chip("R² (time-aware)", f"{primary_model.r2:.3f}"), unsafe_allow_html=True)
                with mc2:
                    st.markdown(metric_chip("MAE", f"{primary_model.mae:.2f}"), unsafe_allow_html=True)
                with mc3:
                    st.markdown(metric_chip("RMSE", f"{primary_model.rmse:.2f}"), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        section_header("Random 80/20 Split · Reference Only", "Model Comparison")
        st.caption("Shown for comparison. Not used to select the primary deployed model — see the Time-Aware tab.")
        st.plotly_chart(viz.model_comparison_chart(results), use_container_width=True)
        st.dataframe(
            results.style.format({"R2": "{:.4f}", "MAE": "{:.3f}", "RMSE": "{:.3f}"}),
            use_container_width=True,
        )
        st.caption("R² is a goodness-of-fit measure (0-1, higher is better) — not 'accuracy'. MAE and RMSE are in tonnes/hectare. Comparable-unit crops only (Coconut excluded).")
        st.markdown('</div>', unsafe_allow_html=True)

        random_best_model = trained[random_best_name]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            section_header(f"Random-Split Best: {random_best_name}", "Actual vs Predicted Yield")
            st.plotly_chart(viz.actual_vs_predicted_chart(random_best_model.y_test, random_best_model.y_pred), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            section_header(f"Random-Split Best: {random_best_name}", "Feature Importance")
            fi = ml_model.get_feature_importance(random_best_model, cat_cols, num_cols)
            if len(fi):
                st.plotly_chart(viz.feature_importance_chart(fi), use_container_width=True)
                st.caption("Feature importance reflects influence on the model's predictions. It does not prove a causal relationship.")
            else:
                st.info(f"{random_best_name} does not expose feature importances (e.g. a linear model — see coefficients instead).")
            st.markdown('</div>', unsafe_allow_html=True)


# ========================================================================
# PAGE 6 — WHAT-IF ANALYSIS
# ========================================================================

def render_what_if():
    st.markdown("## 🔬 What-If Analysis")
    st.caption("Model-based scenario analysis — sweep one input and see how the primary model's prediction changes.")

    callout(
        "This tool shows what the trained model *predicts* as one input changes, holding all "
        "others fixed. It reflects the model's learned association, not a guaranteed real-world "
        "causal effect. Example: changing rainfall from X to Y changes the model's predicted "
        "yield from A to B — that does not mean changing rainfall will actually cause that "
        "exact change in a real field.",
        "info",
    )

    required = {"Crop", "State", "Season", "Crop_Year", "Area", "Yield"}
    if not required.issubset(df_full.columns):
        empty_state("The dataset is missing columns required for scenario analysis.")
        return

    with st.spinner("Preparing model..."):
        model, primary_name, (cat_cols, num_cols), primary_source = ml_model.get_primary_model(df_full)

    comparable_crops = sorted(data_loader.comparable_yield_df(df_full)["Crop"].unique())
    if not comparable_crops:
        empty_state("No comparable-unit crops available for scenario analysis.")
        return

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header(f"Base Scenario · Primary Model: {primary_name} (time-aware selected)", "Set Base Inputs")
    st.caption("Crop selection is limited to comparable tonnes/hectare crops — Coconut is excluded for the same unit reason as the general prediction model.")
    c1, c2, c3 = st.columns(3)
    with c1:
        base_crop = st.selectbox("Crop", comparable_crops, key="wi_crop")
        base_state = st.selectbox("State", sorted(df_full["State"].unique()), key="wi_state")
    with c2:
        base_season = st.selectbox("Season", sorted(df_full["Season"].unique()), key="wi_season")
        base_year = st.number_input("Year", min_value=1997, max_value=2035, value=2024, key="wi_year")
    with c3:
        base_area = st.number_input("Area (hectares)", min_value=0.1, value=1000.0, key="wi_area")

    base_rain = float(df_full["Annual_Rainfall"].median()) if "Annual_Rainfall" in num_cols else None
    base_fert = float(df_full["Fertilizer"].median()) if "Fertilizer" in num_cols else None
    base_pest = float(df_full["Pesticide"].median()) if "Pesticide" in num_cols else None
    base_input = {"Crop": base_crop, "State": base_state, "Season": base_season, "Crop_Year": base_year,
                  "Area": base_area, "Annual_Rainfall": base_rain, "Fertilizer": base_fert, "Pesticide": base_pest}

    vary_options = [c for c in ["Annual_Rainfall", "Fertilizer", "Pesticide", "Area"] if c in num_cols]
    vary_col = st.selectbox("Variable to vary", vary_options, key="wi_vary")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("Model-Based Scenario Analysis", f"Predicted Yield (tonnes/hectare) vs {vary_col.replace('_', ' ')}")

    base_val = base_input[vary_col]
    lo, hi = base_val * 0.3, base_val * 2.0
    sweep_values = list(pd.RangeIndex(0, 9).map(lambda i: lo + (hi - lo) * i / 8))

    scenario_df = ml_model.what_if_scenario(model, base_input, cat_cols + num_cols, vary_col, sweep_values)
    st.plotly_chart(viz.what_if_chart(scenario_df, vary_col), use_container_width=True)
    st.dataframe(scenario_df.style.format({vary_col: "{:.1f}", "Predicted Yield": "{:.2f}"}), use_container_width=True)
    st.caption(
        "\"Model-based scenario analysis\" using the primary (time-aware selected) model: each "
        f"point is the model's prediction with only {vary_col.replace('_', ' ')} changed. "
        "Not a guaranteed real-world outcome."
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ========================================================================
# PAGE 7 — METHODOLOGY
# ========================================================================

def render_methodology():
    st.markdown("## 📚 Methodology")
    st.caption("A judge-friendly explanation of how this platform's numbers are produced.")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("Dataset", "Source & Scope")
    st.markdown(
        f"- **Records:** {cleaning_report.original_rows:,} raw → {cleaning_report.cleaned_rows:,} after cleaning\n"
        f"- **Coverage:** {int(df_full['Crop_Year'].min())}–{int(df_full['Crop_Year'].max())}, "
        f"{df_full['State'].nunique()} states, {df_full['Crop'].nunique()} crops\n"
        f"- **Columns:** {', '.join([k for k, v in available.items() if v])}"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("Data Cleaning", "What Was Done — And Why")
    st.markdown(
        f"- Text columns stripped of whitespace: {', '.join(cleaning_report.text_columns_stripped) or 'none needed'}\n"
        f"- Duplicate rows removed: **{cleaning_report.duplicate_rows_removed}**\n"
        f"- Rows removed for missing essential fields: **{cleaning_report.missing_value_rows_removed}**\n"
        f"- Rows removed for invalid (negative) numeric values: **{cleaning_report.invalid_numeric_rows_removed}**\n"
        f"- Total rows removed: **{cleaning_report.rows_removed}** of {cleaning_report.original_rows:,}"
    )
    for note in cleaning_report.notes:
        callout(note, "info")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("Data Quality", "Unit Limitation — Coconut")
    callout(
        "Yield units differ in the source dataset. <b>Coconut</b> is reported in "
        "<b>nuts/hectare</b> (and total nuts for Production), while every other crop is "
        "reported in <b>tonnes/hectare</b> (and total tonnes). Direct numerical comparison "
        "is therefore not meaningful. This platform excludes Coconut from every cross-crop "
        "Yield and Production ranking, total, and KPI by default, while keeping it available "
        "for crop-specific analysis (its own boxplot, stability score, etc.).",
        "warn",
    )
    callout(
        "This exclusion extends to the <b>machine learning models</b>: the general yield "
        "prediction model is trained only on the comparable tonnes/hectare crop subset, so "
        "Coconut's nuts-based Yield never enters the shared regression target. Coconut is not "
        "removed from the dataset — selecting it in the Predict Yield tab shows a clear "
        "explanatory message instead of a misleading cross-unit prediction.",
        "warn",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("Feature Engineering & Leakage Prevention", "Why Production Is Excluded from ML")
    leak_check = analysis.yield_area_relationship_check(df_full)
    st.markdown(
        f"Yield in this dataset is (almost) exactly Production ÷ Area — verified live on the "
        f"current data: correlation between reported Yield and computed Production/Area = "
        f"**{leak_check['correlation']:.4f}** across {leak_check['n_rows_checked']:,} rows. "
        f"Because Yield can be reconstructed almost exactly from Production, including "
        f"Production as a model feature would let the model 'cheat' via near-exact division "
        f"rather than learn genuine agronomic relationships — producing an inflated R² that "
        f"would not generalize to real forecasting (where the season's Production is not yet "
        f"known). **Production is therefore excluded from every model's feature set.** Area is "
        f"kept: on its own, Area does not determine Yield, and it is a genuine, independently-"
        f"known agronomic input."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("Model Evaluation & Selection", "Random vs. Time-Aware Validation")
    st.markdown(
        "This dataset spans multiple years (1997–2020). A random 80/20 split can let the model "
        "train on records from later years and get evaluated on earlier years — unrealistic for "
        "forecasting, since future data is never available at training time. This platform "
        "reports **both**: a standard random split, and a **time-aware split** (train on the "
        "earliest years, test on the most recent years).\n\n"
        "**Because yield prediction is intended for future use, AGRIVISTA uses chronological "
        "validation as the primary model-selection strategy.** The model type deployed to the "
        "interactive Predict Yield and What-If Analysis tools is whichever model scores highest "
        "on the time-aware evaluation — not whichever happens to score highest on the random "
        "split. Random-split results remain visible on the AI Yield Prediction page for "
        "comparison, but are not used as the basis for model selection. See the AI Yield "
        "Prediction page's *Time-Aware Validation (Primary)* tab for live numbers."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("ML Models Compared", "Algorithms & Metrics")
    st.markdown(
        "- **Linear Regression** — simple, interpretable baseline\n"
        "- **Decision Tree Regressor** — captures non-linear splits\n"
        "- **Random Forest Regressor** — ensemble of trees, typically strong on tabular data\n"
        "- **Gradient Boosting Regressor** — sequential ensemble, often the strongest fit\n\n"
        "Evaluated with **R²** (goodness of fit, 0–1, higher is better — never called "
        "'accuracy'), **MAE** (mean absolute error, same units as Yield), and **RMSE** "
        "(root mean squared error, penalizes larger errors more)."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    section_header("Limitations", "What This Platform Does Not Claim")
    st.markdown(
        "- Correlations shown (rainfall, fertilizer, pesticide vs. yield) are associations, "
        "**not proof of causation**.\n"
        "- The What-If tool shows model-based scenario predictions, not guaranteed real-world "
        "agronomic outcomes.\n"
        "- Coconut's raw yield/production numbers are not comparable to other crops due to "
        "differing units (see above).\n"
        "- The dataset does not include soil quality, irrigation infrastructure, crop variety, "
        "or farming practice — factors that materially affect real yields but are not present "
        "here."
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ========================================================================
# ROUTER
# ========================================================================

PAGES = {
    "🏠 Overview": render_overview,
    "🌱 Crop Intelligence": render_crop_intelligence,
    "🗺️ Regional Intelligence": render_regional_intelligence,
    "🌦️ Agricultural Factors": render_agricultural_factors,
    "🤖 AI Yield Prediction": render_prediction,
    "🔬 What-If Analysis": render_what_if,
    "📚 Methodology": render_methodology,
}

PAGES[page]()
