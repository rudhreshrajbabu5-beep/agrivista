"""
visualizations.py
-------------------
Every chart in AGRIVISTA is built here so the app has one consistent
visual language: a shared Plotly template and the AgriVista color palette.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------------------------
# Design tokens
# ----------------------------------------------------------------------

DEEP_GREEN = "#1B4332"
FOREST_GREEN = "#2D6A4F"
SAGE_GREEN = "#74A57F"
SOFT_GREEN = "#B7E4C7"
AMBER = "#D98E04"
CLAY = "#BC6C25"
SLATE = "#4A5859"
OFF_WHITE = "#FAF6EE"
CARD_WHITE = "#FFFFFF"
MUTED_GRID = "#E7E2D6"

SEQUENTIAL_GREEN = ["#F1F8F1", "#D8ECDD", "#B7E4C7", "#95D5B2", "#74C69D",
                     "#52B788", "#40916C", "#2D6A4F", "#1B4332"]

CATEGORICAL_PALETTE = [
    "#2D6A4F", "#D98E04", "#4A5859", "#74A57F", "#BC6C25",
    "#40916C", "#A98467", "#1B4332", "#E9C46A", "#588157",
]

FONT_FAMILY = "'Inter', 'Segoe UI', sans-serif"


def _base_layout(fig: go.Figure, title: str = None, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=17, color=DEEP_GREEN, family=FONT_FAMILY), x=0.02, xanchor="left") if title else None,
        font=dict(family=FONT_FAMILY, color=SLATE, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=55 if title else 20, b=10),
        height=height,
        hoverlabel=dict(bgcolor=CARD_WHITE, font_size=13, font_family=FONT_FAMILY, bordercolor=SAGE_GREEN),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=True, gridcolor=MUTED_GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=MUTED_GRID, zeroline=False)
    return fig


# 1. Production bar chart -----------------------------------------------

def production_bar_chart(df: pd.DataFrame, top_n: int = 12, height: int = 440) -> go.Figure:
    data = df.sort_values("Production", ascending=False).head(top_n).sort_values("Production")
    fig = go.Figure(go.Bar(
        x=data["Production"], y=data["Crop"], orientation="h",
        marker=dict(color=data["Production"], colorscale=SEQUENTIAL_GREEN, line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>Production (tonnes): %{x:,.0f}<extra></extra>",
    ))
    fig = _base_layout(fig, title="Production by Crop (tonnes)", height=height)
    fig.update_xaxes(title="Total Production (tonnes)")
    fig.update_yaxes(title=None)
    return fig


# 2. Average yield lollipop chart ---------------------------------------

def yield_lollipop_chart(df: pd.DataFrame, top_n: int = 12, height: int = 440) -> go.Figure:
    data = df.sort_values("Yield", ascending=False).head(top_n).sort_values("Yield")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["Yield"], y=data["Crop"], mode="markers",
        marker=dict(size=12, color=AMBER, line=dict(width=1, color=DEEP_GREEN)),
        hovertemplate="<b>%{y}</b><br>Avg Yield (t/ha): %{x:,.2f}<extra></extra>", showlegend=False,
    ))
    for _, row in data.iterrows():
        fig.add_shape(type="line", x0=0, x1=row["Yield"], y0=row["Crop"], y1=row["Crop"],
                       line=dict(color=SAGE_GREEN, width=2))
    fig = _base_layout(fig, title="Average Yield by Crop (tonnes/hectare)", height=height)
    fig.update_xaxes(title="Average Yield (tonnes/hectare)")
    fig.update_yaxes(title=None)
    return fig


# 3. Yield boxplot --------------------------------------------------------

def yield_boxplot(df: pd.DataFrame, crops: list, height: int = 480, unit_label: str = "tonnes/hectare") -> go.Figure:
    data = df[df["Crop"].isin(crops)] if crops else df
    order = data.groupby("Crop")["Yield"].median().sort_values(ascending=False).index.tolist()
    fig = px.box(
        data, x="Crop", y="Yield", color="Crop",
        category_orders={"Crop": order}, color_discrete_sequence=CATEGORICAL_PALETTE,
        points="outliers",
    )
    fig.update_traces(marker=dict(size=4, opacity=0.6), line=dict(width=1.5))
    fig = _base_layout(fig, title=f"Yield Distribution & Consistency by Crop ({unit_label})", height=height)
    fig.update_xaxes(title=None, tickangle=-30)
    fig.update_yaxes(title=f"Yield ({unit_label})")
    fig.update_layout(showlegend=False)
    return fig


# 4. Yearly trend (dual-axis) ---------------------------------------------

def yearly_trend_chart(df: pd.DataFrame, height: int = 420) -> go.Figure:
    fig = go.Figure()
    if "Production" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Crop_Year"], y=df["Production"], name="Total Production (tonnes)",
            mode="lines+markers", line=dict(color=FOREST_GREEN, width=3), marker=dict(size=5), yaxis="y1",
            hovertemplate="Year %{x}<br>Production (tonnes): %{y:,.0f}<extra></extra>",
        ))
    if "Yield" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Crop_Year"], y=df["Yield"], name="Average Comparable Yield (t/ha)",
            mode="lines+markers", line=dict(color=AMBER, width=3, dash="dot"), marker=dict(size=5), yaxis="y2",
            hovertemplate="Year %{x}<br>Avg Yield (t/ha): %{y:,.2f}<extra></extra>",
        ))
    fig.update_layout(
        yaxis=dict(title="Total Production (tonnes)", showgrid=True, gridcolor=MUTED_GRID),
        yaxis2=dict(title="Average Yield (t/ha)", overlaying="y", side="right", showgrid=False),
        xaxis=dict(title="Year"),
    )
    fig = _base_layout(fig, title="Yearly Production (tonnes) & Yield (t/ha) Trend", height=height)
    return fig


# 5. Crop x State heatmap (signature viz) ---------------------------------

def crop_state_heatmap(pivot: pd.DataFrame, height: int = 560) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=SEQUENTIAL_GREEN, colorbar=dict(title="Avg Yield (t/ha)"),
        hovertemplate="Crop: %{y}<br>State: %{x}<br>Avg Yield (t/ha): %{z:,.2f}<extra></extra>",
    ))
    fig = _base_layout(fig, title="Crop × State Average Yield Heatmap (tonnes/hectare)", height=height)
    fig.update_xaxes(tickangle=-45, title=None)
    fig.update_yaxes(title=None)
    return fig


# 6. Bubble chart -----------------------------------------------------------

def yield_production_bubble(df: pd.DataFrame, height: int = 500) -> go.Figure:
    agg = df.groupby("Crop", as_index=False).agg(
        Avg_Yield=("Yield", "mean"), Total_Production=("Production", "sum"), Total_Area=("Area", "sum"),
    )
    fig = px.scatter(
        agg, x="Avg_Yield", y="Total_Production", size="Total_Area", color="Crop",
        color_discrete_sequence=CATEGORICAL_PALETTE, size_max=55, hover_name="Crop",
        labels={"Avg_Yield": "Average Yield (t/ha)", "Total_Production": "Total Production (tonnes)", "Total_Area": "Total Area"},
    )
    fig.update_traces(marker=dict(line=dict(width=1, color="white"), opacity=0.85))
    fig = _base_layout(fig, title="Productivity vs Scale: Yield (t/ha), Production (tonnes) & Area by Crop", height=height)
    fig.update_yaxes(type="log", title="Total Production (tonnes, log scale)")
    fig.update_xaxes(title="Average Yield (tonnes/hectare)")
    return fig


# 7. State ranking bar -------------------------------------------------------

_METRIC_UNITS = {"Production": "tonnes", "Yield": "tonnes/hectare"}


def state_ranking_chart(df: pd.DataFrame, metric: str, top_n: int = 15, height: int = 460) -> go.Figure:
    data = df.sort_values(metric, ascending=False).head(top_n).sort_values(metric)
    unit = _METRIC_UNITS.get(metric, "")
    axis_label = f"{metric.replace('_', ' ')} ({unit})" if unit else metric.replace("_", " ")
    fig = go.Figure(go.Bar(
        x=data[metric], y=data["State"], orientation="h",
        marker=dict(color=data[metric], colorscale=SEQUENTIAL_GREEN),
        hovertemplate="<b>%{y}</b><br>" + axis_label + ": %{x:,.2f}<extra></extra>",
    ))
    fig = _base_layout(fig, title=f"State Ranking by {axis_label}", height=height)
    fig.update_xaxes(title=axis_label)
    fig.update_yaxes(title=None)
    return fig


# 8. Factor scatter -----------------------------------------------------------

def factor_scatter(df: pd.DataFrame, factor: str, height: int = 460) -> go.Figure:
    sample = df.sample(min(len(df), 4000), random_state=42) if len(df) > 4000 else df
    color = "Crop" if sample["Crop"].nunique() <= 12 else None
    fig = px.scatter(
        sample, x=factor, y="Yield", color=color, color_discrete_sequence=CATEGORICAL_PALETTE,
        opacity=0.6, trendline="ols" if len(sample) > 5 else None,
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0.3, color="white")))
    fig = _base_layout(fig, title=f"{factor.replace('_', ' ')} vs Yield (tonnes/hectare)", height=height)
    fig.update_xaxes(title=factor.replace("_", " "))
    fig.update_yaxes(title="Yield (tonnes/hectare)")
    return fig


def correlation_heatmap(corr: pd.DataFrame, height: int = 420) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale=[[0, CLAY], [0.5, OFF_WHITE], [1, FOREST_GREEN]], zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}", colorbar=dict(title="r"),
    ))
    fig = _base_layout(fig, title="Correlation Matrix of Agricultural Factors", height=height)
    return fig


# 9. Stability chart -----------------------------------------------------------

def stability_chart(ranking: pd.DataFrame, top_n: int = 15, height: int = 460) -> go.Figure:
    data = ranking.head(top_n).sort_values("stability_score")
    fig = go.Figure(go.Bar(
        x=data["stability_score"], y=data["Crop"], orientation="h",
        marker=dict(color=data["stability_score"], colorscale=SEQUENTIAL_GREEN),
        hovertemplate="<b>%{y}</b><br>Stability Score: %{x:.3f}<extra></extra>",
    ))
    fig = _base_layout(fig, title="Crop Stability Ranking (higher = more consistent yield)", height=height)
    fig.update_xaxes(title="Stability Score (0-1)", range=[0, 1])
    fig.update_yaxes(title=None)
    return fig


# 10. ML result charts -----------------------------------------------------------

def model_comparison_chart(results: pd.DataFrame, height: int = 380) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=results["Model"], y=results["R2"], marker=dict(color=CATEGORICAL_PALETTE[:len(results)]),
        text=results["R2"].round(3), textposition="outside",
        hovertemplate="<b>%{x}</b><br>R²: %{y:.4f}<extra></extra>",
    ))
    fig = _base_layout(fig, title="Model Comparison — R² (higher is better)", height=height)
    fig.update_yaxes(title="R² Score")
    fig.update_xaxes(title=None)
    return fig


def actual_vs_predicted_chart(y_true, y_pred, height: int = 460) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred, mode="markers", marker=dict(color=FOREST_GREEN, opacity=0.5, size=6),
        name="Predictions", hovertemplate="Actual: %{x:,.2f}<br>Predicted: %{y:,.2f}<extra></extra>",
    ))
    lo, hi = float(np.min(y_true)), float(np.max(y_true))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                              line=dict(color=CLAY, dash="dash", width=2), name="Perfect Prediction"))
    fig = _base_layout(fig, title="Actual vs Predicted Yield (Test Set)", height=height)
    fig.update_xaxes(title="Actual Yield")
    fig.update_yaxes(title="Predicted Yield")
    return fig


def feature_importance_chart(importances: pd.DataFrame, height: int = 420) -> go.Figure:
    data = importances.sort_values("Importance").tail(15)
    fig = go.Figure(go.Bar(
        x=data["Importance"], y=data["Feature"], orientation="h",
        marker=dict(color=data["Importance"], colorscale=SEQUENTIAL_GREEN),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig = _base_layout(fig, title="Feature Importance", height=height)
    fig.update_xaxes(title="Relative Importance")
    fig.update_yaxes(title=None)
    return fig


def what_if_chart(scenario_df: pd.DataFrame, vary_col: str, height: int = 420) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=scenario_df[vary_col], y=scenario_df["Predicted Yield"], mode="lines+markers",
        line=dict(color=FOREST_GREEN, width=3), marker=dict(size=8, color=AMBER),
        hovertemplate=f"{vary_col}: " + "%{x}<br>Predicted Yield: %{y:,.2f}<extra></extra>",
    ))
    fig = _base_layout(fig, title=f"Model-Based Scenario: Predicted Yield vs {vary_col.replace('_', ' ')}", height=height)
    fig.update_xaxes(title=vary_col.replace("_", " "))
    fig.update_yaxes(title="Predicted Yield")
    return fig
