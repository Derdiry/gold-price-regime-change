"""
Evaluation plots for model comparison, walk-forward validation,
learning curves, and Sudan-specific visualisations.

All functions return Plotly Figure objects.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────

MODEL_COLORS = {
    "lstm":      "#1f77b4",
    "gru":       "#ff7f0e",
    "tft":       "#2ca02c",
    "tsmixer":   "#d62728",
    "patchtst":  "#9467bd",
    "chronos":   "#8c564b",
    "ensemble":  "#e377c2",
    "actual":    "#17becf",
}


def _color(model_name: str) -> str:
    return MODEL_COLORS.get(model_name.lower(), "#7f7f7f")


# ─────────────────────────────────────────────────────────────────────────────
# Prediction overlay chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_predictions_overlay(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    title: str = "Gold Price: Actual vs Predictions",
    y_label: str = "Price",
) -> go.Figure:
    """
    Overlay actual prices with predictions from multiple models.

    predictions: {model_name: y_pred array}
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=y_true,
        name="Actual",
        line=dict(color=_color("actual"), width=2),
    ))

    for model_name, y_pred in predictions.items():
        fig.add_trace(go.Scatter(
            x=dates, y=y_pred,
            name=model_name.upper(),
            line=dict(color=_color(model_name), width=1.5, dash="dash"),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        template="plotly_dark",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty bands (MC Dropout)
# ─────────────────────────────────────────────────────────────────────────────

def plot_with_uncertainty(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    model_name: str = "Model",
    title: str = "Prediction with Uncertainty (MC Dropout)",
    y_label: str = "Price (SDG/gram)",
) -> go.Figure:
    """Plot predictions with 90% confidence interval from MC Dropout."""
    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=np.concatenate([dates, dates[::-1]]),
        y=np.concatenate([y_upper, y_lower[::-1]]),
        fill="toself",
        fillcolor="rgba(148,103,189,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        name="90% Interval",
        showlegend=True,
    ))

    fig.add_trace(go.Scatter(
        x=dates, y=y_true,
        name="Actual",
        line=dict(color=_color("actual"), width=2),
    ))

    fig.add_trace(go.Scatter(
        x=dates, y=y_mean,
        name=f"{model_name} (mean)",
        line=dict(color=_color(model_name), width=2),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        hovermode="x unified",
        template="plotly_dark",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Learning curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_learning_curves(
    histories: dict[str, dict],
    metric: str = "mae",
    title: str = "Training vs Validation Loss",
) -> go.Figure:
    """
    Plot train/val learning curves for multiple models.

    histories: {model_name: keras_history.history dict}
    """
    fig = make_subplots(
        rows=1, cols=len(histories),
        subplot_titles=list(histories.keys()),
        shared_yaxes=True,
    )

    for col, (name, hist) in enumerate(histories.items(), start=1):
        color = _color(name)
        train_key = metric
        val_key = f"val_{metric}"

        if train_key in hist:
            fig.add_trace(go.Scatter(
                y=hist[train_key], name=f"{name} train",
                line=dict(color=color, width=2),
                legendgroup=name,
            ), row=1, col=col)

        if val_key in hist:
            fig.add_trace(go.Scatter(
                y=hist[val_key], name=f"{name} val",
                line=dict(color=color, width=2, dash="dash"),
                legendgroup=name,
            ), row=1, col=col)

    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title=metric.upper(),
        template="plotly_dark",
        height=400,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Model comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(
    metrics_df: pd.DataFrame,
    metric: str = "mae",
    title: Optional[str] = None,
) -> go.Figure:
    """
    Bar chart comparing models on a single metric.

    metrics_df: DataFrame with columns [model, mae, rmse, mape, directional_accuracy]
    """
    title = title or f"Model Comparison — {metric.upper()}"
    df_sorted = metrics_df.sort_values(metric)

    colors = [_color(m) for m in df_sorted["model"]]

    fig = go.Figure(go.Bar(
        x=df_sorted["model"].str.upper(),
        y=df_sorted[metric],
        marker_color=colors,
        text=df_sorted[metric].round(3),
        textposition="outside",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Model",
        yaxis_title=metric.upper(),
        template="plotly_dark",
        showlegend=False,
    )
    return fig


def plot_metrics_heatmap(
    metrics_df: pd.DataFrame,
    title: str = "Model Metrics Heatmap",
) -> go.Figure:
    """
    Heatmap of all metrics across all models.
    Lower MAE/RMSE/MAPE is better; higher directional_accuracy is better.
    Values are rank-normalised so colour scale is consistent.
    """
    metric_cols = ["mae", "rmse", "mape", "directional_accuracy"]
    available = [c for c in metric_cols if c in metrics_df.columns]

    z = metrics_df[available].values.astype(float)

    # Rank-normalise each column to [0,1]
    # For MAE/RMSE/MAPE: lower rank → darker (better)
    # For dir_acc: higher rank → darker (better) — invert
    z_norm = np.zeros_like(z)
    for i, col in enumerate(available):
        col_vals = z[:, i]
        ranks = col_vals.argsort().argsort().astype(float)
        ranks /= max(ranks.max(), 1)
        if col == "directional_accuracy":
            ranks = 1 - ranks  # invert so higher acc = darker = better
        z_norm[:, i] = ranks

    fig = go.Figure(go.Heatmap(
        z=z_norm,
        x=[c.upper() for c in available],
        y=metrics_df["model"].str.upper().tolist(),
        colorscale="RdYlGn_r",
        text=np.round(z, 2),
        texttemplate="%{text}",
        showscale=False,
    ))

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=max(300, 60 * len(metrics_df)),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward validation chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_walk_forward(
    wf_df: pd.DataFrame,
    model_name: str,
    title: str = "Walk-Forward Validation",
    y_label: str = "Price",
) -> go.Figure:
    """
    Plot walk-forward predictions vs actuals, colour-coded by fold.

    wf_df: DataFrame from load_walk_forward_result() with columns
           [fold, pred_start, step, y_true, y_pred, model, track, horizon]
    """
    fig = go.Figure()

    # Plot actual (use first model's y_true as ground truth)
    actual = wf_df[["pred_start", "y_true"]].drop_duplicates(subset=["pred_start"])
    fig.add_trace(go.Scatter(
        x=actual["pred_start"], y=actual["y_true"],
        name="Actual",
        line=dict(color=_color("actual"), width=2),
        mode="lines",
    ))

    # Plot each fold's predictions
    n_folds = wf_df["fold"].nunique()
    for fold_id in sorted(wf_df["fold"].unique()):
        fold_df = wf_df[wf_df["fold"] == fold_id]
        opacity = 0.4 + 0.6 * (fold_id / max(n_folds - 1, 1))
        fig.add_trace(go.Scatter(
            x=fold_df["pred_start"],
            y=fold_df["y_pred"],
            name=f"Fold {fold_id}",
            line=dict(color=_color(model_name), width=1),
            opacity=opacity,
            showlegend=(fold_id == 0),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        hovermode="x unified",
        template="plotly_dark",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sudan-specific: FX premium and price timeline
# ─────────────────────────────────────────────────────────────────────────────

def plot_sudan_overview(
    df: pd.DataFrame,
    event_markers: Optional[list[dict]] = None,
) -> go.Figure:
    """
    Two-panel Sudan overview:
      Top: local price (SDG) and the international reference price in SDG
      Bottom: FX rate (SDG/USD) and FX premium

    event_markers: list of {"date": "YYYY-MM-DD", "label": "Event name"}
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=["Gold Price: Local and International (SDG)", "SDG/USD Exchange Rate"],
        row_heights=[0.6, 0.4],
        vertical_spacing=0.08,
    )

    if "price_sdg" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["price_sdg"],
            name="Local Price (SDG/gram)",
            line=dict(color="#e377c2", width=2),
        ), row=1, col=1)

    if "parity_price" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["parity_price"],
            name="International price in SDG",
            line=dict(color="#17becf", width=1.5, dash="dot"),
        ), row=1, col=1)

    if "fx_rate" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["fx_rate"],
            name="SDG/USD Rate",
            line=dict(color="#ff7f0e", width=2),
        ), row=2, col=1)

    # Political event markers
    default_events = [
        {"date": "2019-04-11", "label": "Bashir ousted"},
        {"date": "2021-10-25", "label": "Military coup"},
    ]
    events = event_markers or default_events

    for ev in events:
        for row in [1, 2]:
            fig.add_vline(
                x=pd.Timestamp(ev["date"]).timestamp() * 1000,
                line_dash="dash",
                line_color="rgba(255,255,0,0.6)",
                row=row, col=1,
                annotation_text=ev["label"] if row == 1 else "",
                annotation_position="top right",
            )

    fig.update_layout(
        title="Sudan Gold Market Overview",
        hovermode="x unified",
        template="plotly_dark",
        height=600,
    )
    return fig


def plot_regime_accuracy(
    regime_metrics: pd.DataFrame,
    title: str = "Model Accuracy by Political Regime (Sudan)",
) -> go.Figure:
    """
    Grouped bar chart showing MAE per model per political regime.

    regime_metrics columns: [model, regime, mae, directional_accuracy]
    """
    fig = px.bar(
        regime_metrics,
        x="regime",
        y="mae",
        color="model",
        barmode="group",
        title=title,
        color_discrete_map=MODEL_COLORS,
        template="plotly_dark",
    )
    fig.update_layout(xaxis_title="Regime", yaxis_title="MAE")
    return fig
