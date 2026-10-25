"""
OptiDrug - Plotly Interactive Visualizations
Interactive charts for bioinformatics dashboard:
  - Actual vs Predicted regression plot with R² and MAE annotations
  - Residuals distribution plot
  - Random Forest feature importance bar chart
  - Multi-objective candidate radar chart (5 dimensions)
  - 2D/3D Pareto Frontier scatter plot
  - Molecular descriptor distributions
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Optional

def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, r2: float, mae: float) -> go.Figure:
    """Actual vs Predicted activity plot with 1:1 identity line."""
    min_val = min(y_true.min(), y_pred.min()) - 0.05
    max_val = max(y_true.max(), y_pred.max()) + 0.05
    
    fig = go.Figure()
    
    # Scatter points
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode="markers",
        marker=dict(
            color="#2563eb",
            size=9,
            opacity=0.75,
            line=dict(width=1, color="#1e3a8a")
        ),
        name="Test Candidates",
        hovertemplate="Actual: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>"
    ))
    
    # 1:1 Reference Line
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color="#ef4444", dash="dash", width=2),
        name="Perfect Identity (y = x)"
    ))
    
    # Annotation box for metrics
    fig.add_annotation(
        x=0.05,
        y=0.95,
        xref="paper",
        yref="paper",
        text=f"<b>Held-out Test Performance</b><br>R² = {r2:.4f}<br>MAE = {mae:.4f}",
        showarrow=False,
        align="left",
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="#cbd5e1",
        borderwidth=1,
        borderpad=8,
        font=dict(size=12, color="#0f172a")
    )
    
    fig.update_layout(
        title="<b>Random Forest: Actual vs. Predicted Activity Score</b>",
        xaxis_title="Actual Activity Score (Held-out Test Set)",
        yaxis_title="Predicted Computational Activity Score",
        template="plotly_white",
        height=450,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )
    return fig

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Distribution of residuals (prediction errors)."""
    residuals = y_true - y_pred
    fig = px.histogram(
        x=residuals,
        nbins=20,
        marginal="box",
        labels={"x": "Residual Error (Actual - Predicted)"},
        color_discrete_sequence=["#0284c7"]
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#dc2626", annotation_text="Zero Error")
    fig.update_layout(
        title="<b>Error Residual Distribution (Held-Out Test Set)</b>",
        xaxis_title="Residual Error",
        yaxis_title="Count",
        template="plotly_white",
        height=450,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_feature_importances(fi_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of feature importances explaining model decisions."""
    fi_sorted = fi_df.sort_values(by="importance", ascending=True)
    
    # Format labels cleanly
    labels = [
        f.replace("_", " ").title()
        for f in fi_sorted["feature"]
    ]
    
    fig = go.Figure(go.Bar(
        x=fi_sorted["importance_pct"],
        y=labels,
        orientation="h",
        marker=dict(
            color=fi_sorted["importance_pct"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Importance %")
        ),
        text=[f"{v:.1f}%" for v in fi_sorted["importance_pct"]],
        textposition="outside"
    ))
    
    fig.update_layout(
        title="<b>Random Forest Feature Importance (% Variance Explained)</b>",
        xaxis_title="Relative Importance (%)",
        yaxis_title="Molecular Descriptor / Property",
        template="plotly_white",
        height=450,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_pareto_frontier(df: pd.DataFrame) -> go.Figure:
    """
    2D Scatter plot visualizing the Pareto Trade-off:
    Predicted Bioactivity vs. Safety Profile (1 - Safety Risk), colored by Composite Score.
    """
    df_plot = df.copy()
    df_plot["safety_profile"] = 1.0 - df_plot["safety_risk"]
    
    fig = go.Figure()
    
    # Non-Pareto candidates
    non_pareto = df_plot[~df_plot["is_pareto_optimal"]]
    fig.add_trace(go.Scatter(
        x=non_pareto["safety_profile"],
        y=non_pareto["predicted_pic50"],
        mode="markers",
        marker=dict(
            size=8,
            color=non_pareto["composite_score"],
            colorscale="Blues",
            opacity=0.65,
            showscale=False
        ),
        name="Candidate Compounds",
        customdata=non_pareto[["compound_id", "compound_name", "composite_score"]],
        hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>pIC50: %{y:.2f}<br>Safety Profile: %{x:.2f}<br>Composite Score: %{customdata[2]:.1f}<extra></extra>"
    ))
    
    # Pareto-Optimal frontier candidates
    pareto = df_plot[df_plot["is_pareto_optimal"]].sort_values(by="safety_profile")
    fig.add_trace(go.Scatter(
        x=pareto["safety_profile"],
        y=pareto["predicted_pic50"],
        mode="markers+lines",
        line=dict(color="#f59e0b", width=2, dash="dot"),
        marker=dict(
            size=14,
            symbol="star",
            color="#d97706",
            line=dict(color="#78350f", width=1.5)
        ),
        name="Pareto Optimal Frontier ⭐",
        customdata=pareto[["compound_id", "compound_name", "composite_score", "optimized_rank"]],
        hovertemplate="⭐ <b>Pareto Frontier: %{customdata[0]}</b><br>Rank: #%{customdata[3]}<br>pIC50: %{y:.2f}<br>Safety Profile: %{x:.2f}<br>Composite Score: %{customdata[2]:.1f}<extra></extra>"
    ))
    
    fig.update_layout(
        title="<b>Multi-Objective Optimization: Pareto Efficiency Trade-off</b>",
        xaxis_title="Safety Profile (1 - Toxicological Risk, Higher is Safer)",
        yaxis_title="Predicted Bioactivity (pIC50, Higher is Potent)",
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_radar_comparison(top_candidates: List[pd.Series], target_name: str) -> go.Figure:
    """
    Radar chart comparing top candidates across the 5 optimization dimensions:
    Effectiveness, Safety, Low Interaction, Low Cost, Drug-likeness.
    """
    categories = [
        "Effectiveness",
        "Safety Profile",
        "Low Interaction",
        "Synthetic Economy",
        "Drug-likeness"
    ]
    
    fig = go.Figure()
    colors = ["#2563eb", "#10b981", "#f59e0b", "#8b5cf6"]
    
    for idx, c in enumerate(top_candidates):
        values = [
            float(c["norm_effectiveness"]) * 100.0,
            float(c["norm_safety"]) * 100.0,
            float(c["norm_interaction"]) * 100.0,
            float(c["norm_cost"]) * 100.0,
            float(c["norm_druglikeness"]) * 100.0
        ]
        # Close polygon
        values.append(values[0])
        cat_closed = categories + [categories[0]]
        
        color = colors[idx % len(colors)]
        label = f"#{c.get('optimized_rank', idx+1)} {c['compound_id']} ({c['compound_name']})"
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=cat_closed,
            fill="toself",
            name=label,
            line=dict(color=color, width=2.5),
            fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.15)"
        ))
        
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=9)
            )
        ),
        title=f"<b>Multi-Attribute Profile: Top Prioritized Candidates ({target_name})</b>",
        showlegend=True,
        template="plotly_white",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_property_distributions(df: pd.DataFrame) -> go.Figure:
    """Distribution of key molecular descriptors."""
    fig = go.Figure()
    
    fig.add_trace(go.Box(y=df["molecular_weight"], name="MW (Da)", marker_color="#3b82f6"))
    fig.add_trace(go.Box(y=df["tpsa"], name="TPSA (Å²)", marker_color="#10b981"))
    fig.add_trace(go.Box(y=df["logp"] * 50.0, name="LogP (x50)", marker_color="#f59e0b"))
    fig.add_trace(go.Box(y=df["rotatable_bonds"] * 20.0, name="Rotatable Bonds (x20)", marker_color="#8b5cf6"))
    
    fig.update_layout(
        title="<b>Molecular Descriptor Boxplots (Demo Compound Library)</b>",
        yaxis_title="Property Value (Scaled for Comparison)",
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig
