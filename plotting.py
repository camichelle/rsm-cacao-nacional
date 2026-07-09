"""
Visualizaciones interactivas (Plotly) para la app RSM:
  - Gráfico de contorno 2D (para un par de factores, resto fijo en el centro)
  - Superficie de respuesta 3D
  - Diagrama de Pareto de efectos estandarizados
  - Gráfico de perturbación (perturbation plot)
  - Diagnóstico de residuos (QQ-plot, residuos vs ajustados, histograma)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from .modeling import predict


def _grid_prediction(beta, names, factor_names, fixed, fx, fy, n=60,
                      lo=-2.0, hi=2.0):
    g1 = np.linspace(lo, hi, n)
    g2 = np.linspace(lo, hi, n)
    Z = np.zeros((n, n))
    for i, v2 in enumerate(g2):
        for j, v1 in enumerate(g1):
            point = dict(fixed)
            point[fx] = v1
            point[fy] = v2
            Z[i, j] = predict(beta, names, point)
    return g1, g2, Z


def contour_plot(beta, names, factor_names, fx, fy, fixed=None,
                  response_name="Y", lo=-2.0, hi=2.0, n=60):
    fixed = fixed or {n_: 0.0 for n_ in factor_names}
    g1, g2, Z = _grid_prediction(beta, names, factor_names, fixed, fx, fy, n, lo, hi)
    fig = go.Figure(data=go.Contour(
        x=g1, y=g2, z=Z,
        colorscale="RdYlGn",
        contours=dict(showlabels=True, labelfont=dict(size=10, color="white")),
        colorbar=dict(title=response_name),
    ))
    fig.update_layout(
        title=f"Contorno de {response_name}: {fx} vs {fy}",
        xaxis_title=fx, yaxis_title=fy,
        template="plotly_white",
    )
    return fig


def surface_plot(beta, names, factor_names, fx, fy, fixed=None,
                  response_name="Y", lo=-2.0, hi=2.0, n=50):
    fixed = fixed or {n_: 0.0 for n_ in factor_names}
    g1, g2, Z = _grid_prediction(beta, names, factor_names, fixed, fx, fy, n, lo, hi)
    fig = go.Figure(data=[go.Surface(x=g1, y=g2, z=Z, colorscale="RdYlGn",
                                      colorbar=dict(title=response_name))])
    fig.update_layout(
        title=f"Superficie de respuesta 3D: {response_name}",
        scene=dict(xaxis_title=fx, yaxis_title=fy, zaxis_title=response_name),
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def pareto_chart(coef_table, exclude_intercept=True, alpha=0.05):
    """
    Diagrama de Pareto de efectos estandarizados (usando el estadístico t
    de cada coeficiente como medida del efecto estandarizado), con línea de
    referencia de significancia (t crítico).
    """
    df = coef_table.copy()
    if exclude_intercept:
        df = df[df["Término"] != "Intercepto"]
    df["abs_t"] = df["t"].abs()
    df = df.sort_values("abs_t", ascending=True)

    colors = ["#2E7D32" if p < alpha else "#B0BEC5" for p in df["p-valor"]]

    fig = go.Figure(go.Bar(
        x=df["abs_t"], y=df["Término"], orientation="h",
        marker_color=colors,
        text=[f"p={p:.3f}" for p in df["p-valor"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Diagrama de Pareto de efectos estandarizados (|t|)",
        xaxis_title="|t| (efecto estandarizado)",
        yaxis_title="Término",
        template="plotly_white",
    )
    return fig


def perturbation_plot(beta, names, factor_names, response_name="Y",
                       lo=-2.0, hi=2.0, n=50):
    """
    Gráfico de perturbación: muestra cómo cambia la respuesta cuando cada
    factor se perturba individualmente desde el punto central del diseño,
    manteniendo los demás fijos en 0 (nivel medio).
    """
    xs = np.linspace(lo, hi, n)
    fig = go.Figure()
    for f in factor_names:
        ys = []
        for v in xs:
            point = {ff: 0.0 for ff in factor_names}
            point[f] = v
            ys.append(predict(beta, names, point))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f))
    fig.update_layout(
        title=f"Gráfico de perturbación: {response_name}",
        xaxis_title="Desviación codificada respecto al centro",
        yaxis_title=response_name,
        template="plotly_white",
    )
    return fig


def residual_diagnostic_plots(diag_df):
    """Panel 2x2: residuos vs ajustados, QQ-plot, histograma, residuos vs orden."""
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Residuos vs Ajustados", "QQ-Plot (normalidad)",
        "Histograma de residuos", "Residuos vs Orden de corrida"))

    fig.add_trace(go.Scatter(x=diag_df["Ajustado"], y=diag_df["Residuo_studentizado"],
                              mode="markers", marker=dict(color="#1565C0")), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", row=1, col=1)

    osm, osr = stats.probplot(diag_df["Residuo_studentizado"].dropna(), dist="norm", fit=False)
    fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", marker=dict(color="#EF6C00")), row=1, col=2)
    lims = [min(osm.min(), osr.min()), max(osm.max(), osr.max())]
    fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", line=dict(dash="dash", color="gray")), row=1, col=2)

    fig.add_trace(go.Histogram(x=diag_df["Residuo"], marker=dict(color="#6A1B9A")), row=2, col=1)

    fig.add_trace(go.Scatter(x=diag_df["Orden"], y=diag_df["Residuo_studentizado"],
                              mode="lines+markers", marker=dict(color="#2E7D32")), row=2, col=2)
    fig.add_hline(y=0, line_dash="dash", row=2, col=2)

    fig.update_layout(showlegend=False, template="plotly_white", height=650,
                       title="Diagnóstico de residuos del modelo")
    return fig
