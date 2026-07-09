"""
Ajuste de modelos de primer y segundo orden por mínimos cuadrados (OLS),
tabla ANOVA con descomposición de falta de ajuste / error puro, R2, R2
ajustado y R2 predicho, y diagnóstico de residuos.

Implementado con numpy/scipy puro (sin dependencia obligatoria de
statsmodels) para que la lógica estadística sea 100% transparente y
verificable línea por línea (requisito de la defensa oral).
"""

from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats


def build_design_matrix(X, order="second", interactions=True):
    """
    Construye la matriz de diseño (con intercepto) a partir de un DataFrame
    de factores en unidades codificadas o reales.

    order: 'first' (solo lineal), 'first_int' (lineal + interacciones 2FI),
           'second' (lineal + interacciones + cuadráticos puros)
    """
    X = X.copy()
    names = list(X.columns)
    cols = {"Intercepto": np.ones(len(X))}

    for n in names:
        cols[n] = X[n].values.astype(float)

    if order in ("first_int", "second") and interactions:
        for a, b in combinations(names, 2):
            cols[f"{a}*{b}"] = X[a].values * X[b].values

    if order == "second":
        for n in names:
            cols[f"{n}^2"] = X[n].values ** 2

    design = pd.DataFrame(cols)
    return design


def fit_ols(y, design_matrix):
    """
    Ajuste OLS clásico vía ecuaciones normales con pseudo-inversa (estable
    numéricamente). Devuelve coeficientes, matriz (X'X)^-1, residuos, etc.
    """
    Xmat = design_matrix.values
    yv = np.asarray(y, dtype=float)
    n, p = Xmat.shape

    XtX_inv = np.linalg.pinv(Xmat.T @ Xmat)
    beta = XtX_inv @ Xmat.T @ yv
    y_hat = Xmat @ beta
    resid = yv - y_hat

    return {
        "beta": beta,
        "XtX_inv": XtX_inv,
        "y_hat": y_hat,
        "resid": resid,
        "n": n,
        "p": p,
        "names": list(design_matrix.columns),
    }


def anova_with_lof(y, design_matrix, replicate_groups):
    """
    Tabla ANOVA completa con descomposición de Falta de Ajuste (Lack of Fit)
    y Error Puro (Pure Error), usando las réplicas exactas del diseño
    (típicamente los puntos centrales).

    replicate_groups: array/Series con un identificador de "punto único" del
    diseño (p. ej. una tupla de niveles redondeados) para poder agrupar las
    réplicas y calcular el error puro (SSPE).
    """
    fit = fit_ols(y, design_matrix)
    yv = np.asarray(y, dtype=float)
    n, p = fit["n"], fit["p"]

    y_bar = yv.mean()
    ss_total = np.sum((yv - y_bar) ** 2)
    ss_resid = np.sum(fit["resid"] ** 2)
    ss_reg = ss_total - ss_resid

    df_total = n - 1
    df_reg = p - 1
    df_resid = n - p

    ms_reg = ss_reg / df_reg if df_reg > 0 else np.nan
    ms_resid = ss_resid / df_resid if df_resid > 0 else np.nan
    f_reg = ms_reg / ms_resid if ms_resid > 0 else np.nan
    p_reg = 1 - stats.f.cdf(f_reg, df_reg, df_resid) if not np.isnan(f_reg) else np.nan

    # --- Error puro (Pure Error) y Falta de ajuste ---
    groups = pd.Series(replicate_groups).reset_index(drop=True)
    yv_s = pd.Series(yv)
    ss_pe = 0.0
    df_pe = 0
    for _, idx in groups.groupby(groups).groups.items():
        if len(idx) > 1:
            sub = yv_s.iloc[list(idx)]
            ss_pe += np.sum((sub - sub.mean()) ** 2)
            df_pe += len(idx) - 1

    ss_lof = ss_resid - ss_pe
    df_lof = df_resid - df_pe

    ms_lof = ss_lof / df_lof if df_lof > 0 else np.nan
    ms_pe = ss_pe / df_pe if df_pe > 0 else np.nan
    f_lof = ms_lof / ms_pe if (ms_pe and ms_pe > 0 and not np.isnan(ms_lof)) else np.nan
    p_lof = 1 - stats.f.cdf(f_lof, df_lof, df_pe) if not np.isnan(f_lof) and df_lof > 0 and df_pe > 0 else np.nan

    r2 = ss_reg / ss_total if ss_total > 0 else np.nan
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p) if (n - p) > 0 else np.nan

    # R2 predicho (PRESS) usando "leave-one-out" hat matrix
    Xmat = design_matrix.values
    H = Xmat @ fit["XtX_inv"] @ Xmat.T
    h = np.diag(H)
    with np.errstate(divide="ignore", invalid="ignore"):
        press_resid = fit["resid"] / (1 - h)
    press = np.nansum(press_resid ** 2)
    r2_pred = 1 - press / ss_total if ss_total > 0 else np.nan

    anova_table = pd.DataFrame([
        {"Fuente": "Regresión (Modelo)", "SS": ss_reg, "df": df_reg, "MS": ms_reg, "F": f_reg, "p-valor": p_reg},
        {"Fuente": "Residual", "SS": ss_resid, "df": df_resid, "MS": ms_resid, "F": np.nan, "p-valor": np.nan},
        {"Fuente": "  Falta de ajuste", "SS": ss_lof, "df": df_lof, "MS": ms_lof, "F": f_lof, "p-valor": p_lof},
        {"Fuente": "  Error puro", "SS": ss_pe, "df": df_pe, "MS": ms_pe, "F": np.nan, "p-valor": np.nan},
        {"Fuente": "Total", "SS": ss_total, "df": df_total, "MS": np.nan, "F": np.nan, "p-valor": np.nan},
    ])

    # Errores estándar y p-valores individuales de los coeficientes
    se_beta = np.sqrt(np.diag(fit["XtX_inv"]) * ms_resid) if ms_resid and ms_resid > 0 else np.full(p, np.nan)
    t_vals = fit["beta"] / se_beta
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_vals), df_resid)) if df_resid > 0 else np.full(p, np.nan)

    coef_table = pd.DataFrame({
        "Término": fit["names"],
        "Coeficiente": fit["beta"],
        "Error Std": se_beta,
        "t": t_vals,
        "p-valor": p_vals,
    })

    result = {
        "fit": fit,
        "anova": anova_table,
        "coefficients": coef_table,
        "r2": r2,
        "r2_adj": r2_adj,
        "r2_pred": r2_pred,
        "ss_total": ss_total,
        "leverage": h,
        "press_resid": press_resid,
    }
    return result


def residual_diagnostics(result):
    """
    Devuelve un DataFrame con: valores ajustados, residuos, residuos
    estandarizados y residuos studentizados (para gráficos de diagnóstico).
    """
    fit = result["fit"]
    resid = fit["resid"]
    n, p = fit["n"], fit["p"]
    ms_resid = result["anova"].loc[result["anova"]["Fuente"] == "Residual", "MS"].values[0]
    h = result["leverage"]

    std_resid = resid / np.sqrt(ms_resid) if ms_resid > 0 else resid * np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        stud_resid = resid / np.sqrt(ms_resid * (1 - h))

    df = pd.DataFrame({
        "Ajustado": fit["y_hat"],
        "Residuo": resid,
        "Residuo_estandarizado": std_resid,
        "Residuo_studentizado": stud_resid,
        "Leverage": h,
        "Orden": np.arange(1, n + 1),
    })
    return df


def predict(beta, names, X_new):
    """Predice la respuesta para nuevos puntos usando el vector de coeficientes."""
    row = {"Intercepto": 1.0}
    for n in names:
        if n == "Intercepto":
            continue
        if "*" in n:
            a, b = n.split("*")
            row[n] = X_new[a] * X_new[b]
        elif n.endswith("^2"):
            base = n[:-2]
            row[n] = X_new[base] ** 2
        else:
            row[n] = X_new[n]
    xv = np.array([row[n] for n in names])
    return float(xv @ beta)
