"""
Función de deseabilidad de Derringer & Suich (1980) para optimización
multirespuesta.

Cada respuesta Yi se transforma a una deseabilidad individual d_i en [0, 1]
según su meta (maximizar, minimizar o alcanzar un valor objetivo), y luego
se combinan mediante la media geométrica ponderada (deseabilidad global D).

Referencia:
  Derringer, G., & Suich, R. (1980). Simultaneous optimization of several
  response variables. Journal of Quality Technology, 12(4), 214-219.
"""

import numpy as np
import pandas as pd


def desirability_max(y, low, high, weight=1.0):
    """Deseabilidad para 'entre más alto, mejor' (larger-the-better)."""
    y = np.asarray(y, dtype=float)
    d = np.where(y <= low, 0.0,
        np.where(y >= high, 1.0, ((y - low) / (high - low)) ** weight))
    return np.clip(d, 0.0, 1.0)


def desirability_min(y, low, high, weight=1.0):
    """Deseabilidad para 'entre más bajo, mejor' (smaller-the-better)."""
    y = np.asarray(y, dtype=float)
    d = np.where(y <= low, 1.0,
        np.where(y >= high, 0.0, ((high - y) / (high - low)) ** weight))
    return np.clip(d, 0.0, 1.0)


def desirability_target(y, low, target, high, weight_low=1.0, weight_high=1.0):
    """Deseabilidad para 'valor objetivo' (nominal-the-best), con posible
    asimetría entre el tramo ascendente y descendente."""
    y = np.asarray(y, dtype=float)
    d = np.where(
        y < low, 0.0,
        np.where(
            y <= target, ((y - low) / (target - low)) ** weight_low,
            np.where(
                y <= high, ((high - y) / (high - target)) ** weight_high,
                0.0,
            ),
        ),
    )
    return np.clip(d, 0.0, 1.0)


def individual_desirability(y, goal, low, high, target=None, weight=1.0):
    """
    Despacha al tipo de deseabilidad correspondiente.
    goal: 'max', 'min' o 'target'
    """
    if goal == "max":
        return desirability_max(y, low, high, weight)
    if goal == "min":
        return desirability_min(y, low, high, weight)
    if goal == "target":
        if target is None:
            raise ValueError("Se requiere 'target' para goal='target'")
        return desirability_target(y, low, target, high, weight, weight)
    raise ValueError("goal debe ser 'max', 'min' o 'target'")


def overall_desirability(d_matrix, importances=None):
    """
    Media geométrica ponderada de las deseabilidades individuales:

        D = ( prod( d_i ^ w_i ) ) ^ (1 / sum(w_i))

    Si cualquier d_i = 0, D = 0 (una respuesta totalmente inaceptable anula
    la solución completa, tal como lo define Derringer-Suich).

    d_matrix: array (n_puntos, n_respuestas)
    importances: pesos relativos de cada respuesta (por defecto, iguales)
    """
    d_matrix = np.asarray(d_matrix, dtype=float)
    n_resp = d_matrix.shape[1]
    if importances is None:
        importances = np.ones(n_resp)
    importances = np.asarray(importances, dtype=float)

    with np.errstate(divide="ignore"):
        log_d = np.where(d_matrix > 0, np.log(np.clip(d_matrix, 1e-300, None)), -np.inf)
    weighted = (log_d * importances).sum(axis=1) / importances.sum()
    D = np.exp(weighted)
    D = np.where(np.any(d_matrix <= 0, axis=1), 0.0, D)
    return D


def build_desirability_table(responses_df, specs):
    """
    responses_df: DataFrame con una columna por respuesta (valores predichos
                  o experimentales para un conjunto de puntos candidatos).
    specs: dict { nombre_respuesta: {"goal":..., "low":..., "high":...,
                  "target":..., "weight":..., "importance":...} }

    Devuelve un DataFrame con las deseabilidades individuales, la
    deseabilidad global D, y ordenado de mayor a menor D.
    """
    d_cols = {}
    importances = []
    for name, spec in specs.items():
        d_cols[f"d_{name}"] = individual_desirability(
            responses_df[name],
            goal=spec.get("goal", "max"),
            low=spec["low"],
            high=spec["high"],
            target=spec.get("target"),
            weight=spec.get("weight", 1.0),
        )
        importances.append(spec.get("importance", 1.0))

    d_df = pd.DataFrame(d_cols)
    D = overall_desirability(d_df.values, importances=importances)

    out = pd.concat([responses_df.reset_index(drop=True), d_df], axis=1)
    out["D_global"] = D
    out = out.sort_values("D_global", ascending=False).reset_index(drop=True)
    return out
