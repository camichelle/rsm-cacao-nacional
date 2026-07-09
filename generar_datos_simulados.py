"""
Genera el dataset SIMULADO del caso de aplicación:
"Optimización del tostado de cacao Nacional (Ecuador)".

Factores (diseño CCD rotable, k=3):
  X1: Temperatura de tostado      [120 - 160] °C
  X2: Tiempo de tostado           [10  - 30]  min
  X3: Humedad inicial del grano   [5   - 8]   %

Respuestas (modelo cuadrático verdadero + ruido aleatorio, simulando un
proceso agroindustrial real de tostado de cacao):
  Y1: Puntaje sensorial de sabor (panel de catadores, escala 0-10)  -> maximizar
  Y2: Polifenoles totales (mg equivalentes de ácido gálico / g)     -> maximizar
  Y3: Acidez titulable (% ácido acético)                            -> minimizar

Los coeficientes del modelo verdadero son plausibles según la literatura de
tostado de cacao (el sabor mejora con temperatura/tiempo hasta un óptimo y
luego decae por sobretostado; los polifenoles se degradan monótonamente con
mayor severidad térmica; la acidez del grano fresco disminuye con el
tostado). Esto es un DATASET SIMULADO con fines académicos, no mediciones
de laboratorio reales.
"""

import numpy as np
import pandas as pd
from src.design import build_design

SEED = 2026

def main():
    design = build_design(
        "CCD", k=3,
        factor_bounds=[(120, 160), (10, 30), (5, 8)],
        factor_names=["Temperatura_C", "Tiempo_min", "Humedad_pct"],
        alpha="rotatable", n_center=6, randomize=True, seed=SEED,
    )

    rng = np.random.default_rng(SEED)
    x1, x2, x3 = design["x1"], design["x2"], design["x3"]

    # --- Y1: Sabor (0-10), modelo verdadero ---
    y1 = (7.4 + 0.85 * x1 + 0.55 * x2 - 0.30 * x3
          - 0.95 * x1**2 - 0.75 * x2**2 - 0.20 * x3**2
          + 0.30 * x1 * x2 - 0.12 * x1 * x3 + 0.05 * x2 * x3
          + rng.normal(0, 0.18, len(design)))
    y1 = np.clip(y1, 0, 10)

    # --- Y2: Polifenoles (mg GAE/g), decrecen con severidad de tostado ---
    y2 = (17.8 - 2.6 * x1 - 1.7 * x2 + 0.35 * x3
          - 1.0 * x1**2 - 0.55 * x2**2 - 0.15 * x3**2
          - 0.40 * x1 * x2
          + rng.normal(0, 0.45, len(design)))
    y2 = np.clip(y2, 0, None)

    # --- Y3: Acidez titulable (%), disminuye con tostado, sube con humedad ---
    y3 = (1.15 - 0.14 * x1 - 0.09 * x2 + 0.10 * x3
          + 0.05 * x1**2 + 0.03 * x2**2 + 0.015 * x1 * x2
          + rng.normal(0, 0.025, len(design)))
    y3 = np.clip(y3, 0.1, None)

    design["Sabor_puntaje"] = np.round(y1, 2)
    design["Polifenoles_mgGAE_g"] = np.round(y2, 2)
    design["Acidez_titulable_pct"] = np.round(y3, 3)

    out = design[[
        "StdOrder", "RunOrder", "PtType", "Block",
        "Temperatura_C", "Tiempo_min", "Humedad_pct",
        "Sabor_puntaje", "Polifenoles_mgGAE_g", "Acidez_titulable_pct",
    ]].sort_values("RunOrder").reset_index(drop=True)

    out.to_csv("data/cacao_tostado_simulado.csv", index=False, encoding="utf-8")
    print(out)
    print("\nGuardado en data/cacao_tostado_simulado.csv  (", len(out), "corridas )")


if __name__ == "__main__":
    main()
