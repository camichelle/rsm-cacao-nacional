"""
Herramientas de optimización de superficies de respuesta de segundo orden:

  - Ascenso/descenso más pronunciado (steepest ascent/descent)
  - Análisis canónico: punto estacionario, autovalores/autovectores, naturaleza
    (máximo, mínimo, silla)
  - Análisis de cresta (ridge analysis, método de Draper/Hoerl vía multiplicador
    de Lagrange)
  - Optimización numérica restringida a la región experimental (scipy.optimize)

Convención: el modelo de segundo orden se escribe como

    y_hat(x) = b0 + b' x + x' B x

donde b es el vector de coeficientes lineales y B es la matriz simétrica de
coeficientes cuadráticos (con los términos de interacción divididos entre 2
fuera de la diagonal).
"""

import numpy as np
from scipy.optimize import minimize


def coefficients_to_vector_matrix(coef_table, factor_names):
    """
    Extrae b0 (escalar), b (vector k), B (matriz kxk simétrica) a partir de la
    tabla de coeficientes producida por modeling.anova_with_lof (columna
    'Término' con nombres tipo 'x1', 'x1*x2', 'x1^2').
    """
    k = len(factor_names)
    b0 = 0.0
    b = np.zeros(k)
    B = np.zeros((k, k))
    idx = {name: i for i, name in enumerate(factor_names)}

    for _, row in coef_table.iterrows():
        term = row["Término"]
        coef = row["Coeficiente"]
        if term == "Intercepto":
            b0 = coef
        elif term.endswith("^2"):
            base = term[:-2]
            B[idx[base], idx[base]] = coef
        elif "*" in term:
            a, c = term.split("*")
            B[idx[a], idx[c]] = coef / 2.0
            B[idx[c], idx[a]] = coef / 2.0
        else:
            b[idx[term]] = coef
    return b0, b, B


def steepest_path(b0, b, step=0.5, n_steps=8, maximize=True):
    """
    Genera el camino de ascenso (o descenso) más pronunciado desde el centro
    del diseño (origen en coordenadas codificadas), moviéndose en la
    dirección del gradiente del modelo de PRIMER orden.

    Devuelve un DataFrame con la distancia radial recorrida, las coordenadas
    codificadas x1..xk y la predicción del modelo de primer orden en ese
    punto (nota: para la predicción real del proceso se recomienda correr
    experimentos confirmatorios a lo largo del camino).
    """
    import pandas as pd

    direction = b / np.linalg.norm(b)
    if not maximize:
        direction = -direction

    rows = []
    for i in range(n_steps + 1):
        r = i * step
        x = direction * r
        y_lin = b0 + b @ x
        row = {"paso": i, "distancia_radial": r}
        for j, val in enumerate(x):
            row[f"x{j+1}"] = val
        row["y_pred_1er_orden"] = y_lin
        rows.append(row)
    return pd.DataFrame(rows)


def canonical_analysis(b0, b, B, factor_names=None):
    """
    Análisis canónico del modelo de segundo orden.

    Punto estacionario:  x_s = -0.5 * B^-1 b
    Valor ajustado en el punto estacionario: y_s = b0 + 0.5 * b' x_s
    Forma canónica: y = y_s + sum(lambda_i * w_i^2)  (autovalores de B)

    Naturaleza:
      - Todos los autovalores < 0  -> Máximo
      - Todos los autovalores > 0  -> Mínimo
      - Signos mixtos              -> Punto de silla (saddle point)
    """
    k = len(b)
    names = factor_names or [f"x{i+1}" for i in range(k)]

    B_inv = np.linalg.pinv(B)
    x_s = -0.5 * B_inv @ b
    y_s = b0 + 0.5 * (b @ x_s)

    eigvals, eigvecs = np.linalg.eigh(B)

    if np.all(eigvals < -1e-8):
        nature = "Máximo"
    elif np.all(eigvals > 1e-8):
        nature = "Mínimo"
    else:
        nature = "Punto de silla (saddle point)"

    return {
        "x_stationary": dict(zip(names, x_s)),
        "y_stationary": y_s,
        "eigenvalues": eigvals,
        "eigenvectors": eigvecs,
        "nature": nature,
        "B_inv": B_inv,
    }


def ridge_analysis(b0, b, B, radii, maximize=True):
    """
    Análisis de cresta (ridge analysis) de Hoerl/Draper: para cada radio r,
    encuentra el punto sobre la esfera ||x|| = r que optimiza y_hat,
    resolviendo (B - mu*I) x = -0.5 b  para el multiplicador de Lagrange mu
    apropiado (búsqueda numérica de mu que satisface ||x(mu)|| = r).

    Devuelve un DataFrame con radio, coordenadas óptimas y predicción.
    """
    import pandas as pd
    k = len(b)
    eigvals = np.linalg.eigvalsh(B)

    def x_of_mu(mu):
        # (2B - 2*mu*I) x = -b   <=>  x = -(2B - 2 mu I)^-1 b
        M = 2 * B - 2 * mu * np.eye(k)
        return np.linalg.solve(M, -b)

    rows = []
    for r in radii:
        # mu debe estar fuera del rango de autovalores de B para que la
        # solución sea real; para maximizar nos movemos por encima del mayor
        # autovalor, para minimizar por debajo del menor.
        if maximize:
            lo, hi = eigvals.max() + 1e-6, eigvals.max() + 50
        else:
            lo, hi = eigvals.min() - 50, eigvals.min() - 1e-6

        # búsqueda binaria de mu tal que ||x(mu)|| = r
        for _ in range(100):
            mid = (lo + hi) / 2
            x_mid = x_of_mu(mid)
            norm_mid = np.linalg.norm(x_mid)
            if maximize:
                if norm_mid > r:
                    lo = mid
                else:
                    hi = mid
            else:
                if norm_mid > r:
                    hi = mid
                else:
                    lo = mid
        mu_star = (lo + hi) / 2
        x_star = x_of_mu(mu_star)
        x_star = x_star * (r / (np.linalg.norm(x_star) + 1e-12))  # normaliza exacto al radio
        y_star = b0 + b @ x_star + x_star @ B @ x_star

        row = {"radio": r, "mu": mu_star, "y_pred": y_star}
        for j, val in enumerate(x_star):
            row[f"x{j+1}"] = val
        rows.append(row)
    return pd.DataFrame(rows)


def numerical_optimize(predict_fn, k, bounds=None, maximize=True, x0=None,
                        n_restarts=20, seed=42):
    """
    Optimización numérica restringida (por defecto a la región experimental
    codificada [-1, 1]^k, o a los límites que se especifiquen) usando
    múltiples reinicios aleatorios + SLSQP para evitar óptimos locales.

    predict_fn: función que recibe un array x (longitud k) y devuelve un
                escalar (la predicción del modelo).
    """
    if bounds is None:
        bounds = [(-1.0, 1.0)] * k

    sign = -1.0 if maximize else 1.0

    def obj(x):
        return sign * predict_fn(x)

    rng = np.random.default_rng(seed)
    best = None
    starts = [x0] if x0 is not None else []
    starts += [rng.uniform([b[0] for b in bounds], [b[1] for b in bounds]) for _ in range(n_restarts)]

    for s in starts:
        res = minimize(obj, s, method="SLSQP", bounds=bounds)
        if res.success:
            val = sign * res.fun
            if best is None or (maximize and val > best["y_opt"]) or (not maximize and val < best["y_opt"]):
                best = {"x_opt": res.x, "y_opt": val}
    return best
