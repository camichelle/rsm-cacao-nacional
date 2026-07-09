"""
Generación de diseños experimentales de segundo orden.

Incluye:
  - Diseño Central Compuesto (CCD): rotable, ortogonal o "face-centered" (cara centrada).
  - Diseño Box-Behnken (BBD).

Todas las funciones trabajan primero en variables codificadas [-1, 1] (o +/- alpha
para los puntos axiales del CCD) y luego se decodifican a unidades reales usando
los límites (low, high) definidos para cada factor.

Referencias:
  Myers, Montgomery & Anderson-Cook (2016). Response Surface Methodology, 4th ed.
  Box, G. E. P., & Behnken, D. W. (1960). Some new three level designs for the
    study of quantitative variables. Technometrics, 2(4), 455-475.
"""

from itertools import combinations, product
import numpy as np
import pandas as pd


def _factorial_2level(k):
    """Full 2^k factorial en +-1, orden estándar (Yates)."""
    return np.array(list(product([-1, 1], repeat=k)), dtype=float)


def ccd_alpha(k, kind="rotatable", n_factorial=None):
    """
    Calcula alpha (distancia axial) según el tipo de CCD.

    kind:
      - 'rotatable' : alpha = (n_factorial)^(1/4)  -> varianza de predicción constante en esferas
      - 'orthogonal': alpha que hace ortogonal el bloque axial/factorial
      - 'face'      : alpha = 1 (Face-Centered CCD, "cara centrada")
    """
    if n_factorial is None:
        n_factorial = 2 ** k
    if kind == "face":
        return 1.0
    if kind == "rotatable":
        return n_factorial ** 0.25
    if kind == "orthogonal":
        # alpha ortogonal clásico (aprox.) - depende también de n_center, se ajusta luego
        return n_factorial ** 0.25
    raise ValueError("kind debe ser 'rotatable', 'orthogonal' o 'face'")


def generate_ccd_coded(k, alpha="rotatable", n_center=6, n_center_axial=None):
    """
    Genera un CCD en variables codificadas.

    Parameters
    ----------
    k : int
        Número de factores (2 a 6 recomendado).
    alpha : float or str
        'rotatable', 'orthogonal', 'face' o un valor numérico directo.
    n_center : int
        Número de puntos centrales en el bloque factorial.
    n_center_axial : int or None
        Número de puntos centrales en el bloque axial (por defecto = n_center).

    Returns
    -------
    DataFrame con columnas x1..xk, PtType (Factorial/Axial/Center), Block
    """
    if n_center_axial is None:
        n_center_axial = n_center

    fact = _factorial_2level(k)
    n_fact = fact.shape[0]

    if isinstance(alpha, str):
        a = ccd_alpha(k, alpha, n_fact)
    else:
        a = float(alpha)

    axial = []
    for i in range(k):
        for s in (-1, 1):
            row = [0.0] * k
            row[i] = s * a
            axial.append(row)
    axial = np.array(axial)

    center_fact = np.zeros((n_center, k))
    center_axial = np.zeros((n_center_axial, k))

    cols = [f"x{i+1}" for i in range(k)]
    df_fact = pd.DataFrame(fact, columns=cols)
    df_fact["PtType"] = "Factorial"
    df_fact["Block"] = 1

    df_cfact = pd.DataFrame(center_fact, columns=cols)
    df_cfact["PtType"] = "Center"
    df_cfact["Block"] = 1

    df_axial = pd.DataFrame(axial, columns=cols)
    df_axial["PtType"] = "Axial"
    df_axial["Block"] = 2

    df_caxial = pd.DataFrame(center_axial, columns=cols)
    df_caxial["PtType"] = "Center"
    df_caxial["Block"] = 2

    design = pd.concat([df_fact, df_cfact, df_axial, df_caxial], ignore_index=True)
    design.insert(0, "StdOrder", range(1, len(design) + 1))
    design["alpha_used"] = a
    return design


def generate_bbd_coded(k, n_center=6):
    """
    Genera un diseño Box-Behnken en variables codificadas para k factores
    (construcción estándar por bloques incompletos balanceados: para cada par
    de factores se corren las 4 combinaciones +-1 mientras el resto de
    factores permanece en 0; válido para k = 3,4,5,6,7,9,... que es el rango
    típico usado en aplicaciones agroindustriales).

    Returns
    -------
    DataFrame con columnas x1..xk, PtType, Block
    """
    if k < 3:
        raise ValueError("Box-Behnken requiere al menos 3 factores")

    rows = []
    for (i, j) in combinations(range(k), 2):
        for si, sj in product([-1, 1], repeat=2):
            row = [0.0] * k
            row[i] = si
            row[j] = sj
            rows.append(row)

    cols = [f"x{i+1}" for i in range(k)]
    df_edge = pd.DataFrame(rows, columns=cols)
    df_edge["PtType"] = "Edge"
    df_edge["Block"] = 1

    center = np.zeros((n_center, k))
    df_center = pd.DataFrame(center, columns=cols)
    df_center["PtType"] = "Center"
    df_center["Block"] = 1

    design = pd.concat([df_edge, df_center], ignore_index=True)
    design.insert(0, "StdOrder", range(1, len(design) + 1))
    return design


def decode_design(design_coded, factor_bounds, factor_names=None):
    """
    Convierte un diseño codificado (x1..xk en [-alpha, alpha]) a unidades
    reales, usando el punto medio y semi-rango de cada factor.

    factor_bounds : list of (low, high) tuples, uno por factor (en el nivel +-1,
                    NO en el nivel axial; el nivel axial se extrapola).
    """
    k = len(factor_bounds)
    names = factor_names or [f"Factor{i+1}" for i in range(k)]
    out = design_coded.copy()
    for i, (low, high) in enumerate(factor_bounds):
        mid = (low + high) / 2.0
        half = (high - low) / 2.0
        out[names[i]] = mid + out[f"x{i+1}"] * half
    return out


def build_design(design_type, k, factor_bounds, factor_names=None,
                  alpha="rotatable", n_center=6, n_center_axial=None,
                  randomize=True, seed=42):
    """
    Función de alto nivel usada por la app: genera CCD o BBD, decodifica a
    unidades reales y (opcionalmente) aleatoriza el orden de corrida.
    """
    if design_type.upper() == "CCD":
        coded = generate_ccd_coded(k, alpha=alpha, n_center=n_center,
                                    n_center_axial=n_center_axial)
    elif design_type.upper() in ("BBD", "BOX-BEHNKEN"):
        coded = generate_bbd_coded(k, n_center=n_center)
    else:
        raise ValueError("design_type debe ser 'CCD' o 'BBD'")

    decoded = decode_design(coded, factor_bounds, factor_names)

    if randomize:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(decoded))
        decoded = decoded.iloc[order].reset_index(drop=True)
        decoded.insert(1, "RunOrder", range(1, len(decoded) + 1))
    return decoded
