# -*- coding: utf-8 -*-
"""
Aplicativo RSM - Optimización del tostado de cacao Nacional (Ecuador)
Curso de Optimización Estadística - Universidad Central del Ecuador

Implementa de forma integrada:
  - Generación de diseños experimentales de segundo orden (CCD, Box-Behnken)
  - Ajuste de modelos de primer/segundo orden, ANOVA, falta de ajuste,
    R2/R2 ajustado/R2 predicho, diagnóstico de residuos
  - Optimización: ascenso más pronunciado, análisis canónico, análisis de
    cresta, optimización numérica
  - Optimización multirespuesta (deseabilidad de Derringer-Suich)
  - Visualización: contornos, superficies 3D, Pareto, perturbación
  - Exportación de reporte y memorándum gerencial en lenguaje natural
"""

import io
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

from src.design import build_design
from src.modeling import (build_design_matrix, anova_with_lof,
                           residual_diagnostics, predict)
from src.optimization import (coefficients_to_vector_matrix, steepest_path,
                               canonical_analysis, ridge_analysis,
                               numerical_optimize)
from src.desirability import build_desirability_table
from src.plotting import (contour_plot, surface_plot, pareto_chart,
                           perturbation_plot, residual_diagnostic_plots)

st.set_page_config(page_title="RSM Cacao Nacional", page_icon="🍫", layout="wide")

# ------------------------------------------------------------------ #
# Estado de sesión
# ------------------------------------------------------------------ #
if "df" not in st.session_state:
    st.session_state.df = None
if "coding" not in st.session_state:
    st.session_state.coding = {}
if "models" not in st.session_state:
    st.session_state.models = {}
if "factor_cols" not in st.session_state:
    st.session_state.factor_cols = []
if "response_cols" not in st.session_state:
    st.session_state.response_cols = []


def code_column(series, low, high):
    mid = (low + high) / 2.0
    half = (high - low) / 2.0
    return (series - mid) / half


def get_coded_df(df, factor_cols, coding):
    coded = pd.DataFrame(index=df.index)
    for f in factor_cols:
        low, high = coding[f]
        coded[f] = code_column(df[f], low, high)
    return coded


# ==================================================================== #
st.title("🍫 Aplicativo RSM — Optimización del tostado de cacao Nacional")
st.caption("Metodología de Superficie de Respuesta (RSM) aplicada a un caso agroindustrial "
           "ecuatoriano simulado · Curso de Optimización Estadística UCE")

tabs = st.tabs([
    "1. Datos", "2. Diseño experimental", "3. Ajuste del modelo",
    "4. Optimización", "5. Multirespuesta", "6. Visualización", "7. Reporte",
])

# ==================================================================== #
# TAB 1: DATOS
# ==================================================================== #
with tabs[0]:
    st.header("1. Carga de datos")
    st.markdown(
        "Cargue su propio archivo CSV o utilice el **dataset simulado** de "
        "tostado de cacao Nacional (diseño CCD, 3 factores, 3 respuestas) "
        "incluido en `data/cacao_tostado_simulado.csv`."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded = st.file_uploader("Sube tu archivo CSV", type=["csv"])
    with col2:
        use_sample = st.button("📂 Usar dataset simulado de cacao", use_container_width=True)

    if uploaded is not None:
        st.session_state.df = pd.read_csv(uploaded)
        st.success(f"Archivo cargado: {uploaded.name} ({len(st.session_state.df)} filas)")
    elif use_sample:
        st.session_state.df = pd.read_csv("data/cacao_tostado_simulado.csv")
        st.success("Dataset simulado de cacao cargado correctamente.")

    if st.session_state.df is not None:
        df = st.session_state.df
        st.dataframe(df, use_container_width=True, height=300)

        st.subheader("Selección de variables")
        all_cols = list(df.columns)
        default_factors = [c for c in ["Temperatura_C", "Tiempo_min", "Humedad_pct"] if c in all_cols]
        default_resp = [c for c in ["Sabor_puntaje", "Polifenoles_mgGAE_g", "Acidez_titulable_pct"] if c in all_cols]

        factor_cols = st.multiselect("Factores (variables independientes, X)", all_cols,
                                      default=default_factors or all_cols[:3])
        response_cols = st.multiselect("Respuestas (variables dependientes, Y)", all_cols,
                                        default=default_resp or [])
        st.session_state.factor_cols = factor_cols
        st.session_state.response_cols = response_cols

        st.subheader("Niveles de codificación (-1 / +1) por factor")
        st.caption(
            "Definen el nivel factorial bajo/alto de su diseño (NO el axial). "
            "Por defecto se usan los valores presentes en el propio dataset simulado."
        )
        default_bounds = {
            "Temperatura_C": (120, 160),
            "Tiempo_min": (10, 30),
            "Humedad_pct": (5, 8),
        }
        coding = {}
        cols = st.columns(min(3, max(1, len(factor_cols))) or 1)
        for i, f in enumerate(factor_cols):
            with cols[i % len(cols)]:
                lo_def, hi_def = default_bounds.get(f, (float(df[f].min()), float(df[f].max())))
                lo = st.number_input(f"{f} — bajo (-1)", value=float(lo_def), key=f"lo_{f}")
                hi = st.number_input(f"{f} — alto (+1)", value=float(hi_def), key=f"hi_{f}")
                coding[f] = (lo, hi)
        st.session_state.coding = coding
    else:
        st.info("Cargue un archivo o use el dataset simulado para continuar.")

# ==================================================================== #
# TAB 2: DISEÑO EXPERIMENTAL
# ==================================================================== #
with tabs[1]:
    st.header("2. Generación de diseños de segundo orden")
    st.markdown("Genere un nuevo diseño **Central Compuesto (CCD)** o **Box-Behnken (BBD)** "
                "para planificar experimentos futuros (por ejemplo, una ronda confirmatoria "
                "tras el ascenso más pronunciado).")

    c1, c2, c3 = st.columns(3)
    with c1:
        design_type = st.selectbox("Tipo de diseño", ["CCD", "BBD"])
        k = st.number_input("Número de factores (k)", min_value=2, max_value=6, value=3)
    with c2:
        if design_type == "CCD":
            alpha_kind = st.selectbox("Tipo de alpha", ["rotatable", "face", "orthogonal"])
        n_center = st.number_input("Puntos centrales", min_value=2, max_value=12, value=6)
    with c3:
        randomize = st.checkbox("Aleatorizar orden de corrida", value=True)
        seed = st.number_input("Semilla aleatoria", value=42)

    st.subheader("Límites reales de cada factor")
    bounds = []
    names = []
    bcols = st.columns(int(k))
    for i in range(int(k)):
        with bcols[i]:
            nm = st.text_input(f"Nombre factor {i+1}", value=f"X{i+1}", key=f"dname_{i}")
            lo = st.number_input(f"Bajo (-1) {i+1}", value=0.0, key=f"dlo_{i}")
            hi = st.number_input(f"Alto (+1) {i+1}", value=1.0, key=f"dhi_{i}")
            names.append(nm)
            bounds.append((lo, hi))

    if st.button("🧪 Generar diseño"):
        kwargs = dict(design_type=design_type, k=int(k), factor_bounds=bounds,
                      factor_names=names, n_center=int(n_center),
                      randomize=randomize, seed=int(seed))
        if design_type == "CCD":
            kwargs["alpha"] = alpha_kind
        design = build_design(**kwargs)
        st.dataframe(design, use_container_width=True)
        st.download_button("⬇️ Descargar diseño (CSV)", design.to_csv(index=False),
                            file_name=f"diseno_{design_type}_{int(k)}factores.csv")
        if design_type == "CCD":
            st.info(f"Puntos: {len(design)} — Factorial: {(design.PtType=='Factorial').sum()}, "
                    f"Axial: {(design.PtType=='Axial').sum()}, Centrales: {(design.PtType=='Center').sum()}, "
                    f"alpha = {design.alpha_used.iloc[0]:.4f}")
        else:
            st.info(f"Puntos: {len(design)} — Vértices/aristas: {(design.PtType=='Edge').sum()}, "
                    f"Centrales: {(design.PtType=='Center').sum()}")

# ==================================================================== #
# TAB 3: AJUSTE DEL MODELO
# ==================================================================== #
with tabs[2]:
    st.header("3. Ajuste del modelo de segundo orden")
    df = st.session_state.df
    factor_cols = st.session_state.factor_cols
    response_cols = st.session_state.response_cols
    coding = st.session_state.coding

    if df is None or not factor_cols or not response_cols:
        st.warning("Vaya a la pestaña **1. Datos**, cargue el dataset y seleccione factores/respuestas.")
    else:
        resp_sel = st.selectbox("Seleccione la respuesta a modelar", response_cols)
        order = st.selectbox("Orden del modelo", ["second", "first_int", "first"],
                              format_func=lambda o: {"second": "Segundo orden (completo)",
                                                      "first_int": "Primer orden + interacciones",
                                                      "first": "Primer orden"}[o])

        if st.button("📈 Ajustar modelo"):
            Xcoded = get_coded_df(df, factor_cols, coding)
            dm = build_design_matrix(Xcoded, order=order)
            y = df[resp_sel].values
            rep_group = Xcoded.round(4).apply(tuple, axis=1)
            result = anova_with_lof(y, dm, rep_group)
            st.session_state.models[resp_sel] = {
                "result": result, "factor_cols": factor_cols, "order": order,
            }
            st.success(f"Modelo ajustado para **{resp_sel}**.")

        if resp_sel in st.session_state.models:
            result = st.session_state.models[resp_sel]["result"]
            m1, m2, m3 = st.columns(3)
            m1.metric("R²", f"{result['r2']:.4f}")
            m2.metric("R² ajustado", f"{result['r2_adj']:.4f}")
            m3.metric("R² predicho", f"{result['r2_pred']:.4f}")

            st.subheader("Tabla ANOVA")
            st.dataframe(result["anova"].style.format(precision=4), use_container_width=True)
            lof_p = result["anova"].loc[result["anova"]["Fuente"].str.contains("Falta"), "p-valor"].values[0]
            if pd.notna(lof_p):
                if lof_p > 0.05:
                    st.success(f"Falta de ajuste NO significativa (p = {lof_p:.4f} > 0.05): "
                               "el modelo cuadrático describe adecuadamente los datos.")
                else:
                    st.error(f"Falta de ajuste SIGNIFICATIVA (p = {lof_p:.4f} ≤ 0.05): "
                             "considere un modelo de mayor orden o revisar el diseño.")

            st.subheader("Coeficientes del modelo")
            st.dataframe(result["coefficients"].style.format(precision=5), use_container_width=True)

            st.subheader("Diagnóstico de residuos")
            diag = residual_diagnostics(result)
            fig = residual_diagnostic_plots(diag)
            st.plotly_chart(fig, use_container_width=True)

# ==================================================================== #
# TAB 4: OPTIMIZACIÓN
# ==================================================================== #
with tabs[3]:
    st.header("4. Optimización de la superficie de respuesta")
    models = st.session_state.models
    if not models:
        st.warning("Ajuste al menos un modelo en la pestaña **3. Ajuste del modelo** primero.")
    else:
        resp_sel = st.selectbox("Respuesta a optimizar", list(models.keys()), key="opt_resp")
        entry = models[resp_sel]
        result = entry["result"]
        factor_cols = entry["factor_cols"]
        coef_table = result["coefficients"]
        b0, b, B = coefficients_to_vector_matrix(coef_table, factor_cols)
        maximize = st.radio("Meta de optimización", ["Maximizar", "Minimizar"], horizontal=True) == "Maximizar"

        st.subheader("🚀 Ascenso / descenso más pronunciado")
        st.caption("Válido cuando el modelo de primer orden domina (útil como paso previo, antes de "
                   "llegar a la región de curvatura donde aplica el análisis canónico).")
        step = st.slider("Tamaño de paso (unidades codificadas)", 0.1, 1.0, 0.3)
        n_steps = st.slider("Número de pasos", 3, 15, 6)
        path = steepest_path(b0, b, step=step, n_steps=n_steps, maximize=maximize)
        st.dataframe(path.style.format(precision=4), use_container_width=True)

        st.subheader("🎯 Análisis canónico")
        canon = canonical_analysis(b0, b, B, factor_cols)
        cc1, cc2 = st.columns(2)
        with cc1:
            st.write("**Punto estacionario (unidades codificadas):**")
            st.json({k_: round(v_, 4) for k_, v_ in canon["x_stationary"].items()})
            st.metric("Valor de respuesta en el punto estacionario", f"{canon['y_stationary']:.4f}")
        with cc2:
            st.write("**Autovalores (naturaleza del punto):**")
            st.write(np.round(canon["eigenvalues"], 4))
            if canon["nature"] == "Máximo":
                st.success(f"Naturaleza: **{canon['nature']}**")
            elif canon["nature"] == "Mínimo":
                st.info(f"Naturaleza: **{canon['nature']}**")
            else:
                st.warning(f"Naturaleza: **{canon['nature']}** — considere análisis de cresta.")

        st.subheader("🪢 Análisis de cresta (ridge analysis)")
        max_r = st.slider("Radio máximo a explorar (unidades codificadas)", 0.5, 3.0, 1.7)
        radii = np.linspace(0.1, max_r, 10)
        ridge = ridge_analysis(b0, b, B, radii, maximize=maximize)
        st.dataframe(ridge.style.format(precision=4), use_container_width=True)
        st.line_chart(ridge.set_index("radio")["y_pred"])

        st.subheader("🔢 Optimización numérica restringida")
        bound_choice = st.selectbox("Región de búsqueda", ["Cubo [-1,1]", "Región del CCD [-alpha,alpha]", "Personalizada"])
        if bound_choice == "Cubo [-1,1]":
            bnds = [(-1.0, 1.0)] * len(factor_cols)
        elif bound_choice == "Región del CCD [-alpha,alpha]":
            a = (2 ** len(factor_cols)) ** 0.25
            bnds = [(-a, a)] * len(factor_cols)
        else:
            bnds = []
            bc = st.columns(len(factor_cols))
            for i, f in enumerate(factor_cols):
                with bc[i]:
                    lo = st.number_input(f"{f} min", value=-1.68, key=f"nlo_{f}")
                    hi = st.number_input(f"{f} max", value=1.68, key=f"nhi_{f}")
                    bnds.append((lo, hi))

        if st.button("Optimizar numéricamente"):
            def pfn(x, names=coef_table["Término"].tolist(), beta=result["fit"]["beta"], fc=factor_cols):
                point = dict(zip(fc, x))
                return predict(beta, names, point)
            opt = numerical_optimize(pfn, len(factor_cols), bounds=bnds, maximize=maximize)
            if opt:
                st.success(f"Óptimo encontrado: y = {opt['y_opt']:.4f}")
                st.json({f: round(v, 4) for f, v in zip(factor_cols, opt["x_opt"])})
                coding = st.session_state.coding
                if coding:
                    real = {}
                    for f, xc in zip(factor_cols, opt["x_opt"]):
                        if f in coding:
                            lo, hi = coding[f]
                            mid, half = (lo + hi) / 2, (hi - lo) / 2
                            real[f] = round(mid + xc * half, 3)
                    st.write("**En unidades reales:**")
                    st.json(real)

# ==================================================================== #
# TAB 5: MULTIRRESPUESTA (DESEABILIDAD)
# ==================================================================== #
with tabs[4]:
    st.header("5. Optimización multirespuesta — Deseabilidad de Derringer-Suich")
    models = st.session_state.models
    if len(models) < 2:
        st.warning("Ajuste al menos DOS modelos (una respuesta por pestaña 3) para combinarlos aquí.")
    else:
        st.markdown("Defina la meta y los límites de aceptación para cada respuesta modelada:")
        specs = {}
        for r in models.keys():
            with st.expander(f"Especificación para: {r}", expanded=True):
                goal = st.selectbox(f"Meta para {r}", ["max", "min", "target"], key=f"goal_{r}")
                lo = st.number_input(f"Límite inferior aceptable — {r}", key=f"dlo_{r}")
                hi = st.number_input(f"Límite superior aceptable — {r}", key=f"dhi_{r}", value=1.0)
                target = None
                if goal == "target":
                    target = st.number_input(f"Valor objetivo — {r}", key=f"dtar_{r}")
                imp = st.slider(f"Importancia relativa — {r}", 1, 5, 1, key=f"imp_{r}")
                specs[r] = {"goal": goal, "low": lo, "high": hi, "target": target, "importance": imp}

        factor_cols = st.session_state.factor_cols
        n_grid = st.slider("Densidad de la malla de búsqueda por factor", 5, 25, 12)

        if st.button("⚖️ Calcular deseabilidad global sobre malla candidata"):
            grids = [np.linspace(-1.68, 1.68, n_grid) for _ in factor_cols]
            mesh = np.array(np.meshgrid(*grids)).reshape(len(factor_cols), -1).T
            cand = pd.DataFrame(mesh, columns=factor_cols)

            preds = {}
            for r, entry in models.items():
                result = entry["result"]
                beta = result["fit"]["beta"]
                names = result["coefficients"]["Término"].tolist()
                preds[r] = [predict(beta, names, row) for _, row in cand.iterrows()]
            preds_df = pd.DataFrame(preds)

            table = build_desirability_table(preds_df, specs)
            full = pd.concat([cand.reset_index(drop=True), table], axis=1)
            st.subheader("Top 10 combinaciones (mayor deseabilidad global D)")
            st.dataframe(full.head(10).style.format(precision=4), use_container_width=True)

            best = full.iloc[0]
            st.success(f"Mejor combinación: D = {best['D_global']:.4f}")
            coding = st.session_state.coding
            if coding:
                real = {}
                for f in factor_cols:
                    if f in coding:
                        lo, hi = coding[f]
                        mid, half = (lo + hi) / 2, (hi - lo) / 2
                        real[f] = round(mid + best[f] * half, 3)
                st.write("**Punto recomendado en unidades reales:**")
                st.json(real)

# ==================================================================== #
# TAB 6: VISUALIZACIÓN
# ==================================================================== #
with tabs[5]:
    st.header("6. Visualización de la superficie de respuesta")
    models = st.session_state.models
    if not models:
        st.warning("Ajuste al menos un modelo en la pestaña **3. Ajuste del modelo** primero.")
    else:
        resp_sel = st.selectbox("Respuesta a visualizar", list(models.keys()), key="viz_resp")
        entry = models[resp_sel]
        result = entry["result"]
        factor_cols = entry["factor_cols"]
        beta = result["fit"]["beta"]
        names = result["coefficients"]["Término"].tolist()

        c1, c2 = st.columns(2)
        with c1:
            fx = st.selectbox("Factor eje X", factor_cols, index=0)
        with c2:
            fy = st.selectbox("Factor eje Y", factor_cols, index=min(1, len(factor_cols) - 1))

        vc1, vc2 = st.columns(2)
        with vc1:
            st.plotly_chart(contour_plot(beta, names, factor_cols, fx, fy, response_name=resp_sel),
                             use_container_width=True)
        with vc2:
            st.plotly_chart(surface_plot(beta, names, factor_cols, fx, fy, response_name=resp_sel),
                             use_container_width=True)

        st.plotly_chart(pareto_chart(result["coefficients"]), use_container_width=True)
        st.plotly_chart(perturbation_plot(beta, names, factor_cols, response_name=resp_sel),
                         use_container_width=True)

# ==================================================================== #
# TAB 7: REPORTE
# ==================================================================== #
with tabs[6]:
    st.header("7. Exportación de reporte y memorándum gerencial")
    models = st.session_state.models
    if not models:
        st.warning("Ajuste al menos un modelo para poder generar el reporte.")
    else:
        if st.button("📄 Generar reporte técnico (Markdown)"):
            lines = []
            lines.append("# Reporte técnico — Optimización RSM del tostado de cacao Nacional")
            lines.append(f"*Generado automáticamente el {dt.datetime.now():%Y-%m-%d %H:%M}*\n")
            lines.append("## 1. Resumen de modelos ajustados\n")
            for r, entry in models.items():
                res = entry["result"]
                lines.append(f"### Respuesta: {r}")
                lines.append(f"- R² = {res['r2']:.4f} | R² ajustado = {res['r2_adj']:.4f} | "
                              f"R² predicho = {res['r2_pred']:.4f}")
                lof_row = res["anova"][res["anova"]["Fuente"].str.contains("Falta")]
                lof_p = lof_row["p-valor"].values[0]
                veredicto = "no significativa (modelo adecuado)" if (pd.notna(lof_p) and lof_p > 0.05) else "significativa (revisar modelo)"
                lines.append(f"- Falta de ajuste: p = {lof_p:.4f} → {veredicto}")
                lines.append("\n**Coeficientes:**\n")
                lines.append(res["coefficients"].to_markdown(index=False, floatfmt=".4f"))
                lines.append("")
            report_md = "\n".join(lines)
            st.download_button("⬇️ Descargar reporte (Markdown)", report_md,
                                file_name="reporte_tecnico_rsm_cacao.md")
            st.markdown(report_md)

        st.divider()
        st.subheader("📝 Memorándum gerencial (lenguaje natural)")
        if st.button("Generar memorándum"):
            memo = ["**MEMORÁNDUM TÉCNICO — RECOMENDACIÓN OPERATIVA**\n"]
            memo.append(f"Fecha: {dt.date.today():%d de %B de %Y}\n")
            memo.append(
                "Con base en el análisis de superficie de respuesta aplicado al proceso de "
                "tostado de cacao Nacional, se ajustaron modelos cuadráticos para las variables "
                "de calidad evaluadas. A continuación se resumen los hallazgos y recomendaciones "
                "operativas para la planta:\n"
            )
            for r, entry in models.items():
                res = entry["result"]
                memo.append(f"- Para **{r}**, el modelo explica el {res['r2']*100:.1f}% de la "
                            f"variabilidad observada (R² ajustado = {res['r2_adj']*100:.1f}%), "
                            "lo que indica un ajuste adecuado para fines de toma de decisiones.")
            memo.append(
                "\nSe recomienda validar experimentalmente el punto óptimo identificado en la "
                "pestaña de Optimización/Multirespuesta antes de escalar el proceso a nivel "
                "industrial, mediante corridas confirmatorias por triplicado."
            )
            memo_text = "\n".join(memo)
            st.markdown(memo_text)
            st.download_button("⬇️ Descargar memorándum", memo_text, file_name="memo_gerencial.md")

st.divider()
st.caption("Aplicativo desarrollado para la asignación final del curso de Optimización Estadística — "
           "Universidad Central del Ecuador. Dataset de cacao SIMULADO con fines académicos.")
