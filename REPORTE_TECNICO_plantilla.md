# Reporte Técnico — Optimización RSM del Tostado de Cacao Nacional
*(Plantilla — completar y exportar a PDF, máx. 10 páginas, antes de entregar)*

**Grupo:** _____  **Integrantes:** _____  **Fecha:** _____

## 1. Contexto
- Problema agroindustrial abordado (tostado de cacao Nacional) y su relevancia
  para el sector ecuatoriano.
- Objetivo del aplicativo desarrollado.

## 2. Métodos implementados
- Diseño experimental: CCD rotable, k=3 factores (Temperatura, Tiempo, Humedad).
- Ajuste de modelo cuadrático por mínimos cuadrados, ANOVA, falta de ajuste.
- Optimización: ascenso más pronunciado, análisis canónico, cresta, numérica.
- Deseabilidad de Derringer-Suich para conciliar Sabor / Polifenoles / Acidez.

## 3. Arquitectura del aplicativo
- Diagrama o descripción de `app.py` (interfaz) vs `src/` (lógica estadística).
- Tecnologías: Python, Streamlit, NumPy, SciPy, Plotly.
- (Insertar captura de pantalla de la app aquí)

## 4. Caso de prueba
- Descripción del dataset simulado: 26 corridas, CCD rotable, 3 factores, 3 respuestas.
- Resultados del ajuste (R², R² ajustado, ANOVA) — insertar tablas/capturas.
- Punto óptimo identificado (canónico + numérico + deseabilidad global).

## 5. Resultados y recomendación operativa
- Interpretación práctica del óptimo encontrado (temperatura/tiempo/humedad recomendados).
- Justificación agroindustrial de la recomendación.

## 6. Limitaciones
- Datos simulados (no medición real de laboratorio).
- Rango experimental válido solo dentro de la región estudiada (no extrapolar).
- (Agregar otras limitaciones identificadas por el grupo)

## 7. Referencias (mínimo 8, al menos 4 artículos científicos)
1. Myers, Montgomery & Anderson-Cook (2016). *Response Surface Methodology*, 4th ed. Wiley.
2. Box & Behnken (1960). *Technometrics*, 2(4), 455-475.
3. Box & Wilson (1951). *JRSS Series B*, 13(1), 1-45.
4. Derringer & Suich (1980). *Journal of Quality Technology*, 12(4), 214-219.
5. _(agregar artículos científicos sobre tostado de cacao / RSM en alimentos)_
6. _____
7. _____
8. _____

## 8. Declaración de uso de IA
Detallar qué herramientas de IA se usaron (p. ej. Claude), para qué tareas
específicas (generación de código base, redacción de README, etc.), y confirmar
que cada integrante puede explicar cualquier línea del código en la defensa oral.
