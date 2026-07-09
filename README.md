# 🍫 Aplicativo RSM — Optimización del tostado de cacao Nacional

Aplicativo interactivo en **Python + Streamlit** que implementa de forma integrada
la Metodología de Superficie de Respuesta (RSM) para un caso simulado del sector
agroindustrial ecuatoriano: **optimización del proceso de tostado de cacao Nacional**.

Proyecto desarrollado para la asignación final del curso **Optimización Estadística**
(Universidad Central del Ecuador — Docente: Christian Franco Crespo, Ph.D.).

> ⚠️ **Los datos en `data/cacao_tostado_simulado.csv` son SIMULADOS** (generados con
> `generar_datos_simulados.py` a partir de un modelo cuadrático conocido + ruido
> aleatorio), con fines exclusivamente académicos y de demostración del aplicativo.

---

## 1. ¿Qué hace el aplicativo?

| Módulo | Contenido |
|---|---|
| **Diseño experimental** | Genera diseños Central Compuesto (CCD, rotable/cara-centrada/ortogonal) y Box-Behnken (BBD) para cualquier número de factores. |
| **Ajuste del modelo** | Regresión de 1er/2do orden por mínimos cuadrados, tabla ANOVA con descomposición de **Falta de Ajuste / Error Puro**, R², R² ajustado, R² predicho (PRESS), diagnóstico de residuos. |
| **Optimización** | Ascenso más pronunciado, **análisis canónico** (punto estacionario + autovalores), **análisis de cresta**, optimización numérica restringida a la región experimental. |
| **Multirespuesta** | Función de deseabilidad de **Derringer–Suich** (individual y global) para conciliar varias respuestas en conflicto. |
| **Visualización** | Gráficos de contorno, superficies 3D interactivas, diagrama de Pareto de efectos, gráfico de perturbación. |
| **Reporte** | Exportación de reporte técnico (Markdown) y memorándum gerencial en lenguaje natural. |

Caso de aplicación: 3 factores (Temperatura de tostado, Tiempo de tostado, Humedad
inicial del grano) y 3 respuestas (Sabor sensorial ↑, Polifenoles ↑, Acidez titulable ↓).

---

## 2. Estructura de archivos — ¿qué es cada uno?

```
cacao_rsm_app/
├── app.py                          # App principal de Streamlit (interfaz de usuario, 7 pestañas)
├── generar_datos_simulados.py      # Script que crea el CCD simulado de cacao (data/*.csv)
├── requirements.txt                # Dependencias exactas para instalar con pip
├── README.md                       # Este archivo
├── .gitignore                      # Archivos/carpetas que Git debe ignorar
├── .streamlit/
│   └── config.toml                 # Tema visual de la app (colores tipo cacao)
├── data/
│   └── cacao_tostado_simulado.csv  # Dataset de prueba (26 corridas, CCD rotable k=3)
└── src/                            # Lógica estadística, separada de la interfaz
    ├── __init__.py
    ├── design.py                   # Generadores de diseños CCD y Box-Behnken
    ├── modeling.py                 # OLS, ANOVA, falta de ajuste, R², diagnóstico de residuos
    ├── optimization.py             # Ascenso más pronunciado, análisis canónico, cresta, opt. numérica
    ├── desirability.py             # Función de deseabilidad de Derringer-Suich
    └── plotting.py                 # Gráficos interactivos con Plotly
```

**Por qué esta separación:** `app.py` solo contiene la interfaz (botones, pestañas,
inputs). Toda la estadística vive en `src/`, como funciones puras de Python
(numpy/scipy) que **no dependen de Streamlit**. Esto permite:
1. Probar cada función matemática de forma aislada (más fácil de explicar en la defensa oral).
2. Reutilizar la lógica en un notebook, script de línea de comandos, u otra interfaz (Shiny, Dash, etc.) sin reescribir nada.

---

## 3. Cómo correr el proyecto — paso a paso

### 3.1. Requisitos previos
- Tener instalado **Git** ([descargar aquí](https://git-scm.com/downloads)).
- Tener instalado **Python 3.10 o superior** ([descargar aquí](https://www.python.org/downloads/)).
- Tener una cuenta de **GitHub** (gratuita, [crear aquí](https://github.com/signup)).

### 3.2. Subir el proyecto a GitHub (si aún no existe el repositorio)

Abra una terminal (CMD, PowerShell, Terminal de Mac/Linux) dentro de la carpeta
`cacao_rsm_app` y ejecute:

```bash
# 1. Inicializar el repositorio local
git init

# 2. Agregar todos los archivos
git add .

# 3. Crear el primer commit
git commit -m "Aplicativo RSM - optimizacion tostado de cacao"

# 4. Crear el repositorio remoto en GitHub
#    (opción A: hacerlo manualmente en github.com -> New repository)
#    (opción B: usando GitHub CLI, si lo tiene instalado)
gh repo create cacao-rsm-app --public --source=. --remote=origin

# 5. Si creó el repo manualmente en la web, conecte el remoto (reemplace <su-usuario>):
git remote add origin https://github.com/<su-usuario>/cacao-rsm-app.git

# 6. Subir el código
git branch -M main
git push -u origin main
```

### 3.3. Clonar el repositorio (si ya existe en GitHub) y correrlo localmente

```bash
# 1. Clonar el repositorio
git clone https://github.com/<su-usuario>/cacao-rsm-app.git
cd cacao-rsm-app

# 2. Crear un entorno virtual (recomendado, evita conflictos de versiones)
python -m venv venv

# 3. Activar el entorno virtual
#    En Windows:
venv\Scripts\activate
#    En Mac/Linux:
source venv/bin/activate

# 4. Instalar las dependencias
pip install -r requirements.txt

# 5. Ejecutar la aplicación
streamlit run app.py
```

Streamlit abrirá automáticamente su navegador en `http://localhost:8501`. Si no se
abre solo, copie esa dirección y péguela en su navegador.

Para detener la app: en la terminal, presione `Ctrl + C`.

### 3.4. Uso rápido dentro de la app
1. Pestaña **1. Datos** → clic en "📂 Usar dataset simulado de cacao" (o suba su propio CSV).
2. Seleccione los factores (X) y respuestas (Y), confirme los niveles ±1.
3. Pestaña **3. Ajuste del modelo** → elija una respuesta → "Ajustar modelo".
4. Repita el paso 3 para cada respuesta que quiera analizar.
5. Pestaña **4. Optimización** → revise ascenso más pronunciado, análisis canónico, cresta y optimización numérica.
6. Pestaña **5. Multirespuesta** → defina metas/límites de cada respuesta → "Calcular deseabilidad global".
7. Pestaña **6. Visualización** → explore contornos y superficies 3D.
8. Pestaña **7. Reporte** → descargue el reporte técnico y el memorándum gerencial.

---

## 4. Despliegue en la nube (opcional, recomendado para el entregable)

### Opción A: Streamlit Community Cloud (gratuita)
1. Suba el proyecto a un repositorio **público** de GitHub (ver sección 3.2).
2. Vaya a [share.streamlit.io](https://share.streamlit.io) e inicie sesión con su cuenta de GitHub.
3. Clic en "New app" → seleccione el repositorio, la rama `main` y el archivo `app.py`.
4. Clic en "Deploy". En 1-2 minutos obtendrá una URL pública tipo
   `https://<su-app>.streamlit.app` para compartir en el reporte y video tutorial.

### Opción B: Hugging Face Spaces
1. Cree una cuenta en [huggingface.co](https://huggingface.co).
2. Cree un nuevo "Space" con SDK = **Streamlit**.
3. Suba los mismos archivos de este repositorio (o conecte el Space a su repo de GitHub).
4. El Space instalará `requirements.txt` y correrá `app.py` automáticamente.

---

## 5. Regenerar el dataset simulado (opcional)

Si desea cambiar la semilla aleatoria, el número de puntos centrales, o los rangos
de los factores, edite `generar_datos_simulados.py` y vuelva a ejecutarlo:

```bash
python generar_datos_simulados.py
```

Esto sobreescribirá `data/cacao_tostado_simulado.csv`.

---

## 6. Fundamento estadístico y referencias

- Myers, R. H., Montgomery, D. C., & Anderson-Cook, C. M. (2016). *Response Surface
  Methodology: Process and Product Optimization Using Designed Experiments* (4th ed.).
  Wiley.
- Box, G. E. P., & Behnken, D. W. (1960). Some new three level designs for the study
  of quantitative variables. *Technometrics, 2*(4), 455-475.
- Box, G. E. P., & Wilson, K. B. (1951). On the experimental attainment of optimum
  conditions. *Journal of the Royal Statistical Society: Series B, 13*(1), 1-45.
- Derringer, G., & Suich, R. (1980). Simultaneous optimization of several response
  variables. *Journal of Quality Technology, 12*(4), 214-219.

## 7. Declaración de uso de IA

Este aplicativo fue desarrollado con asistencia de **Claude (Anthropic)** para:
generación de la estructura del proyecto, implementación de las funciones de diseño
experimental/ANOVA/optimización/deseabilidad, la interfaz Streamlit, y este README.
El equipo debe reemplazar esta sección en el reporte técnico final por su propia
declaración detallada, según lo exige la rúbrica del docente (sección 3.d).
