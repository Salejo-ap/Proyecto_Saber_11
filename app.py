# ============================================================
# Saber 11 · Desempeño relativo de sedes educativas
# App para Secretarías de Educación
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ------------------------------------------------------------
st.set_page_config(
    page_title="Saber 11 · Desempeño relativo",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------
# ESTILOS · Paleta corporativa para Secretaría
# ------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --primary:    #2C5F8D;
    --primary-2:  #3A7CA5;
    --secondary:  #4A9D9C;
    --accent:     #E8A87C;
    --accent-2:   #F4B860;
    --bg:         #F8F9FA;
    --bg-soft:    #F1F3F5;
    --card:       #FFFFFF;
    --text:       #2D3748;
    --text-soft:  #718096;
    --success:    #48BB78;
    --error:      #E53E3E;
    --warning:    #ECC94B;
    --shadow:     0 4px 16px rgba(45, 55, 72, 0.08);
    --shadow-lg:  0 8px 28px rgba(45, 55, 72, 0.12);
    --radius:     12px;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
    color: var(--text);
}
h1, h2, h3, h4 {
    font-family: 'Poppins', sans-serif !important;
    color: var(--primary) !important;
    letter-spacing: -0.01em;
}
h1 { font-size: 34px !important; font-weight: 700 !important; }
h2 { font-size: 26px !important; font-weight: 600 !important; }
h3 { font-size: 20px !important; font-weight: 600 !important; }

/* Fondo general */
.stApp { background: var(--bg); }

/* Header principal */
.hero {
    background: linear-gradient(135deg, #2C5F8D 0%, #3A7CA5 60%, #4A9D9C 100%);
    color: white;
    padding: 40px 44px;
    border-radius: 16px;
    margin-bottom: 28px;
    box-shadow: var(--shadow-lg);
}
.hero h1 { color: white !important; margin: 0 0 8px 0; font-size: 36px !important; }
.hero p  { color: rgba(255,255,255,0.92); font-size: 17px; margin: 0; line-height: 1.55; }

/* Tarjetas de servicios */
.service-card {
    background: var(--card);
    border-radius: var(--radius);
    padding: 22px 20px;
    box-shadow: var(--shadow);
    border-top: 4px solid var(--secondary);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    height: 100%;
}
.service-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-lg);
}
.service-card .icon { font-size: 28px; margin-bottom: 8px; }
.service-card h4 {
    font-family: 'Poppins', sans-serif;
    color: var(--primary) !important;
    margin: 0 0 6px 0;
    font-size: 17px;
}
.service-card p {
    color: var(--text-soft);
    font-size: 14px;
    line-height: 1.5;
    margin: 0;
}

/* Metric containers */
[data-testid="stMetric"] {
    background: var(--card);
    border-radius: var(--radius);
    padding: 18px 22px;
    box-shadow: var(--shadow);
    border-left: 5px solid var(--primary-2);
}
[data-testid="stMetricLabel"] { color: var(--text-soft) !important; }
[data-testid="stMetricValue"] {
    color: var(--primary) !important;
    font-family: 'Poppins', sans-serif !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: transparent;
    border-bottom: 2px solid var(--bg-soft);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 10px 10px 0 0;
    padding: 12px 20px;
    font-family: 'Poppins', sans-serif;
    font-weight: 500;
    color: var(--text-soft);
    transition: all 0.3s ease;
}
.stTabs [aria-selected="true"] {
    background: var(--card) !important;
    color: var(--primary) !important;
    border-bottom: 3px solid var(--accent) !important;
}

/* Botones primarios (accent) */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #E8A87C 0%, #F4B860 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 11px 26px;
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(232, 168, 124, 0.35);
    transition: all 0.3s ease;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(232, 168, 124, 0.45);
}

/* Botones secundarios */
.stButton > button {
    border-radius: 10px;
    font-family: 'Poppins', sans-serif;
    font-weight: 500;
    transition: all 0.3s ease;
}

/* Inputs */
.stTextInput input, .stNumberInput input, .stSelectbox div {
    border-radius: 10px !important;
}

/* Divisor sutil */
hr { border-color: var(--bg-soft); }

/* Alertas */
.stAlert { border-radius: var(--radius); }

/* Expander */
.streamlit-expanderHeader {
    font-family: 'Poppins', sans-serif;
    font-weight: 500;
    color: var(--primary);
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# CARGA DE MODELOS Y DATOS
# ------------------------------------------------------------
RUTA = Path("models")

@st.cache_resource
def cargar_modelos():
    meta = joblib.load(RUTA / "metadata.pkl")
    reg = {
        "Regresión Lineal": joblib.load(RUTA / "regresion_lineal.pkl"),
        "SVR (RBF)":        joblib.load(RUTA / "svr.pkl"),
        "CatBoost":         joblib.load(RUTA / "catboost_reg.pkl"),
    }
    clf = {
        "Regresión Logística": joblib.load(RUTA / "clasif_regresion_logistica.pkl"),
        "SVM Lineal":          joblib.load(RUTA / "clasif_svm_lineal.pkl"),
        "Gradient Boosting":   joblib.load(RUTA / "clasif_gradient_boosting.pkl"),
    }
    return meta, reg, clf

@st.cache_data
def cargar_catalogo():
    return pd.read_parquet(RUTA / "catalogo_sedes.parquet")

@st.cache_data
def cargar_historico():
    return pd.read_parquet(RUTA / "historico_sedes.parquet")

@st.cache_data
def cargar_base_modelo():
    # Solo para la pestaña de priorización y lote
    return pd.read_csv(RUTA / "base_modelo_para_app.csv")

try:
    meta, modelos_reg, modelos_clf = cargar_modelos()
    catalogo   = cargar_catalogo()
    historico  = cargar_historico()
except FileNotFoundError as e:
    st.error(
        f"No se encontraron los archivos del modelo en `{RUTA}`. "
        f"Verifica que hayas ejecutado las celdas de exportación del notebook. "
        f"Detalle: {e}"
    )
    st.stop()


# ------------------------------------------------------------
# UTILIDADES
# ------------------------------------------------------------
CATEGORICAS = meta["categoricas_contexto"]
NUMERICAS   = meta["numericas_contexto"]
TEMPORALES  = meta["variables_temporales"]
TEMPORALES_B = meta["variables_temporales_B"]
FEATURES_B  = meta["features_B"]
UMBRAL_B    = meta["umbral_B"]


def asegurar_tipos(df):
    df = df.copy()
    for c in CATEGORICAS:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


def derivar_temporales(historial: pd.DataFrame):
    """Devuelve puntaje_previo, variacion_previa, cohortes_previas desde historial."""
    if historial is None or len(historial) == 0:
        return {"puntaje_previo": np.nan,
                "variacion_previa": np.nan,
                "cohortes_previas": 0}

    h = historial.sort_values("periodo").reset_index(drop=True)
    puntaje_previo   = float(h["puntaje_observado"].iloc[-1])
    cohortes_previas = len(h)
    variacion_previa = (
        float(h["puntaje_observado"].iloc[-1])
        - float(h["puntaje_observado"].iloc[-2])
        if len(h) >= 2 else np.nan
    )
    return {"puntaje_previo": puntaje_previo,
            "variacion_previa": variacion_previa,
            "cohortes_previas": cohortes_previas}


def predecir_regresion(modelo, entrada):
    X = entrada[NUMERICAS + TEMPORALES + CATEGORICAS]
    return float(modelo.predict(X)[0])


def predecir_clasificacion(modelo, entrada):
    X = entrada[FEATURES_B]
    if hasattr(modelo, "predict_proba"):
        return float(modelo.predict_proba(X)[0, 1])
    score = modelo.decision_function(X)[0]
    return float(1 / (1 + np.exp(-score)))


def nivel_riesgo(proba):
    if proba >= 0.6:  return "alto",    "🔴 Riesgo ALTO de deterioro"
    if proba >= 0.35: return "medio",   "🟡 Riesgo MEDIO de deterioro"
    return "bajo", "🟢 Riesgo BAJO de deterioro"


def explicar_shap_regresion(modelo, entrada, nombre_modelo):
    """Intenta SHAP; si falla, muestra importancia global."""
    try:
        if "CatBoost" in nombre_modelo:
            imp = modelo.get_feature_importance()
            nombres = NUMERICAS + TEMPORALES + CATEGORICAS
            df_imp = pd.DataFrame({"Variable": nombres[:len(imp)],
                                   "Importancia": imp})
            df_imp = df_imp.sort_values("Importancia").tail(12)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.barh(df_imp["Variable"], df_imp["Importancia"], color="#3A7CA5")
            ax.set_xlabel("Importancia relativa")
            st.pyplot(fig, bbox_inches="tight")
            plt.close(fig)
        else:
            X = entrada[NUMERICAS + TEMPORALES + CATEGORICAS]
            pre = modelo.named_steps["prep"]
            X_trans = pre.transform(X)
            explainer = shap.Explainer(modelo.named_steps["model"], X_trans)
            sv = explainer(X_trans)
            fig, ax = plt.subplots(figsize=(8, 4))
            shap.plots.waterfall(sv[0], show=False)
            st.pyplot(fig, bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        st.info(f"No se pudo generar el gráfico SHAP: {e}")


def mostrar_acciones_regresion(brecha):
    if brecha <= -20:
        st.markdown("""
        - Programar visita técnica de acompañamiento.
        - Revisar prácticas pedagógicas y gestión institucional.
        - Verificar posibles cambios en la composición del estudiantado.
        - Priorizar la sede en el plan de mejoramiento territorial.
        """)
    elif brecha >= 20:
        st.markdown("""
        - Documentar prácticas exitosas de la sede.
        - Considerarla como referente para sedes con contexto similar.
        - Socializar experiencias en mesas territoriales de calidad educativa.
        """)
    else:
        st.markdown("""
        - Mantener el monitoreo periódico habitual.
        - Registrar cambios relevantes de contexto para el siguiente análisis.
        """)


def mostrar_acciones_clasificacion(proba):
    if proba >= 0.6:
        st.markdown("""
        - Priorizar la sede en la agenda de visitas del próximo semestre.
        - Contactar al rector para revisar planes de mejoramiento.
        - Activar alerta temprana en el sistema territorial.
        - Recolectar información cualitativa adicional.
        """)
    elif proba >= 0.35:
        st.markdown("""
        - Incluir la sede en la lista de seguimiento preventivo.
        - Verificar cambios recientes en planta docente o directivos.
        - Solicitar reporte breve de gestión al establecimiento.
        """)
    else:
        st.markdown("""
        - Monitoreo habitual.
        - Sin acciones especiales por ahora.
        """)


# ------------------------------------------------------------
# ENCABEZADO
# ------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🎓 Desempeño relativo de sedes educativas en Saber 11</h1>
    <p>Herramienta de apoyo para Secretarías de Educación: estime el desempeño
    esperado de una sede según su contexto, identifique sedes que requieren
    acompañamiento y anticipe posibles deterioros antes de que ocurran.</p>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# PESTAÑAS
# ------------------------------------------------------------
tab_inicio, tab_consulta, tab_prioriza, tab_lote = st.tabs([
    "🏠 Inicio",
    "🏫 Consultar sede",
    "📊 Priorización",
    "📁 Análisis por lote",
])


# ============================================================
# TAB 1 · INICIO / CONTEXTO
# ============================================================
with tab_inicio:
    st.markdown("## ¿Qué hace esta herramienta?")
    st.markdown("""
    Esta aplicación ayuda a **interpretar el desempeño de una sede-jornada** en
    Saber 11 de forma más justa, comparando sus resultados con lo que sería
    esperable dado su **contexto socioeconómico, territorial e institucional**.
    Además, estima la **probabilidad de deterioro** en la siguiente cohorte a
    partir de la trayectoria reciente de la sede.
    """)

    st.markdown("## ¿Por qué es útil?")
    st.markdown("""
    Comparar colegios únicamente por sus puntajes puede ser injusto: no todos
    atienden estudiantes bajo las mismas condiciones. Esta herramienta permite
    identificar **sedes que rinden mejor o peor de lo esperado**, y **anticipar
    deterioros** antes de que sean evidentes en los resultados.
    """)

    st.markdown("## Servicios de la aplicación")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="service-card">
            <div class="icon">🏫</div>
            <h4>Consultar una sede</h4>
            <p>Elige una sede existente o registra una nueva para obtener su
            desempeño esperado y su alerta de deterioro.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="service-card">
            <div class="icon">📊</div>
            <h4>Priorizar acompañamiento</h4>
            <p>Rankings de sedes que requieren acompañamiento y sedes referentes,
            filtrables por departamento y naturaleza.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="service-card">
            <div class="icon">📁</div>
            <h4>Análisis por lote</h4>
            <p>Carga un CSV con varias sedes y obtén su predicción y ranking
            interno en un solo paso.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## Cómo interpretar los resultados")
    st.markdown(f"""
    | Resultado | Interpretación |
    |---|---|
    | **Puntaje por encima de lo esperado** (+20 pts) | La sede rinde mejor de lo previsible. Candidata a referente. |
    | **En línea con lo esperado** | Desempeño consistente con su contexto. |
    | **Por debajo de lo esperado** (−20 pts) | Requiere revisión y posible acompañamiento. |
    | **Probabilidad de deterioro ≥ 60 %** | Priorizar seguimiento en la próxima cohorte. |
    | **Probabilidad de deterioro 35–60 %** | Seguimiento preventivo. |
    | **Probabilidad < 35 %** | Monitoreo habitual. |
    """)

    st.markdown("## Cobertura y fuentes")
    st.markdown(f"""
    - **Fuente**: resultados públicos de Saber 11 (ICFES), cohortes 2014-2 a 2022-2.
    - **Unidad de análisis**: sede-jornada-cohorte.
    - **Modelo de regresión**: {list(modelos_reg.keys())}.
    - **Modelo de clasificación**: {list(modelos_clf.keys())}.
    - **Umbral de deterioro**: caída ≥ {abs(UMBRAL_B):.1f} puntos (percentil 25 histórico).
    """)

    st.info(
        "⚠️ Esta herramienta es un **apoyo al diagnóstico**. No constituye un "
        "mecanismo automático de sanción, clasificación institucional ni "
        "asignación de recursos. Las decisiones finales deben pasar por "
        "revisión humana."
    )


# ============================================================
# TAB 2 · CONSULTAR SEDE (3 modos)
# ============================================================
with tab_consulta:
    st.markdown("## Consultar una sede-jornada")

    modo = st.radio(
        "¿Cómo quieres ingresar los datos?",
        ["🏫 Elegir sede existente",
         "🆕 Registrar sede nueva",
         "🧪 Simular sede hipotética"],
        horizontal=True,
        key="modo_consulta",
    )

    entrada = None
    historial_mostrado = None
    puntaje_observado = None
    contexto_full = None

    # --------------------------------------------------------
    # MODO 1 · SEDE EXISTENTE
    # --------------------------------------------------------
    if modo == "🏫 Elegir sede existente":
        col1, col2 = st.columns(2)
        with col1:
            deptos = ["(Todos)"] + sorted(catalogo["departamento"].dropna().unique())
            depto = st.selectbox("Departamento", deptos)
        filtro = catalogo if depto == "(Todos)" else catalogo[catalogo["departamento"] == depto]

        with col2:
            municipios = ["(Todos)"] + sorted(filtro["municipio"].dropna().unique())
            muni = st.selectbox("Municipio", municipios)
        if muni != "(Todos)":
            filtro = filtro[filtro["municipio"] == muni]

        texto = st.text_input(
            "Buscar por código DANE",
            placeholder="Ej: 105001000001",
            key="busqueda_dane"
        )
        if texto:
            filtro = filtro[
                filtro["cole_cod_dane_sede"].astype(str).str.contains(texto, case=False)
            ]

        if len(filtro) == 0:
            st.warning("No hay sedes que coincidan con el filtro.")
        else:
            filtro = filtro.copy()
            filtro["etiqueta"] = (
                filtro["cole_cod_dane_sede"].astype(str)
                + " · " + filtro["cole_jornada"].astype(str)
                + " · " + filtro["municipio"].astype(str)
            )
            opciones = filtro["etiqueta"].sort_values().tolist()
            if len(opciones) > 2000:
                st.info(f"Hay {len(opciones)} sedes con estos filtros. "
                        "Refina el buscador para ver menos resultados.")
                opciones = opciones[:2000]

            sede_sel = st.selectbox("Sede-jornada", opciones, key="sede_sel")
            fila = filtro[filtro["etiqueta"] == sede_sel].iloc[0]
            codigo = fila["cole_cod_dane_sede"]
            jornada = fila["cole_jornada"]

            hist = historico[
                (historico["cole_cod_dane_sede"] == codigo) &
                (historico["cole_jornada"] == jornada)
            ].sort_values("periodo")

            if len(hist) > 0:
                st.markdown("**Trayectoria observada**")
                st.line_chart(
                    hist.set_index("periodo")["punt_global_promedio"],
                    height=220,
                )
                historial_mostrado = hist

                # Recuperar contexto completo de la última cohorte
                ultima = hist.iloc[-1]
                contexto_full = (
                    cargar_base_modelo()[
                        (cargar_base_modelo()["cole_cod_dane_sede"] == codigo) &
                        (cargar_base_modelo()["cole_jornada"] == jornada) &
                        (cargar_base_modelo()["periodo"] == ultima["periodo"])
                    ]
                )
                if len(contexto_full) == 0:
                    st.warning(
                        "No se encontró el contexto detallado de esta sede. "
                        "Usa el modo 'Simular sede hipotética'."
                    )
                else:
                    contexto_full = contexto_full.iloc[0]
                    puntaje_observado = float(contexto_full["punt_global_promedio"])

                    entrada = pd.DataFrame([{
                        "log_n_estudiantes":   contexto_full["log_n_estudiantes"],
                        "prop_internet":       contexto_full["prop_internet"],
                        "prop_computador":     contexto_full["prop_computador"],
                        "prop_lavadora":       contexto_full["prop_lavadora"],
                        "prop_automovil":      contexto_full["prop_automovil"],
                        "prop_estrato_1_2":    contexto_full["prop_estrato_1_2"],
                        "prop_madre_superior": contexto_full["prop_madre_superior"],
                        "prop_padre_superior": contexto_full["prop_padre_superior"],
                        "prop_mujeres":        contexto_full["prop_mujeres"],
                        "puntaje_previo":      contexto_full["puntaje_previo"],
                        "variacion_previa":    contexto_full["variacion_previa"],
                        "cohortes_previas":    contexto_full["cohortes_previas"],
                        "primera_cohorte":     int(contexto_full["cohortes_previas"] == 0),
                        "area":                contexto_full["area"],
                        "calendario":          contexto_full["calendario"],
                        "caracter":            contexto_full["caracter"],
                        "naturaleza":          contexto_full["naturaleza"],
                        "departamento":        contexto_full["departamento"],
                        "municipio":           contexto_full["municipio"],
                        "aplicacion":          contexto_full["aplicacion"],
                    }])
                    entrada = asegurar_tipos(entrada)
                    st.caption(
                        f"Se usa la información de la cohorte "
                        f"**{contexto_full['periodo']}** como punto de partida."
                    )

    # --------------------------------------------------------
    # MODO 2 · REGISTRAR SEDE NUEVA (Opción A)
    # --------------------------------------------------------
    elif modo == "🆕 Registrar sede nueva":

        if "sedes_nuevas" not in st.session_state:
            st.session_state["sedes_nuevas"] = []

        st.markdown("### Identificación de la sede")
        col1, col2, col3 = st.columns(3)
        with col1:
            codigo = st.text_input("Código DANE nuevo", placeholder="999999999999")
            nombre = st.text_input("Nombre (opcional)")
        with col2:
            depto = st.text_input("Departamento", "ANTIOQUIA")
            muni  = st.text_input("Municipio", "MEDELLIN")
        with col3:
            jornada = st.selectbox("Jornada",
                ["MAÑANA", "TARDE", "NOCHE", "COMPLETA", "ÚNICA"])
            aplicacion = st.selectbox("Aplicación", [1, 2], index=1)

        col4, col5, col6 = st.columns(3)
        with col4:
            area = st.selectbox("Área", ["URBANO", "RURAL"])
        with col5:
            naturaleza = st.selectbox("Naturaleza", ["OFICIAL", "NO OFICIAL"])
        with col6:
            calendario = st.selectbox("Calendario", ["A", "B", "OTRO"])
        caracter = st.selectbox("Carácter", ["ACADÉMICO", "TÉCNICO", "OTRO"])

        st.markdown("### Contexto de la última cohorte")
        col1, col2 = st.columns(2)
        with col1:
            n_est = st.number_input("N° de estudiantes", 10, 2000, 60, step=5)
            prop_internet       = st.slider("Proporción con internet", 0.0, 1.0, 0.55)
            prop_computador     = st.slider("Proporción con computador", 0.0, 1.0, 0.54)
            prop_lavadora       = st.slider("Proporción con lavadora", 0.0, 1.0, 0.70)
            prop_automovil      = st.slider("Proporción con automóvil", 0.0, 1.0, 0.24)
        with col2:
            prop_estrato_1_2    = st.slider("Proporción estrato 1-2", 0.0, 1.0, 0.73)
            prop_madre_superior = st.slider("Madres con educación superior", 0.0, 1.0, 0.26)
            prop_padre_superior = st.slider("Padres con educación superior", 0.0, 1.0, 0.21)
            prop_mujeres        = st.slider("Proporción de mujeres", 0.0, 1.0, 0.54)

        st.markdown("### Historial de cohortes")
        st.caption("Agrega las cohortes previas que ya conozcas. "
                   "Con dos o más, el modelo calcula la variación previa.")

        historial_default = pd.DataFrame({
            "periodo":           ["20211", "20221"],
            "puntaje_observado": [245.0,   250.0],
            "n_estudiantes":     [50,      60],
        })

        historial = st.data_editor(
            historial_default,
            num_rows="dynamic",
            use_container_width=True,
            key="editor_historial",
        )
        historial = historial.dropna(how="all").reset_index(drop=True)
        historial = historial[
            historial["periodo"].notna() &
            historial["puntaje_observado"].notna()
        ]

        temporales = derivar_temporales(historial)

        registro = {
            "cole_cod_dane_sede": codigo,
            "cole_jornada":       jornada,
            "log_n_estudiantes":  np.log1p(n_est),
            "prop_internet":       prop_internet,
            "prop_computador":     prop_computador,
            "prop_lavadora":       prop_lavadora,
            "prop_automovil":      prop_automovil,
            "prop_estrato_1_2":    prop_estrato_1_2,
            "prop_madre_superior": prop_madre_superior,
            "prop_padre_superior": prop_padre_superior,
            "prop_mujeres":        prop_mujeres,
            "puntaje_previo":      temporales["puntaje_previo"],
            "variacion_previa":    temporales["variacion_previa"],
            "cohortes_previas":    temporales["cohortes_previas"],
            "primera_cohorte":     int(temporales["cohortes_previas"] == 0),
            "area":                area,
            "calendario":          calendario,
            "caracter":            caracter,
            "naturaleza":          naturaleza,
            "departamento":        depto,
            "municipio":           muni,
            "aplicacion":          aplicacion,
        }
        entrada = asegurar_tipos(pd.DataFrame([registro]))
        puntaje_observado = temporales["puntaje_previo"]
        historial_mostrado = historial

        if temporales["cohortes_previas"] == 0:
            st.warning(
                "⚠️ Esta sede no tiene cohortes previas. El modelo usará la "
                "mediana del entrenamiento para `puntaje_previo` y "
                "`variacion_previa`, lo que reduce la precisión."
            )
        elif temporales["cohortes_previas"] == 1:
            st.info(
                "ℹ️ Solo hay una cohorte previa. `variacion_previa` se imputará "
                "con la mediana del entrenamiento."
            )

    # --------------------------------------------------------
    # MODO 3 · SIMULAR SEDE HIPOTÉTICA
    # --------------------------------------------------------
    else:
        st.markdown("### Simular una sede con condiciones específicas")
        col1, col2, col3 = st.columns(3)
        with col1:
            area = st.selectbox("Área", ["URBANO", "RURAL"], key="sim_area")
            calendario = st.selectbox("Calendario", ["A", "B", "OTRO"], key="sim_cal")
            caracter = st.selectbox("Carácter", ["ACADÉMICO", "TÉCNICO", "OTRO"], key="sim_car")
            naturaleza = st.selectbox("Naturaleza", ["OFICIAL", "NO OFICIAL"], key="sim_nat")
        with col2:
            depto = st.text_input("Departamento", "ANTIOQUIA", key="sim_dep")
            muni  = st.text_input("Municipio", "MEDELLIN", key="sim_mun")
            aplicacion = st.selectbox("Aplicación", [1, 2], index=1, key="sim_app")
            n_est = st.number_input("N° de estudiantes", 10, 2000, 60, step=5, key="sim_n")
        with col3:
            prop_internet       = st.slider("Internet", 0.0, 1.0, 0.55, key="sim_int")
            prop_computador     = st.slider("Computador", 0.0, 1.0, 0.54, key="sim_comp")
            prop_lavadora       = st.slider("Lavadora", 0.0, 1.0, 0.70, key="sim_lav")
            prop_automovil      = st.slider("Automóvil", 0.0, 1.0, 0.24, key="sim_auto")

        col4, col5, col6, col7 = st.columns(4)
        with col4:
            prop_estrato_1_2    = st.slider("Estrato 1-2", 0.0, 1.0, 0.73, key="sim_est")
        with col5:
            prop_madre_superior = st.slider("Madre superior", 0.0, 1.0, 0.26, key="sim_mad")
        with col6:
            prop_padre_superior = st.slider("Padre superior", 0.0, 1.0, 0.21, key="sim_pad")
        with col7:
            prop_mujeres        = st.slider("Mujeres", 0.0, 1.0, 0.54, key="sim_muj")

        st.markdown("### Trayectoria histórica (opcional)")
        col1, col2, col3 = st.columns(3)
        with col1:
            puntaje_previo = st.number_input(
                "Puntaje cohorte anterior", 0.0, 500.0, 250.0, step=5.0
            )
        with col2:
            variacion_previa = st.number_input(
                "Variación entre las dos previas", -100.0, 100.0, 0.0
            )
        with col3:
            cohortes_previas = st.number_input("N° cohortes previas", 0, 20, 1)

        registro = {
            "log_n_estudiantes":   np.log1p(n_est),
            "prop_internet":       prop_internet,
            "prop_computador":     prop_computador,
            "prop_lavadora":       prop_lavadora,
            "prop_automovil":      prop_automovil,
            "prop_estrato_1_2":    prop_estrato_1_2,
            "prop_madre_superior": prop_madre_superior,
            "prop_padre_superior": prop_padre_superior,
            "prop_mujeres":        prop_mujeres,
            "puntaje_previo":      puntaje_previo if cohortes_previas > 0 else np.nan,
            "variacion_previa":    variacion_previa if cohortes_previas >= 2 else np.nan,
            "cohortes_previas":    cohortes_previas,
            "primera_cohorte":     int(cohortes_previas == 0),
            "area":                area,
            "calendario":          calendario,
            "caracter":            caracter,
            "naturaleza":          naturaleza,
            "departamento":        depto,
            "municipio":           muni,
            "aplicacion":          aplicacion,
        }
        entrada = asegurar_tipos(pd.DataFrame([registro]))
        puntaje_observado = puntaje_previo if cohortes_previas > 0 else None

    # --------------------------------------------------------
    # PREDICCIÓN (común a los 3 modos)
    # --------------------------------------------------------
    if entrada is not None:
        st.divider()
        st.markdown("### Selección de modelo")
        c1, c2 = st.columns(2)
        with c1:
            reg_nombre = st.selectbox("Modelo de regresión",
                                       list(modelos_reg.keys()), index=2)
        with c2:
            clf_nombre = st.selectbox("Modelo de clasificación",
                                       list(modelos_clf.keys()), index=2)

        if st.button("🔍 Calcular predicción y alerta",
                     type="primary", key="btn_predecir"):

            modelo_reg = modelos_reg[reg_nombre]
            modelo_clf = modelos_clf[clf_nombre]

            pred   = predecir_regresion(modelo_reg, entrada)
            proba  = predecir_clasificacion(modelo_clf, entrada)

            # ---------- MÉTRICAS ----------
            st.markdown("### 📊 Resultado")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Puntaje esperado", f"{pred:.1f}")
            with col2:
                if puntaje_observado is not None and not pd.isna(puntaje_observado):
                    delta = puntaje_observado - pred
                    st.metric("Puntaje observado (última cohorte)",
                              f"{puntaje_observado:.1f}",
                              delta=f"{delta:+.1f} vs esperado")

            if puntaje_observado is not None and not pd.isna(puntaje_observado):
                brecha = puntaje_observado - pred
                if brecha >= 20:
                    st.success(f"🟢 La sede está **{brecha:.0f} puntos por encima** de lo esperado.")
                elif brecha <= -20:
                    st.error(f"🔴 La sede está **{abs(brecha):.0f} puntos por debajo** de lo esperado.")
                else:
                    st.info(f"🟡 La sede está **en línea** con lo esperado ({brecha:+.0f} pts).")

            nivel, mensaje = nivel_riesgo(proba)
            st.metric("Probabilidad de deterioro en la siguiente cohorte",
                      f"{proba:.1%}")
            st.caption(f"Umbral de referencia: caída ≥ {abs(UMBRAL_B):.1f} puntos (percentil 25 histórico)")

            if nivel == "alto":
                st.error(mensaje)
            elif nivel == "medio":
                st.warning(mensaje)
            else:
                st.success(mensaje)

            # ---------- ACCIONES ----------
            st.markdown("### ✅ Acciones recomendadas")
            if puntaje_observado is not None and not pd.isna(puntaje_observado):
                with st.expander("Sobre desempeño esperado", expanded=True):
                    mostrar_acciones_regresion(puntaje_observado - pred)
            with st.expander("Sobre riesgo de deterioro", expanded=True):
                mostrar_acciones_clasificacion(proba)

            # ---------- EXPLICABILIDAD ----------
            st.markdown("### 🔎 ¿Qué variables influyeron más en la predicción?")
            explicar_shap_regresion(modelo_reg, entrada, reg_nombre)

            # ---------- HISTORIAL GRÁFICO ----------
            if historial_mostrado is not None and len(historial_mostrado) > 0:
                st.markdown("### 📈 Trayectoria observada")
                h = historial_mostrado.copy()
                if "punt_global_promedio" in h.columns:
                    h = h.rename(columns={"punt_global_promedio": "puntaje_observado"})
                st.line_chart(
                    h.sort_values("periodo").set_index("periodo")["puntaje_observado"],
                    height=240,
                )

            # ---------- GUARDAR (solo modo nueva sede) ----------
            if modo == "🆕 Registrar sede nueva":
                st.divider()
                if st.button("💾 Guardar esta sede nueva en la lista"):
                    nuevo = {
                        "codigo":     codigo,
                        "nombre":     nombre,
                        "jornada":    jornada,
                        "depto":      depto,
                        "municipio":  muni,
                        "n_cohortes": int(temporales["cohortes_previas"]),
                        "pred_reg":   round(pred, 2),
                        "proba_clf":  round(proba, 4),
                        "modelo_reg": reg_nombre,
                        "modelo_clf": clf_nombre,
                        "fecha":      pd.Timestamp.now().isoformat(timespec="seconds"),
                        "contexto":   registro,
                        "historial":  historial.to_dict("records"),
                    }
                    st.session_state["sedes_nuevas"].append(nuevo)
                    st.success("Sede guardada en la lista de sedes nuevas.")

    # --------------------------------------------------------
    # PANEL DE SEDES NUEVAS (visible siempre en este modo)
    # --------------------------------------------------------
    if modo == "🆕 Registrar sede nueva" and st.session_state.get("sedes_nuevas"):
        st.divider()
        st.markdown("### 📚 Sedes registradas en esta sesión")
        tabla = pd.DataFrame([{
            "Código":       s["codigo"],
            "Nombre":       s["nombre"],
            "Jornada":      s["jornada"],
            "Depto":        s["depto"],
            "N° cohortes":  s["n_cohortes"],
            "Puntaje esp.": s["pred_reg"],
            "Prob. det.":   f"{s['proba_clf']:.1%}",
            "Modelo reg.":  s["modelo_reg"],
        } for s in st.session_state["sedes_nuevas"]])
        st.dataframe(tabla, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            csv = tabla.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Descargar sedes nuevas (CSV)",
                data=csv,
                file_name="sedes_nuevas.csv",
                mime="text/csv",
            )
        with col2:
            if st.button("🗑️ Limpiar lista"):
                st.session_state["sedes_nuevas"] = []
                st.rerun()


# ============================================================
# TAB 3 · PRIORIZACIÓN
# ============================================================
with tab_prioriza:
    st.markdown("## Rankings de priorización")
    st.caption(
        "A partir del histórico consolidado, esta sección prioriza sedes que "
        "requieren acompañamiento y sedes referentes, según su desempeño "
        "relativo a su contexto."
    )

    @st.cache_data
    def calcular_rankings():
        base = cargar_base_modelo().copy()
        base = base.dropna(subset=["punt_global_promedio"]).copy()
        # Brecha: usar el mejor modelo (CatBoost) para predecir el esperado
        modelo_reg = modelos_reg["CatBoost"]
        X = base[NUMERICAS + TEMPORALES + CATEGORICAS]
        base["puntaje_esperado"] = modelo_reg.predict(X)
        base["brecha"] = base["punt_global_promedio"] - base["puntaje_esperado"]
        return base

    try:
        base_rank = calcular_rankings()
    except Exception as e:
        st.warning(f"No se pudo calcular el ranking: {e}")
        base_rank = None

    if base_rank is not None:
        # Filtros
        col1, col2, col3 = st.columns(3)
        with col1:
            deptos = ["(Todos)"] + sorted(base_rank["departamento"].dropna().unique())
            f_depto = st.selectbox("Departamento", deptos, key="pri_depto")
        with col2:
            f_nat = st.selectbox("Naturaleza",
                                 ["(Todas)", "OFICIAL", "NO OFICIAL"],
                                 key="pri_nat")
        with col3:
            f_per = st.selectbox("Periodo",
                                 ["(Todos)"] + sorted(base_rank["periodo"].astype(str).unique(),
                                                       reverse=True),
                                 key="pri_per")

        f = base_rank.copy()
        if f_depto != "(Todos)":     f = f[f["departamento"] == f_depto]
        if f_nat != "(Todas)":       f = f[f["naturaleza"] == f_nat]
        if f_per != "(Todos)":       f = f[f["periodo"].astype(str) == f_per]

        st.markdown("### 🚨 Sedes que requieren acompañamiento")
        st.caption("Bajo desempeño relativo (brecha negativa) en el contexto analizado.")
        cols_mostrar = ["cole_cod_dane_sede", "cole_jornada", "departamento",
                        "municipio", "punt_global_promedio", "puntaje_esperado",
                        "brecha", "n_estudiantes"]
        top_acomp = f.sort_values("brecha").head(50)[cols_mostrar]
        st.dataframe(top_acomp.round(2), use_container_width=True)

        st.download_button(
            "⬇️ Descargar ranking de acompañamiento",
            top_acomp.to_csv(index=False).encode("utf-8-sig"),
            file_name="ranking_acompanamiento.csv",
            mime="text/csv",
        )

        st.markdown("### 🌟 Sedes referentes")
        st.caption("Alto desempeño relativo (brecha positiva) en el contexto analizado.")
        top_ref = f.sort_values("brecha", ascending=False).head(50)[cols_mostrar]
        st.dataframe(top_ref.round(2), use_container_width=True)

        st.download_button(
            "⬇️ Descargar ranking de referentes",
            top_ref.to_csv(index=False).encode("utf-8-sig"),
            file_name="ranking_referentes.csv",
            mime="text/csv",
        )


# ============================================================
# TAB 4 · ANÁLISIS POR LOTE
# ============================================================
with tab_lote:
    st.markdown("## Análisis por lote")
    st.caption(
        "Carga un CSV con códigos DANE y jornadas para obtener la predicción "
        "de cada sede, sin ingresarlas una por una."
    )

    st.markdown("### Formato esperado del CSV")
    st.code(
        "cole_cod_dane_sede,cole_jornada\n"
        "105001000001,TARDE\n"
        "105001000043,MAÑANA",
        language="csv"
    )

    subido = st.file_uploader("Sube tu CSV", type=["csv"], key="lote_csv")
    if subido is not None:
        df_up = pd.read_csv(subido, dtype={"cole_cod_dane_sede": str})
        st.success(f"Se cargaron {len(df_up)} filas.")

        base_lote = cargar_base_modelo()
        base_lote["cole_cod_dane_sede"] = base_lote["cole_cod_dane_sede"].astype(str)

        df_up["cole_cod_dane_sede"] = df_up["cole_cod_dane_sede"].astype(str)
        df_up["cole_jornada"] = df_up["cole_jornada"].astype(str)

        ultima = (
            base_lote.sort_values("periodo")
            .groupby(["cole_cod_dane_sede", "cole_jornada"], as_index=False)
            .last()
        )
        merge = df_up.merge(
            ultima, on=["cole_cod_dane_sede", "cole_jornada"], how="left"
        )

        if merge[NUMERICAS].isna().all(axis=1).any():
            faltan = merge[merge[NUMERICAS].isna().all(axis=1)]
            st.warning(
                f"{len(faltan)} sedes del CSV no se encontraron en la base histórica "
                "y se omitirán."
            )
            merge = merge.dropna(subset=NUMERICAS)

        if len(merge) > 0:
            modelo_reg = modelos_reg["CatBoost"]
            modelo_clf = modelos_clf["Gradient Boosting"]

            X_reg = merge[NUMERICAS + TEMPORALES + CATEGORICAS]
            merge["puntaje_esperado"] = modelo_reg.predict(X_reg)
            merge["brecha"] = merge["punt_global_promedio"] - merge["puntaje_esperado"]

            X_clf = merge[FEATURES_B]
            if hasattr(modelo_clf, "predict_proba"):
                merge["proba_deterioro"] = modelo_clf.predict_proba(X_clf)[:, 1]
            else:
                sc = modelo_clf.decision_function(X_clf)
                merge["proba_deterioro"] = 1 / (1 + np.exp(-sc))

            salida = merge[[
                "cole_cod_dane_sede", "cole_jornada", "departamento",
                "municipio", "punt_global_promedio", "puntaje_esperado",
                "brecha", "proba_deterioro",
            ]].sort_values("brecha")
            salida["brecha"] = salida["brecha"].round(2)
            salida["puntaje_esperado"] = salida["puntaje_esperado"].round(2)
            salida["proba_deterioro"] = salida["proba_deterioro"].round(3)

            st.markdown("### Resultados")
            st.dataframe(salida, use_container_width=True)

            st.download_button(
                "⬇️ Descargar resultados completos",
                salida.to_csv(index=False).encode("utf-8-sig"),
                file_name="predicciones_lote.csv",
                mime="text/csv",
            )


# ------------------------------------------------------------
# PIE
# ------------------------------------------------------------
st.divider()
st.caption(
    "Modelo entrenado con datos públicos de Saber 11 (ICFES, cohortes 2014-2 a 2022-2). "
    "Esta herramienta es un apoyo al diagnóstico y no reemplaza la decisión de un analista."
)