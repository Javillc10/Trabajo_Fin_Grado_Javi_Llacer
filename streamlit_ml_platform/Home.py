# Home.py (La Guía de Inicio Rápido para la Arquitectura de Paneles)
import streamlit as st

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Guía de Uso de la Plataforma",
    page_icon="🚀",
    layout="wide"
)

# --- CSS Personalizado ---
st.markdown("""
<style>
    .step-box {
        border: 2px solid #e6e6e6; border-radius: 10px; padding: 25px;
        margin-bottom: 25px; background-color: #f8f9fa;
    }
    .step-icon { font-size: 50px; line-height: 1; }
    .step-title { font-size: 24px; font-weight: bold; color: #007BFF; }
</style>
""", unsafe_allow_html=True)

# --- Encabezado ---
st.title("🚀 Bienvenido a la Plataforma de Mantenimiento Predictivo")
st.header("Guía Rápida para el Ciclo Completo de Machine Learning")
st.info(
    "Esta plataforma te permite controlar cada fase del pipeline de mantenimiento predictivo. "
    "Sigue estos pasos para ir de los datos en bruto a los resultados accionables."
)
st.divider()

# --- Flujo de Trabajo Explicado ---
st.header("El Flujo de Trabajo: De Datos a Decisiones")

# --- PASO 1: GENERAR DATOS ---
with st.container(border=True):
    col1, col2 = st.columns([0.1, 0.9])
    with col1:
        st.markdown('<p class="step-icon">🔬</p>', unsafe_allow_html=True)
    with col2:
        st.markdown('<p class="step-title">Paso 1: Generación de Datos</p>', unsafe_allow_html=True)
        st.markdown(
            """
            Todo empieza con los datos. En este panel, puedes crear datasets sintéticos a medida.
            
            - **Tu Control:** Ajusta el tipo de generador (`enriquecido` para entrenar, `database` para validar) y el volumen de datos (horas de simulación).
            - **Acción:** Navega a la página **`1_🔬_Generación_de_Datos`** desde el menú lateral para empezar.
            """
        )

# --- PASO 2: ENTRENAR EL MODELO ---
with st.container(border=True):
    col1, col2 = st.columns([0.1, 0.9])
    with col1:
        st.markdown('<p class="step-icon">🧠</p>', unsafe_allow_html=True)
    with col2:
        st.markdown('<p class="step-title">Paso 2: Entrenamiento y Evaluación</p>', unsafe_allow_html=True)
        st.markdown(
            """
            Con los datos listos, es hora de entrenar los modelos de inteligencia artificial.
            
            - **Tu Control:** Selecciona el dataset a usar, dale un nombre único a tu sesión de entrenamiento y lanza el proceso.
            - **Resultado:** Al finalizar, la página mostrará un **dashboard de métricas** con gráficos (matrices de confusión, importancia de features, etc.) específicos de esa sesión.
            - **Acción:** Ve a la página **`2_🧠_Entrenamiento_y_Métricas`**.
            """
        )

# --- PASO 3: EJECUTAR INFERENCIA Y VER RESULTADOS ---
with st.container(border=True):
    col1, col2 = st.columns([0.1, 0.9])
    with col1:
        st.markdown('<p class="step-icon">✨</p>', unsafe_allow_html=True)
    with col2:
        st.markdown('<p class="step-title">Paso 3: Inferencia y Dashboards de Resultados</p>', unsafe_allow_html=True)
        st.markdown(
            """
            Este es el paso final: usar un modelo entrenado para hacer predicciones sobre datos nuevos y visualizar los resultados.
            
            - **Tu Control:** Elige una sesión de entrenamiento guardada y un dataset de prueba.
            - **Resultado:** Tras ejecutar la inferencia, la misma página se convertirá en un **panel de visualización con dos pestañas**: una vista de alto nivel para gerencia y una vista técnica para mantenimiento.
            - **Acción:** Ve a la página **`3_✨_Inferencia_y_Resultados`**.
            """
        )

st.divider()
st.success("¡Todo listo! Ya puedes empezar a explorar el pipeline completo usando el menú de la izquierda.")
