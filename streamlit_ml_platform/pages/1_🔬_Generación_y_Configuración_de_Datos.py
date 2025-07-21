import streamlit as st
import pandas as pd
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

st.set_page_config(page_title="Generación de Datos", page_icon="🔬", layout="wide")

# --- Lógica de Rutas ---
try:
    from utils.paths import TFG_SINTETICO_PATH
except (ImportError, ModuleNotFoundError):
    st.error("No se pudo importar 'utils.paths'. Asegúrate de que la estructura es correcta.")
    TFG_SINTETICO_PATH = None

# --- Lógica de Backend ---
def run_generator(generator_type: str, hours: int):
    if not TFG_SINTETICO_PATH: return None
    st.session_state.is_generating = True
    
    module_name = "Generador_enriquecido" if generator_type == 'enriquecido' else "Generador_database"
    command = [sys.executable, "-m", module_name, f"--total-hours={hours}"]
    
    st.info(f"Ejecutando desde el directorio: `{TFG_SINTETICO_PATH}`")
    st.code(f"> {' '.join(command)}", language="bash")
    
    log_area = st.empty()
    log_content = []

    try:
        process = subprocess.Popen(command, cwd=TFG_SINTETICO_PATH, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', bufsize=1)
        with st.spinner(f"Generando datos ({hours} horas)..."):
            for line in iter(process.stdout.readline, ''):
                log_content.append(line)
                log_area.text_area("Log de Generación", "".join(log_content), height=300)
        if process.wait() == 0:
            st.success("¡Generación completada!")
            output_dir = TFG_SINTETICO_PATH / ('dataset_rul_variable_ENRIQUECIDO' if generator_type == 'enriquecido' else 'dataset_rul_variable_DEFINITIVO')
            latest_file = max(output_dir.glob('*.parquet'), key=os.path.getctime)
            return pd.read_parquet(latest_file)
        else:
            st.error(f"El proceso de generación falló.")
            return None
    except Exception as e:
        st.error(f"Ocurrió un error crítico: {e}")
        return None
    finally:
        st.session_state.is_generating = False

# --- Renderizado de la Página ---
st.title("🔬 Panel de Generación de Datos")
st.markdown("Crea datasets sintéticos para alimentar el pipeline de Machine Learning.")

if 'generated_dataset' not in st.session_state: st.session_state.generated_dataset = None
if 'is_generating' not in st.session_state: st.session_state.is_generating = False

with st.container(border=True):
    st.subheader("Configuración del Generador")
    generator_type = st.radio("Tipo de Generador:", ['enriquecido', 'database'], captions=["Ideal para ENTRENAR (más fallos).", "Ideal para VALIDAR (realista)."], horizontal=True)
    total_hours = st.slider("Horas de simulación a generar:", 1, 100, 10)

if st.button("🚀 Iniciar Generación", type="primary", disabled=st.session_state.is_generating, use_container_width=True):
    df = run_generator(generator_type, total_hours)
    if df is not None:
        st.session_state.generated_dataset = df
        st.rerun()

if st.session_state.generated_dataset is not None:
    st.subheader("📊 Vista Previa y Descarga")
    st.dataframe(st.session_state.generated_dataset.head())
    st.success(f"Dataset generado con {len(st.session_state.generated_dataset):,} filas. Ya está disponible en la página de Entrenamiento.")
    
    st.download_button(
        "📥 Descargar Dataset (Parquet)",
        data=st.session_state.generated_dataset.to_parquet(index=False),
        file_name=f"synthetic_dataset_{datetime.now().strftime('%Y%m%d_%H%M')}.parquet"
    )
