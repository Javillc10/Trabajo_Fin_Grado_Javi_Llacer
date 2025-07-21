import streamlit as st
import pandas as pd
import os
import sys
import subprocess
import tempfile
import re
from datetime import datetime
from pathlib import Path
import time

st.set_page_config(page_title="Entrenamiento y Métricas", page_icon="🧠", layout="wide")

# --- Lógica de Rutas ---
try:
    from utils.paths import TFG_MODELS_PROJECT_PATH
except (ImportError, ModuleNotFoundError):
    st.error("No se pudo importar 'utils.paths'.")
    TFG_MODELS_PROJECT_PATH = None

# --- Lógica de Backend ---
def run_training(dataset_path: str, session_id: str):
    # (Tu función run_training se queda aquí, sin cambios)
    if not TFG_MODELS_PROJECT_PATH: return False
    st.session_state.is_training = True
    st.session_state.training_log = [f"Iniciando entrenamiento para sesión: '{session_id}'...\n"]
    command = [sys.executable, "main.py", f"--train-data={dataset_path}", f"--session-id={session_id}"]
    st.code(f"> {' '.join(command)}", language="bash")
    log_area = st.empty()
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.Popen(command, cwd=TFG_MODELS_PROJECT_PATH, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', bufsize=1, env=env)
        log_content = []
        with st.spinner(f"Entrenando modelos..."):
            for line in iter(process.stdout.readline, ''):
                log_content.append(line)
                log_area.text_area("Log de Entrenamiento", "".join(log_content), height=400)
        st.session_state.training_log = log_content
        return process.wait() == 0
    except Exception as e:
        st.error(f"Error crítico al lanzar el entrenamiento: {e}")
        return False
    finally:
        st.session_state.is_training = False

# --- Renderizado de la Página ---
st.title("🧠 Panel de Entrenamiento y Evaluación de Modelos")

if 'is_training' not in st.session_state: st.session_state.is_training = False
if 'last_trained_session' not in st.session_state: st.session_state.last_trained_session = None

with st.container(border=True):
    st.subheader("Configuración del Entrenamiento")
    
    # Selección de Dataset
    dataset_options = {}
    if st.session_state.get('generated_dataset') is not None:
        dataset_options["✅ Dataset Generado en Paso 1"] = st.session_state.generated_dataset
    uploaded_file = st.file_uploader("O sube un nuevo archivo Parquet:", type=['parquet'], key="train_uploader")
    if uploaded_file:
        dataset_options[f"📂 {uploaded_file.name}"] = pd.read_parquet(uploaded_file)
    
    if not dataset_options:
        st.warning("No hay datasets disponibles. Genera uno en la página anterior o sube un archivo.")
        st.stop()

    selected_option = st.selectbox("Elige el dataset para entrenar:", options=list(dataset_options.keys()))
    df_to_train = dataset_options[selected_option]

    # Nombre de la sesión
    session_name = st.text_input("Dale un nombre a esta sesión de entrenamiento:", "entrenamiento_web")

if st.button("🚀 Iniciar Entrenamiento", ...):
    safe_name = re.sub(r'[^\w\-. ]', '', session_name).strip().replace(' ', '_')
    final_session_id = f"{datetime.now().strftime('%Y%m%d')}_{safe_name}"
    
    # Creamos un puntero para el nombre del archivo temporal
    temp_file_path = None
    try:
        # Usamos el gestor de contexto para crear y escribir en el archivo
        with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp:
            df_to_train.to_parquet(tmp.name, index=False)
            temp_file_path = tmp.name # Guardamos la ruta para usarla después
        
        # Ahora que hemos salido del bloque 'with', el archivo está cerrado y liberado.
        # Ya podemos llamar a la función de entrenamiento de forma segura.
        success = run_training(dataset_path=temp_file_path, session_id=final_session_id)
        
        if success:
            st.session_state.last_trained_session = final_session_id
            st.rerun()
            
    finally:
        # Este bloque se ejecutará SIEMPRE, incluso si hay un error.
        # Es el lugar perfecto para limpiar archivos temporales.
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            print(f"Archivo temporal {temp_file_path} eliminado correctamente.")

# --- Dashboard de Resultados del Entrenamiento ---
if st.session_state.last_trained_session:
    st.divider()
    session_id = st.session_state.last_trained_session
    st.header(f"📊 Resultados para la Sesión: `{session_id}`")
    session_path = TFG_MODELS_PROJECT_PATH / "results" / session_id
    
    if not session_path.exists():
        st.error(f"No se encontró la carpeta de resultados para la sesión '{session_id}'.")
    else:
        st.info("Mostrando artefactos generados por el pipeline de entrenamiento.")
        image_files = sorted(list(session_path.glob("*.png")))
        
        if not image_files:
            st.warning("No se encontraron gráficos (.png) en la carpeta de resultados.")
        else:
            for img_path in image_files:
                st.image(str(img_path), caption=img_path.name, use_container_width=True)
        
        log_file = session_path / "training_log.log"
        if log_file.exists():
            with st.expander("Ver Log de Entrenamiento Completo"):
                st.text(log_file.read_text(encoding="utf-8"))
