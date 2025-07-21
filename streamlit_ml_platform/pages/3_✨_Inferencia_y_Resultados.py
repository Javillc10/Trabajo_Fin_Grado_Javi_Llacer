import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys
import subprocess
import tempfile
from pathlib import Path

st.set_page_config(page_title="Inferencia y Resultados", page_icon="✨", layout="wide")

# --- Lógica de Rutas y Backend ---
try:
    from utils.paths import TFG_MODELS_PROJECT_PATH
except (ImportError, ModuleNotFoundError):
    st.error("No se pudo importar 'utils.paths'.")
    TFG_MODELS_PROJECT_PATH = None

@st.cache_data(ttl=300)
def find_completed_sessions():
    if not TFG_MODELS_PROJECT_PATH: return []
    results_dir = TFG_MODELS_PROJECT_PATH / "results"
    return sorted([p.name for p in results_dir.iterdir() if p.is_dir() and (p / "training_log.log").exists()], reverse=True)

def run_inference(model_session_id: str, data_path: str):
    if not TFG_MODELS_PROJECT_PATH: return None
    
    # Ensure we're running from the TFGModels directory
    working_dir = str(TFG_MODELS_PROJECT_PATH)
    command = [sys.executable, "inference.py", f"--model-session={model_session_id}", f"--input-file={data_path}"]
    st.code(f"> {' '.join(command)}", language="bash")
    
    try:
        # Run command in the correct working directory
        result = subprocess.run(
            command,
            cwd=working_dir,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Log output for debugging
        if result.stdout:
            st.code(result.stdout, language="text")
        if result.stderr:
            st.error(result.stderr)
            
        # Check for output file
        output_path = TFG_MODELS_PROJECT_PATH / "results" / "inference_output" / f"dashboard_data_{model_session_id}.parquet"
        if output_path.exists():
            return pd.read_parquet(output_path)
            
        st.error("No se generó el archivo de resultados")
        return None
        
    except subprocess.CalledProcessError as e:
        st.error(f"Error al ejecutar inferencia (código {e.returncode}):")
        st.code(e.stderr, language="text")
        return None
    except Exception as e:
        st.error(f"Error inesperado: {str(e)}")
        return None

# --- Renderizado de la Página ---
st.title("✨ Panel de Inferencia y Visualización de Resultados")

if 'inference_df' not in st.session_state: st.session_state.inference_df = None

with st.container(border=True):
    st.subheader("Configuración de la Inferencia")
    completed_sessions = find_completed_sessions()
    if not completed_sessions:
        st.error("No se han encontrado sesiones de entrenamiento válidas. Entrena un modelo primero.")
        st.stop()
    
    inference_session = st.selectbox("1. Elige la sesión de modelos a utilizar:", completed_sessions)
    uploaded_file_infer = st.file_uploader("2. Sube un dataset para predecir (formato Parquet):", type=['parquet'])

if st.button("▶️ Ejecutar Inferencia", type="primary", disabled=not uploaded_file_infer, use_container_width=True):
    with st.spinner("Ejecutando inferencia..."):
        try:
            # Create output directory if it doesn't exist
            output_dir = Path(TFG_MODELS_PROJECT_PATH) / "results" / "inference_output"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp:
                tmp.write(uploaded_file_infer.getvalue())
                tmp.flush()
                tmp.close()
                
                st.info(f"Archivo temporal creado en: {tmp.name}")
                
                df_resultados = run_inference(model_session_id=inference_session, data_path=tmp.name)
                
                if df_resultados is not None:
                    st.session_state.inference_df = df_resultados
                    st.rerun()
                else:
                    st.error("La inferencia no devolvió resultados. Verifica los logs para más detalles.")
        except FileNotFoundError as e:
            st.error(f"Error: No se encontró el archivo o directorio requerido: {e}")
        except PermissionError as e:
            st.error(f"Error de permisos: {e}")
        except Exception as e:
            st.error(f"Error inesperado durante la inferencia: {str(e)}")
        finally:
            try:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
            except Exception as e:
                st.warning(f"No se pudo eliminar el archivo temporal: {e}")

# --- Visualización de Resultados en Pestañas ---
if st.session_state.inference_df is not None:
    st.divider()
    st.header("Resultados de la Inferencia")
    df = st.session_state.inference_df
    
    tab1, tab2 = st.tabs(["📈 Vista Gerencial", "🔧 Vista Técnica"])
    
    with tab1:
        # --- Dashboard Gerencial ---
        st.subheader("KPIs Clave y Análisis de Riesgo")
        ultimo_estado = df.iloc[-1]
        col1, col2, col3 = st.columns(3)
        col1.metric("Estado General Actual", ultimo_estado.get('predicted_estado', 'N/A'))
        alertas_altas = (df['prioridad'] == 'ALTA').sum()
        col2.metric("Nº Alertas Críticas", f"{alertas_altas}")
        coste_evitado = alertas_altas * 10000 # Simulación
        col3.metric("Coste Evitado (Estimado)", f"€{coste_evitado:,.0f}")
        
        if 'prioridad' in df.columns:
            prioridad_counts = df['prioridad'].value_counts()
            fig_donut = px.pie(prioridad_counts, values=prioridad_counts.values, names=prioridad_counts.index, hole=.4, color_discrete_map={'ALTA':'#d9534f', 'MEDIA':'#f0ad4e', 'BAJA':'#5cb85c'})
            st.plotly_chart(fig_donut, use_container_width=True)

    with tab2:
        # --- Dashboard Técnico ---
        st.subheader("Análisis Detallado para Mantenimiento")
        componentes_disponibles = [c.replace('predicted_rul_', '') for c in df.columns if 'predicted_rul_' in c]
        if componentes_disponibles:
            comp_sel = st.selectbox("Analizar RUL del componente:", componentes_disponibles)
            fig_line = px.line(df, x='timestamp', y=f'predicted_rul_{comp_sel}', color='prioridad', title=f'Historial de RUL para {comp_sel.upper()}')
            st.plotly_chart(fig_line, use_container_width=True)
        
        with st.expander("Ver registro de eventos completo"):
            st.dataframe(df)
