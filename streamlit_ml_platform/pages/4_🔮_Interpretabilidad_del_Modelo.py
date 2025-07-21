import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import os
import numpy as np # Import numpy for array checks
import matplotlib.pyplot as plt # For bar plot fallback

try:
    import streamlit_shap as st_shap
    import shap
except ImportError:
    st.error("Error: Los paquetes streamlit-shap o shap no están instalados. Ejecuta: pip install streamlit-shap shap")
    st.stop()

st.set_page_config(page_title="Interpretabilidad de Modelos (XAI)", page_icon="🔮", layout="wide")

# --- Lógica de Rutas y Carga de Datos ---
try:
    from utils.paths import TFG_MODELS_PROJECT_PATH
except (ImportError, ModuleNotFoundError):
    st.error("No se pudo importar 'utils.paths'.")
    TFG_MODELS_PROJECT_PATH = None

@st.cache_data(ttl=300)
def find_sessions_with_shap():
    """Busca sesiones que contengan un análisis SHAP completo."""
    if not TFG_MODELS_PROJECT_PATH: return []
    results_dir = TFG_MODELS_PROJECT_PATH / "results"
    valid_sessions = []
    if results_dir.exists():
        for session_dir in results_dir.iterdir():
            if session_dir.is_dir():
                # Check both possible SHAP locations and naming patterns
                shap_dir = session_dir / "interpretability_analysis"
                if not shap_dir.exists():
                    shap_dir = session_dir / "shap_analysis"
                
                if (shap_dir / "shap_explainer_xgboost.joblib").exists() or \
                   (shap_dir / "shap_explainer_clf.joblib").exists():
                    valid_sessions.append(session_dir.name)
    return sorted(valid_sessions, reverse=True)

@st.cache_data
def load_shap_data(session_id):
    """Carga todos los 'ingredientes' SHAP para una sesión dada."""
    if not TFG_MODELS_PROJECT_PATH: return None
    
    # Try both possible directory structures
    shap_dir = TFG_MODELS_PROJECT_PATH / "results" / session_id / "interpretability_analysis"
    if not shap_dir.exists():
        shap_dir = TFG_MODELS_PROJECT_PATH / "results" / session_id / "shap_analysis"
    
    try:
        # Try xgboost naming pattern first
        explainer_path = shap_dir / 'shap_explainer_xgboost.joblib'
        shap_values_path = shap_dir / 'shap_values_xgboost.joblib'
        sample_path = shap_dir / 'X_shap_sample_xgboost.parquet'
        
        if not explainer_path.exists():
            # Fall back to clf naming pattern
            explainer_path = shap_dir / 'shap_explainer_clf.joblib'
            shap_values_path = shap_dir / 'shap_values_clf.joblib'
            sample_path = shap_dir / 'X_test_sample_clf.parquet'
        
        explainer = joblib.load(explainer_path)
        shap_values = joblib.load(shap_values_path)
        X_test_sample = pd.read_parquet(sample_path)
        return explainer, shap_values, X_test_sample
        
    except FileNotFoundError as e:
        st.error(f"No se encontraron los archivos SHAP en: {shap_dir}")
        st.error(f"Error detallado: {str(e)}")
        return None, None, None

# --- Renderizado de la Página ---
st.title("🔮 Interpretabilidad de Modelos (Explainable AI)")
st.markdown("Analiza *por qué* el modelo toma sus decisiones. Esta sección te permite 'abrir la caja negra'.")

sessions = find_sessions_with_shap()
if not sessions:
    st.warning("No se encontraron sesiones de entrenamiento con análisis de interpretabilidad guardado.")
    st.info("Asegúrate de que el pipeline de entrenamiento se ha ejecutado y ha guardado los artefactos SHAP.")
    st.stop()

selected_session = st.selectbox("Selecciona una sesión de entrenamiento para analizar:", sessions)

if selected_session:
    explainer, shap_values, X_test_sample = load_shap_data(selected_session)

    if explainer is not None:
        st.header("Análisis Global de Features")
        st.markdown("Este gráfico muestra el impacto de cada feature en las predicciones del modelo. Las features en la parte superior son las más importantes.")
        
        st_shap.st_shap(shap.summary_plot(shap_values, X_test_sample))

        st.divider()

        st.header("Análisis de Predicción Individual")
        st.markdown("Selecciona una predicción específica para entender qué factores la impulsaron.")
        
        # Slider para seleccionar una predicción de la muestra
        instance_index = st.slider(
            "Selecciona un punto de datos para analizar:", 
            0, len(X_test_sample) - 1, 0
        )

        if instance_index is not None:
            st.subheader(f"Análisis para la Predicción #{instance_index}")
            
            # --- FIX: Ensure base_value is a scalar and create Explanation object for force plot ---
            # Ensure base_value_to_use is a scalar
            if isinstance(explainer.expected_value, (list, np.ndarray)):
                if isinstance(explainer.expected_value, np.ndarray) and explainer.expected_value.ndim == 0:
                    base_value_to_use = explainer.expected_value.item() # Extract scalar from 0-dim array
                elif len(explainer.expected_value) > 1:
                    base_value_to_use = explainer.expected_value[0] # For multi-output, take the first class
                else: # Handles single-element list or 1-element numpy array
                    base_value_to_use = explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value
            else:
                base_value_to_use = explainer.expected_value # Already a scalar float

            # Check if shap_values is 3D (samples, features, outputs)
            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                # If it's 3D, select the SHAP values for the specific instance and the first output (class 0)
                shap_values_to_use = shap_values[instance_index, :, 0]
            else:
                # If it's 2D (samples, features), select the SHAP values for the specific instance
                shap_values_to_use = shap_values[instance_index, :]

            # Get feature values for the selected instance
            features_to_use = X_test_sample.iloc[instance_index, :]
            feature_names_to_use = X_test_sample.columns.tolist()

            # Debug output for SHAP values
            with st.expander("Debug Info (SHAP Values)"):
                st.write(f"SHAP values type: {type(shap_values_to_use)}")
                if hasattr(shap_values_to_use, 'shape'):
                    st.write(f"SHAP values shape: {shap_values_to_use.shape}")
                st.write(f"Base value: {base_value_to_use} (type: {type(base_value_to_use)})")
                st.write(f"Features shape: {features_to_use.values.shape}")

            # For multi-class models, let user select which class to visualize
            if len(shap_values_to_use.shape) == 2 and shap_values_to_use.shape[1] > 1:
                class_idx = st.selectbox("Select class to visualize:", 
                                       range(shap_values_to_use.shape[1]),
                                       format_func=lambda x: f"Class {x}")
                shap_values_to_use = shap_values_to_use[:, class_idx]
                if isinstance(base_value_to_use, (list, np.ndarray)):
                    base_value_to_use = base_value_to_use[class_idx]

            # Check if we already have an Explanation object
            if isinstance(shap_values_to_use, shap.Explanation):
                explanation_instance = shap_values_to_use
            else:
                # Create a new Explanation object for the single instance
                explanation_instance = shap.Explanation(
                    values=np.array([shap_values_to_use]),  # Wrap in array for single instance
                    base_values=base_value_to_use,
                    data=np.array([features_to_use.values]),  # Wrap in array
                    feature_names=feature_names_to_use
                )

            # Try different visualization methods
            try:
                # First ensure we have a valid Explanation object
                if not isinstance(explanation_instance, shap.Explanation):
                    raise ValueError("Invalid SHAP explanation format")
                
                # Configure matplotlib backend
                import matplotlib
                matplotlib.use('agg')  # Set non-interactive backend
                
                # Handle different Explanation shapes
                if len(explanation_instance.shape) == 3:  # (samples, features, outputs)
                    plot_data = explanation_instance[0,:,0]  # First sample, all features, first output
                elif len(explanation_instance.shape) == 2:  # (samples, features)
                    plot_data = explanation_instance[0,:]    # First sample, all features
                else:  # Single sample
                    plot_data = explanation_instance
                
                # Create figure explicitly
                fig, ax = plt.subplots()
                shap.plots.waterfall(plot_data, show=False)
                st.pyplot(fig)
                    
            except Exception as e1:
                st.warning(f"Waterfall plot failed: {str(e1)}")
                try:
                    # Try simple bar plot of absolute SHAP values
                    fig, ax = plt.subplots()
                    shap.plots.bar(explanation_instance, show=False)
                    st.pyplot(fig)
                except Exception as e2:
                    st.warning(f"Bar plot failed: {str(e2)}")
                    try:
                        # Fallback to simple force text display
                        st.text(shap.force_text(
                            base_value_to_use,
                            shap_values_to_use,
                            features_to_use.values,
                            feature_names=feature_names_to_use
                        ))
                    except Exception as e3:
                        st.error("All visualizations failed. Showing raw SHAP values:")
                        st.write("Base value:", float(base_value_to_use))
                        st.write("Feature contributions:")
                        
                        # Extract raw values if it's an Explanation object
                        if isinstance(shap_values_to_use, shap.Explanation):
                            raw_values = shap_values_to_use.values
                        else:
                            raw_values = shap_values_to_use
                            
                        contrib_df = pd.DataFrame({
                            'Feature': feature_names_to_use,
                            'Value': features_to_use.values,
                            'SHAP Value': np.abs(raw_values) if hasattr(raw_values, '__abs__') else raw_values
                        })
                        st.dataframe(contrib_df.sort_values('SHAP Value', ascending=False))
            # --- END FIX ---

            st.markdown("""
            **Cómo leer este gráfico:**
            - **`base value`**: Es la predicción promedio del modelo sobre todo el dataset.
            - **Flechas rojas (features en rosa)**: Impulsan la predicción hacia un valor más alto (ej. hacia 'Crítico').
            - **Flechas azules (features en azul)**: Impulsan la predicción hacia un valor más bajo (ej. hacia 'Normal').
            - La longitud de la flecha indica la magnitud del impacto de esa feature.
            """)
            
            with st.expander("Ver los valores de las features para esta predicción"):
                st.dataframe(X_test_sample.iloc[[instance_index]])
