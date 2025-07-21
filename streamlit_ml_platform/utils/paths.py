# utils/paths.py (VERSIÓN FINAL CORREGIDA)
import os
from pathlib import Path

# 1. Raíz del Proyecto Completo (TFGFinal)
# Sube dos niveles desde este archivo (utils/ -> streamlit_ml_platform/ -> TFGFinal/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 2. Ruta al proyecto del Generador Sintético
# El directorio de trabajo (CWD) para ejecutar el generador.
TFG_SINTETICO_PATH = PROJECT_ROOT / 'TFGSintetico'

# 3. Ruta al proyecto de Modelos
# --- CORRECCIÓN CLAVE ---
# Eliminamos '_final' para que la ruta coincida con tu estructura real.
TFG_MODELS_PROJECT_PATH = PROJECT_ROOT / 'TFGModels'

# 4. Verificación (opcional pero útil para depurar)
# Puedes descomentar estas líneas para ver si las rutas son correctas en la terminal.
# print(f"✅ [paths.py] PROJECT_ROOT: {PROJECT_ROOT}")
# print(f"✅ [paths.py] TFG_SINTETICO_PATH: {TFG_SINTETICO_PATH}")
# print(f"✅ [paths.py] TFG_MODELS_PROJECT_PATH: {TFG_MODELS_PROJECT_PATH}")
