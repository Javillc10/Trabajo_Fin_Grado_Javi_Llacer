# Sistema de Mantenimiento Predictivo Industrial

Plataforma integral para el mantenimiento predictivo en entornos industriales que combina:
- Generación de datos sintéticos
- Modelado de machine learning
- Interfaz interactiva de visualización

## Estructura del Proyecto

```
TFG/
├── streamlit_ml_platform/          # Interfaz web interactiva
│   ├── Home.py                     # Página principal
│   ├── pages/                      # Páginas específicas
│   │   ├── 1_🔬_Generación_y_Configuración_de_Datos.py
│   │   ├── 2_🧠_Entrenamiento_y_Métricas.py
│   │   ├── 3_✨_Inferencia_y_Resultados.py
│   │   └── 4_🔮_Interpretabilidad_del_Modelo.py
│   └── utils/                      # Utilidades
│       └── paths.py
│
├── TFGModels/                      # Núcleo de machine learning
│   ├── config.py                   # Configuración global
│   ├── main.py                     # Pipeline principal
│   ├── inference.py                # Lógica de inferencia
│   ├── split_data.py               # Manejo de datos
│   └── src/                        # Módulos del pipeline
│       ├── _01_EDA_and_Preprocessing.py
│       ├── _02_Fault_Classification_Models.py
│       ├── _03_RUL_Estimation_Models.py
│       ├── _04_Model_Interpretation_and_Business_Impact.py
│       └── utils/                  # Utilidades ML
│           ├── checkpoint_manager.py
│           ├── data_adapter.py
│           ├── feature_engineering.py
│           ├── training_logger.py
│           └── unified_metrics.py
│
└── TFGSintetico/                   # Generación de datos sintéticos
    ├── Generador_database.py       # Generador base
    └── Generador_enriquecido.py    # Generador mejorado
```

## Requisitos

- Python 3.10+
- Dependencias listadas en `requirements.txt`

Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Funcionalidades Principales

### 1. Pipeline de Machine Learning
- **Preprocesamiento avanzado**: Feature engineering y preparación de datos
- **Modelado dual**:
  - Clasificación de fallos (XGBoost, LightGBM)
  - Estimación de RUL (Remaining Useful Life)
- **Interpretabilidad**: SHAP values y análisis de impacto

### 2. Interfaz Streamlit
- Visualización interactiva de resultados
- Control del pipeline completo
- Análisis de métricas y modelos

### 3. Generación de Datos Sintéticos
- Simulación de escenarios industriales
- Configuración flexible de parámetros
- Exportación a múltiples formatos

## Uso

### Ejecutar el pipeline ML
```bash
python TFGModels/main.py --mode train --train-data datos_entrenamiento.parquet
```

### Lanzar la interfaz Streamlit
```bash
streamlit run streamlit_ml_platform/Home.py
```

### Generar datos sintéticos
```bash
python TFGSintetico/Generador_enriquecido.py #11% de fallos
python TFGSintetico/Generador_database.py # 0,04% de fallos

```

## Configuración

El archivo `TFGModels/config.py` contiene:
- Parámetros globales del proyecto
- Mapeo de columnas y variables
- Configuración de modelos
- Rutas de almacenamiento

