# config.py
"""
Configuración central del proyecto de mantenimiento predictivo industrial
"""
import os
from datetime import datetime

class Config:
    """Configuración principal del proyecto"""
        
    # Configuración general
    PROJECT_NAME = "Mantenimiento Predictivo Industrial"
    VERSION = "1.0.0"
    RANDOM_SEED = 42
    
    # Rutas del proyecto
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))    
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')
        
    # Configuración flexible de columnas (se detectarán automáticamente)
    COLUMN_MAPPING = {
        'timestamp': 'timestamp',
        'id_maquina': 'vehiculo_id', # Se usa para el muestreo en modo desarrollo
        'target_classification': 'estado_sistema', # Objetivo para la clasificación
        'RUL_TARGETS': {
        'aceite': 'rul_dias_aceite',
        'frenos': 'rul_dias_frenos',
        'refrigerante': 'rul_dias_refrigerante'
        }
    }
    
    # Configuración de sensores
    SENSOR_CONFIG = {
        'sampling_frequency_hz': 20,
        'sensor_columns': [
            'sensor_presion_aceite_bar',
            'sensor_presion_frenos_bar', 
            'sensor_presion_refrigerante_bar'
        ],
        'operational_columns': [
            'horas_operacion_acumuladas_componente_X',
            'temperatura_ambiente_celsius'
        ],
        'temporal_windows_hours': [1, 6, 24],  # Ventanas para feature engineering
        'disable_spectral_analysis': False
    }
    
    # Configuración de preprocesamiento
    PREPROCESSING_CONFIG = {
        'scaling': {
            'method': 'robust',  # Usar RobustScaler en lugar de StandardScaler
            'quantile_range': (5.0, 95.0),  # Rango para RobustScaler
            'with_centering': True,
            'with_scaling': True
        },
        'binning': {
            'n_bins': 10,
            'strategy': 'quantile',  # Usar quantile para mejor distribución
            'encode': 'ordinal',
            'numeric_features': [
                'sensor_presion_aceite_bar',
                'sensor_presion_frenos_bar',
                'sensor_presion_refrigerante_bar'
            ]
        }
    }

    # Configuración de selección de features
    FEATURE_SELECTION = {
        'variance_threshold': 0.0,       # Umbral para eliminar features con baja varianza
        'correlation_threshold': 1.0,    # Umbral para eliminar features altamente correlacionadas
        'k_best_features': 40,   
        'importance_sample_size': 200000 # Número de features a seleccionar con SelectKBest
    }
    
    # Configuración de modelos
    MODEL_CONFIG = {
        'classification': {
            'random_forest': {
                'n_estimators': 200,      
                'max_depth': 20,         
                'min_samples_split': 15, 
                'min_samples_leaf': 8,    
                'random_state': RANDOM_SEED,
                'n_jobs': -1,
                'class_weight': {
                    "Normal": 1,
                    "Advertencia": 10,    
                    "Critico": 35         
                }
            },
            'xgboost': {
                'n_estimators': 50,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': RANDOM_SEED,
                'n_jobs': -1
            },
            'logistic_regression': {
                'max_iter': 1000,
                'random_state': RANDOM_SEED
            },
            'decision_tree': {
                'max_depth': 10,
                'min_samples_split': 100,
                'random_state': RANDOM_SEED
            }
        },
        'regression': {
            'random_forest': {
                'n_estimators': 50,
                'max_depth': 10,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'random_state': RANDOM_SEED,
                'n_jobs': -1
            },
            'xgboost': {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'early_stopping_rounds': 10,
                'random_state': RANDOM_SEED,
                'n_jobs': -1,
                'warm_start': True
            },
            'linear_regression': {}
        }
    }
    
    # Configuración de métricas
    METRICS_CONFIG = {
        'classification': {
            'target_f1_macro': 0.70,
            'target_precision_critical': 0.75,
            'target_recall_critical': 0.80,
            'max_false_positive_rate': 0.15,
            'precision_at_k_values': [5, 10]
        },
        'regression': {
            'target_mae_days': 3.5,
            'target_r2': 0.60,
            'accuracy_bands_days': [1, 3, 7],
            'target_accuracy_3d': 0.70
        }
    }

    # Configuración específica para RUL (Remaining Useful Life)
    RUL_CONFIG = {
        'prognostic_horizon_threshold': 10, # Días para considerar un RUL válido
        'alpha_bounds': (0.8, 1.2), # Factor de ajuste para RUL
        'feature_selection_sample_size': 500000, # Tamaño de muestra para selección de features
        'prediction_ceiling_days': 15.0  # Límite de predicción para RUL
    }

    # Configuración de costos de negocio
    BUSINESS_CONFIG = {
        'operational_costs': {
            'false_positive_cost': 50,      # Costo de una inspección innecesaria
            'false_negative_cost': 1000,    # Costo de un fallo no detectado
            'sensor_monitoring_cost_per_day': 5
        },
        'maintenance_optimization': {
            'planned_maintenance_savings': 0.15,  # Ahorro del 15% por mantenimiento planificado
            'inventory_optimization': 0.20,       # Reducción del 20% en costos de inventario
            'downtime_reduction': 0.25            # Reducción del 25% en costos por tiempo de inactividad
        }
        # ### FIN DE LA CORRECCIÓN ###
    }
    
    # Configuración de visualización
    VISUALIZATION_CONFIG = {
        'figure_size': (12, 8),
        'dpi': 300,
        'style': 'seaborn-v0_8',
        'color_palette': 'viridis',
        'save_format': 'png'
    }
    
    # Development configuration
    DEVELOPMENT_CONFIG = {
        'development_mode': False,  # Activar modo desarrollo
        'id_column_for_sampling': 'vehiculo_id', 
        'sample_fraction_by_id': 0.1 
    }

    # Add development validation settings
    VALIDATION_CONFIG = {
        'gap_hours': 12,
        'n_splits': 2,
        'test_size': 0.2,
        'shuffle': False,  # Importante para series temporales
        'development_mode': {
            'use_smaller_splits': True,
            'skip_expensive_metrics': False, # Considerar si esto debe ser configurable por DEVELOPMENT_CONFIG
            'early_stopping_minutes': 10
        }
    }
    
    PROCESSING_MODES = {
        'development': {
            'description': 'Rápido para desarrollo e iteración',
            'max_dataset_size': 200000,        # Muestra máxima
            'use_all_features': True,          # Todas las features
            'expected_time_minutes': 5,
            'memory_limit_gb': 2
        },
        'balanced': {
            'description': 'Balance velocidad-precisión',
            'chunk_size': 500000,             # Chunks medianos
            'feature_reduction': 'moderate',   # Reducción moderada
            'expected_time_minutes': 30,
            'memory_limit_gb': 4
        },
        'optimized': {
            'description': 'Máxima velocidad para datasets grandes',
            'chunk_size': 200000,             # Chunks pequeños
            'feature_reduction': 'aggressive', # Máxima optimización
            'expected_time_minutes': 20,
            'memory_limit_gb': 3
        },
        'complete': {
            'description': 'Máxima precisión, procesamiento por chunks',
            'chunk_size': 300000,             # Chunks con overlap
            'overlap_samples': 20000,          # Overlap para rolling windows
            'use_all_features': True,          # Sin reducción de features
            'expected_time_minutes': 60,
            'memory_limit_gb': 6
        }
    }

    # Thresholds automáticos para selección de modo
    AUTO_MODE_SELECTION = {
        'thresholds': {
            'small': 100000,      # < 100K → development
            'medium': 1000000,    # 100K-1M → balanced  
            'large': 5000000,     # 1M-5M → optimized
            'huge': float('inf')  # >5M → complete with chunks
        },
        'force_mode': 'complete',  # Forzar modo específico (para testing)
        'consider_memory': True,  # Considerar memoria disponible
        'consider_time': True,    # Considerar tiempo disponible
    }

    # Configuración de features por modo
    FEATURE_CONFIGURATION = {
        'development': {
            'temporal_windows_hours': [1, 6, 24, 72],
            'use_spectral_features': True,
            'use_cycle_features': True,
            'use_advanced_degradation': True,
            'max_rolling_window_samples': 100000
        },
        'balanced': {
            'temporal_windows_hours': [1, 6, 24],
            'use_spectral_features': True,
            'use_cycle_features': True,
            'use_advanced_degradation': False,
            'max_rolling_window_samples': 50000
        },
        'optimized': {
            'temporal_windows_hours': [1, 6],
            'use_spectral_features': False,
            'use_cycle_features': False,
            'use_advanced_degradation': False,
            'max_rolling_window_samples': 20000
        },
        'complete': {
            'temporal_windows_hours': [1, 6, 24, 72, 168],  # Hasta 1 semana
            'use_spectral_features': True,
            'use_cycle_features': True,
            'use_advanced_degradation': True,
            'max_rolling_window_samples': 200000
        }
    }
    
    AUTO_MODEL_SELECTION_CONFIG = {
        'classification': {
            'key_metric': 'f1_macro', # Métrica a optimizar (debe existir en los logs y resultados)
            'goal': 'maximize'        # 'maximize' o 'minimize'
        },
        'rul': {
            'key_metric': 'MAE', # Usaremos el Error Absoluto Medio para RUL
            'goal': 'minimize'     # Queremos minimizar el error
        },
        # Modelos a usar si la selección automática falla por alguna razón
        'fallback': {
            'classification': 'random_forest',
            'rul': 'xgboost' # O el que consideres un buen baseline
        }
    }

    def get_optimal_processing_mode(dataset_size, available_memory_gb=None, target_time_minutes=None):
        """Determina automáticamente el modo óptimo de procesamiento"""
        
        if AUTO_MODE_SELECTION['force_mode']:
            return AUTO_MODE_SELECTION['force_mode']
        
        # Selección por tamaño
        if dataset_size < AUTO_MODE_SELECTION['thresholds']['small']:
            suggested_mode = 'development'
        elif dataset_size < AUTO_MODE_SELECTION['thresholds']['medium']:
            suggested_mode = 'balanced'
        elif dataset_size < AUTO_MODE_SELECTION['thresholds']['large']:
            suggested_mode = 'optimized'
        else:
            suggested_mode = 'complete'
        
        # Ajustar por memoria disponible
        if available_memory_gb and AUTO_MODE_SELECTION['consider_memory']:
            mode_memory = PROCESSING_MODES[suggested_mode]['memory_limit_gb']
            if available_memory_gb < mode_memory:
                # Downgrade a modo menos intensivo
                if suggested_mode == 'complete':
                    suggested_mode = 'optimized'
                elif suggested_mode == 'balanced':
                    suggested_mode = 'optimized'
        
        # Ajustar por tiempo objetivo
        if target_time_minutes and AUTO_MODE_SELECTION['consider_time']:
            mode_time = PROCESSING_MODES[suggested_mode]['expected_time_minutes']
            if target_time_minutes < mode_time:
                # Downgrade a modo más rápido
                if suggested_mode == 'complete':
                    suggested_mode = 'optimized'
                elif suggested_mode == 'balanced':
                    suggested_mode = 'optimized'
        
        return suggested_mode
    
    @classmethod
    def create_directories(cls):
        """Crear directorios del proyecto si no existen"""
        directories = [
            cls.MODELS_DIR,
            cls.RESULTS_DIR,
            cls.CHECKPOINTS_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print(f"✅ Directorios del proyecto creados en: {cls.BASE_DIR}")
    
    @classmethod
    def get_timestamp(cls):
        """Obtener timestamp para nombrado de archivos"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @classmethod
    def validate_config(cls):
        """Validar configuración del proyecto - versión robusta"""
        try:
            # Crear directorios
            cls.create_directories()
            
            # Verificar que existe algún archivo CSV en data/raw/
            if not os.path.exists(cls.RAW_DATA_DIR):
                print(f"❌ Directorio de datos no existe: {cls.RAW_DATA_DIR}")
                return False
            
            csv_files = [f for f in os.listdir(cls.RAW_DATA_DIR) if f.endswith('.csv')]
            
            if not csv_files:
                print(f"❌ No se encontraron archivos CSV en: {cls.RAW_DATA_DIR}")
                print(f"💡 Coloca tu dataset CSV en esta carpeta")
                return False
            
            # Si el archivo específico no existe, usar el primero disponible
            if not os.path.exists(cls.DATASET_PATH):
                if csv_files:
                    cls.DATASET_FILE = csv_files[0]
                    cls.DATASET_PATH = os.path.join(cls.RAW_DATA_DIR, cls.DATASET_FILE)
                    print(f"⚠️  Usando archivo encontrado: {cls.DATASET_FILE}")
                else:
                    print(f"❌ No se encontró archivo de datos: {cls.DATASET_PATH}")
                    return False
            
            print(f"✅ Configuración validada exitosamente")
            print(f"📊 Dataset: {cls.DATASET_PATH}")
            return True
            
        except Exception as e:
            print(f"❌ Error validando configuración: {str(e)}")
            return False
        
    # Configuración de balanceo de datos
    BALANCING_CONFIG = {
        'strategy_thresholds': {
            'small': 100_000,    # <100K: SMOTE completo
            'medium': 500_000,   # 100K-500K: SMOTE por lotes
            'large': 1_000_000,  # 500K-1M: Undersampling + SMOTE selectivo
            'huge': float('inf') # >1M: Solo undersampling conservador
        },
        'max_samples_per_class': {
            'small': None,       # Sin límite para datasets pequeños
            'medium': 50_000,    # Máximo por clase para datasets medianos
            'large': 30_000,     # Máximo por clase para datasets grandes
            'huge': 25_000      # Máximo conservador para datasets enormes
        },
        'smote_params': {
            'small': {'k_neighbors': 5, 'sampling_strategy': 'auto'},
            'medium': {'k_neighbors': 4, 'sampling_strategy': 'auto'},
            'large': {'k_neighbors': 3, 'sampling_strategy': 'minority'},
            'huge': {'k_neighbors': 3, 'sampling_strategy': 'minority'}
        },
        'rare_class_threshold': 1000,  # Umbral para considerar una clase como rara
        'critical_class_threshold': 500,  # Umbral para SMOTE en clases críticas
        'batch_size': {
            'small': None,      # Sin batcheo para datasets pequeños
            'medium': 50_000,   # Tamaño de batch para datasets medianos
            'large': 30_000,    # Tamaño de batch para datasets grandes
            'huge': 25_000     # Tamaño de batch conservador
        },
        'memory_safety_factor': 0.25,  # Factor de seguridad para uso de memoria
        'fallback_enabled': True,      # Habilitar degradación automática
        'preserve_minority_classes': True,  # Preservar clases minoritarias completas
        'logging_enabled': True  # Habilitar logging detallado
    }

    CLASSIFICATION_CONFIG = {
        'critical_fault_classes': ['Critico']    
        }

    @classmethod
    def validate_config_sqlite(cls):
        """Validar configuración específica para SQLite"""
        try:
            from src.utils.data_adapter import DataAdapter

            # Intentar crear adaptador
            data_adapter = DataAdapter()
            if data_adapter.source_type != 'sqlite':
                raise ValueError("DataAdapter no está configurado para SQLite")
            info = data_adapter.get_database_info()

            print(f"✅ SQLite configurado exitosamente")
            print(f"📊 Base de datos: {data_adapter.data_path}")
            print(f"📈 Registros disponibles: {info.get('total_records', 0):,}")
            print(f"📅 Rango fechas: {info.get('date_range', 'N/A')}")

            return True, data_adapter

        except Exception as e:
            print(f"❌ Error configurando SQLite: {str(e)}")

            if cls.DATA_CONFIG['fallback_to_csv']:
                print("🔄 Intentando fallback a CSV...")
                return cls.validate_config(), None
            else:
                return False, None
        
# Crear instancia global de configuración
config = Config()
