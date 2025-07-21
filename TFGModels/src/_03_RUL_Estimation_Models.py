# EN: src/_03_RUL_Estimation_Models.py
# REEMPLAZA EL FICHERO ENTERO CON ESTE CÓDIGO FINAL

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 

from src.utils.model_persistence import ModelPersistence
from src.utils.unified_metrics import UnifiedMetrics
from src.utils.validation_strategies import TemporalValidator
from src.utils.feature_selection import FeatureSelector

class RULEstimator:
    def __init__(self, config, logger, session_id=None):
        self.config = config
        self.logger = logger
        self.session_id = session_id
        self.models = {}
        self.trained_feature_columns = []
        self.validator = TemporalValidator(config, self.logger)
        self.metrics_calculator = UnifiedMetrics(config, self.logger)
        self.model_persister = ModelPersistence(config)

    def train(self, df_full_processed, target_rul_column, feature_columns):
        self.logger.log_section(f"INICIO PIPELINE RUL PARA: {target_rul_column}")

        # Pasos 1 y 2 no cambian
        valid_rul_mask = (df_full_processed[target_rul_column].notna()) & (df_full_processed[target_rul_column] > 0)
        df_filtered = df_full_processed[valid_rul_mask].copy()
        if df_filtered.empty:
            self.logger.log_warning(f"No hay datos válidos para {target_rul_column}. Saltando.")
            return
        RUL_CEILING = self.config.RUL_CONFIG.get('prediction_ceiling_days', 15.0)
        df_filtered[target_rul_column] = df_filtered[target_rul_column].clip(upper=RUL_CEILING)

        ### --- INICIO DE LA CORRECCIÓN FINAL --- ###
        self.logger.log_subsection("Paso 3: Eliminación de Features con Fuga de Datos (Data Leakage)")
        
        # Lista 1: Targets de RUL
        all_rul_targets = list(self.config.COLUMN_MAPPING['RUL_TARGETS'].values())
        
        # Lista 2: Proxies de tiempo de vida
        proxy_leakage_features = [col for col in feature_columns if 'horas_operacion_comp_' in col]

        # Lista 3 (LA CLAVE): Features que revelan la secuencia o el orden de los datos
        sequence_leakage_features = [
            'id', 'timestamp_unix', 'horas_operacion_acumuladas_maquina',
            'ciclo_id', 'posicion_en_ciclo', 'num_muestras', 'duracion_segundos',
            'vehiculo_id', # Prohibir el ID del vehículo
            'bloque_productivo', # Prohibir el turno de trabajo
            'semana_del_mes' # Prohibir la semana
        ]
        
        # Combinamos TODAS las listas para la exclusión definitiva.
        leakage_features_to_remove = list(set(all_rul_targets + proxy_leakage_features + sequence_leakage_features))
        
        feature_columns_clean = [col for col in feature_columns if col not in leakage_features_to_remove]
        self.logger.log_warning(f"Detectadas y eliminadas {len(leakage_features_to_remove)} features con potencial de data leakage para RUL: {leakage_features_to_remove}")
        ### --- FIN DE LA CORRECCIÓN FINAL --- ###

        # 4. SEPARACIÓN DE DATOS (usando las features limpias)
        self.logger.log_subsection("Paso 4: Separando Features y Metadatos")
        y_data = df_filtered[target_rul_column]
        X_features = df_filtered[feature_columns_clean]
        self.logger.log_step(f"Extraídas {len(X_features.columns)} columnas de features limpias.")

        # 5. FEATURE SELECTION
        self.logger.log_subsection("Paso 5: Selección de Features para Regresión")
        feature_selector = FeatureSelector(self.config, self.logger, task_type='regression')
        feature_selector.fit(X_features, y_data)
        self.trained_feature_columns = feature_selector.selected_features_
        self.logger.log_critical(f"Features seleccionadas para {target_rul_column}: {self.trained_feature_columns}")
        X_selected = X_features[self.trained_feature_columns]
        self.logger.log_success(f"Selección completada. Se usarán {len(self.trained_feature_columns)} features.")

        # 6. ENTRENAMIENTO
        id_col = self.config.COLUMN_MAPPING['id_maquina']
        timestamp_col = self.config.COLUMN_MAPPING['timestamp']
        self.train_rul_models(X_selected, y_data, df_filtered[[timestamp_col, id_col]], target_rul_column)

        # 7. GUARDADO
        self.save_trained_models(self.trained_feature_columns, component_name=target_rul_column.replace('rul_dias_', ''))

    def train_rul_models(self, X_features_only, y_data, df_meta, target_rul_column):
        
        self.logger.log_subsection("Paso 6: Entrenamiento de Modelos con Validación Cruzada")
        id_col = self.config.COLUMN_MAPPING['id_maquina']
        df_validation = pd.DataFrame({'RUL_days': y_data, id_col: df_meta[id_col]})
        splits_info = self.validator.create_temporal_splits(df_validation, 'RUL_days', id_col, task_type='regression')
        
        models_to_train = self.config.MODEL_CONFIG['regression']

        # --- INICIO CORRECCIÓN 1: Iteramos por modelo PRIMERO ---
        for name, params in models_to_train.items():
            self.logger.log_step(f"Construyendo y entrenando pipeline para: {name}")
            
            # Listas para recopilar resultados de TODOS los folds para ESTE modelo
            all_metrics_list = []
            all_y_test_folds = []
            all_y_pred_folds = []
            final_trained_model = None # Para guardar el modelo del último fold

            for i, split in enumerate(splits_info):
                self.logger.log_subsection(f"Fold {i + 1}/{len(splits_info)} para {name} en {target_rul_column}")
                train_idx, test_idx = split['train_idx'], split['test_idx']
                X_train, X_test = X_features_only.iloc[train_idx], X_features_only.iloc[test_idx]
                y_train, y_test = y_data.iloc[train_idx], y_data.iloc[test_idx]
                
                if 'linear' in name: base_model = LinearRegression(**params)
                elif 'random_forest' in name: base_model = RandomForestRegressor(**params)
                elif 'xgboost' in name: base_model = xgb.XGBRegressor(**params)
                else: continue

                full_model_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('regressor', base_model)])
                
                # Lógica de entrenamiento (sin cambios)
                if name == 'xgboost' and 'early_stopping_rounds' in params:
                    transformer_pipeline = Pipeline(full_model_pipeline.steps[:-1])
                    X_test_transformed = transformer_pipeline.fit(X_train).transform(X_test)
                    fit_params = {'regressor__eval_set': [(X_test_transformed, y_test)], 'regressor__verbose': False}
                    full_model_pipeline.fit(X_train, y_train, **fit_params)
                else:
                    full_model_pipeline.fit(X_train, y_train)

                y_pred = np.maximum(0, full_model_pipeline.predict(X_test))
                metrics = self.metrics_calculator.calculate_rul_metrics(y_test, y_pred)
                
                # --- INICIO CORRECCIÓN 2: Guardamos los resultados de CADA fold ---
                all_metrics_list.append(metrics)
                all_y_test_folds.append(y_test)
                all_y_pred_folds.append(y_pred)
                
                # Guardamos el modelo del último fold como el modelo "representativo"
                if i == len(splits_info) - 1:
                    final_trained_model = full_model_pipeline
            
            # --- INICIO CORRECCIÓN 3: Procesamos y guardamos los resultados AGREGADOS ---
            # Este bloque se ejecuta DESPUÉS de haber procesado todos los folds para un modelo
            if all_metrics_list:
                # Calculamos las métricas promedio
                avg_metrics = pd.DataFrame(all_metrics_list).mean().to_dict()
                self.logger.log_metrics(avg_metrics, f"Métricas Promedio CV - {name}", component=target_rul_column)

                # Concatenamos los resultados de todos los folds en un único DataFrame
                final_y_true = pd.concat(all_y_test_folds)
                final_y_pred = np.concatenate(all_y_pred_folds)
                predictions_df = pd.DataFrame({'y_true': final_y_true, 'y_pred': final_y_pred})

                # Guardamos todo lo que necesitamos en el diccionario de resultados del modelo
                self.models[name] = {
                    'model': final_trained_model, 
                    'metrics': avg_metrics,
                    'predictions_df': predictions_df # ¡El dato que necesita la fase de reporte!
                }

    def save_trained_models(self, feature_cols, component_name):
        if not self.session_id: return
        for name, info in self.models.items():
            # --- CORRECCIÓN FINAL DE NOMBRES ---
            # El nombre del archivo ahora será limpio y predecible.
            # Ej: xgboost_rul_dias_aceite.joblib
            target_rul_column = self.config.COLUMN_MAPPING['RUL_TARGETS'][component_name]
            model_full_name = f"{name}_{target_rul_column}"
            
            self.model_persister.save_model(
                model=info['model'], 
                model_name=model_full_name, # Usamos el nombre completo y limpio
                model_type='rul', 
                metrics=info['metrics'], 
                session_id=self.session_id, 
                feature_cols=feature_cols,
                predictions_df=info.get('predictions_df')  # Add predictions data
            )
        self.logger.log_success(f"Modelos para {component_name} guardados.")
        
def run_rul_step(df_processed, feature_columns, config, logger, session_id_base, force_recompute):
    
    logger.log_section("INICIO DEL PASO DE ESTIMACIÓN DE RUL (DISEÑO ENCAPSULADO Y SEGURO)")
    all_rul_results = {}
    for component_name, target_rul_column in config.COLUMN_MAPPING['RUL_TARGETS'].items():
        
        rul_pipeline = RULEstimator(config, logger, session_id=session_id_base)
        
        rul_pipeline.train(df_processed, target_rul_column, feature_columns)
        
        all_rul_results[component_name] = {
            'models': rul_pipeline.models, 
            'session_id': session_id_base, 
            'feature_columns': rul_pipeline.trained_feature_columns
        }
    
    logger.log_success("Paso de estimación RUL finalizado para todos los componentes.")
    
    return all_rul_results
