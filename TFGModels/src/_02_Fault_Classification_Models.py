# ==============================================================================
# ARCHIVO COMPLETO Y CORREGIDO: _02_Fault_Classification_Models.py
# VERSIÓN FINAL CON CORRECCIONES DE CONSISTENCIA Y TIPO DE DATOS
# ==============================================================================

import pandas as pd
import numpy as np
import time
import os
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from config import Config
from src.utils.training_logger import TrainingLogger
from src.utils.model_persistence import ModelPersistence
from src.utils.validation_strategies import TemporalValidator
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.feature_selection import FeatureSelector
from src.utils.unified_metrics import UnifiedMetrics

class IndustrialFaultClassifier:
    def __init__(self, config, logger, session_id):
        self.config = config
        self.logger = logger
        self.session_id = session_id
        self.models = {}
        self.feature_importance = {}
        self.selected_features_ = None
        self.checkpoint_manager = CheckpointManager(config, session_id)
        self.model_persister = ModelPersistence(config)
        self.validator = TemporalValidator(config, self.logger) 
        self.metrics_calculator = UnifiedMetrics(self.config, self.logger)
        self.logger.log_step("Clase IndustrialFaultClassifier inicializada.")

    def train(self, df_full, y_target_full_raw, df_timestamps_full_raw, force_recompute=False):
        self.logger.log_section("INICIO DEL PIPELINE DE ENTRENAMIENTO DE CLASIFICACIÓN")
        try:
            if isinstance(df_timestamps_full_raw, pd.DataFrame):
                df_timestamps_full_raw = df_timestamps_full_raw.iloc[:, 0]
            self.train_advanced_models(df_full, y_target_full_raw, df_timestamps_full_raw, force_recompute)
            self.save_trained_models(feature_cols=self.selected_features_)
            self.logger.log_success("Pipeline de entrenamiento de clasificación completado exitosamente.")
        except Exception as e:
            self.logger.log_critical(f"Error fatal en el pipeline de clasificación: {e}", exc_info=True)
            raise
        
    def train_advanced_models(self, X_data_full, y_data, df_timestamps, force_recompute=False):
        self.logger.log_subsection("Entrenamiento de Modelos Avanzados (Selección de Features por Fold)")

        # --- INICIO DE LA CORRECCIÓN DE ORDEN ---

        # 1. GUARDA LA COLUMNA DE GRUPOS ANTES DE HACER NADA
        grouping_col_name = 'vehiculo_id'
        groups = X_data_full[grouping_col_name].copy() if grouping_col_name in X_data_full else pd.Series(index=X_data_full.index)

        # 2. AHORA, REALIZA TODA LA LIMPIEZA SOBRE X_data_full
        
        # Limpieza de RULs (como ya la tenías)
        rul_target_cols = self.config.COLUMN_MAPPING['RUL_TARGETS'].values()
        leakage_cols = [col for col in X_data_full.columns if col in rul_target_cols]
        if leakage_cols:
            self.logger.log_warning(f"¡DATA LEAKAGE DETECTADO! Eliminando: {leakage_cols}")
            X_data_full = X_data_full.drop(columns=leakage_cols)

        # Limpieza de Identidad (la que añadiste)
        identity_leakage_cols = [
            'id', 'vehiculo_id', 'timestamp_unix', 'bloque_productivo',
            'semana_del_mes', 'ciclo_id', 'posicion_en_ciclo',
            'num_muestras', 'duracion_segundos'
        ]
        cols_to_drop_clf = [col for col in identity_leakage_cols if col in X_data_full.columns]
        if cols_to_drop_clf:
            self.logger.log_warning(f"¡DATA LEAKAGE DE IDENTIDAD DETECTADO! Eliminando: {cols_to_drop_clf}")
            X_data_full = X_data_full.drop(columns=cols_to_drop_clf)

        # 3. CREA EL DATAFRAME DE VALIDACIÓN USANDO LA COLUMNA DE GRUPOS GUARDADA
        target_col = self.config.COLUMN_MAPPING['target_classification']
        df_validation = pd.DataFrame({
            'timestamp': df_timestamps, 
            target_col: y_data, 
            grouping_col_name: groups  # <--- Usamos la copia guardada
        })
        
        splits_info = self.validator.create_temporal_splits(df_validation, target_column=target_col, group_column=grouping_col_name)       
        if not self.validator.validate_no_data_leakage(splits_info, df_validation, group_column='vehiculo_id'):
            self.logger.log_warning("Posible data leakage detectado.")

        fold_metrics = {'random_forest': [], 'xgboost': []}

        for i, split in enumerate(splits_info):
            self.logger.log_subsection(f"Fold de Entrenamiento Avanzado {i + 1}/{len(splits_info)}")
            train_idx, test_idx = split['train_idx'], split['test_idx']
            X_train_fold, X_test_fold = X_data_full.iloc[train_idx], X_data_full.iloc[test_idx]
            y_train, y_test = y_data.iloc[train_idx], y_data.iloc[test_idx]

            feature_selector = FeatureSelector(self.config, self.logger)
            feature_selector.fit(X_train_fold, y_train)
            
            current_fold_features = feature_selector.selected_features_
            if i == len(splits_info) - 1:
                self.logger.log_step(f"Guardando la lista de features del último fold ({len(current_fold_features)} features).")
                self.selected_features_ = current_fold_features

            X_train_selected = feature_selector.transform(X_train_fold)
            X_test_selected = feature_selector.transform(X_test_fold)
            
            imputer_fold = SimpleImputer(strategy='median')
            X_train_imputed = pd.DataFrame(imputer_fold.fit_transform(X_train_selected), columns=X_train_selected.columns)
            X_test_imputed = pd.DataFrame(imputer_fold.transform(X_test_selected), columns=X_test_selected.columns)
            
            X_train_balanced, y_train_balanced = self.process_with_strategy(X_train_imputed, y_train, 'hybrid')

            # RandomForest
            rf_model = RandomForestClassifier(**self.config.MODEL_CONFIG['classification']['random_forest'])
            rf_model.fit(X_train_balanced, y_train_balanced)
            y_pred_rf = rf_model.predict(X_test_imputed)
            y_proba_rf = rf_model.predict_proba(X_test_imputed)
            
            # Crear LabelEncoder para RandomForest similar a XGBoost
            le_rf = LabelEncoder()
            le_rf.fit(y_train_balanced)
            metrics_rf = self.metrics_calculator.calculate_classification_metrics(
                y_true=y_test, 
                y_pred=y_pred_rf, 
                y_proba=y_proba_rf, 
                class_labels=rf_model.classes_,
                label_encoder=le_rf
            )
            fold_metrics['random_forest'].append(metrics_rf)
            self.logger.log_metrics(metrics_rf, f"Métricas - RandomForest - Fold {i+1}", component='classification')
            
            # XGBoost
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train_balanced)
            y_test_encoded = le.transform(y_test)
            xgb_model = xgb.XGBClassifier(**self.config.MODEL_CONFIG['classification']['xgboost'])
            xgb_model.fit(X_train_balanced, y_train_encoded, eval_set=[(X_test_imputed, y_test_encoded)], verbose=False)
            y_pred_xgb_encoded = xgb_model.predict(X_test_imputed)
            y_pred_xgb = le.inverse_transform(y_pred_xgb_encoded)
            y_proba_xgb = xgb_model.predict_proba(X_test_imputed)
            metrics_xgb = self.metrics_calculator.calculate_classification_metrics(
                y_true=y_test, 
                y_pred=y_pred_xgb, 
                y_proba=y_proba_xgb, 
                class_labels=le.classes_,
                label_encoder=le
            )
            fold_metrics['xgboost'].append(metrics_xgb)
            self.logger.log_metrics(metrics_xgb, f"Métricas - XGBoost - Fold {i+1}", component='classification')

            if i == len(splits_info) - 1:
                self.logger.log_step("Guardando los modelos y predicciones finales del último fold...")
                self.models['random_forest'] = {'model': rf_model, 'metrics': metrics_rf, 'type': 'advanced', 'predictions': y_pred_rf.tolist(), 'y_test': y_test.tolist()}
                self.models['xgboost'] = {
                    'model': xgb_model, 
                    'label_encoder': le, 
                    'metrics': metrics_xgb, 
                    'type': 'advanced', 
                    'predictions': y_pred_xgb.tolist(), 
                    'y_test': y_test.tolist()
                }

        self.logger.log_section("Resumen Métricas Promedio (Validación Cruzada - Avanzado)")
        for model_name, metrics_list in fold_metrics.items():
            if metrics_list:
                avg_metrics = pd.DataFrame(metrics_list).mean().to_dict()
                self.logger.log_metrics(avg_metrics, f"Métricas Promedio CV - {model_name}")
                
    def save_trained_models(self, feature_cols):

        if not self.session_id:
            self.logger.log_warning("Session ID no definido. No se guardarán modelos.")
            return
        self.logger.log_subsection(f"Guardando modelos de clasificación para la sesión: {self.session_id}")
        for model_name, model_info in self.models.items():
            if 'model' in model_info:
                self.model_persister.save_model(
                    model=model_info['model'],
                    model_name=model_name,
                    model_type='classification',
                    metrics=model_info.get('metrics'),
                    session_id=self.session_id,
                    label_encoder=model_info.get('label_encoder'),
                    feature_cols=feature_cols
                )
        self.logger.log_success("Todos los modelos de clasificación han sido guardados.")

    def process_with_strategy(self, X, y, strategy):

        self.logger.log_step(f"Aplicando la estrategia de balanceo HÍBRIDA ('{strategy}')...")
        if y.nunique() <= 1:
            self.logger.log_warning("Solo una clase en el target. No se aplica balanceo.")
            return X, y
        try:
            n_muestras_max = 150000 
            class_counts = y.value_counts()
            sampling_strategy_under = {cls: min(count, n_muestras_max) for cls, count in class_counts.items()}
            rus = RandomUnderSampler(sampling_strategy=sampling_strategy_under, random_state=self.config.RANDOM_SEED)
            X_under, y_under = rus.fit_resample(X, y)
            self.logger.log_success(f"Under-sampling completado. Shape: {X_under.shape}")
            smote = SMOTE(random_state=self.config.RANDOM_SEED, sampling_strategy='auto')
            X_balanced, y_balanced = smote.fit_resample(X_under, y_under)
            self.logger.log_success(f"Balanceo HÍBRIDO completado. Shape final: {X_balanced.shape}")
            return X_balanced, y_balanced
        except Exception as e:
            self.logger.log_error(f"Falló la estrategia de balanceo HÍBRIDA: {e}. Devolviendo datos originales.", exc_info=True)
            return X, y

def run_classification_step(df_full, y_target_full, df_timestamps_full, config, logger, session_id, force_recompute):
    # ... (sin cambios en esta función) ...
    logger.log_step("Iniciando el paso de clasificación de fallos.")
    try:
        classifier = IndustrialFaultClassifier(config, logger, session_id)
        classifier.train(df_full, y_target_full, df_timestamps_full, force_recompute)
        logger.log_success("Paso de clasificación de fallos finalizado exitosamente.")
        selected_features = getattr(classifier, 'selected_features_', []) 
        return {'models': classifier.models, 'feature_importance': classifier.feature_importance, 'feature_columns': selected_features}
    except Exception as e:
        logger.log_critical(f"Error durante el paso de clasificación de fallos: {e}", exc_info=True)
        raise
