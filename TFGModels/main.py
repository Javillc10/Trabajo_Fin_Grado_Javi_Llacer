#!/usr/bin/env python3
"""
Main script for Industrial Predictive Maintenance Project
VERSIÓN REFACTORIZADA Y ROBUSTA
"""

import argparse
import logging
import os
import pandas as pd
import traceback
from datetime import datetime
import json
import numpy as np


# --- Importaciones de Módulos del Proyecto ---
from config import Config
from src.utils.data_adapter import DataAdapter
from src.utils.training_logger import TrainingLogger
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.unified_metrics import UnifiedMetrics

# --- Importaciones de Funciones de Pipeline ---
from src._01_EDA_and_Preprocessing import run_preprocessing_step
from src._02_Fault_Classification_Models import run_classification_step
from src._03_RUL_Estimation_Models import run_rul_step
from src._04_Model_Interpretation_and_Business_Impact import run_final_reporting_step

# --- Configuración de Logging Base ---
# Este logging captura errores que podrían ocurrir antes de que nuestro logger de sesión se inicialice.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
base_logger = logging.getLogger(__name__)

def parse_args():
    """Parsea los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description='Sistema de Mantenimiento Predictivo Industrial')
    parser.add_argument('--mode', choices=['train', 'infer'], default='train', help='Modo de operación: train o infer')
    parser.add_argument('--train-data', required=True, help='Ruta al fichero Parquet de ENTRENAMIENTO.')
    parser.add_argument('--test-data', default=None, help='(Opcional) Ruta al fichero Parquet de TEST para la evaluación final.')
    parser.add_argument('--session-id', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"), help='ID de sesión para logs y checkpoints. Se genera uno si no se proporciona.')
    parser.add_argument('--force-recompute', type=str, nargs='*', choices=['all', 'features', 'classification', 'rul'], help='Forzar recálculo de fases específicas.')
    return parser.parse_args()

def run_training_pipeline(args, config, logger, checkpoint_manager):
    """
    Ejecuta el pipeline completo de entrenamiento (Fases 0 a 4).
    VERSIÓN FINAL CON CORRECCIÓN DE CHECKPOINTS.
    """
    logger.log_step(f"Inicializando adaptador de datos con la ruta: {args.train_data}")
    data_adapter = DataAdapter(data_path=args.train_data)

    # =====================================================================
    # FASE 0: CARGA Y MUESTREO DE DATOS
    # =====================================================================
    logger.log_section("FASE 0: CARGA DE DATOS")
    df_completo = data_adapter.load_dataset()
    logger.log_step(f"Columnas disponibles: {df_completo.columns.tolist()}")

    df_master = df_completo
    if config.DEVELOPMENT_CONFIG['development_mode']:
        logger.log_section("MODO DESARROLLO ACTIVADO")
        id_col = config.DEVELOPMENT_CONFIG['id_column_for_sampling']
        sample_frac = config.DEVELOPMENT_CONFIG['sample_fraction_by_id']
        if id_col not in df_master.columns:
            raise KeyError(f"La columna de muestreo '{id_col}' no existe.")
        
        unique_ids = df_master[id_col].unique()
        num_ids_to_sample = int(len(unique_ids) * sample_frac)
        sampled_ids = pd.Series(unique_ids).sample(n=num_ids_to_sample, random_state=config.RANDOM_SEED)
        df_master = df_master[df_master[id_col].isin(sampled_ids)].copy()
        logger.log_step(f"Dataset de trabajo reducido a {len(df_master):,} filas.")

    # =====================================================================
    # FASE 1: PREPROCESAMIENTO Y FEATURE ENGINEERING
    # =====================================================================
    logger.log_section("FASE 1: PREPROCESAMIENTO Y FEATURE ENGINEERING")
    force_features = args.force_recompute and ('all' in args.force_recompute or 'features' in args.force_recompute)
    
    preprocessed_data_paket = checkpoint_manager.load_checkpoint('preprocessed_data') if not force_features else None
    engineer_paket = checkpoint_manager.load_checkpoint('feature_engineer') if not force_features else None
    
    feature_engineer_fitted = None
    if engineer_paket:
        feature_engineer_fitted = engineer_paket.get('engineer_object')
        if feature_engineer_fitted:
            feature_engineer_fitted.logger = logger

    if preprocessed_data_paket and feature_engineer_fitted:
        logger.log_success("Checkpoints de datos y feature engineer cargados.")
        df_processed = preprocessed_data_paket['dataframe']
        preparation_info = preprocessed_data_paket['prep_info']
    else:
        logger.log_step("Ejecutando preprocesamiento y feature engineering...")
        df_processed, preparation_info, feature_engineer_fitted = run_preprocessing_step(
            df_raw=df_master, config=config, logger=logger, session_id=args.session_id
        )
        if df_processed is None: raise RuntimeError("El preprocesamiento falló.")
        
        # Guardamos los checkpoints aquí, en el orquestador.
        checkpoint_manager.save_checkpoint('preprocessed_data', {'dataframe': df_processed, 'prep_info': preparation_info})
        if feature_engineer_fitted:
            checkpoint_manager.save_checkpoint('feature_engineer', {'engineer_object': feature_engineer_fitted})

    # Extraer los datos necesarios para las siguientes fases
    if 'feature_columns' not in preparation_info or not preparation_info['feature_columns']:
         raise ValueError("No se encontraron 'feature_columns' en la información de preparación.")
         
    X_features_master = df_processed[preparation_info['feature_columns']]
    y_class_fe = df_processed[preparation_info['target_column']]
    df_timestamps_fe = df_processed[preparation_info['timestamp_column']]
    
    # =====================================================================
    # FASE 2: ENTRENAMIENTO DE MODELOS DE CLASIFICACIÓN
    # =====================================================================
    logger.log_section("FASE 2: ENTRENAMIENTO DE MODELOS DE CLASIFICACIÓN")
    force_classification = 'all' in (args.force_recompute or []) or 'classification' in (args.force_recompute or [])
    
    classification_results = None
    if not force_classification:
        logger.log_step("Buscando checkpoint de resultados de clasificación...")
        classification_results = checkpoint_manager.load_checkpoint('classification_results')

    if classification_results:
        logger.log_success("Checkpoint de clasificación cargado. Saltando recálculo.")
    else:
        if force_classification:
            logger.log_warning("Forzando recálculo de los modelos de clasificación.")            
        else:
            logger.log_step("No se encontró checkpoint. Ejecutando entrenamiento de clasificación...")

        classification_results = run_classification_step(
            df_full=X_features_master, y_target_full=y_class_fe, df_timestamps_full=df_timestamps_fe,
            config=config, logger=logger, session_id=args.session_id, force_recompute=force_classification
            )
            # Guardamos el checkpoint solo si lo hemos calculado
        checkpoint_manager.save_checkpoint('classification_results', classification_results)

    # =====================================================================
    # FASE 3: ENTRENAMIENTO DE MODELOS DE ESTIMACIÓN DE RUL
    # =====================================================================
    logger.log_section("FASE 3: ENTRENAMIENTO DE MODELOS DE ESTIMACIÓN DE RUL")
    force_rul = 'all' in (args.force_recompute or []) or 'rul' in (args.force_recompute or [])

    rul_results = None
    if not force_rul:
        logger.log_step("Buscando checkpoint de resultados de RUL...")
        rul_results = checkpoint_manager.load_checkpoint('rul_results')

    if rul_results:
        logger.log_success("Checkpoint de RUL cargado. Saltando recálculo.")
    else:
        if force_rul:
            logger.log_warning("Forzando recálculo de los modelos de RUL.")
        else:
            logger.log_step("No se encontró checkpoint. Ejecutando entrenamiento de RUL...")

        # Obtenemos la "fuente de la verdad" de las features desde la Fase 1
        feature_columns_list = preparation_info['feature_columns']
        
        # La llamada ahora pasa el DataFrame completo Y la lista de features
        rul_results = run_rul_step(
            df_processed=df_processed,
            feature_columns=feature_columns_list, 
            config=config, 
            logger=logger,
            session_id_base=args.session_id, 
            force_recompute=force_rul
        )
        
        checkpoint_manager.save_checkpoint('rul_results', rul_results)
    
    # Registro de métricas de clasificación
    if classification_results and 'models' in classification_results:
        for model_name, model_data in classification_results['models'].items():
            if 'metrics' in model_data:
                logger.log_metrics(model_data['metrics'], f"Clasificación - {model_name}")

    # Registro de métricas de RUL
    if rul_results:
        for component, component_data in rul_results.items():
            if 'models' in component_data:
                for model_name, model_data in component_data['models'].items():
                    if 'metrics' in model_data:
                        logger.log_metrics(model_data['metrics'], f"RUL - {model_name}", component=component)
    
    # =====================================================================
    # FASE 4: INTERPRETACIÓN Y ANÁLISIS DE IMPACTO
    # =====================================================================
    logger.log_section("FASE 4: GENERANDO REPORTE FINAL Y VISUALIZACIONES")
    run_final_reporting_step(
        df_processed, classification_results, rul_results, config, logger, args.session_id
    )

    return True, classification_results, rul_results, feature_engineer_fitted
    
def run_final_evaluation_step(classification_results, rul_results, feature_engineer_fitted, args, config, logger):
    """
    Ejecuta una evaluación final de los modelos entrenados sobre un dataset de test separado y realista.
    VERSIÓN FINAL A PRUEBA DE BALAS.
    """
    logger.log_section("FASE FINAL: EVALUACIÓN SOBRE DATASET DE TEST")

    try:
        # 1. Cargar datos de test
        logger.log_step(f"Cargando dataset de test desde: {args.test_data}")
        df_test_raw = pd.read_parquet(args.test_data)

        # 2. Aplicar Feature Engineering y Ensamblar
        logger.log_step("Aplicando el mismo Feature Engineering al dataset de test...")
        if feature_engineer_fitted is None:
            raise ValueError("El objeto FeatureEngineer ajustado no está disponible.")
        
        df_new_test_features = feature_engineer_fitted.create_all_features(df_test_raw, is_inference=True)
        df_test_features = pd.concat([df_test_raw.reset_index(drop=True), df_new_test_features.reset_index(drop=True)], axis=1)
        logger.log_success(f"Ensamblaje de datos de test completado. Shape: {df_test_features.shape}")

        metrics_calculator = UnifiedMetrics(config, logger)

        # 3. EVALUACIÓN DEL MEJOR MODELO DE CLASIFICACIÓN
        logger.log_subsection("Evaluación Final - Modelo de Clasificación")
        if classification_results and 'models' in classification_results:
            best_model_name = max(classification_results['models'], key=lambda k: classification_results['models'][k].get('metrics', {}).get('f1_macro', 0))
            model_data = classification_results['models'][best_model_name]
            
            logger.log_step(f"Mejor modelo de clasificación para evaluación: '{best_model_name}'")
            
            clf_model = model_data['model']
            clf_features = classification_results['feature_columns']
            
            target_clf_col = config.COLUMN_MAPPING.get('target_classification', 'estado_agrupado')
            def map_state(estado):
                if pd.isna(estado): return estado
                estado_str = str(estado)
                if 'Normal' in estado_str: return 'Normal'
                if any(crit in estado_str for crit in ['Inminente', 'Critico', 'Severa', 'Falla', 'Error', 'Perdida']): return 'Critico'
                return 'Advertencia'
            df_test_features[target_clf_col] = df_test_features['estado_sistema'].apply(map_state)

            # =================================================================
            # INICIO DE LA CORRECCIÓN
            logger.log_step(f"Alineando las columnas del dataset de test con las {len(clf_features)} columnas esperadas por el modelo...")
            X_test_clf = df_test_features.reindex(columns=clf_features)
            # FIN DE LA CORRECCIÓN
            # =================================================================
            
            y_test_clf = df_test_features[target_clf_col]

            logger.log_step("Realizando predicciones de clasificación...")
            y_pred_clf = clf_model.predict(X_test_clf)
            
            # Get label_encoder for XGBoost predictions
            label_encoder = model_data.get('label_encoder') if best_model_name == 'xgboost' else None
            
            final_clf_metrics = metrics_calculator.calculate_classification_metrics(
                y_test_clf, 
                y_pred_clf,
                label_encoder=label_encoder
            )
            logger.log_metrics(final_clf_metrics, f"Métricas Finales Clasificación - {best_model_name} (sobre dataset de test)")
        else:
            logger.log_warning("No se encontraron resultados de clasificación para evaluar.")

        # 4. EVALUACIÓN DE LOS MEJORES MODELOS RUL
        logger.log_subsection("Evaluación Final - Modelos RUL")
        if rul_results:
            for component_name, component_data in rul_results.items():
                logger.log_step(f"--- Evaluando RUL para componente: {component_name} ---")
                if not ('models' in component_data and component_data['models']):
                    logger.log_warning(f"No se encontraron modelos para el componente RUL '{component_name}'.")
                    continue
                
                best_model_name = min(component_data['models'], key=lambda k: component_data['models'][k].get('metrics', {}).get('MAE', float('inf')))
                model_data = component_data['models'][best_model_name]

                logger.log_step(f"Mejor modelo RUL para '{component_name}': '{best_model_name}'")

                rul_model = model_data['model']
                rul_features = component_data['feature_columns']
                target_rul_col = config.COLUMN_MAPPING['RUL_TARGETS'][component_name]

                # =================================================================
                # INICIO DE LA CORRECCIÓN
                logger.log_step(f"Alineando las columnas del dataset de test con las {len(rul_features)} columnas esperadas por el modelo RUL...")
                X_test_rul = df_test_features.reindex(columns=rul_features)
                # FIN DE LA CORRECCIÓN
                # =================================================================

                y_test_rul = df_test_features[target_rul_col]

                RUL_CEILING = config.RUL_CONFIG.get('prediction_ceiling_days', 15.0)
                y_test_rul = y_test_rul.clip(upper=RUL_CEILING)

                logger.log_step("Realizando predicciones RUL...")
                y_pred_rul = rul_model.predict(X_test_rul)

                final_rul_metrics = metrics_calculator.calculate_rul_metrics(y_test_rul, y_pred_rul)
                logger.log_metrics(final_rul_metrics, f"Métricas Finales RUL - {component_name} - {best_model_name} (sobre dataset de test)", component=component_name)
        else:
            logger.log_warning("No se encontraron resultados RUL para evaluar.")

    except Exception as e:
        logger.log_critical(f"Error durante la fase de evaluación final: {e}", exc_info=True)
        raise

def select_and_save_best_models(clf_results, rul_results, config, session_id, logger):
    """
    Analiza los resultados, selecciona el mejor modelo para cada tarea y
    guarda la selección en un archivo 'best_models.json'.
    AHORA USA EL LOGGER DE LA SESIÓN.
    """
    selection_config = config.AUTO_MODEL_SELECTION_CONFIG
    best_models = {}

    # --- Selección para Clasificación ---
    try:
        task = 'classification'
        metric = selection_config[task]['key_metric']
        goal = selection_config[task]['goal']
        
        best_model_name = selection_config['fallback'][task]
        best_score = -float('inf') if goal == 'maximize' else float('inf')

        if clf_results and 'models' in clf_results:
            for model_name, model_data in clf_results['models'].items():
                avg_score = model_data.get('metrics', {}).get(metric, best_score)
                if (goal == 'maximize' and avg_score > best_score) or \
                   (goal == 'minimize' and avg_score < best_score):
                    best_score = avg_score
                    best_model_name = model_name
        best_models[task] = best_model_name
        logger.log_finding(f"Mejor modelo de clasificación seleccionado: '{best_model_name}' con un {metric} de {best_score:.4f}")
    except Exception as e:
        logger.log_error(f"Error seleccionando el mejor modelo de clasificación: {e}")
        best_models[task] = selection_config['fallback'][task]

    # --- Selección para RUL ---
    try:
        task = 'rul'
        metric = selection_config[task]['key_metric']
        goal = selection_config[task]['goal']
        
        best_model_name = selection_config['fallback'][task]
        best_score = -float('inf') if goal == 'maximize' else float('inf')

        if rul_results:
            model_scores = {}
            for component_data in rul_results.values():
                for model_name, model_data in component_data['models'].items():
                    if model_name not in model_scores: model_scores[model_name] = []
                    model_scores[model_name].append(model_data.get('metrics', {}).get(metric, best_score))
            
            for model_name, scores in model_scores.items():
                avg_score = np.mean(scores)
                if (goal == 'maximize' and avg_score > best_score) or \
                   (goal == 'minimize' and avg_score < best_score):
                    best_score = avg_score
                    best_model_name = model_name
        best_models[task] = best_model_name
        logger.log_finding(f"Mejor modelo RUL seleccionado: '{best_model_name}' con un {metric} promedio de {best_score:.4f}")
    except Exception as e:
        logger.log_error(f"Error seleccionando el mejor modelo RUL: {e}")
        best_models[task] = selection_config['fallback'][task]

    # --- Guardar el archivo de selección ---
    results_dir = os.path.join(config.RESULTS_DIR, session_id)
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, 'best_models.json')
    try:
        with open(output_path, 'w') as f:
            json.dump(best_models, f, indent=4)
        # --- REGISTRO DEL ARTEFACTO ---
        logger.log_artifact("Archivo de Selección de Mejores Modelos", output_path)
    except Exception as e:
        logger.log_error(f"❌ No se pudo guardar el archivo de selección de modelos: {e}")

def main():
    """Función principal que orquesta todo el proceso."""
    args = parse_args()
    config = Config()
    
    # --- Inicialización de Herramientas de Sesión ---
    logger = TrainingLogger(args.session_id, config)
    checkpoint_manager = CheckpointManager(config, session_id=args.session_id)
    
    try:
        logger.log_section("INICIO DE SESIÓN DE ENTRENAMIENTO")
        logger.log_step(f"Session ID: {args.session_id}")
        logger.log_step(f"Mode: {args.mode}")
        logger.log_step(f"Train Data: {args.train_data}")
        logger.log_step(f"Test Data: {args.test_data or 'N/A'}")
        logger.log_step(f"Force Recompute: {args.force_recompute or 'None'}")

        # --- Ejecutar el pipeline de entrenamiento ---
        success, clf_results, rul_results, fe_fitted = run_training_pipeline(
            args, config, logger, checkpoint_manager
        )

        # --- Ejecutar la evaluación final si se proporciona un dataset de test ---
        if success and args.test_data:
            run_final_evaluation_step(
                clf_results, rul_results, fe_fitted, args, config, logger
            )
        
        logger.log_section("FASE 5: SELECCIÓN AUTOMÁTICA DEL MEJOR MODELO")
        if success:
            # Pasamos el logger a la función para un logging unificado
            select_and_save_best_models(clf_results, rul_results, config, args.session_id, logger)
        
        logger.log_success("PIPELINE COMPLETADO EXITOSAMENTE")

    except Exception as e:
        # Captura cualquier error no manejado en los sub-módulos
        logger.log_critical(f"ERROR FATAL EN EL PIPELINE: {e}", exc_info=True)
        # Imprimir en consola para asegurar visibilidad
        print("\n" + "="*70)
        print(">>> ¡ERROR CRÍTICO DETECTADO! EL PIPELINE SE HA DETENIDO. <<<")
        print("="*70)
        traceback.print_exc()
        print("="*70)
        exit(1) # Salir con un código de error

    finally:
        # Este bloque se ejecuta siempre, haya o no errores,
        # asegurando que los resúmenes se generen.
        logger.log_section("FINALIZANDO SESIÓN")
        logger.close()
        print(f"\n📄 Resúmenes de entrenamiento guardados en el directorio: results/{args.session_id}")

if __name__ == "__main__":
    main()
