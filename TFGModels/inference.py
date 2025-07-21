# inference.py (Versión final, limpia y sin duplicados)
import numpy as np
import pandas as pd
from datetime import datetime
import os
import logging
import argparse
import json
from src.utils.model_persistence import ModelPersistence
from src.utils.checkpoint_manager import CheckpointManager
from config import Config

def generar_plan_de_accion(df_predicciones, config):
    """
    Toma un DataFrame con predicciones y lo enriquece con columnas de decisión.
    Esta es la función central que usará tanto el reporte como el dashboard.

    Args:
        df_predicciones (pd.DataFrame): DataFrame con las columnas de predicciones.
        config (Config): Objeto de configuración para acceder a umbrales y nombres.

    Returns:
        pd.DataFrame: El DataFrame enriquecido con 'prioridad' y 'accion_recomendada'.
    """
    df = df_predicciones.copy()
    
    # --- 1. Definir Prioridad de Mantenimiento ---
    # Usamos np.select para una lógica limpia y eficiente
    
    # Condiciones para cada nivel de prioridad
    # ALTA: Estado es 'Critico' O cualquier RUL es menor de 3 días.
    cond_alta = (df['predicted_estado'] == 'Critico')
    for comp in config.COLUMN_MAPPING['RUL_TARGETS'].keys():
        rul_col = f'predicted_rul_{comp}'
        if rul_col in df.columns:
            cond_alta |= (df[rul_col] < 3)

    # MEDIA: Estado es 'Advertencia' O cualquier RUL es menor de 7 días.
    cond_media = (df['predicted_estado'] == 'Advertencia')
    for comp in config.COLUMN_MAPPING['RUL_TARGETS'].keys():
        rul_col = f'predicted_rul_{comp}'
        if rul_col in df.columns:
            cond_media |= (df[rul_col] < 7)

    # Lista de condiciones y sus resultados correspondientes
    condiciones = [cond_alta, cond_media]
    prioridades = ['ALTA', 'MEDIA']
    
    # Asignar la columna 'prioridad'
    df['prioridad'] = np.select(condiciones, prioridades, default='BAJA')

    # --- 2. Definir Acción Recomendada ---
    df['accion_recomendada'] = 'Seguimiento estándar' # Valor por defecto

    # Encontrar el componente con RUL más bajo para cada vehículo
    rul_cols = [col for col in df.columns if 'predicted_rul' in col]
    if rul_cols:
        # idxmin(axis=1) nos da el nombre de la columna con el valor mínimo por fila
        componente_urgente = df[rul_cols].idxmin(axis=1)
        
        # Asignar acción basada en el componente más urgente y la prioridad
        for comp_name in config.COLUMN_MAPPING['RUL_TARGETS'].keys():
            rul_col_name = f"predicted_rul_{comp_name}"
            # Máscara: filas donde la prioridad es ALTA y el componente urgente es este
            mask = (df['prioridad'] == 'ALTA') & (componente_urgente == rul_col_name)
            df.loc[mask, 'accion_recomendada'] = f"INSPECCIONAR {comp_name.upper()} URGENTE"

    # Acción para estado Crítico sin un RUL específico
    mask_critico_generico = (df['prioridad'] == 'ALTA') & (df['accion_recomendada'] == 'Seguimiento estándar')
    df.loc[mask_critico_generico, 'accion_recomendada'] = 'FALLO GENERAL CRÍTICO - INSPECCIONAR'
    
    return df


def imprimir_reporte_de_alertas(df_enriquecido, config):
    """
    Imprime en consola un reporte legible para un jefe de mantenimiento.
    """
    id_col = config.COLUMN_MAPPING.get('id_maquina', 'vehiculo_id')

    print("\n" + "="*60)
    print("[REPORT] REPORTE DE ESTADO DE FLOTA Y MANTENIMIENTO PREDICTIVO")
    print(f"   Fecha del Reporte: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # KPIs principales
    num_alta = (df_enriquecido['prioridad'] == 'ALTA').sum()
    num_media = (df_enriquecido['prioridad'] == 'MEDIA').sum()

    print(f"\nRESUMEN EJECUTIVO:")
    print(f"  - [URGENT] {num_alta} vehículos requieren ATENCIÓN INMEDIATA (Prioridad ALTA).")
    print(f"  - [WARN] {num_media} vehículos requieren seguimiento cercano (Prioridad MEDIA).")

    if num_alta > 0:
        print("\n--- VEHÍCULOS CON PRIORIDAD ALTA ---")
        alertas_altas = df_enriquecido[df_enriquecido['prioridad'] == 'ALTA']
        # Mostramos solo la última entrada por vehículo para no saturar
        for _, alerta in alertas_altas.drop_duplicates(subset=[id_col], keep='last').iterrows():
            print(f"  - Vehículo ID: {alerta[id_col]:<15} | Acción: {alerta['accion_recomendada']}")

    if num_media > 0:
        print("\n--- VEHÍCULOS CON PRIORIDAD MEDIA ---")
        alertas_medias = df_enriquecido[df_enriquecido['prioridad'] == 'MEDIA']
        for _, alerta in alertas_medias.drop_duplicates(subset=[id_col], keep='last').iterrows():
            print(f"  - Vehículo ID: {alerta[id_col]:<15} | Estado: {alerta['predicted_estado']}")

    print("\n" + "="*60)
    print("Fin del Reporte.")
    
# --- Configuración del Logging ---
# Lo configuramos aquí para que esté disponible en todo el script.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Clase Predictor (Definición Única y Correcta) ---
class Predictor:
    def __init__(self, session_id, config):
        self.session_id = session_id
        self.config = config
        self.model_persister = ModelPersistence(config)
        self.checkpoint_manager = CheckpointManager(config, session_id)
        self.feature_engineer = None

    def load_feature_engineer(self):
        """Carga el FeatureEngineer ajustado."""
        logging.info("Cargando FeatureEngineer ajustado...")
        engineer_paket = self.checkpoint_manager.load_checkpoint('feature_engineer')
        if engineer_paket is None:
            raise FileNotFoundError("El checkpoint 'feature_engineer.pkl' no se encontró para esta sesión.")
        
        self.feature_engineer = engineer_paket.get('engineer_object')
        if self.feature_engineer is None:
            raise ValueError("El checkpoint 'feature_engineer' se cargó, pero no contenía la clave 'engineer_object'.")
        
        logging.info("FeatureEngineer cargado exitosamente.")

    def generate_features(self, raw_data_df):
        """Genera características usando el FeatureEngineer cargado."""
        if self.feature_engineer is None:
            raise RuntimeError("El FeatureEngineer no ha sido cargado. Llama a load_feature_engineer() primero.")
        
        # Protegemos el logger para que la inferencia no falle si no está presente
        if hasattr(self.feature_engineer, 'logger'):
            self.feature_engineer.logger = None

        logging.info("Generando nuevas features a partir de los datos brutos...")
        df_new_features = self.feature_engineer.create_all_features(raw_data_df, is_inference=True)
        
        df_full_features = pd.concat([raw_data_df.reset_index(drop=True), df_new_features.reset_index(drop=True)], axis=1)
        logging.info(f"Ensamblaje completado. Shape final de features: {df_full_features.shape}")
        return df_full_features

    def load_model_and_predict(self, component_name, features_df, model_type, session_id_override=None):
        """Carga un modelo, realiza la predicción y decodifica si es necesario."""
        
        session_to_use = session_id_override if session_id_override else self.session_id

        model, metadata = self.model_persister.load_model(
            model_name=component_name, model_type=model_type, session_id=session_to_use
        )
        
        if model is None or metadata is None:
            raise FileNotFoundError(f"No se pudo cargar el bundle del modelo para '{component_name}' en sesión '{session_to_use}'.")

        required_features = metadata.get('feature_columns', [])
        if not required_features:
            raise ValueError(f"No se encontraron 'feature_columns' en los metadatos de '{component_name}'.")

        # Asegurarse de que las columnas están en el orden correcto y manejar las que falten
        X_inference = features_df.reindex(columns=required_features, fill_value=0)

        # --- INICIO DE LA CORRECCIÓN FINAL ---
        # Realizamos la predicción
        predictions_numeric = model.predict(X_inference)
        
        # Si es un modelo de clasificación y tiene un decodificador, lo usamos
        if model_type == 'classification' and 'label_encoder' in metadata:
            logging.info("Decodificando predicciones numéricas a etiquetas de texto...")
            # Usamos el decodificador cargado para traducir los números a texto
            predictions_decoded = metadata['label_encoder'].inverse_transform(predictions_numeric)
            return predictions_decoded
        else:
            # Para RUL u otros modelos, devolvemos la predicción numérica directamente
            return predictions_numeric
        # --- FIN DE LA CORRECCIÓN FINAL ---

# --- Función Principal (Orquestador) ---
def main():
    """ Punto de entrada para ejecutar la inferencia con los modelos entrenados.
    """
    logging.info("Iniciando el proceso de inferencia...")
    parser = argparse.ArgumentParser(description="Ejecutar inferencia con modelos TFGModels")
    parser.add_argument('--model-session', required=True, help='ID de sesión de entrenamiento a usar.')
    parser.add_argument('--input-file', required=True, help='Ruta al fichero Parquet con los datos de entrada.')
    parser.add_argument('--output-dir', type=str, default='results/inference_output', help='Directorio para guardar las predicciones.')
    args = parser.parse_args()

    logging.info(f"--- Iniciando Inferencia con la sesión de modelos: {args.model_session} ---")
    
    # --- Lógica de Selección de Modelo Inteligente ---
    model_selection = Config.AUTO_MODEL_SELECTION_CONFIG['fallback'] # Empezamos con el fallback
    best_models_path = os.path.join(Config.RESULTS_DIR, args.model_session, 'best_models.json')
    
    try:
        with open(best_models_path, 'r') as f:
            model_selection = json.load(f)
            logging.info(f"[OK] Selección de modelos cargada automáticamente desde: {best_models_path}")
    except FileNotFoundError:
        logging.warning(f"⚠️ No se encontró 'best_models.json'. Usando la selección de fallback de config.py: {model_selection}")
    except Exception as e:
        logging.error(f"❌ Error al cargar 'best_models.json', usando fallback. Error: {e}")

    try:
        df_raw = pd.read_parquet(args.input_file)
        config = Config()
        predictor = Predictor(args.model_session, config)
        
        predictor.load_feature_engineer()
        df_features = predictor.generate_features(df_raw)
        df_results = df_raw.copy()
        
        # --- Clasificación (Usa el modelo seleccionado automáticamente) ---
        model_name_clf = model_selection.get('classification')
        if model_name_clf:
            try:
                logging.info(f"Buscando el mejor modelo de clasificación ('{model_name_clf}') en sesión '{args.model_session}'...")
                df_results['predicted_estado'] = predictor.load_model_and_predict(model_name_clf, df_features, 'classification')
                logging.info("[OK] Predicciones de clasificación generadas.")
            except FileNotFoundError as e:
                logging.warning(f"⚠️  Saltando clasificación: {e}")
                df_results['predicted_estado'] = 'N/A'
        else:
            logging.warning("⚠️ No se especificó un modelo de clasificación en la selección.")

        # --- RUL (Usa el modelo seleccionado automáticamente) ---
        base_model_name_rul = model_selection.get('rul')
        if base_model_name_rul:
            for component in config.COLUMN_MAPPING['RUL_TARGETS'].keys():
                target_col = config.COLUMN_MAPPING['RUL_TARGETS'][component]
                try:
                    # Construimos el nombre final del modelo que fue guardado por el entrenador
                    model_name_rul = f"{base_model_name_rul}_{target_col}"
                    
                    logging.info(f"Buscando el mejor modelo RUL ('{model_name_rul}') en sesión '{args.model_session}'...")
                    
                    df_results[f'predicted_rul_{component}'] = predictor.load_model_and_predict(
                        model_name_rul, 
                        df_features, 
                        'rul'
                    )
                    logging.info(f"[OK] Predicciones RUL para '{component}' generadas.")
                except FileNotFoundError as e:
                    logging.warning(f"⚠️  Saltando RUL para '{component}': {e}")
        else:
            logging.warning("⚠️ No se especificó un modelo RUL en la selección.")
        
        # --- Flujo de Output ---
        logging.info("Traduciendo predicciones a un plan de acción...")
        df_final = generar_plan_de_accion(df_results, config)
        imprimir_reporte_de_alertas(df_final, config)
        
        os.makedirs(args.output_dir, exist_ok=True)
        output_filename = f"dashboard_data_{args.model_session}.parquet"
        output_path = os.path.join(args.output_dir, output_filename)
        df_final.to_parquet(output_path, index=False)
        logging.info(f"[DONE] Datos para el dashboard guardados en: {output_path}")

    except Exception as e:
        logging.critical(f"Error fatal durante la inferencia: {e}", exc_info=True)

# --- Punto de Entrada del Script ---
if __name__ == '__main__':
    main()
