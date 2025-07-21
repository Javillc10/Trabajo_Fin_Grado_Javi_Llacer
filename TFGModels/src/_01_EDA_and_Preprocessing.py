# src/01_EDA_and_Preprocessing.py
"""
Módulo de preprocesamiento y análisis exploratorio robusto
Maneja automáticamente diferentes formatos de dataset
"""

import traceback 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
import warnings
from src.utils.feature_engineering import FeatureEngineer 
from src.utils.visualization_tools import IndustrialVisualizer 
from src.utils.checkpoint_manager import CheckpointManager
import os

warnings.filterwarnings('ignore')

class IndustrialDataPreprocessor:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.timestamp_column = None
        self.target_column = None
        
    def _group_system_states(self, df):
        """Agrupa los 17 estados del sistema en 3 categorías de negocio."""
        if self.target_column not in df.columns:
            return df

        self.logger.log_step("Agrupando estados del sistema en categorías: Normal, Advertencia, Crítico...")

        def map_state(estado):
            if 'Normal' in estado:
                return 'Normal'
            elif any(crit in estado for crit in ['Inminente', 'Critico', 'Severa', 'Falla', 'Error', 'Perdida']):
                return 'Critico'
            else:  # Degradacion, Fuga, Obstruccion, Calibracion, etc.
                return 'Advertencia'

        # Crea la nueva columna agrupada
        new_target_col = 'estado_agrupado'
        df[new_target_col] = df[self.target_column].apply(map_state)
        
        # Actualiza el target_column para el resto del pipeline
        self.logger.log_success(f"Estados agrupados. Nueva columna target: '{new_target_col}'")
        self.target_column = new_target_col
        self.config.COLUMN_MAPPING['target_classification'] = new_target_col
        
        return df

    def robust_date_converter(self, date_string):
        """
        Intenta convertir un string a datetime probando una lista de formatos.
        Es resistente a formatos mixtos (con y sin microsegundos).
        """
        # Si el valor es nulo o ya ha sido convertido, lo ignoramos.
        if pd.isna(date_string):
            return pd.NaT

        formats_to_try = [
            "%Y-%m-%dT%H:%M:%S.%f",  # Formato con microsegundos
            "%Y-%m-%dT%H:%M:%S",      # Formato sin microsegundos
        ]

        for fmt in formats_to_try:
            try:
                # Intenta convertir con el formato actual
                return pd.to_datetime(date_string, format=fmt)
            except (ValueError, TypeError):
                # Si falla, pasa al siguiente formato de la lista
                continue
                
        # Si ningún formato funcionó, devuelve NaT (Not a Time)
        return pd.NaT
    
    def process_dataframe(self, df_raw):
        """
        Procesa un DataFrame en memoria, detecta columnas automáticamente,
        realiza análisis exploratorio y prepara para el modelado.
        """
        self.logger.log_section("PROCESAMIENTO DE DATAFRAME EN MEMORIA")
        
        df = df_raw.copy() # Trabajar con una copia para no modificar el original
        
        self.logger.log_success(f"DataFrame recibido. Shape inicial: {df.shape}")
        
        # Mostrar información básica del dataset
        self.logger.log_step("INFORMACIÓN DEL DATASET:")
        self.logger.log_step(f"Columnas disponibles: {list(df.columns)}")
        self.logger.log_step("Tipos de datos:")
        for col, dtype in df.dtypes.items():
            self.logger.log_step(f"   {col}: {dtype}")
        
        # Detectar columna temporal automáticamente
        self.timestamp_column = self._detect_timestamp_column(df)
        
        # Detectar columna target automáticamente  
        self.target_column = self._detect_target_column(df)
        
        df = self._group_system_states(df)

        # Convertir columna temporal a datetime (requerida)
        if not self.timestamp_column:
            raise ValueError("Se requiere columna timestamp en los datos de entrada")
            
        self.logger.log_step(f"Usando columna temporal: {self.timestamp_column}")

        # 1. VERIFICAR si la columna ya es del tipo datetime.
        if pd.api.types.is_datetime64_any_dtype(df[self.timestamp_column]):
            self.logger.log_success("Columna temporal ya está en formato datetime. No se necesita conversión.")
        else:
            # 2. Si NO lo es, la convertimos de forma robusta.
            self.logger.log_warning("Columna temporal no es datetime. Aplicando conversión vectorizada optimizada...")
            
            original_rows = len(df)
            # Guardamos los valores originales para poder mostrar los que fallan
            original_timestamps = df[self.timestamp_column].copy() 

            # Intentamos la conversión. 'coerce' es la clave: los errores se convierten en NaT (Not a Time).
            df[self.timestamp_column] = pd.to_datetime(df[self.timestamp_column], format='ISO8601', errors='coerce')
            
            # 3. VERIFICAMOS SI HUBO FALLOS
            failed_mask = df[self.timestamp_column].isnull()
            failed_conversions = failed_mask.sum()
            
            if failed_conversions > 0:
                # Si hubo fallos, informamos y actuamos.
                failure_rate = (failed_conversions / original_rows) * 100
                problematic_samples = original_timestamps[failed_mask].head(5).to_list()
                
                # DECISIÓN INTELIGENTE: Si fallan demasiados (>90%), es un error crítico.
                if failure_rate > 90:
                    error_msg = (f"¡ERROR CRÍTICO! Más del 90% ({failure_rate:.1f}%) de las fechas en '{self.timestamp_column}' no se pudieron convertir. "
                                f"Ejemplos: {problematic_samples}. El pipeline no puede continuar.")
                    self.logger.log_critical(error_msg)
                    raise ValueError(error_msg) # Esto detendrá el programa con un error claro.
                else:
                    # Si fallan pocos, es una advertencia. Los eliminamos y continuamos.
                    warning_msg = (f"¡ADVERTENCIA! Se encontraron {failed_conversions} ({failure_rate:.1f}%) fechas no válidas "
                                f"en '{self.timestamp_column}'. Ejemplos: {problematic_samples}. "
                                f"Estas {failed_conversions} filas se eliminarán del dataset.")
                    self.logger.log_warning(warning_msg)
                    df.dropna(subset=[self.timestamp_column], inplace=True)
                    self.logger.log_step(f"Shape del DataFrame después de eliminar filas con fechas inválidas: {df.shape}")

        self.logger.log_success("Columna temporal parseada y limpiada correctamente.")

        # Este paso se puede quedar como está
        self.config.COLUMN_MAPPING['timestamp'] = self.timestamp_column
        
        self.logger.log_step(f"Columna target detectada: {self.target_column}")
                
        # Análisis exploratorio
        self._basic_data_exploration(df)
        
        # Análisis de calidad
        self._analyze_data_quality(df)
        
        # Análisis temporal si hay timestamp
        if self.timestamp_column in df.columns and pd.api.types.is_datetime64_any_dtype(df[self.timestamp_column]):
            self._analyze_temporal_patterns(df)
        
        return df
    
    def _detect_timestamp_column(self, df):
        """Detectar automáticamente la columna de timestamp"""
        
        # Candidatos comunes para timestamp
        timestamp_candidates = [
            'timestamp', 'time', 'fecha', 'date', 'datetime', 
            'fecha_hora', 'time_stamp', 'ts', 'created_at'
        ]
        
        # Buscar por nombre exacto
        for candidate in timestamp_candidates:
            if candidate in df.columns:
                return candidate
        
        # Buscar por coincidencia parcial
        for col in df.columns:
            col_lower = col.lower()
            if any(candidate in col_lower for candidate in ['time', 'fecha', 'date']):
                return col
        
        # Buscar por tipo de dato (si parece fecha)
        for col in df.columns:
            if df[col].dtype == 'object':
                # Intentar convertir muestra a datetime
                try:
                    sample_values = df[col].dropna().head(100)
                    pd.to_datetime(sample_values)
                    # Si no da error, probablemente es timestamp
                    if len(sample_values) > 0:
                        return col
                except:
                    continue
        
        return None
    
    def _detect_target_column(self, df):
        """Detectar automáticamente la columna target"""
        
        # Candidatos comunes para target
        target_candidates = [
            'estado_sistema', 'estado', 'status', 'state', 'target', 
            'label', 'class', 'classification', 'fallo', 'failure'
        ]
        
        # Buscar por nombre exacto
        for candidate in target_candidates:
            if candidate in df.columns:
                return candidate
        
        # Buscar por coincidencia parcial
        for col in df.columns:
            col_lower = col.lower()
            if any(candidate in col_lower for candidate in ['estado', 'status', 'target', 'label']):
                return col
        
        # Si no encuentra, usar la primera columna categórica
        for col in df.columns:
            if df[col].dtype == 'object' and col != self.timestamp_column:
                unique_values = df[col].nunique()
                # Si tiene entre 2 y 50 valores únicos, podría ser target
                if 2 <= unique_values <= 50:
                    return col
        
        return None
    
    def _basic_data_exploration(self, df):
        """Análisis exploratorio básico"""
        
        self.logger.log_section("ANÁLISIS EXPLORATORIO BÁSICO")
        self.logger.log_step(f"Número de filas: {len(df):,}")
        self.logger.log_step(f"Número de columnas: {len(df.columns)}")
        
        # Tipos de columnas
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        self.logger.log_step(f"Columnas numéricas: {len(numeric_cols)}")
        self.logger.log_step(f"Columnas categóricas: {len(categorical_cols)}")
        self.logger.log_step(f"Columnas de fecha: {len(datetime_cols)}")
        
        # Análisis del target si se detectó
        if self.target_column and self.target_column in df.columns:
            self.logger.log_subsection(f"ANÁLISIS DEL TARGET ({self.target_column})")
            target_counts = df[self.target_column].value_counts()
            self.logger.log_step(f"Valores únicos: {len(target_counts)}")
            self.logger.log_step("Distribución:")
            for value, count in target_counts.head(10).items():
                pct = count / len(df) * 100
                self.logger.log_step(f"    - {value}: {count:,} ({pct:.1f}%)")
            
            if len(target_counts) > 10:
                self.logger.log_step(f"      ... y {len(target_counts) - 10} más")
        
        # Análisis de sensores si existen
        sensor_keywords = ['sensor', 'presion', 'pressure', 'temperatura', 'temp']
        sensor_columns = [col for col in df.columns 
                        if any(keyword in col.lower() for keyword in sensor_keywords) and pd.api.types.is_numeric_dtype(df[col])]
        
        if sensor_columns:
            self.logger.log_subsection("SENSORES DETECTADOS")
            for sensor in sensor_columns:
                stats = df[sensor].describe()
                self.logger.log_step(f"Sensor: {sensor}")
                self.logger.log_step(f"    Min: {stats['min']:.3f}, Max: {stats['max']:.3f}, Media: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
    
    def _analyze_data_quality(self, df):
        """Análisis crítico de calidad para datos industriales"""
        self.logger.log_section("ANÁLISIS DE CALIDAD DE DATOS")
        
        # Missing values
        missing_analysis = df.isnull().sum()
        missing_pct = (missing_analysis / len(df) * 100).round(2)
        
        print(f"   Missing values:")
        if missing_analysis.sum() == 0:
            self.logger.log_step("No hay valores faltantes")
        else:
            for col, missing_count in missing_analysis[missing_analysis > 0].items():
                pct = missing_pct[col]
                status = "⚠️" if pct > 5 else "📝"
                self.logger.log_step(f"{status} {col}: {missing_count:,} ({pct}%)")
        
        # Duplicados
        duplicates = df.duplicated().sum()
        self.logger.log_step(f"Filas duplicadas: {duplicates:,}")
        
        # Análisis de sensores
        sensor_columns = [col for col in df.columns 
                         if any(keyword in col.lower() for keyword in ['sensor', 'presion', 'pressure'])]
        
        if sensor_columns:
            self.logger.log_step("Análisis de sensores:")
            for sensor in sensor_columns:
                if pd.api.types.is_numeric_dtype(df[sensor]):
                    # Outliers usando IQR
                    Q1 = df[sensor].quantile(0.25)
                    Q3 = df[sensor].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((df[sensor] < (Q1 - 1.5 * IQR)) | 
                               (df[sensor] > (Q3 + 1.5 * IQR))).sum()
                    outlier_pct = outliers / len(df) * 100
                    
                    # Valores constantes
                    unique_values = df[sensor].nunique()
                    
                    status = "✅" if outlier_pct < 5 and unique_values > 10 else "⚠️"
                    self.logger.log_step(f"{status} {sensor}:")
                    self.logger.log_step(f"   Outliers: {outliers:,} ({outlier_pct:.1f}%)")
                    self.logger.log_step(f"   Valores únicos: {unique_values:,}")
    
    def _analyze_temporal_patterns(self, df):
        """Análisis de patrones temporales"""
        
        if self.timestamp_column not in df.columns:
            return
        
        self.logger.log_section("ANÁLISIS TEMPORAL")
        
        # Período total
        time_span = df[self.timestamp_column].max() - df[self.timestamp_column].min()
        self.logger.log_step(f"Período total: {time_span}")
        self.logger.log_step(f"Desde: {df[self.timestamp_column].min()}")
        self.logger.log_step(f"Hasta: {df[self.timestamp_column].max()}")
        
        # Frecuencia de muestreo
        if len(df) > 1:
            time_diffs = df[self.timestamp_column].diff().dropna()
            if len(time_diffs) > 0:
                median_interval = time_diffs.median()
                freq_hz = 1 / median_interval.total_seconds() if median_interval.total_seconds() > 0 else 0
                self.logger.log_step(f"Intervalo mediano: {median_interval}")
                self.logger.log_step(f"Frecuencia estimada: {freq_hz:.2f} Hz")
        
        # Gaps en los datos
        if len(df) > 1:
            time_diffs = df[self.timestamp_column].diff().dropna()
            expected_interval = time_diffs.mode().iloc[0] if len(time_diffs) > 0 else pd.Timedelta(seconds=1)
            large_gaps = time_diffs[time_diffs > expected_interval * 2]
            
            self.logger.log_step(f"Gaps detectados: {len(large_gaps)}")
            if len(large_gaps) > 0:
                self.logger.log_step(f"Gap máximo: {large_gaps.max()}")
        
        # Análisis por períodos (si target disponible)
        if self.target_column and self.target_column in df.columns:
            # Agregar información temporal
            df_temp = df.copy()
            df_temp['hora'] = df_temp[self.timestamp_column].dt.hour
            df_temp['dia_semana'] = df_temp[self.timestamp_column].dt.day_name()
            
            # Distribución por hora
            hourly_dist = df_temp.groupby('hora')[self.target_column].apply(
                lambda x: x.value_counts().to_dict()
            )
            
            self.logger.log_step(f"Patrones horarios detectados: {len(hourly_dist)} horas únicas")
    
    def analyze_data_quality(self, df):
        """Método público para análisis de calidad (compatibilidad)"""
        return self._analyze_data_quality(df)
    
    def create_temporal_features(self, df):
        """Crear features temporales básicos - versión robusta"""
        
        self.logger.log_section("CREANDO FEATURES TEMPORALES")
        
        df_features = df.copy()
        
        # Verificar que tenemos columna temporal
        if self.timestamp_column not in df_features.columns:
            self.logger.log_warning("No hay columna temporal disponible, saltando features temporales")
            return df_features
        
        try:
            # Features temporales básicos
            df_features['hora_dia'] = df_features[self.timestamp_column].dt.hour
            df_features['dia_semana'] = df_features[self.timestamp_column].dt.dayofweek
            df_features['mes'] = df_features[self.timestamp_column].dt.month
            df_features['es_fin_semana'] = (df_features['dia_semana'] >= 5).astype(int)
            
            # Semana operacional
            df_features['semana_operacion'] = (
                df_features[self.timestamp_column].dt.isocalendar().week - 
                df_features[self.timestamp_column].dt.isocalendar().week.min() + 1
            )
            
            # Features adicionales de tiempo
            df_features['es_horario_laboral'] = (
                (df_features['hora_dia'] >= 8) & (df_features['hora_dia'] <= 17)
            ).astype(int)
            
            df_features['minuto_hora'] = df_features[self.timestamp_column].dt.minute
            df_features['segundo_minuto'] = df_features[self.timestamp_column].dt.second
            
            self.logger.log_success("Features temporales creadas: 7 nuevas columnas")
            
        except Exception as e:
            self.logger.log_error(f"Error creando features temporales: {str(e)}")
        
        return df_features
    
    def get_sensor_columns(self, df):
        """Obtener columnas de sensores automáticamente"""
        
        sensor_keywords = ['sensor', 'presion', 'pressure', 'temperatura', 'temp']
        sensor_columns = []
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in sensor_keywords):
                if pd.api.types.is_numeric_dtype(df[col]):
                    sensor_columns.append(col)
        
        return sensor_columns
    
    def get_feature_columns(self, df):
        """Obtener columnas de features para modelos"""
        
        # Excluir columnas no predictivas
        exclude_cols = [self.timestamp_column, self.target_column, 'index']
        exclude_cols = [col for col in exclude_cols if col is not None]
        
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and 
                       pd.api.types.is_numeric_dtype(df[col])]
        
        return feature_cols
    
    def stratified_sampling(self, df):
        """Implementar muestreo estratificado inteligente"""
        if not self.config.DEVELOPMENT_CONFIG['development_mode']:
            return df
            
        self.logger.log_step(f"Aplicando muestreo estratificado (n={self.config.DEVELOPMENT_CONFIG['sample_fraction_by_id']})...")
        
        # Use the id_column_for_sampling from DEVELOPMENT_CONFIG
        id_column = self.config.DEVELOPMENT_CONFIG['id_column_for_sampling']
        sample_fraction = self.config.DEVELOPMENT_CONFIG['sample_fraction_by_id']

        # Get unique IDs and sample a fraction of them
        unique_ids = df[id_column].unique()
        num_ids_to_sample = max(1, int(len(unique_ids) * sample_fraction))
        sampled_ids = np.random.choice(unique_ids, num_ids_to_sample, replace=False)
        
        # Filter the DataFrame to include only data from sampled IDs
        stratified_sample = df[df[id_column].isin(sampled_ids)]
        
        self.logger.log_success(f"Muestra estratificada creada: {len(stratified_sample):,} registros (de {len(df):,} originales)")
        return stratified_sample
    
    def prepare_for_modeling(self, df, logger):
        """Preparar datos para modelado con soporte para modo desarrollo"""
        
        logger.log_step("PREPARANDO DATOS PARA MODELADO...")
        
        # Crear features temporales
        df_processed = self.create_temporal_features(df)
        
        # Obtener columnas relevantes
        feature_columns = self.get_feature_columns(df_processed)
        sensor_columns = self.get_sensor_columns(df_processed)
        
        logger.log_step(f"Features disponibles: {len(feature_columns)}")
        logger.log_step(f"Sensores detectados: {len(sensor_columns)}")
        logger.log_step(f"Target column: {self.target_column}")
        
        # Aplicar muestreo si estamos en modo desarrollo
        if self.config.DEVELOPMENT_CONFIG['development_mode']: # Usar la configuración correcta
            df_processed = self.stratified_sampling(df_processed)
        
        # Información de preparación
        preparation_info = {
            'timestamp_column': self.timestamp_column,
            'target_column': self.target_column,
            'feature_columns': feature_columns,
            'sensor_columns': sensor_columns,
            'total_samples': len(df_processed),
            'numeric_features': len(feature_columns)
        }
        
        return df_processed, preparation_info

def run_preprocessing_step(df_raw, config, logger, session_id):
    """
    Función orquestadora para el paso de preprocesamiento y feature engineering.
    """
    logger.log_step("Iniciando el paso de preprocesamiento y feature engineering.")
    try:
        # ETAPA 1: Análisis y limpieza inicial
        logger.log_step("Etapa 1: Ejecutando preprocesador básico y EDA...")
        preprocessor = IndustrialDataPreprocessor(config, logger)
        df_cleaned = preprocessor.process_dataframe(df_raw)
        logger.log_success("Etapa 1 completada.")

        # ETAPA 2: Creación de nuevas features
        logger.log_step("Etapa 2: Solicitando nuevas features al FeatureEngineer...")
        feature_engineer = FeatureEngineer(config, logger, session_id=session_id)
        df_new_features = feature_engineer.create_all_features(df_cleaned, is_inference=False)
        logger.log_success(f"Etapa 2 completada. Se recibieron {df_new_features.shape[1]} nuevas features.")

        # ETAPA 3: Ensamblaje y preparación de la salida
        logger.log_step("Etapa 3: Ensamblando DataFrame final...")
        df_final_processed = pd.concat([df_cleaned.reset_index(drop=True), df_new_features.reset_index(drop=True)], axis=1)
        
        target_col = preprocessor.target_column
        if target_col in df_final_processed.columns:
            df_final_processed.dropna(subset=[target_col], inplace=True)
        
        original_numeric_cols = list(df_raw.select_dtypes(include=np.number).columns)
        all_feature_columns = original_numeric_cols + list(df_new_features.columns)
        final_feature_list = list(dict.fromkeys([col for col in all_feature_columns if col in df_final_processed.columns]))

        preparation_info = {
            'timestamp_column': preprocessor.timestamp_column,
            'target_column': target_col,
            'feature_columns': final_feature_list,
        }
        
        logger.log_success("Paso de preprocesamiento y feature engineering finalizado exitosamente.")
        
        # Devolvemos los tres objetos que main.py necesita.
        return df_final_processed, preparation_info, feature_engineer

    except Exception as e:
        error_details = traceback.format_exc()
        logger.log_critical(f"Error fatal durante el paso de preprocesamiento: {e}\nDETALLES COMPLETOS:\n{error_details}")
        return None, None, None

if __name__ == '__main__':
    def main_preprocessing():
        from datetime import datetime
        from config import Config
        from src.utils.training_logger import TrainingLogger
        from src.utils.data_adapter import DataAdapter
        
        # a. Inicializar config, logger y el SQLAdapter
        config = Config()
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = TrainingLogger(config, session_id)
        db_adapter = DataAdapter(config.DB_PATH)

        logger.log_step("Iniciando script de preprocesamiento (modo independiente).")
        
        try:
            # b. Cargar los datos desde la base de datos para obtener df_raw
            logger.log_step("Cargando datos desde la base de datos...")
            df_raw = db_adapter.fetch_data_to_dataframe(config.TABLE_NAME)
            logger.log_success(f"Datos crudos cargados exitosamente. Shape: {df_raw.shape}")

            # c. Llamar a la nueva función run_preprocessing_step(df_raw, ...)
            df_processed, prep_info = run_preprocessing_step(df_raw, config, logger, session_id)

            # d. Imprimir el shape o las primeras filas del DataFrame procesado para verificar que funcionó
            logger.log_step(f"DataFrame procesado final shape: {df_processed.shape}")
            logger.log_step("Primeras 5 filas del DataFrame procesado:")
            logger.log_dataframe(df_processed.head())
            logger.log_step(f"Columnas de features: {prep_info['feature_columns']}")
            logger.log_step(f"Columna target: {prep_info['target_column']}")
            logger.log_step(f"Columna timestamp: {prep_info['timestamp_column']}")

            logger.log_success("Proceso de preprocesamiento finalizado (modo independiente).")
            
        except Exception as e:
            logger.log_critical(f"Error crítico durante la ejecución del script principal de preprocesamiento (modo independiente): {e}", exc_info=True)
            
    main_preprocessing()
