# src/utils/feature_engineering.py
"""
Módulo especializado de feature engineering para datos industriales
VERSIÓN HÍBRIDA INTELIGENTE
"""

import pandas as pd
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
from .feature_selection import FeatureSelector
import warnings
import time
import psutil
import gc
import logging
import re
from sklearn.base import BaseEstimator
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer(BaseEstimator):
    """Clase especializada en feature engineering híbrido para mantenimiento predictivo (VERSIÓN REFACTORIZADA)"""
    
    def __init__(self, config, logger=None, mode='auto', session_id=None):
        self.config = config
        self.logger = logger
        self.scaler = StandardScaler()
        self.baseline_values = {}
        self.timestamp_col_name = self.config.COLUMN_MAPPING['timestamp'] # Get from config
        self.mode = mode
        self.processing_stats = {}
        self.current_feature_config = None
        self.session_id = session_id
        
        if session_id:
            from .checkpoint_manager import CheckpointManager
            self.checkpoint_manager = CheckpointManager(config, session_id)
        else:
            self.checkpoint_manager = None
    
    def create_all_features(self, df, is_inference=False):
        """
        VERSIÓN FINAL Y ROBUSTA: Crea y devuelve ÚNICAMENTE las nuevas features.
        No devuelve ninguna columna del DataFrame original.
        """
        # ### INICIO DE LA CORRECCIÓN ###
        # Comprobamos si el logger existe antes de usarlo.
        if self.logger:
            self.logger.log_step("Iniciando la creación de features (modo simplificado)...")
        # ### FIN DE LA CORRECCIÓN ###
        
        original_cols = set(df.columns)
        df_features = df.copy()

        # --- Cadena de creación de features ---
        df_features = self.create_temporal_features_optimized(df_features)
        df_features = self._create_sensor_features_complete(df_features)
        df_features = self._create_degradation_features_selective(df_features)
        df_features = self.create_advanced_degradation_features(df_features)
        df_features = self.create_operational_stress_features_optimized(df_features)
        
        if 'tipo_vehiculo' in df_features.columns:
            df_features = self.create_cycle_efficiency_features(df_features)
        
        df_features = self._create_spectral_features_optimized(df_features)
        df_features = self.remove_outliers(df_features)
        
        new_cols = set(df_features.columns)
        added_feature_cols = list(new_cols - original_cols)
        
        # ### INICIO DE LA CORRECCIÓN ###
        # Comprobamos de nuevo si el logger existe.
        if self.logger:
            self.logger.log_success(f"Se han creado {len(added_feature_cols)} nuevas features.")
        # ### FIN DE LA CORRECCIÓN ###
        
        return df_features[added_feature_cols]
    
    def _create_all_features_hybrid(self, df, is_inference, target_time_minutes, show_progress, y=None):
        print(f"🧠 FEATURE ENGINEERING HÍBRIDO INTELIGENTE")
        print("=" * 60)
        start_time = time.time()
        dataset_size = len(df)
        available_memory = psutil.virtual_memory().available / (1024**3)
        optimal_mode = self._get_optimal_processing_mode(dataset_size, available_memory, target_time_minutes)
        mode_config = self._get_processing_mode_config('complete')
        feature_config = self._get_feature_configuration('complete')
        print(f"📊 ANÁLISIS DEL DATASET:")
        print(f"   📏 Tamaño: {dataset_size:,} registros")
        print(f"   💾 Memoria disponible: {available_memory:.1f} GB")
        print(f"   ⏱️  Tiempo objetivo: {target_time_minutes or 'No especificado'} min")
        print(f"\n🎯 MODO SELECCIONADO: {optimal_mode.upper()}")
        print(f"   📝 Descripción: {mode_config['description']}")
        print(f"   ⏱️  Tiempo estimado: {mode_config['expected_time_minutes']} min")
        print(f"   💾 Memoria requerida: {mode_config['memory_limit_gb']} GB")
        self._apply_mode_configuration(feature_config)
        df_features = self._create_all_features_specific_mode(df, is_inference, optimal_mode, show_progress, y)
        end_time = time.time()
        self._generate_processing_report(df, df_features, start_time, end_time, optimal_mode)
        return df_features
    
    def _create_all_features_specific_mode(self, df, is_inference, mode, show_progress, y=None):
        if mode == 'development':
            return self._process_development_mode(df, is_inference, show_progress, y)
        elif mode == 'balanced':
            return self._process_balanced_mode(df, is_inference, show_progress)
        elif mode == 'optimized':
            return self._process_optimized_mode(df, is_inference, show_progress)
        elif mode == 'complete':
            return self._process_complete_mode(df, is_inference, show_progress)
        else:
            print("⚠️ Modo no reconocido, usando método completo refactorizado...")
            return self._create_all_features_original(df, is_inference, y)

    def _process_development_mode(self, df, is_inference, show_progress, y=None):
        print(f"\n🔬 PROCESANDO EN MODO DESARROLLO")
        max_samples = 200000
        if len(df) > max_samples:
            print(f"📊 Creando muestra estratificada de {max_samples:,} registros...")
            target_col = None
            for candidate in self.config.COLUMN_MAPPING['target_candidates']:
                if candidate in df.columns:
                    target_col = candidate
                    break
            if target_col and df[target_col].nunique() > 1:
                df_sample = df.groupby(target_col, group_keys=False).apply(
                    lambda x: x.sample(min(len(x), max_samples // df[target_col].nunique()))
                )
            else:
                df_sample = df.sample(n=max_samples, random_state=42)
            print(f"✅ Muestra creada: {len(df_sample):,} registros")
        else:
            df_sample = df.copy()
        
        # El 'y' para la muestra también debe ser muestreado si existe
        y_sample = y.loc[df_sample.index] if y is not None else None
        return self._create_all_features_original(df_sample, is_inference, y_sample)

    def _process_balanced_mode(self, df, is_inference, show_progress):
        print(f"\n⚖️  PROCESANDO EN MODO BALANCEADO")
        chunk_size = 500000
        if len(df) <= chunk_size:
            return self._create_features_balanced(df, is_inference)
        else:
            return self._process_by_chunks(df, chunk_size, 'balanced', is_inference, show_progress)

    def _process_optimized_mode(self, df, is_inference, show_progress):
        print(f"\n🚀 PROCESANDO EN MODO OPTIMIZADO")
        df_optimized = self._optimize_dataframe_memory(df)
        chunk_size = 200000
        if len(df_optimized) <= chunk_size:
            return self._create_features_optimized(df_optimized, is_inference)
        else:
            return self._process_by_chunks(df_optimized, chunk_size, 'optimized', is_inference, show_progress)

    def _process_complete_mode(self, df, is_inference, show_progress):
        print(f"\n🎯 PROCESANDO EN MODO COMPLETO")
        chunk_size = 300000
        overlap = 20000
        return self._process_by_chunks_optimized_complete(df, chunk_size, overlap, is_inference, show_progress)

    
    def _process_by_chunks(self, df, chunk_size, mode_type, is_inference, show_progress):
        chunks_processed = []
        n_chunks = (len(df) // chunk_size) + 1
        print(f"📊 Procesando {n_chunks} chunks de {chunk_size:,} registros...")
        for i in range(n_chunks):
            start_idx, end_idx = i * chunk_size, min((i + 1) * chunk_size, len(df))
            if start_idx >= len(df): break
            chunk = df.iloc[start_idx:end_idx].copy()
            if show_progress: print(f"   📦 Chunk {i+1}/{n_chunks}: {start_idx:,} - {end_idx:,}")
            if mode_type == 'balanced':
                chunk_processed = self._create_features_balanced(chunk, is_inference)
            elif mode_type == 'optimized':
                chunk_processed = self._create_features_optimized(chunk, is_inference)
            else:
                chunk_processed = self._create_all_features_original(chunk, is_inference)
            chunks_processed.append(chunk_processed)
            del chunk; gc.collect()
        print("🔗 Concatenando chunks...")
        result = pd.concat(chunks_processed, ignore_index=True)
        del chunks_processed; gc.collect()
        return result

    def _process_by_chunks_with_overlap(self, df, chunk_size, overlap, is_inference, show_progress):
        chunks_processed = []
        n_chunks = (len(df) // chunk_size) + 1
        print(f"📊 Procesando {n_chunks} chunks con overlap de {overlap:,} muestras...")
        for i in range(n_chunks):
            start_idx = max(0, i * chunk_size - overlap if i > 0 else 0)
            end_idx = min(len(df), (i + 1) * chunk_size + overlap)
            if start_idx >= len(df): break
            chunk = df.iloc[start_idx:end_idx].copy()
            if show_progress: print(f"   📦 Chunk {i+1}/{n_chunks}: {start_idx:,} - {end_idx:,} (con overlap)")
            chunk_processed = self._create_all_features_original(chunk, is_inference)
            if i > 0: chunk_processed = chunk_processed.iloc[overlap:].copy()
            if end_idx < len(df): chunk_processed = chunk_processed.iloc[:-overlap].copy()
            chunks_processed.append(chunk_processed)
            del chunk; gc.collect()
        print("🔗 Concatenando chunks con overlap...")
        result = pd.concat(chunks_processed, ignore_index=True)
        del chunks_processed; gc.collect()
        return result

    def create_advanced_degradation_features(self, df):
        """
        Crea features avanzadas que capturan la "velocidad" y "aceleración"
        de la degradación de los sensores.
        """
        # ### INICIO DE LA CORRECCIÓN ###
        if self.logger:
            self.logger.log_step("  📈 Creando features de degradación avanzada (velocidad/aceleración)...")
        # ### FIN DE LA CORRECCIÓN ###
        df_features = df.copy()
        
        # Usaremos una ventana de 24 horas como base para las tendencias
        window_samples = int(24 * 3600 * self.config.SENSOR_CONFIG['sampling_frequency_hz'])
        
        sensor_columns = self.config.SENSOR_CONFIG.get('sensor_columns', [])
        for sensor in sensor_columns:
            if sensor not in df_features.columns:
                continue

            # Usamos una media móvil más suave para calcular tendencias
            rolling_mean_col = f'{sensor}_ma_24h'
            if rolling_mean_col not in df_features.columns:
                 # Si no existe, la calculamos aquí
                 df_features[rolling_mean_col] = df_features[sensor].rolling(window_samples, min_periods=100).mean()

            # 1. Velocidad de Degradación (Primera Derivada)
            # Cuán rápido está cambiando el sensor ahora mismo.
            df_features[f'{sensor}_trend_velocity'] = df_features[rolling_mean_col].diff().fillna(0)

            # 2. Aceleración de Degradación (Segunda Derivada)
            # ¿Se está acelerando el cambio? Una señal muy potente de fallo inminente.
            df_features[f'{sensor}_trend_acceleration'] = df_features[f'{sensor}_trend_velocity'].diff().fillna(0)

        return df_features
    
    def _create_features_balanced(self, df, is_inference):
        df_features = df.copy()
        print("  ⏰ Features temporales optimizados...")
        df_features = self.create_temporal_features_optimized(df_features)
        print("  📡 Features de sensores básicos...")
        df_features = self._create_sensor_features_basic(df_features)
        print("  📉 Features de degradación selectivos...")
        df_features = self._create_degradation_features_selective(df_features)
        print("  ⚙️  Features operacionales...")
        df_features = self.create_operational_stress_features_optimized(df_features)
        if not is_inference and 'tipo_vehiculo' in df_features.columns:
            print("  🔄 Features de ciclo...")
            df_features = self.create_cycle_efficiency_features(df_features)
        if self._should_use_spectral_features():
            print("  🌊 Features espectrales básicos...")
            df_features = self._create_spectral_features_basic(df_features)
        return df_features

    def _create_features_optimized(self, df, is_inference):
        df_features = df.copy()
        print("  ⏰ Features temporales optimizados...")
        df_features = self.create_temporal_features_optimized(df_features)
        print("  📡 Features de sensores esenciales...")
        df_features = self._create_sensor_features_essential(df_features)
        print("  📉 Features de degradación mínimos...")
        df_features = self._create_degradation_features_minimal(df_features)
        print("  ⚙️  Features operacionales básicos...")
        df_features = self.create_operational_stress_features_optimized(df_features)
        return df_features


    def _create_all_features_original(self, df, is_inference, y=None):
        # Esta función ahora es interna y no se llama desde fuera
        df_features = df.copy()
        df_features = self.create_temporal_features_optimized(df_features)
        df_features = self._create_sensor_features_complete(df_features)
        df_features = self._create_degradation_features_selective(df_features)
        df_features = self.create_advanced_degradation_features(df_features)
        df_features = self.create_operational_stress_features_optimized(df_features)
        if 'tipo_vehiculo' in df_features.columns:
            print("  🔄 Creando features de eficiencia...")
        df_features = self.create_cycle_efficiency_features(df_features)
        df_features = self._create_spectral_features_optimized(df_features)
        df_features = self.remove_outliers(df_features)
        return df_features


    def create_temporal_features_optimized(self, df):
        df_features = df.copy()
        if self.timestamp_col_name not in df_features.columns: return df_features
        if not pd.api.types.is_datetime64_any_dtype(df_features[self.timestamp_col_name]):
            df_features[self.timestamp_col_name] = pd.to_datetime(
                df_features[self.timestamp_col_name],
                format='ISO8601',
                errors='coerce'
            ) 
            
        dt = df_features[self.timestamp_col_name].dt
        df_features['hora_dia'] = dt.hour.astype('int8')
        df_features['dia_semana'] = dt.dayofweek.astype('int8')
        df_features['mes'] = dt.month.astype('int8')
        df_features['es_fin_semana'] = (dt.dayofweek >= 5).astype('int8')
        df_features['es_horario_nocturno'] = ((dt.hour >= 22) | (dt.hour <= 6)).astype('int8')
        semanas = dt.isocalendar().week
        df_features['semana_operacion'] = (semanas - semanas.min() + 1).astype('int16')
        hour_norm = dt.hour / 24.0
        df_features['hora_sin'] = np.sin(2 * np.pi * hour_norm).astype('float32')
        df_features['hora_cos'] = np.cos(2 * np.pi * hour_norm).astype('float32')
        day_norm = dt.dayofweek / 7.0
        df_features['dia_sin'] = np.sin(2 * np.pi * day_norm).astype('float32')
        df_features['dia_cos'] = np.cos(2 * np.pi * day_norm).astype('float32')
        return df_features

    def _create_sensor_features_basic(self, df):
        sensor_columns = self.config.SENSOR_CONFIG.get('sensor_columns', [])
        sensor_cols = [col for col in sensor_columns if col in df.columns]
        if not sensor_cols: return df
        sensor_data = df[sensor_cols].values.astype('float32')
        df['sensor_mean'] = np.mean(sensor_data, axis=1)
        df['sensor_std'] = np.std(sensor_data, axis=1)
        df['sensor_max'] = np.max(sensor_data, axis=1)
        df['sensor_min'] = np.min(sensor_data, axis=1)
        df['sensor_range'] = df['sensor_max'] - df['sensor_min']
        if len(sensor_cols) >= 2: df['sensor_ratio_01'] = sensor_data[:, 0] / (sensor_data[:, 1] + 1e-8)
        return df

    def _create_sensor_features_essential(self, df):
        sensor_columns = self.config.SENSOR_CONFIG.get('sensor_columns', [])
        sensor_cols = [col for col in sensor_columns if col in df.columns]
        if not sensor_cols: return df
        sensor_data = df[sensor_cols].values.astype('float32')
        df['sensor_mean'] = np.mean(sensor_data, axis=1)
        df['sensor_std'] = np.std(sensor_data, axis=1)
        return df

    def _create_degradation_features_selective(self, df):
        """
        VERSIÓN CORREGIDA: Asegura que las ventanas móviles siempre se puedan calcular.
        """
        sensor_columns = self.config.SENSOR_CONFIG.get('sensor_columns', [])
        sampling_freq = self.config.SENSOR_CONFIG.get('sampling_frequency_hz', 1)
        # Usaremos las ventanas de 1 y 6 horas que son más seguras
        windows_hours = [1, 6] 
        
        for sensor in sensor_columns:
            if sensor not in df.columns: continue
            
            # El baseline no cambia
            baseline_samples = min(int(24 * 3600 * sampling_freq), len(df) // 4)
            self.baseline_values[sensor] = df[sensor].iloc[:baseline_samples].mean() if baseline_samples > 0 else df[sensor].mean()

            for window_hours in windows_hours:
                window_samples_config = int(window_hours * 3600 * sampling_freq)
                
                window_samples_effective = min(window_samples_config, len(df) - 1)
                
                if window_samples_effective > 10: # Solo si la ventana es útil
                    df[f'{sensor}_ma_{window_hours}h'] = df[sensor].rolling(window_samples_effective, min_periods=max(1, window_samples_effective//4)).mean().astype('float32')
                    df[f'{sensor}_deg_ratio_{window_hours}h'] = (df[sensor] / self.baseline_values[sensor]).astype('float32')

        return df

    def _create_degradation_features_minimal(self, df):
        sensor_columns = self.config.SENSOR_CONFIG.get('sensor_columns', [])
        sampling_freq = self.config.SENSOR_CONFIG.get('sampling_frequency_hz', 1)
        window_samples = int(1 * 3600 * sampling_freq)
        for sensor in sensor_columns:
            if sensor not in df.columns: continue
            if window_samples < len(df) and window_samples > 10:
                df[f'{sensor}_ma_1h'] = df[sensor].rolling(window_samples, min_periods=max(1, window_samples//4)).mean().astype('float32')
        return df

    def create_operational_stress_features_optimized(self, df):
        df_features = df.copy()
        if self.timestamp_col_name not in df_features.columns: return df_features
        time_elapsed = (df_features[self.timestamp_col_name] - df_features[self.timestamp_col_name].iloc[0])
        df_features['tiempo_operacion_horas'] = (time_elapsed.dt.total_seconds() / 3600).astype('float32')
        if 'horas_operacion_acumuladas_componente_X' in df_features.columns:
            horas_op = df_features['horas_operacion_acumuladas_componente_X']
            tiempo_cal = df_features['tiempo_operacion_horas']
            df_features['ratio_utilizacion'] = (horas_op / tiempo_cal.replace(0, 1)).astype('float32')
            df_features['intensidad_operacional'] = horas_op.diff().fillna(0).astype('float32')
        return df_features

    def _create_spectral_features_basic(self, df):
        sensor_columns = self.config.SENSOR_CONFIG.get('sensor_columns', [])
        if sensor_columns and sensor_columns[0] in df.columns:
            sensor = sensor_columns[0]
            window_samples = min(1000, len(df) // 10)
            if window_samples > 50:
                df[f'{sensor}_spectral_energy'] = df[sensor].rolling(window_samples, min_periods=window_samples//2).var().fillna(0).astype('float32')
        return df
    
    def create_cycle_efficiency_features(self, df):
        
        df_features = df.copy()
        if 'tipo_vehiculo' not in df_features.columns: 
            return df_features
        
        df_features['cambio_vehiculo'] = (df_features['tipo_vehiculo'] != df_features['tipo_vehiculo'].shift()).astype(int)
        df_features['ciclo_id'] = df_features['cambio_vehiculo'].cumsum()
        
        if self.timestamp_col_name not in df_features.columns:
            raise KeyError(f"Timestamp column '{self.timestamp_col_name}' not found for cycle efficiency features.")
        
        ciclo_stats = df_features.groupby('ciclo_id')[self.timestamp_col_name].agg(['min', 'max', 'count'])
        ciclo_stats['duracion_segundos'] = (ciclo_stats['max'] - ciclo_stats['min']).dt.total_seconds()
        ciclo_stats['num_muestras'] = ciclo_stats['count']
        df_features = df_features.merge(ciclo_stats[['duracion_segundos', 'num_muestras']], left_on='ciclo_id', right_index=True, how='left')
        
        if len(ciclo_stats) > 10:
            baseline_duracion = ciclo_stats['duracion_segundos'].quantile(0.1)
            df_features['eficiencia_relativa'] = baseline_duracion / df_features['duracion_segundos'].replace(0, 1)
        
        df_features['posicion_en_ciclo'] = (df_features.groupby('ciclo_id').cumcount() / df_features['num_muestras'].replace(0, 1))
        
        return df_features

    def clean_features_for_rul_prediction(self, df):
            """
            Elimina features que pueden causar data leakage en predicción RUL
            """
            df_clean = df.copy()
            dropped_features = []

            # 1. Eliminar features con 'rul' en el nombre
            rul_features = [col for col in df_clean.columns if 'rul' in col.lower()]
            if rul_features:
                dropped_features.extend(rul_features)
                df_clean = df_clean.drop(columns=rul_features)
                logger.info(f"Removed RUL-related features: {rul_features}")

            # 2. Eliminar features circulares (volumen_real_*)
            circular_features = [col for col in df_clean.columns if 'volumen_real_' in col.lower()]
            if circular_features:
                dropped_features.extend(circular_features)
                df_clean = df_clean.drop(columns=circular_features)
                logger.info(f"Removed circular features: {circular_features}")

            # 3. Eliminar otras features que puedan tener data leakage
            leakage_keywords = ['futuro', 'siguiente', 'proximo', 'predicho']
            leakage_features = [
                col for col in df_clean.columns 
                if any(keyword in col.lower() for keyword in leakage_keywords)
            ]
            if leakage_features:
                dropped_features.extend(leakage_features)
                df_clean = df_clean.drop(columns=leakage_features)
                logger.info(f"Removed other potential leakage features: {leakage_features}")

            # 4. Validar que no quedan features sospechosas
            suspicious_pattern = re.compile(r'(futuro|siguiente|proximo|predicho|rul)', re.IGNORECASE)
            suspicious_features = [
                col for col in df_clean.columns 
                if suspicious_pattern.search(col)
            ]
            if suspicious_features:
                logger.warning(f"Warning: Potentially suspicious features remaining: {suspicious_features}")

            # Log resumen
            logger.info(f"Total features removed: {len(dropped_features)}")
            logger.info(f"Features remaining: {len(df_clean.columns)} of {len(df.columns)}")
            
            return df_clean, dropped_features

    def remove_outliers(self, df, method='iqr', factor=1.5):
        df_clean = df.copy()
        sensor_columns = self.config.SENSOR_CONFIG.get('sensor_columns', [])
        for sensor in sensor_columns:
            if sensor not in df_clean.columns: continue
            if method == 'iqr':
                Q1, Q3 = df_clean[sensor].quantile(0.25), df_clean[sensor].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound, upper_bound = Q1 - factor * IQR, Q3 + factor * IQR
                outliers = (df_clean[sensor] < lower_bound) | (df_clean[sensor] > upper_bound)
                df_clean[f'{sensor}_is_outlier'] = outliers.astype(int)
                extreme_outliers = (df_clean[sensor] < lower_bound - IQR) | (df_clean[sensor] > upper_bound + IQR)
                df_clean.loc[extreme_outliers, sensor] = np.nan
                df_clean[sensor] = df_clean[sensor].interpolate(method='linear', limit_direction='both')
        return df_clean

    def _process_by_chunks_optimized_complete(self, df, chunk_size, overlap, is_inference, show_progress):
        chunks_processed = []
        n_chunks = (len(df) // chunk_size) + 1
        print(f"📊 Procesando {n_chunks} chunks con overlap optimizado...")
        for i in range(n_chunks):
            start_idx = max(0, i * chunk_size - overlap if i > 0 else 0)
            end_idx = min(len(df), (i + 1) * chunk_size + overlap)
            if start_idx >= len(df): break
            chunk = df.iloc[start_idx:end_idx].copy()
            if show_progress: print(f"   📦 Chunk {i+1}/{n_chunks}: {start_idx:,} - {end_idx:,}")
            chunk_processed = self._create_features_complete_optimized(chunk, is_inference)
            if i > 0: chunk_processed = chunk_processed.iloc[overlap:].copy()
            if end_idx < len(df): chunk_processed = chunk_processed.iloc[:-overlap].copy()
            chunks_processed.append(chunk_processed)
            del chunk; gc.collect()
        print("🔗 Concatenando chunks optimizados...")
        result = pd.concat(chunks_processed, ignore_index=True)
        del chunks_processed; gc.collect()
        return result

    def _create_features_complete_optimized(self, df, is_inference):
        df_features = df.copy()
        print("  ⏰ Features temporales optimizados...")
        df_features = self.create_temporal_features_optimized(df_features)
        print("  📡 Features de sensores completos...")
        df_features = self._create_sensor_features_complete(df_features)
        print("  📉 Features de degradación selectivos...")
        df_features = self._create_degradation_features_selective(df_features)
        print("  ⚙️ Features operacionales...")
        df_features = self.create_operational_stress_features_optimized(df_features)
        if not is_inference and 'tipo_vehiculo' in df_features.columns:
            print("  🔄 Features de ciclo...")
            df_features = self.create_cycle_efficiency_features(df_features)
        if self._should_use_spectral_features():
            print("  🌊 Features espectrales optimizados...")
            df_features = self._create_spectral_features_optimized(df_features)
        return df_features

    def _create_sensor_features_complete(self, df):
        sensor_columns = self.config.SENSOR_CONFIG.get('sensor_columns', [])
        sensor_cols = [col for col in sensor_columns if col in df.columns]
        if not sensor_cols: return df
        sensor_data = df[sensor_cols].values.astype('float32')
        df['sensor_mean'] = np.mean(sensor_data, axis=1)
        df['sensor_std'] = np.std(sensor_data, axis=1)
        df['sensor_max'] = np.max(sensor_data, axis=1)
        df['sensor_min'] = np.min(sensor_data, axis=1)
        df['sensor_range'] = df['sensor_max'] - df['sensor_min']
        df['sensor_median'] = np.median(sensor_data, axis=1)
        # Skew can be slow, using pandas for robustness
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            df['sensor_skew'] = pd.DataFrame(sensor_data).skew(axis=1).values
        for i in range(len(sensor_cols)):
            for j in range(i+1, len(sensor_cols)):
                df[f'sensor_ratio_{i}{j}'] = sensor_data[:, i] / (sensor_data[:, j] + 1e-8)
        return df

    def _create_spectral_features_optimized(self, df):
        sensor_columns = self.config.SENSOR_CONFIG.get('sensor_columns', [])
        for sensor in sensor_columns[:2]:
            if sensor not in df.columns: continue
            window_samples = min(500, len(df) // 20)
            if window_samples > 50:
                df[f'{sensor}_spectral_energy'] = df[sensor].rolling(window_samples, min_periods=window_samples//2).var().fillna(0).astype('float32')
        return df

    def _get_optimal_processing_mode(self, dataset_size, available_memory, target_time_minutes):
        if dataset_size < 100000: suggested_mode = 'development'
        elif dataset_size < 1000000: suggested_mode = 'balanced'
        elif dataset_size < 5000000: suggested_mode = 'optimized'
        else: suggested_mode = 'complete'
        if available_memory < 4 and suggested_mode in ['complete', 'balanced']: suggested_mode = 'optimized'
        if target_time_minutes and target_time_minutes < 30 and suggested_mode in ['complete', 'balanced']: suggested_mode = 'optimized'
        return suggested_mode

    def _get_processing_mode_config(self, mode):
        configs = {
            'development': {'description': 'Rápido para desarrollo', 'expected_time_minutes': 5, 'memory_limit_gb': 2},
            'balanced': {'description': 'Balance velocidad-precisión', 'expected_time_minutes': 30, 'memory_limit_gb': 4},
            'optimized': {'description': 'Máxima velocidad', 'expected_time_minutes': 20, 'memory_limit_gb': 3},
            'complete': {'description': 'Máxima precisión', 'expected_time_minutes': 60, 'memory_limit_gb': 6}
        }
        return configs.get(mode, configs['balanced'])

    def _get_feature_configuration(self, mode):
        configs = {
            'development': {'temporal_windows_hours': [1, 6, 24, 72], 'use_spectral_features': True, 'use_cycle_features': True},
            'balanced': {'temporal_windows_hours': [1, 6, 24], 'use_spectral_features': True, 'use_cycle_features': True},
            'optimized': {'temporal_windows_hours': [1, 6], 'use_spectral_features': False, 'use_cycle_features': False},
            'complete': {'temporal_windows_hours': [1, 6, 24, 72, 168], 'use_spectral_features': True, 'use_cycle_features': True}
        }
        return configs.get(mode, configs['balanced'])

    def _apply_mode_configuration(self, feature_config):
        self.config.SENSOR_CONFIG['temporal_windows_hours'] = feature_config['temporal_windows_hours']
        self.current_feature_config = feature_config
        print(f"⚙️  CONFIGURACIÓN APLICADA:")
        print(f"   🕐 Ventanas temporales: {feature_config['temporal_windows_hours']}")
        print(f"   🌊 Features espectrales: {feature_config['use_spectral_features']}")
        print(f"   🔄 Features de ciclo: {feature_config['use_cycle_features']}")

    def _should_use_spectral_features(self):
        return self.current_feature_config and self.current_feature_config.get('use_spectral_features', False)

    def _optimize_dataframe_memory(self, df):
        memory_before = df.memory_usage(deep=True).sum() / 1024**2
        df_opt = df.copy()
        for col in df_opt.select_dtypes(include=['int64', 'float64']).columns:
            if df_opt[col].dtype == 'int64':
                if df_opt[col].min() >= 0:
                    if df_opt[col].max() < 255: df_opt[col] = df_opt[col].astype('uint8')
                    elif df_opt[col].max() < 65535: df_opt[col] = df_opt[col].astype('uint16')
                else:
                    if df_opt[col].min() >= -128 and df_opt[col].max() <= 127: df_opt[col] = df_opt[col].astype('int8')
                    elif df_opt[col].min() >= -32768 and df_opt[col].max() <= 32767: df_opt[col] = df_opt[col].astype('int16')
            elif df_opt[col].dtype == 'float64':
                df_opt[col] = df_opt[col].astype('float32')
        memory_after = df_opt.memory_usage(deep=True).sum() / 1024**2
        reduction = ((memory_before - memory_after) / memory_before * 100) if memory_before > 0 else 0
        if reduction > 5: print(f"   📉 Memoria optimizada: {memory_before:.1f}MB → {memory_after:.1f}MB ({reduction:.1f}% menos)")
        return df_opt

    def _generate_processing_report(self, df_original, df_features, start_time, end_time, mode):
        elapsed_time = end_time - start_time
        records_per_second = len(df_features) / elapsed_time if elapsed_time > 0 else 0
        added_features = len(df_features.columns) - len(df_original.columns)
        memory_usage_mb = df_features.memory_usage(deep=True).sum() / 1024**2
        print(f"\n📊 REPORTE DE PROCESAMIENTO HÍBRIDO")
        print("=" * 50)
        print(f"🎯 Modo utilizado: {mode.upper()}")
        print(f"⏱️  Tiempo total: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
        print(f"🚀 Velocidad: {records_per_second:,.0f} registros/segundo")
        print(f"📊 Registros procesados: {len(df_features):,}")
        print(f"📈 Features añadidas: {added_features}")
        print(f"📊 Columnas finales: {len(df_features.columns)}")
        print(f"💾 Memoria utilizada: {memory_usage_mb:.1f} MB")
        self.processing_stats = {
            'mode': mode, 'processing_time_seconds': elapsed_time, 'records_per_second': records_per_second,
            'features_added': added_features, 'memory_usage_mb': memory_usage_mb, 'dataset_size': len(df_features)
        }
        print("✅ Feature Engineering Híbrido completado exitosamente!")

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'logger' in state: del state['logger']
        if 'checkpoint_manager' in state: del state['checkpoint_manager']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logger = None
        self.checkpoint_manager = None