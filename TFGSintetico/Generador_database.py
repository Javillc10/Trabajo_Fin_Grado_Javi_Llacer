"""
Módulo Generador Base para el Dataset de Mantenimiento Predictivo.
VERSIÓN CORREGIDA
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from tqdm import tqdm
import gc
import json
import os
import sys
import psutil
import logging
from pathlib import Path
from contextlib import contextmanager
import sqlite3
import codecs

warnings.filterwarnings('ignore')

# =============================================================================
# FUNCIONES DE FÍSICA Y ESTADO (SIN CAMBIOS)
# =============================================================================
def calculate_realistic_rul_definitivo(component_hours: float, stress_factors: dict, base_life: int = 2000, component_type: str = 'bomba_aceite', health_factor: float = 1.0) -> float:
    component_params = {'bomba_aceite': {'base_life': 1800, 'stress_sensitivity': 1.4, 'degradation_factor': 1.2}, 'filtro_frenos': {'base_life': 1200, 'stress_sensitivity': 1.8, 'degradation_factor': 1.5}, 'valvula_refrigerante': {'base_life': 2200, 'stress_sensitivity': 1.0, 'degradation_factor': 1.1}}
    params = component_params.get(component_type, component_params['bomba_aceite'])
    temp_factor = max(0.4, min(2.5, (stress_factors.get('temperatura', 22) - 15) / 15))
    pressure_factor = max(0.6, min(2.0, stress_factors.get('pressure_deviation', 0) + 1))
    load_factor = max(0.7, min(1.6, stress_factors.get('carga_operacional', 1.0)))
    combined_stress = (temp_factor * 0.35 + pressure_factor * 0.35 + load_factor * 0.3) * params['stress_sensitivity']
    health_adjusted_life = params['base_life'] * health_factor
    adjusted_life = health_adjusted_life / combined_stress
    rul_hours = max(0, adjusted_life - component_hours * 1.1)
    rul_hours *= np.random.normal(1.0, 0.25)
    rul_days = rul_hours / 24
    usage_ratio = component_hours / params['base_life']
    if usage_ratio > 0.7:
        acceleration_factor = 1 - (usage_ratio - 0.7) * 2
        rul_days *= max(0.1, acceleration_factor)
    return max(0.5, min(25.0, rul_days))

def generate_realistic_sensor_reading_definitivo(base_value: float, time_hours: float, component_health: float, operational_conditions: dict, degradation_severity: float = 1.0) -> float:
    noise_scale = base_value * (0.025 + (1 - component_health) * 0.015)
    measurement_noise = np.random.normal(0, noise_scale)
    temporal_drift = base_value * (0.002 * (time_hours / 100) + 0.001 * np.sin(time_hours / 24))
    degradation_effect = base_value * (1 - max(0.2, component_health)) * 0.2 * degradation_severity
    temp_factor = operational_conditions.get('temperatura_factor', 0)
    operational_noise = base_value * (0.04 * temp_factor + 0.01 * temp_factor**2)
    final_value = (base_value + measurement_noise + temporal_drift + degradation_effect + operational_noise)
    return max(0.1, final_value)

def determine_system_state_definitivo(rul_days: dict, component_healths: dict) -> str:
    min_rul = min(rul_days.values())
    if min_rul <= 1:
        componente_critico = min(rul_days, key=rul_days.get)
        return f"Fallo_Inminente_{componente_critico.replace('_', '_').title()}"
    if min_rul <= 3: return "RUL_Critico"
    if min_rul <= 7: return "Degradacion_Avanzada"
    if min_rul <= 12: return "Degradacion_Moderada"
    salud_promedio = np.mean(list(component_healths.values()))
    if salud_promedio < 0.5: return 'Degradacion_Severa'
    elif salud_promedio < 0.7: return 'Degradacion_Gradual'
    rand = np.random.random()
    if rand < 0.92: return 'Normal_Operacion'
    elif rand < 0.96: return 'Degradacion_Gradual'
    else: return np.random.choice(['Fuga_Menor', 'Calibracion_Desviada'])

# =============================================================================
# CLASE GENERADORA BASE (PADRE)
# =============================================================================

class EstacionLlenadoDatasetGeneradorDefinitivo:
    def __init__(self, config: dict = None):
        self.config = {
            'chunk_size': 50000, # Reducido para decisiones de estado más frecuentes
            'database_name': 'test_data.db',
            'output_dir': 'dataset_rul_variable_DEFINITIVO',
            'seed': 42
        }
        if config:
            self.config.update(config)
        
        os.makedirs(self.config['output_dir'], exist_ok=True)
        self._setup_logging()
        
        self.FRECUENCIA_HZ = 20
        self.INTERVALO_MS = 1000 / self.FRECUENCIA_HZ
        
        ## CORRECCIÓN ## (Bug #2) - Se calcula MUESTRAS_TOTALES a partir de las horas.
        total_hours = self.config.get('total_hours', 10) # Default a 10 si no se provee
        self.MUESTRAS_TOTALES = int(total_hours * 3600 * self.FRECUENCIA_HZ)
        
        self.FECHA_INICIO = datetime(2024, 5, 27, 6, 0, 0)
        
        self._init_component_states_definitivo()
        self._setup_db_schema()
        
        self.generacion_estado = {'muestras_generadas': 0, 'tiempo_inicio': None, 'registros_insertados': 0}
        self.metricas = {'rul_min': float('inf'), 'rul_max': 0}
        self.db_path = os.path.join(self.config['output_dir'], self.config['database_name'])
        
        self.logger.info(f"✅ Generador BASE (Padre) inicializado para {total_hours} horas ({self.MUESTRAS_TOTALES:,} muestras).")

    # ... (Los métodos _setup_db_schema, _init_component_states_definitivo, etc. se mantienen igual hasta llegar a _generar_muestra_definitiva) ...
    def _setup_db_schema(self):
        self.DB_SCHEMA = {'datos_principales': {'tabla': 'datos_estacion', 'columnas': ['id INTEGER PRIMARY KEY', 'timestamp TEXT', 'bloque_productivo INTEGER', 'vehiculo_id INTEGER', 'tipo_vehiculo TEXT', 'estado_sistema TEXT', 'sensor_presion_aceite_bar REAL', 'sensor_presion_frenos_bar REAL', 'sensor_presion_refrigerante_bar REAL', 'temperatura_ambiente_celsius REAL', 'volumen_real_aceite_ml INTEGER', 'volumen_objetivo_aceite_ml INTEGER', 'volumen_real_frenos_ml INTEGER', 'volumen_objetivo_frenos_ml INTEGER', 'volumen_real_refrigerante_ml INTEGER', 'volumen_objetivo_refrigerante_ml INTEGER', 'horas_operacion_acumuladas_maquina REAL', 'horas_operacion_comp_aceite REAL', 'horas_operacion_comp_frenos REAL', 'horas_operacion_comp_refrigerante REAL', 'rul_dias_aceite REAL', 'rul_dias_frenos REAL', 'rul_dias_refrigerante REAL', 'timestamp_unix INTEGER', 'semana_del_mes INTEGER']}, 'metadatos': {'tabla': 'metadatos_generacion', 'columnas': ['id INTEGER PRIMARY KEY', 'fecha_generacion TEXT', 'version TEXT', 'total_muestras INTEGER', 'configuracion_json TEXT', 'metricas_json TEXT']}}

    def _init_component_states_definitivo(self):
        np.random.seed(self.config['seed'])
        self.component_states = {}
        
        # inicio realista con componentes que tienen algo de uso pero están saludables.
        initial_conditions = {
            'bomba_aceite':       {'hours': (200, 600), 'health': (0.7, 1.0), 'base_life': 1800},
            'filtro_frenos':      {'hours': (100, 500), 'health': (0.7, 1.0), 'base_life': 1500},
            'valvula_refrigerante': {'hours': (500, 1000), 'health': (0.6, 0.9), 'base_life': 2200}
        }        
        
        for name, cond in initial_conditions.items():
            self.component_states[name] = {'hours_operated': np.random.uniform(*cond['hours']), 'health': np.random.uniform(*cond['health']), 'base_life': cond['base_life'], 'degradation_rate': np.random.uniform(0.002, 0.004), 'degradation_acceleration': np.random.uniform(1.2, 1.8), 'stress_accumulator': 0}
        self.logger.info("Estados iniciales BASE de componentes definidos.")

    def _es_horario_productivo(self, timestamp: datetime) -> bool:
        hora = timestamp.hour
        dia_semana = timestamp.weekday()
        if dia_semana == 6: return False
        if 6 <= hora < 22: return True
        if 22 <= hora or hora < 6:
            if dia_semana == 5: return False
            return dia_semana % 2 == 0
        return False

    def _actualizar_estados_componentes_definitivo(self, horas_incremento: float, is_working: bool):
        for comp_state in self.component_states.values():
            if is_working:
                stress_multiplier = 1.0 + comp_state['stress_accumulator']
                current_hours_increment = horas_incremento * stress_multiplier * comp_state['degradation_acceleration']
                comp_state['hours_operated'] += current_hours_increment
                degradation = comp_state['degradation_rate'] * current_hours_increment * np.random.uniform(0.5, 2.0)
                if comp_state['health'] < 0.6: degradation *= 1.5
                comp_state['health'] = max(0.15, comp_state['health'] - degradation)
                comp_state['stress_accumulator'] = min(0.5, comp_state['stress_accumulator'] + np.random.uniform(0.001, 0.005))
            else: # Reposo
                comp_state['stress_accumulator'] = max(0, comp_state['stress_accumulator'] - 0.001)

    ## CORRECCIÓN ## (Bug #1) - Este método ahora solo genera UNA fila, asumiendo que el estado ya fue decidido.
    def _generar_fila_unica(self, timestamp, muestra_idx, is_working, bloque_productivo, vehiculo_id, tipo_vehiculo):
        horas_transcurridas = (muestra_idx * self.INTERVALO_MS) / (3600 * 1000)
        
        temperatura_ambiente = 22 + 8 * np.sin(2 * np.pi * (timestamp.hour / 24 - 0.25)) + np.random.normal(0, 1.5)
        stress_factors = {'temperatura': temperatura_ambiente, 'pressure_deviation': 0, 'carga_operacional': 1.0 if is_working else 0.2}
        
        component_healths, rul_days = {}, {}
        for name, state in self.component_states.items():
            component_healths[name] = state['health']
            rul_days[name] = calculate_realistic_rul_definitivo(state['hours_operated'], stress_factors, state['base_life'], name, state['health'])

        sensor_readings = {}
        comp_map = {'bomba_aceite': ('sensor_presion_aceite_bar', 2.75, {'filtro_frenos': 0.1, 'valvula_refrigerante': 0.15}), 'filtro_frenos': ('sensor_presion_frenos_bar', 2.5, {'bomba_aceite': 0.05, 'valvula_refrigerante': 0.05}), 'valvula_refrigerante': ('sensor_presion_refrigerante_bar', 2.0, {'bomba_aceite': 0.2, 'filtro_frenos': 0.1})}
        for name, (sensor_name, base_val, crosstalk_factors) in comp_map.items():
            state, cond_op = self.component_states[name], {'temperatura_factor': (temperatura_ambiente - 22) / 18}
            main_reading = generate_realistic_sensor_reading_definitivo(base_val, state['hours_operated'], state['health'], cond_op)
            crosstalk_effect = sum((1 - component_healths[other_comp]) * base_val * factor * np.random.uniform(0.5, 1.5) for other_comp, factor in crosstalk_factors.items())
            sensor_readings[sensor_name] = main_reading + crosstalk_effect
        
        estado_sistema = determine_system_state_definitivo(rul_days, component_healths)
        if not is_working: estado_sistema = 'INACTIVO'
        
        volumenes = {'aceite': 4500, 'frenos': 800, 'refrigerante': 10000}
        self._actualizar_metricas_rul(rul_days)

        return {
            'id_muestra': muestra_idx, 'timestamp': timestamp.isoformat(), 'bloque_productivo': bloque_productivo, 'vehiculo_id': vehiculo_id,
            'tipo_vehiculo': tipo_vehiculo, 'estado_sistema': estado_sistema, **sensor_readings,
            'temperatura_ambiente_celsius': temperatura_ambiente,
            'volumen_real_aceite_ml': int(volumenes['aceite'] * np.random.normal(1, 0.05)) if is_working else 0, 'volumen_objetivo_aceite_ml': volumenes['aceite'],
            'volumen_real_frenos_ml': int(volumenes['frenos'] * np.random.normal(1, 0.05)) if is_working else 0, 'volumen_objetivo_frenos_ml': volumenes['frenos'],
            'volumen_real_refrigerante_ml': int(volumenes['refrigerante'] * np.random.normal(1, 0.05)) if is_working else 0, 'volumen_objetivo_refrigerante_ml': volumenes['refrigerante'],
            'horas_operacion_acumuladas_maquina': horas_transcurridas,
            'horas_operacion_comp_aceite': self.component_states['bomba_aceite']['hours_operated'], 'horas_operacion_comp_frenos': self.component_states['filtro_frenos']['hours_operated'],
            'horas_operacion_comp_refrigerante': self.component_states['valvula_refrigerante']['hours_operated'],
            'rul_dias_aceite': rul_days['bomba_aceite'], 'rul_dias_frenos': rul_days['filtro_frenos'], 'rul_dias_refrigerante': rul_days['valvula_refrigerante'],
            'timestamp_unix': int(timestamp.timestamp()), 'semana_del_mes': (timestamp.day - 1) // 7 + 1
        }
    
    ## CORRECCIÓN ## (Bug #1) - Lógica de generación completamente refactorizada.
    def generar_dataset_completo(self):
        self.logger.info(f"🎯 INICIANDO GENERACIÓN DATASET BASE ({self.config['database_name']})")
        self._crear_base_datos()
        self.generacion_estado['tiempo_inicio'] = datetime.now()
        
        total_chunks = (self.MUESTRAS_TOTALES // self.config['chunk_size']) + 1
        
        current_vehiculo_id = 1
        
        for chunk_idx in tqdm(range(total_chunks), desc="Generando chunks BASE", mininterval=1.0):
            inicio = chunk_idx * self.config['chunk_size']
            fin = min(inicio + self.config['chunk_size'], self.MUESTRAS_TOTALES)
            if inicio >= self.MUESTRAS_TOTALES: break
            
            # 1. DECIDIR EL ESTADO PARA TODO EL CHUNK
            timestamp_chunk_start = self.FECHA_INICIO + timedelta(milliseconds=inicio * self.INTERVALO_MS)
            is_working = self._es_horario_productivo(timestamp_chunk_start)
            
            if is_working:
                bloque_productivo = (timestamp_chunk_start.hour % 3) + 1
                tipo_vehiculo = np.random.choice(['Sedan', 'SUV'], p=[0.6, 0.4])
                current_vehiculo_id += 1
            else:
                bloque_productivo = 0
                tipo_vehiculo = 'INACTIVO'

            # 2. ACTUALIZAR ESTADO DE COMPONENTES PARA EL CHUNK
            horas_chunk = (self.config['chunk_size'] * self.INTERVALO_MS) / (1000 * 3600)
            self._actualizar_estados_componentes_definitivo(horas_chunk, is_working)

            # 3. GENERAR FILAS CON EL ESTADO FIJO
            chunk_data = []
            for i in range(inicio, fin):
                timestamp_actual = self.FECHA_INICIO + timedelta(milliseconds=i * self.INTERVALO_MS)
                fila = self._generar_fila_unica(timestamp_actual, i, is_working, bloque_productivo, current_vehiculo_id, tipo_vehiculo)
                chunk_data.append(fila)

            self._insertar_datos_batch(chunk_data)
        
        self._finalizar_metricas()
        self._guardar_metadatos_generacion()
        self.exportar_a_parquet()
        self.logger.info(f"🎉 GENERACIÓN BASE COMPLETADA EXITOSAMENTE")
        return self.db_path, self.metricas

    # --- Métodos Auxiliares (sin cambios) ---
    def _setup_logging(self):
        utf8_writer = codecs.getwriter('utf-8')(sys.stdout.buffer)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(utf8_writer)])
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def get_db_connection(self):
        conn = sqlite3.connect(self.db_path)
        try: yield conn
        finally: conn.close()

    def _crear_base_datos(self):
        if os.path.exists(self.db_path): os.remove(self.db_path)
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            for info in self.DB_SCHEMA.values():
                cursor.execute(f"CREATE TABLE {info['tabla']} ({', '.join(info['columnas'])})")
            conn.commit()
        self.logger.info("Base de datos creada.")

    def _insertar_datos_batch(self, datos):
        if not datos: return
        with self.get_db_connection() as conn:
            df = pd.DataFrame(datos)
            # Renombramos id_muestra a id para que coincida con la PK
            if 'id_muestra' in df.columns:
                df.rename(columns={'id_muestra': 'id'}, inplace=True)
            df.to_sql('datos_estacion', conn, if_exists='append', index=False)
            self.generacion_estado['registros_insertados'] += len(df)

    def _actualizar_metricas_rul(self, rul_days):
        rul_valores = list(rul_days.values())
        self.metricas['rul_min'] = min(self.metricas['rul_min'], min(rul_valores))
        self.metricas['rul_max'] = max(self.metricas['rul_max'], max(rul_valores))

    def _finalizar_metricas(self):
        self.metricas['tiempo_total'] = (datetime.now() - self.generacion_estado['tiempo_inicio']).total_seconds()
        self.logger.info(f"Rango RUL final: {self.metricas['rul_min']:.1f} - {self.metricas['rul_max']:.1f} días")

    def _guardar_metadatos_generacion(self):
        metadatos = {'fecha_generacion': datetime.now().isoformat(), 'version': '5.0_BASE', 'total_muestras': self.generacion_estado['registros_insertados'], 'configuracion_json': json.dumps(self.config, default=str), 'metricas_json': json.dumps(self.metricas, default=str)}
        with self.get_db_connection() as conn:
            pd.DataFrame([metadatos]).to_sql('metadatos_generacion', conn, if_exists='append', index=False)
        self.logger.info("Metadatos guardados.")

    def exportar_a_parquet(self):
        # El nombre del archivo parquet ahora viene del config
        parquet_filename = self.config['database_name'].replace('.db', '.parquet')
        parquet_path = os.path.join(self.config['output_dir'], parquet_filename)
        self.logger.info(f"Exportando a Parquet: {parquet_path}")
        with self.get_db_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM datos_estacion", conn)
        df.to_parquet(parquet_path, engine='pyarrow', compression='snappy', index=False)
        self.logger.info("Exportación a Parquet completada.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generador de datos sintéticos BASE.")
    parser.add_argument('--total-hours', type=int, default=10, help='Número de horas de simulación a generar.')
    args = parser.parse_args()
    config = {'total_hours': args.total_hours}
    print(f"Ejecutando generador BASE (Padre) para {args.total_hours} horas...")
    generador_base = EstacionLlenadoDatasetGeneradorDefinitivo(config)
    generador_base.generar_dataset_completo()

