"""
Módulo Generador Enriquecido para el Dataset de Mantenimiento Predictivo.
VERSIÓN CORREGIDA
"""

import numpy as np
from datetime import datetime
import json
import os
import sqlite3
import pandas as pd
from tqdm import tqdm

# 1. IMPORTAMOS LA CLASE PADRE.
from Generador_database import EstacionLlenadoDatasetGeneradorDefinitivo

# =============================================================================
# CLASE GENERADORA ENRIQUECIDA (HIJA)
# =============================================================================

class EstacionLlenadoDatasetGeneradorOptimizado(EstacionLlenadoDatasetGeneradorDefinitivo):
    def __init__(self, config: dict = None):
        self.enrichment_config = {
            'multiple_runs_per_component': True,
            'target_critical_percentage': 20
        }

        base_config = {
            'database_name': 'train_data.db', # Usamos .db para consistencia
            'output_dir': 'dataset_rul_variable_ENRIQUECIDO',
        }
        if config:
            base_config.update(config)
        
        # ## CORRECCIÓN ## (Bug #2) - Se elimina 'total_samples' fijo.
        # Ahora se usará el cálculo de horas del __init__ del Padre.
        
        super().__init__(base_config)
        
        self.logger.warning("⚠️  Este dataset será ENRIQUECIDO artificialmente para generar más casos críticos.")
        
        self._init_component_states_enriquecido()

    def _init_component_states_enriquecido(self):
        np.random.seed(self.config['seed'])
        self.component_states = {}
        state_probabilities = {'joven': 0.45, 'medio': 0.4, 'viejo': 0.1, 'critico': 0.05}
        initial_state_ranges = {'joven': {'hours': (0, 300), 'health': (0.9, 1.0)}, 'medio': {'hours': (300, 800), 'health': (0.7, 0.9)}, 'viejo': {'hours': (800, 1500), 'health': (0.4, 0.7)}, 'critico': {'hours': (1500, 2000), 'health': (0.15, 0.4)}}
        base_lives = {'bomba_aceite': 1800, 'filtro_frenos': 1200, 'valvula_refrigerante': 2200}

        for name in base_lives.keys():
            chosen_state = np.random.choice(list(state_probabilities.keys()), p=list(state_probabilities.values()))
            ranges = initial_state_ranges[chosen_state]
            self.component_states[name] = {
                'hours_operated': np.random.uniform(*ranges['hours']), 'health': np.random.uniform(*ranges['health']),
                'base_life': base_lives[name],
                'degradation_rate': np.random.uniform(0.01, 0.03),
                'degradation_acceleration': np.random.uniform(3.0, 5.0), 
                'cycle_count': 1, 'stress_accumulator': 0
            }
        self.logger.info("🎯 Estados iniciales ENRIQUECIDOS (mayor probabilidad de degradación) establecidos.")

    def _cycle_component_lifecycle(self, comp_name: str, comp_state: dict):
        if self.enrichment_config['multiple_runs_per_component'] and comp_state['health'] <= 0.15:
            self.logger.info(f"🔄 Componente {comp_name} ha fallado. Iniciando nuevo ciclo de vida.")
            comp_state['cycle_count'] += 1
            comp_state['hours_operated'] = 0
            comp_state['health'] = np.random.uniform(0.9, 1.0)
            comp_state['stress_accumulator'] = 0

    def _actualizar_estados_componentes_definitivo(self, horas_incremento: float, is_working: bool):
        # 1. Llama a la función del PADRE para aplicar la degradación normal.
        super()._actualizar_estados_componentes_definitivo(horas_incremento, is_working)
        
        # 2. AÑADE la nueva lógica de ciclo de vida al final.
        for name, state in self.component_states.items():
            self._cycle_component_lifecycle(name, state)

    ## CORRECCIÓN ## (Bug #3) - Método renombrado y corregido para extender al padre.
    def _guardar_metadatos_generacion(self):
        # 1. Llama al método del padre para que haga todo su trabajo primero.
        super()._guardar_metadatos_generacion()
        
        # 2. Ahora, añade la información específica de enriquecimiento.
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("ALTER TABLE metadatos_generacion ADD COLUMN es_enriquecido BOOLEAN;")
                cursor.execute("ALTER TABLE metadatos_generacion ADD COLUMN config_enriquecimiento_json TEXT;")
            except sqlite3.OperationalError:
                pass # Las columnas ya existen

            update_query = "UPDATE metadatos_generacion SET es_enriquecido = ?, config_enriquecimiento_json = ? WHERE id = (SELECT MAX(id) FROM metadatos_generacion)"
            cursor.execute(update_query, (True, json.dumps(self.enrichment_config, default=str)))
            conn.commit()
        self.logger.info("✅ Metadatos de ENRIQUECIMIENTO guardados.")

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generador de datos sintéticos ENRIQUECIDOS.")
    parser.add_argument('--total-hours', type=int, default=10, help='Número de horas de simulación a generar.')
    args = parser.parse_args()
    config = {'total_hours': args.total_hours}
    print(f"Ejecutando generador ENRIQUECIDO (Hijo) para {args.total_hours} horas...")
    generador_enriquecido = EstacionLlenadoDatasetGeneradorOptimizado(config)
    generador_enriquecido.generar_dataset_completo()
