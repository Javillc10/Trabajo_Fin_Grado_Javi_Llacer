# src/utils/sqlite_adapter.py
"""
Adaptador SQLite para el sistema de mantenimiento predictivo
Proporciona la misma interfaz que CSV pero usando SQLite como backend
"""

import sqlite3
import pandas as pd
import numpy as np
import os
from contextlib import contextmanager
from config import Config
import pandas as pd

class DataAdapter:
    """Adaptador para cargar datos desde diferentes fuentes (Parquet, CSV, SQLite)"""

    def __init__(self, data_path=None):
        """
        Inicializa el DataAdapter.

        Args:
            data_path (str, optional): Ruta al archivo de datos.
        """
        if data_path is None:
            # Get path from config
            self.data_path = Config.DATA_CONFIG['data_path']  # Default to SQLite path
        else:
            self.data_path = data_path

        self.config = Config
        self.source_type = self._determine_source_type()

        print(f"📊 DataAdapter inicializado con: {self.data_path} (Tipo: {self.source_type})")

        if self.source_type == 'sqlite':
            self._validate_database()

    def _determine_source_type(self):
        """Determina el tipo de fuente de datos basado en la extensión del archivo."""
        ext = self.data_path.split('.')[-1].lower()
        if ext == 'parquet':
            return 'parquet'
        elif ext == 'csv':
            return 'csv'
        elif ext == 'db':
            return 'sqlite'
        else:
            raise ValueError(f"Tipo de archivo no soportado: {ext}")

    def _validate_database(self):
        """Valida que la base de datos SQLite tenga las tablas esperadas."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Verificar tabla principal
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='datos_estacion';")
            if not cursor.fetchone():
                raise ValueError("Tabla 'datos_estacion' no encontrada en la base de datos")

            # Contar registros
            cursor.execute("SELECT COUNT(*) FROM datos_estacion")
            total_records = cursor.fetchone()[0]
            print(f"✅ Base de datos validada: {total_records:,} registros disponibles")

    @contextmanager
    def _get_connection(self):
        """Context manager para conexiones SQLite."""
        conn = None
        try:
            conn = sqlite3.connect(self.data_path)
            yield conn
        finally:
            if conn:
                conn.close()

    def load_dataset(self, use_sample=False, sample_size=100000):
        """
        Carga el dataset desde la fuente de datos especificada.

        Args:
            use_sample (bool, optional): Si usar una muestra para desarrollo. Defaults to False.
            sample_size (int, optional): Tamaño de la muestra. Defaults to 100000.

        Returns:
            pd.DataFrame: El dataset cargado.
        """
        print(f"📊 Cargando dataset desde {self.source_type}...")

        if self.source_type == 'parquet':
            df = pd.read_parquet(self.data_path)
        elif self.source_type == 'csv':
            df = pd.read_csv(self.data_path)
        elif self.source_type == 'sqlite':
            if use_sample:
                print(f"🔄 Modo desarrollo: cargando muestra de {sample_size:,} registros")
                query = f"""
                SELECT * FROM datos_estacion
                ORDER BY RANDOM()
                LIMIT {sample_size}
                """
            else:
                print("📈 Cargando dataset completo...")
                query = "SELECT * FROM datos_estacion"

            with self._get_connection() as conn:
                df = pd.read_sql_query(
                    query,
                    conn
                )
        else:
            raise ValueError(f"Tipo de fuente de datos no soportado: {self.source_type}")

        print(f"✅ Dataset cargado: {len(df):,} registros, {len(df.columns)} columnas")
        return df
    
    def get_sensor_data(self, limit=None):
        """Obtiene solo datos de sensores"""
        if self.source_type != 'sqlite':
            raise ValueError("get_sensor_data solo está disponible para bases de datos SQLite")
        
        target_col = self.config.COLUMN_MAPPING['target_candidates'][0]
        query = f"""
        SELECT timestamp_unix, sensor_presion_aceite_bar, 
               sensor_presion_frenos_bar, sensor_presion_refrigerante_bar,
               temperatura_ambiente_celsius, {target_col}
        FROM datos_estacion
        ORDER BY timestamp_unix
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn)
        
        # Convertir timestamp_unix a datetime
        df['timestamp'] = pd.to_datetime(df['timestamp_unix'], unit='ms')
        
        return df
    
    def get_data_by_state(self, state=None, rul_range=None, limit=None):
        """Obtiene datos filtrados por estado del sistema o rango RUL"""
        if self.source_type != 'sqlite':
            raise ValueError("get_data_by_state solo está disponible para bases de datos SQLite")
            
        target_col = self.config.COLUMN_MAPPING['target_candidates'][0]
        
        query = "SELECT * FROM datos_estacion WHERE 1=1"
        params = []
        
        # Filtrar por estado si se especifica
        if state is not None:
            query += f" AND {target_col} = ?"
            params.append(state)
        
        # Filtrar por rango RUL si se especifica
        if rul_range is not None:
            min_rul, max_rul = rul_range
            query += f" AND {target_col} BETWEEN ? AND ?"
            params.extend([min_rul, max_rul])
        
        query += " ORDER BY timestamp_unix"
        
        if limit:
            query += f" LIMIT {limit}"
        
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        return df
    
    def is_target_numeric(self):
        """Detecta si la columna objetivo es numérica o categórica"""
        if self.source_type != 'sqlite':
            raise ValueError("is_target_numeric solo está disponible para bases de datos SQLite")
            
        target_col = self.config.COLUMN_MAPPING['target_candidates'][0]
        
        with self._get_connection() as conn:
            # Intentar convertir algunos valores a float
            query = f"""
            SELECT {target_col} FROM datos_estacion 
            WHERE {target_col} NOT NULL 
            LIMIT 10
            """
            sample_values = pd.read_sql_query(query, conn)[target_col]
            
            try:
                # Intentar conversión a float
                sample_values.astype(float)
                return True
            except (ValueError, TypeError):
                return False
    
    def get_operational_data(self, start_date=None, end_date=None):
        """Obtiene datos operacionales en rango de fechas"""
        if self.source_type != 'sqlite':
            raise ValueError("get_operational_data solo está disponible para bases de datos SQLite")
            
        query = "SELECT * FROM datos_estacion WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND fecha >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND fecha <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp_unix"
        
        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        return df
    
    def get_database_info(self):
        """Obtiene información de la base de datos"""
        if self.source_type != 'sqlite':
            raise ValueError("get_database_info solo está disponible para bases de datos SQLite")
            
        info = {}
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Información de tablas
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            info['tables'] = [row[0] for row in cursor.fetchall()]
            
            # Estadísticas de datos_estacion
            if 'datos_estacion' in info['tables']:
                cursor.execute("SELECT COUNT(*) FROM datos_estacion")
                info['total_records'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT MIN(fecha), MAX(fecha) FROM datos_estacion")
                min_date, max_date = cursor.fetchone()
                info['date_range'] = (min_date, max_date)
                
                target_col = self.config.COLUMN_MAPPING['target_candidates'][0]
                cursor.execute(f"SELECT {target_col}, COUNT(*) FROM datos_estacion GROUP BY {target_col}")
                info['state_distribution'] = dict(cursor.fetchall())
        
        return info
    
    def execute_query(self, query, params=None):
        """Ejecuta consulta SQL personalizada"""
        if self.source_type != 'sqlite':
            raise ValueError("execute_query solo está disponible para bases de datos SQLite")
            
        with self._get_connection() as conn:
            if params:
                df = pd.read_sql_query(query, conn, params=params)
            else:
                df = pd.read_sql_query(query, conn)
        
        return df

    def save_results_to_db(self, df, table_name):
        """Guarda resultados de vuelta a la base de datos"""
        if self.source_type != 'sqlite':
            raise ValueError("save_results_to_db solo está disponible para bases de datos SQLite")
            
        with self._get_connection() as conn:
            df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        print(f"💾 Resultados guardados en tabla: {table_name}")
