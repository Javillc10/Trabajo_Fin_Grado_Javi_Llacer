# src/data/data_splitter.py
"""
Módulo para dividir el dataset en conjuntos de entrenamiento para diferentes tareas
(Clasificación y Estimación de RUL).
"""
import pandas as pd

class DataSplitter:
    """
    Toma un DataFrame y lo divide en características (X) y objetivo (y)
    para las tareas de clasificación y regresión de RUL.
    """
    def __init__(self, dataframe, config):
        """
        Inicializa el DataSplitter.

        Args:
            dataframe (pd.DataFrame): El DataFrame completo y crudo.
            config: El objeto de configuración del proyecto.
        """
        self.df = dataframe.copy()  # Hacemos una copia para evitar efectos secundarios
        self.config = config
        
        # Extraer los nombres de las columnas clave desde la configuración
        self.timestamp_col = self.config.COLUMN_MAPPING.get('timestamp')
        self.target_class_col = self.config.COLUMN_MAPPING.get('target_classification')
        self.target_rul_col = self.config.COLUMN_MAPPING.get('target_rul')
        
        # Lista de todas las columnas que son objetivos para no usarlas como features
        self.all_target_cols = [col for col in [self.target_class_col, self.target_rul_col] if col is not None]

    def split_for_classification(self):
        """
        Divide los datos para la tarea de clasificación de fallos.

        Returns:
            pd.DataFrame: X_class (features)
            pd.Series: y_class (objetivo de clasificación)
            pd.Series: df_timestamps (columna de timestamps)
        """
        if not self.target_class_col or self.target_class_col not in self.df.columns:
            raise ValueError(f"La columna objetivo de clasificación '{self.target_class_col}' no se encuentra en el DataFrame.")
        
        print(f"Dividiendo datos para clasificación. Objetivo: '{self.target_class_col}'")

        # El objetivo 'y' es la columna de clasificación
        y_class = self.df[self.target_class_col]
        
        # Las características 'X' son todo excepto las columnas objetivo y el timestamp
        cols_to_drop = self.all_target_cols + [self.timestamp_col]
        X_class = self.df.drop(columns=[col for col in cols_to_drop if col in self.df.columns])
        
        # La columna de timestamps
        df_timestamps = self.df[self.timestamp_col]
        
        return X_class, y_class, df_timestamps

    def split_for_rul_estimation(self):
        """
        Divide los datos para la tarea de estimación de RUL.

        Returns:
            pd.DataFrame: X_rul (features)
            pd.Series: y_rul (objetivo de RUL)
        """
        if not self.target_rul_col or self.target_rul_col not in self.df.columns:
            raise ValueError(f"La columna objetivo de RUL '{self.target_rul_col}' no se encuentra en el DataFrame.")

        print(f"Dividiendo datos para estimación de RUL. Objetivo: '{self.target_rul_col}'")
        
        # El objetivo 'y' es la columna de RUL
        y_rul = self.df[self.target_rul_col]
        
        # Las características 'X' son todo excepto las columnas objetivo y el timestamp
        cols_to_drop = self.all_target_cols + [self.timestamp_col]
        X_rul = self.df.drop(columns=[col for col in cols_to_drop if col in self.df.columns])
        
        return X_rul, y_rul

