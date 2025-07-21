# EN: src/utils/validation_strategies.py
# REEMPLAZA LA CLASE ENTERA

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

class TemporalValidator:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.n_splits = self.config.VALIDATION_CONFIG.get('n_splits', 2) 

    def create_temporal_splits(self, df_full_info, target_column, group_column='vehiculo_id', task_type='classification'):
        self.logger.log_step(f"Creando {self.n_splits} splits para tarea '{task_type}', agrupando por '{group_column}'.")
        if group_column not in df_full_info.columns or target_column not in df_full_info.columns:
            raise ValueError(f"Las columnas '{group_column}' y/o '{target_column}' no se encuentran en el DataFrame.")
        
        X = df_full_info.drop(columns=[col for col in [target_column, group_column] if col in df_full_info.columns], errors='ignore')
        y = df_full_info[target_column]
        groups = df_full_info[group_column]

        if task_type == 'regression':
            self.logger.log_step("Usando GroupKFold para regresión (sin estratificación).")
            kfold = GroupKFold(n_splits=self.n_splits)
            split_generator = kfold.split(X, y, groups)
        else:
            self.logger.log_step(f"Usando StratifiedGroupKFold para clasificación (estratificando por '{target_column}').")
            kfold = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.config.RANDOM_SEED)
            split_generator = kfold.split(X, y, groups)

        splits_info = [{'train_idx': train_idx, 'test_idx': test_idx} for train_idx, test_idx in split_generator]
        self.logger.log_success(f"Se han creado {len(splits_info)} splits de validación para la tarea '{task_type}'.")
        return splits_info

    def validate_no_data_leakage(self, splits_info, df_full, group_column='vehiculo_id'):
        self.logger.log_step("Validando data leakage a nivel de grupo...")
        for i, split in enumerate(splits_info):
            train_groups = set(df_full.iloc[split['train_idx']][group_column].unique())
            test_groups = set(df_full.iloc[split['test_idx']][group_column].unique())
            if not train_groups.isdisjoint(test_groups):
                self.logger.log_error(f"¡Data Leakage Detectado en Fold {i+1}! Grupos compartidos: {train_groups.intersection(test_groups)}")
                return False
        self.logger.log_success("Validación de data leakage superada. No hay grupos compartidos entre splits.")
        return True
