# checkpoint_manager.py (VERSIÓN FINAL Y CORRECTA CON TUS NOMBRES DE FUNCIÓN)

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from joblib import dump, load
from sklearn.base import BaseEstimator
import shutil

class CheckpointManager:
    """
    Gestor de checkpoints robusto para recuperación y continuación de fases.
    Maneja de forma inteligente DataFrames, diccionarios, listas y modelos de scikit-learn.
    """
    
    def __init__(self, config, session_id=None):
        self.config = config
        self.session_id = session_id or datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_base = getattr(config, 'CHECKPOINTS_DIR', os.path.join(config.BASE_DIR, 'checkpoints'))
        self.checkpoint_dir = Path(checkpoint_base) / self.session_id
        self.metadata_path = self.checkpoint_dir / 'metadata.json'
        self._ensure_checkpoint_dir()
        self._init_or_load_metadata()

    def _ensure_checkpoint_dir(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DIR] Directorio de checkpoints: {self.checkpoint_dir}")

    def _init_or_load_metadata(self):
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'session_id': self.session_id,
                'created_at': datetime.now().isoformat(),
                'phases': {}
            }
            self._save_metadata()

    def _save_metadata(self):
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=4)

    def _json_serializer(self, obj):
        if isinstance(obj, (np.integer, np.int64)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (datetime, pd.Timestamp)): return obj.isoformat()
        if isinstance(obj, pd.Series): return obj.to_dict()

        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def _extract_and_save_models_recursive(self, data, base_path):
        if isinstance(data, dict):
            
            clean_data = {}
            for key, value in data.items():
                
                if isinstance(value, pd.DataFrame):
                    df_path = base_path / f"{key}.parquet"
                    value.to_parquet(df_path)
                    clean_data[key] = {"__dataframe_checkpoint__": True, "path": str(df_path.relative_to(self.checkpoint_dir))}
                    
                elif isinstance(value, BaseEstimator):
                    model_path = base_path / f"{key}.joblib"
                    model_path.parent.mkdir(parents=True, exist_ok=True)

                    dump(value, model_path)
                    clean_data[key] = {"__model_checkpoint__": True, "path": str(model_path.relative_to(self.checkpoint_dir))}

                elif isinstance(value, (dict, list)):
                    clean_data[key] = self._extract_and_save_models_recursive(value, base_path / str(key))
                else:
                    clean_data[key] = value
            return clean_data
        
        elif isinstance(data, list):
            return [self._extract_and_save_models_recursive(item, base_path / str(i)) for i, item in enumerate(data)]
        return data

    def save_checkpoint(self, phase: str, data: any):
        """
        Guarda el checkpoint de una fase, manejando automáticamente cualquier
        estructura de datos que contenga modelos o DataFrames.
        """
        print(f"\n[SAVE] Guardando checkpoint para fase: {phase}")
        try:
            phase_dir = self.checkpoint_dir / phase
            phase_dir.mkdir(exist_ok=True, parents=True)
            
            clean_data_for_json = self._extract_and_save_models_recursive(data, phase_dir)
            
            json_path = phase_dir / "data.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(clean_data_for_json, f, indent=4, default=self._json_serializer)
            
            self.metadata['phases'][phase] = {
                'completed': True,
                'timestamp': datetime.now().isoformat(),
                'data_path': str(json_path.relative_to(self.checkpoint_dir))
            }
            self._save_metadata()
            print(f"[OK] Checkpoint '{phase}' guardado exitosamente.")

        except Exception as e:
            print(f"[ERROR] Error guardando checkpoint {phase}: {e}")
            raise

    def _load_and_reconstruct_models_recursive(self, data):
            if isinstance(data, dict):
                
                if data.get("__dataframe_checkpoint__"):
                    return pd.read_parquet(self.checkpoint_dir / data['path'])
                
                elif data.get("__model_checkpoint__"):
                    return load(self.checkpoint_dir / data['path'])
                
                reconstructed_data = {}
                for key, value in data.items():
                    reconstructed_data[key] = self._load_and_reconstruct_models_recursive(value)
                return reconstructed_data
            
            elif isinstance(data, list):
                return [self._load_and_reconstruct_models_recursive(item) for item in data]
            return data

    def load_checkpoint(self, phase: str):
        """
        Carga el checkpoint de una fase, reconstruyendo automáticamente la
        estructura de datos original, incluyendo modelos y DataFrames.
        """
        if not self.has_checkpoint(phase):
            print(f"[WARN] No existe checkpoint para la fase: {phase}")
            return None
        
        print(f"\n[LOAD] Cargando checkpoint de la fase: {phase}")
        try:
            phase_info = self.metadata['phases'][phase]
            json_path = self.checkpoint_dir / phase_info['data_path']

            with open(json_path, 'r', encoding='utf-8') as f:
                data_from_json = json.load(f)

            reconstructed_data = self._load_and_reconstruct_models_recursive(data_from_json)
            
            print(f"[OK] Checkpoint '{phase}' cargado exitosamente.")
            return reconstructed_data

        except Exception as e:
            print(f"[ERROR] Error cargando checkpoint {phase}: {e}")
            return None

    def has_checkpoint(self, phase: str) -> bool:
        return self.metadata['phases'].get(phase, {}).get('completed', False)

    def clear_checkpoint(self, phase: str):
        if not self.has_checkpoint(phase):
            print(f"[WARN] No hay checkpoint para eliminar en la fase: {phase}")
            return
            
        print(f"\n[DEL] Eliminando checkpoint de la fase: {phase}")
        phase_dir = self.checkpoint_dir / phase
        if phase_dir.exists():
            shutil.rmtree(phase_dir)

        if phase in self.metadata['phases']:
            del self.metadata['phases'][phase]
        self._save_metadata()
        print(f"[OK] Checkpoint '{phase}' eliminado.")

    def clear_all_checkpoints(self):
        print("\n[DEL] Eliminando todos los checkpoints de la sesión...")
        for item in self.checkpoint_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            elif item.is_file():
                item.unlink()
        
        self.metadata['phases'] = {}
        self._save_metadata()
        print("[OK] Todos los checkpoints eliminados.")
