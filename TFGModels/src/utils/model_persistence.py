# TFGModels/src/utils/model_persistence.py
import os
import joblib
import json
import numpy as np
from datetime import datetime

class ModelPersistence:
    def __init__(self, config):
        self.config = config
        self.models_base_dir = self.config.MODELS_DIR
        
    def _convert_numpy_types(self, obj):
        """Recursively converts numpy types to native Python types for JSON serialization."""
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_types(item) for item in obj]
        return obj

    def _get_session_dir(self, model_type, session_id=None):
        """Generates the directory path for a given model type and session."""
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        session_path = os.path.join(self.models_base_dir, model_type, f"session_{session_id}")
        os.makedirs(session_path, exist_ok=True)
        return session_path, session_id

    def save_predictions(self, predictions_df, model_name, model_type, session_id):
        """Saves model predictions dataframe as parquet file.
        
        Args:
            predictions_df (pd.DataFrame): DataFrame containing predictions
            model_name (str): Name of the model
            model_type (str): Type of model
            session_id (str): Session ID
            
        Returns:
            str: Path to saved predictions file
        """
        session_dir = self._get_session_dir(model_type, session_id)[0]
        preds_path = os.path.join(session_dir, f"{model_name}_predictions.parquet")
        predictions_df.to_parquet(preds_path)
        print(f"[PRED] Predictions for {model_type} model '{model_name}' saved to: {preds_path}")
        return preds_path

    def save_model(self, model, model_name, model_type, metrics=None, session_id=None, label_encoder=None, feature_cols=None, predictions_df=None):
        """
        Saves a trained model and its metadata.

        Args:
            model: The trained model object.
            model_name (str): Name of the model (e.g., 'random_forest').
            model_type (str): Type of model ('classification', 'rul', 'preprocessor').
            metrics (dict, optional): Performance metrics of the model.
            session_id (str, optional): Specific session ID. If None, a new one is generated.
            label_encoder (LabelEncoder, optional): Label encoder if used (e.g., for XGBoost classification).
            feature_cols (list, optional): List of feature columns used for training.
            predictions_df (pd.DataFrame, optional): DataFrame containing model predictions.

        Returns:
            str: Path to the saved model.
        """
        session_dir, current_session_id = self._get_session_dir(model_type, session_id)
        
        model_filename = f"{model_name}.joblib"
        model_path = os.path.join(session_dir, model_filename)
        joblib.dump(model, model_path)
        print(f"[SAVE] {model_type.capitalize()} model '{model_name}' saved to: {model_path}")

        metadata = {
            'model_name': model_name,
            'model_type': model_type,
            'saved_at': datetime.now().isoformat(),
            'session_id': current_session_id,
            'config_version': self.config.VERSION,
            'metrics': self._convert_numpy_types(metrics) if metrics else {},
            'feature_columns': feature_cols or []
        }

        if predictions_df is not None:
            preds_path = self.save_predictions(predictions_df, model_name, model_type, current_session_id)
            metadata['predictions_path'] = preds_path

        if label_encoder:
            le_filename = f"{model_name}_label_encoder.joblib"
            le_path = os.path.join(session_dir, le_filename)
            joblib.dump(label_encoder, le_path)
            metadata['label_encoder_path'] = le_path
        print(f"[SAVE] LabelEncoder for {model_type} model '{model_name}' saved to: {le_path}")

        metadata_path = os.path.join(session_dir, f"{model_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"[META] Metadata for {model_type} model '{model_name}' saved to: {metadata_path}")
            
        return model_path, current_session_id

    def load_predictions(self, model_name, model_type, session_id):
        """Loads saved model predictions.
        
        Args:
            model_name (str): Name of the model
            model_type (str): Type of model
            session_id (str): Session ID
            
        Returns:
            pd.DataFrame: Loaded predictions or None if not found
        """
        session_dir = os.path.join(self.models_base_dir, model_type, f"session_{session_id}")
        preds_path = os.path.join(session_dir, f"{model_name}_predictions.parquet")
        
        if not os.path.exists(preds_path):
            return None
            
        return pd.read_parquet(preds_path)

    def load_model(self, model_name, model_type, session_id, load_predictions=False):
        """
        Loads a trained model and its metadata.

        Args:
            model_name (str): Name of the model.
            model_type (str): Type of model ('classification', 'rul', 'preprocessor').
            session_id (str): The session ID from which to load the model.
            load_predictions (bool): Whether to load associated predictions.

        Returns:
            tuple: (model, metadata) or (None, None) if not found.
        """
        session_dir = os.path.join(self.models_base_dir, model_type, f"session_{session_id}")
        
        model_filename = f"{model_name}.joblib"
        model_path = os.path.join(session_dir, model_filename)
        
        metadata_filename = f"{model_name}_metadata.json"
        metadata_path = os.path.join(session_dir, metadata_filename)

        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            print(f"[ERROR] Model or metadata not found for '{model_name}' in session '{session_id}'")
            return None, None

        model = joblib.load(model_path)
        print(f"[OK] Model '{model_name}' loaded from: {model_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"[META] Metadata for '{model_name}' loaded from: {metadata_path}")

        if 'label_encoder_path' in metadata and os.path.exists(metadata['label_encoder_path']):
            label_encoder = joblib.load(metadata['label_encoder_path'])
            metadata['label_encoder'] = label_encoder # Attach to metadata for convenience
            print(f"[OK] LabelEncoder for '{model_name}' loaded from: {metadata['label_encoder_path']}")
            
        if load_predictions and 'predictions_path' in metadata and os.path.exists(metadata['predictions_path']):
            predictions = pd.read_parquet(metadata['predictions_path'])
            metadata['predictions'] = predictions
            print(f"[PRED] Predictions for '{model_name}' loaded from: {metadata['predictions_path']}")
        
        return model, metadata

    def list_sessions(self, model_type=None):
        """Lists available sessions, optionally filtered by model type."""
        sessions = {}
        search_dir = self.models_base_dir
        
        if model_type:
            search_dir = os.path.join(self.models_base_dir, model_type)
            if not os.path.exists(search_dir):
                print(f"No sessions found for type: {model_type}")
                return {}
            
            type_sessions = [d for d in os.listdir(search_dir) if os.path.isdir(os.path.join(search_dir, d)) and d.startswith('session_')]
            sessions[model_type] = sorted([s.replace('session_', '') for s in type_sessions], reverse=True)
        else:
            for m_type in os.listdir(search_dir):
                type_path = os.path.join(search_dir, m_type)
                if os.path.isdir(type_path):
                    type_sessions = [d for d in os.listdir(type_path) if os.path.isdir(os.path.join(type_path, d)) and d.startswith('session_')]
                    if type_sessions:
                        sessions[m_type] = sorted([s.replace('session_', '') for s in type_sessions], reverse=True)
        return sessions

    def get_latest_session_id(self, model_type):
        """Gets the ID of the most recent session for a given model type."""
        sessions = self.list_sessions(model_type)
        if model_type in sessions and sessions[model_type]:
            return sessions[model_type][0] # First one is latest due to sorting
        return None

    def save_preprocessor(self, preprocessor, name, session_id=None):
        """Saves a preprocessor object (e.g., scaler, feature_engineer)."""
        return self.save_model(preprocessor, name, 'preprocessors', session_id=session_id)

    def load_preprocessor(self, name, session_id):
        """Loads a preprocessor object."""
        return self.load_model(name, 'preprocessors', session_id)
