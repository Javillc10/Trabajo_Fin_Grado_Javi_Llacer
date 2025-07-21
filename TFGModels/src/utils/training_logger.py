# src/utils/training_logger.py (VERSIÓN FINAL, LIMPIA Y COMPLETA)

import logging
import os
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

class TrainingLogger:
    """
    Un logger de entrenamiento robusto que registra en un fichero y recopila
    datos estructurados para generar resúmenes fiables.
    """
    def __init__(self, session_id, config):
        self.session_id = session_id
        self.config = config
        self.start_time = datetime.now()

        # Estructura para almacenar todos los datos de la sesión
        self.summary_data = {
            'session_info': {
                'session_id': self.session_id,
                'start_time': self.start_time.isoformat(),
                'end_time': None,
                'duration': None,
            },
            'config': {k: str(v) for k, v in config.__dict__.items() if not k.startswith('_') and not callable(v)},
            'pipeline_steps': [],
            'metrics': {'classification': {}, 'rul': {}},
            'artifacts': {},
            'key_findings': [],
            'errors': []
        }

        self.results_dir = Path(os.path.join('results', self.session_id))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(f"SessionLogger_{self.session_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Configuración de los handlers (fichero y consola)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        log_file = self.results_dir / 'training_log.log'
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def log_section(self, section_name):
        separator = "=" * 50
        self.logger.info(f"\n{separator}\n{section_name}\n{separator}")
        self.summary_data['pipeline_steps'].append(f"SECTION: {section_name}")

    def log_subsection(self, subsection_name):
        separator = "-" * 40
        self.logger.info(f"\n{separator}\n{subsection_name}\n{separator}")
        self.summary_data['pipeline_steps'].append(f"SUBSECTION: {subsection_name}")
        
    def log_step(self, step_name):
        self.logger.info(f"[INFO] {step_name}")
        self.summary_data['pipeline_steps'].append(f"STEP: {step_name}")

    def log_metrics(self, metrics_dict, section_name, component=None):
        self.logger.info(f"--- Métricas para: {section_name} (Componente: {component or 'N/A'}) ---")
        clean_metrics = {k: v for k, v in metrics_dict.items() if v is not None}
        
        for metric, value in clean_metrics.items():
            log_value = f"{value:.4f}" if isinstance(value, (float, np.floating)) else value
            self.logger.info(f"    - {metric}: {log_value}")

        target_dict = self.summary_data['metrics']
        if "clasificación" in section_name.lower() or "classification" in section_name.lower():
            model_key = section_name
            if model_key not in target_dict['classification']:
                 target_dict['classification'][model_key] = {}
            target_dict['classification'][model_key].update(clean_metrics)
        elif "rul" in section_name.lower():
            if component not in target_dict['rul']:
                target_dict['rul'][component] = {}
            model_key = section_name
            if model_key not in target_dict['rul'][component]:
                target_dict['rul'][component][model_key] = {}
            target_dict['rul'][component][model_key].update(clean_metrics)

    def log_artifact(self, artifact_name, artifact_path):
        self.logger.info(f"[ARTIFACT] {artifact_name} guardado en: {artifact_path}")
        self.summary_data['artifacts'][artifact_name] = str(artifact_path)

    def log_finding(self, finding):
        self.logger.info(f"[FINDING] {finding}")
        self.summary_data['key_findings'].append(finding)
        
    def log_error(self, message, exc_info=False):
        self.logger.error(message, exc_info=exc_info)
        self.summary_data['errors'].append(str(message))

    def log_success(self, message):
        self.logger.info(f"[SUCCESS] {message}")
        self.summary_data['key_findings'].append(message)

    def log_warning(self, message):
        self.logger.warning(f"[WARNING] {message}")

    def log_critical(self, message, exc_info=False):
        self.logger.critical(message, exc_info=exc_info)
        self.summary_data['errors'].append(f"CRITICAL: {str(message)}")

    def close(self):
        end_time = datetime.now()
        duration = end_time - self.start_time
        self.summary_data['session_info']['end_time'] = end_time.isoformat()
        self.summary_data['session_info']['duration'] = str(duration)
        
        self._generate_markdown_summary()
        self._generate_json_metrics()

        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

    def _generate_json_metrics(self):
        """Extrae solo las métricas y las guarda en un archivo JSON limpio para la web."""
        all_metrics = {}
        
        for model_name, metrics in self.summary_data['metrics']['classification'].items():
            all_metrics[model_name] = metrics

        for component, models in self.summary_data['metrics']['rul'].items():
            for model_name, metrics in models.items():
                all_metrics[f"{model_name}_{component}"] = metrics

        metrics_path = self.results_dir / 'training_metrics.json'
        
        def default_converter(o):
            if isinstance(o, (np.integer, np.int64)): return int(o)
            if isinstance(o, (np.floating, np.float64)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, Path): return str(o)
            return str(o)
            
        try:
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(all_metrics, f, indent=4, default=default_converter)
            self.logger.info(f"Métricas en JSON guardadas en: {metrics_path}")
        except Exception as e:
            self.log_error(f"No se pudo guardar el JSON de métricas: {e}")


    def _generate_markdown_summary(self):
        """ Genera un resumen en formato Markdown de los datos de entrenamiento.
        """
        lines = [f"# Resumen de Entrenamiento: {self.session_id}"]
        info = self.summary_data['session_info']
        lines.append("\n## Información de la Sesión")
        lines.append(f"- **Inicio:** {info['start_time']}")
        lines.append(f"- **Fin:** {info['end_time']}")
        lines.append(f"- **Duración:** {info['duration']}")
        if self.summary_data['metrics']['classification']:
            lines.append("\n## 🤖 Rendimiento del Modelo de Clasificación")
            for section, metrics in self.summary_data['metrics']['classification'].items():
                lines.append(f"\n### {section}")
                df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Valor'])
                lines.append(df.to_markdown())
        if self.summary_data['metrics']['rul']:
            lines.append("\n## ⏳ Rendimiento de los Modelos RUL")
            for component, sections in self.summary_data['metrics']['rul'].items():
                lines.append(f"\n### Componente: {component.upper()}")
                for section, metrics in sections.items():
                    lines.append(f"\n#### {section}")
                    df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Valor'])
                    lines.append(df.to_markdown())
        lines.append("\n## 📂 Artefactos y Hallazgos Clave")
        lines.extend([f"- **{name}:** `{path}`" for name, path in self.summary_data['artifacts'].items()])
        lines.extend([f"- {finding}" for finding in self.summary_data['key_findings']])
        if self.summary_data['errors']:
            lines.append("\n## ❌ Errores Registrados")
            lines.extend([f"- `{error}`" for error in self.summary_data['errors']])
        summary_path = self.results_dir / 'training_summary.md'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        return summary_path
