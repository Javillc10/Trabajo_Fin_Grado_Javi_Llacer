"""
Módulo unificado de métricas para mantenimiento predictivo
Centraliza funcionalidades de métricas que estaban dispersas
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                           mean_absolute_error, mean_squared_error, r2_score,
                           confusion_matrix, classification_report)
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import seaborn as sns

class UnifiedMetrics:
    """Clase unificada para cálculo y visualización de métricas"""
    
    def __init__(self, config,logger):
        self.config = config
        self.operational_costs = config.BUSINESS_CONFIG['operational_costs']
        self.viz_config = config.VISUALIZATION_CONFIG
        self.logger = logger
    
    def calculate_classification_metrics(self, y_true, y_pred, y_proba=None, critical_classes=None, class_labels=None, label_encoder=None):
        """
        Calcula métricas completas de clasificación.
        VERSIÓN FINAL CON UMBRAL DE DECISIÓN PERSONALIZADO Y CORRECCIÓN DE BUGS.
        """
        metrics = {}
        
        # --- CONVERSIÓN DE TIPOS ---
        # Convertir y_pred a strings si son numéricos y tenemos label_encoder
        if label_encoder is not None and np.issubdtype(y_pred.dtype, np.number):
            y_pred = label_encoder.inverse_transform(y_pred)
        
        # Validar consistencia de tipos
        if not isinstance(y_true.iloc[0], type(y_pred[0])):
            self.logger.log_warning(f"Inconsistencia de tipos: y_true es {type(y_true.iloc[0])} vs y_pred es {type(y_pred[0])}")
        
        # --- LÓGICA DEL UMBRAL DE DECISIÓN ---
        y_pred_adjusted = pd.Series(y_pred, index=y_true.index, dtype=object) # Usamos una copia
        
        # Usamos las clases críticas definidas en la configuración para el umbral.
        critical_class_for_threshold = self.config.CLASSIFICATION_CONFIG.get('critical_fault_classes', [])[0]

        if y_proba is not None and class_labels is not None and critical_class_for_threshold in class_labels:
            self.logger.log_step(f"Aplicando umbral de decisión personalizado para la clase '{critical_class_for_threshold}'...")
            
            critico_idx = np.where(class_labels == critical_class_for_threshold)[0][0]
            CRITICAL_THRESHOLD = 0.35
            prob_critico = y_proba[:, critico_idx]
            override_mask = prob_critico > CRITICAL_THRESHOLD
            
            num_overrides = override_mask.sum()
            if num_overrides > 0:
                y_pred_adjusted[override_mask] = critical_class_for_threshold
                self.logger.log_success(f"Umbral activado: {num_overrides} predicciones forzadas a '{critical_class_for_threshold}'.")
            else:
                self.logger.log_step("Umbral no superado en ninguna predicción.")

        # A partir de aquí, TODAS las métricas se calculan con las predicciones ajustadas
        # Usamos las clases críticas pasadas como argumento para el resto de métricas
        if critical_classes is None:
            critical_classes = self.config.CLASSIFICATION_CONFIG.get('critical_fault_classes', [])

        unique_classes = np.unique(np.concatenate((y_true, y_pred_adjusted)))

        # Métricas básicas
        metrics.update(self._calculate_basic_metrics(y_true, y_pred_adjusted))
        
        # Métricas para clases críticas (usando y_pred_adjusted)
        if critical_classes:
            metrics.update(self._calculate_critical_class_metrics(y_true, y_pred_adjusted, critical_classes))
        
        # Precision@K si hay probabilidades (esta usa las probabilidades originales, es correcto)
        if y_proba is not None:
            metrics.update(self._calculate_precision_at_k(y_true, y_proba, unique_classes, critical_classes))
        
        # Métricas de costo operacional (usando y_pred_adjusted)
        metrics.update(self._calculate_cost_metrics(y_true, y_pred_adjusted, unique_classes))
                
        # Time-to-Detection simulation (usando y_pred_adjusted)
        metrics.update(self._simulate_time_to_detection(y_true, y_pred_adjusted))
        
        return metrics
    
    def calculate_rul_metrics(self, y_true, y_pred):
        """
        Calcula métricas completas de RUL
        
        Unifica métricas de regresión y RUL específicas
        """
        metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred)
        }
        
        # Añadir bandas de precisión
        bands = self.config.METRICS_CONFIG['regression']['accuracy_bands_days']
        for band in bands:
            within_band = np.abs(y_true - y_pred) <= band
            metrics[f'accuracy_band_{band}d'] = np.mean(within_band)
        
        # Prognostic Horizon
        metrics.update(self._calculate_prognostic_horizon(y_true, y_pred))
        
        # Alpha-Lambda metrics
        metrics.update(self._calculate_alpha_lambda_metrics(y_true, y_pred))
        
        # Directional Accuracy
        metrics.update(self._calculate_directional_accuracy(y_true, y_pred))
        
        # Métricas de convergencia cerca del fallo
        metrics.update(self._analyze_convergence_near_failure(y_true, y_pred))
        
        return metrics
    
    def calculate_temporal_stability(self, df, target_column):
        """
        Analiza estabilidad temporal del dataset
        
        Unifica análisis temporal de TemporalValidator
        """
        stability_metrics = {}
        
        # Distribución del target por semana
        df['semana'] = df['timestamp'].dt.isocalendar().week
        df['semana_relativa'] = df['semana'] - df['semana'].min() + 1
        weekly_dist = df.groupby('semana_relativa')[target_column].value_counts(normalize=True)
        
        # Calcular drift temporal
        base_dist = weekly_dist.loc[1] if 1 in weekly_dist.index.get_level_values(0) else None
        if base_dist is not None:
            for semana in df['semana_relativa'].unique()[1:]:
                current_dist = weekly_dist.loc[semana]
                stability_metrics[f'temporal_drift_week_{semana}'] = (
                    self._calculate_distribution_similarity(base_dist, current_dist)
                )
        
        return stability_metrics
    
    def plot_metrics_dashboard(self, metrics_dict, plot_type='classification', save_path=None):
        """
        Genera dashboard visual de métricas
        
        Unifica visualizaciones de métricas
        """
        if plot_type == 'classification':
            self._plot_classification_dashboard(metrics_dict, save_path)
        elif plot_type == 'rul':
            self._plot_rul_dashboard(metrics_dict, save_path)
    
    def _plot_classification_dashboard(self, metrics_dict, save_path=None):
        """Genera dashboard para métricas de clasificación"""
        # Crear figura con múltiples subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dashboard de Métricas de Clasificación', fontsize=16)
        
        # 1. Matriz de confusión
        ax1 = axes[0, 0]
        if 'confusion_matrix' in metrics_dict:
            cm = metrics_dict['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
            ax1.set_title('Matriz de Confusión')
            ax1.set_xlabel('Predicción')
            ax1.set_ylabel('Valor Real')
        
        # 2. Métricas por clase
        ax2 = axes[0, 1]
        class_metrics = {k: v for k, v in metrics_dict.items() 
                        if any(m in k for m in ['precision_', 'recall_', 'f1_'])}
        if class_metrics:
            metrics_df = pd.DataFrame(class_metrics, index=[0]).melt()
            sns.barplot(data=metrics_df, x='variable', y='value', ax=ax2)
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
            ax2.set_title('Métricas por Clase')
        
        # 3. Time-to-Detection
        ax3 = axes[1, 0]
        ttd_metrics = {k: v for k, v in metrics_dict.items() 
                      if 'time_to_detection' in k or 'detection_rate' in k}
        if ttd_metrics:
            ttd_df = pd.DataFrame(ttd_metrics, index=[0]).melt()
            sns.barplot(data=ttd_df, x='variable', y='value', ax=ax3)
            ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
            ax3.set_title('Métricas de Detección Temporal')
        
        # 4. Análisis de costos
        ax4 = axes[1, 1]
        cost_metrics = {k: v for k, v in metrics_dict.items() 
                       if 'cost' in k.lower()}
        if cost_metrics:
            costs_df = pd.DataFrame(cost_metrics, index=[0]).melt()
            sns.barplot(data=costs_df, x='variable', y='value', ax=ax4)
            ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
            ax4.set_title('Análisis de Costos')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
     
    def _plot_rul_dashboard(self, metrics_dict, save_path=None):
        """Genera dashboard para métricas de RUL"""
        if 'y_true' not in metrics_dict or 'y_pred' not in metrics_dict:
            return
        
        y_true = metrics_dict['y_true']
        y_pred = metrics_dict['y_pred']
        
        # Crear figura con múltiples subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dashboard de Métricas RUL', fontsize=16)
        
        # 1. Predicción vs Real con bandas de error
        ax1 = axes[0, 0]
        ax1.scatter(y_true, y_pred, alpha=0.6)
        
        # Línea perfecta y bandas
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', 
                linewidth=2, label='Predicción Perfecta')
        
        # Bandas de error
        for band in [1, 3, 7]:
            ax1.fill_between([min_val, max_val], 
                           [min_val - band, max_val - band],
                           [min_val + band, max_val + band],
                           alpha=0.1, label=f'±{band} días')
        
        ax1.set_xlabel('RUL Real (días)')
        ax1.set_ylabel('RUL Predicho (días)')
        ax1.set_title('Predicción vs Realidad')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribución de errores
        ax2 = axes[0, 1]
        errors = y_pred - y_true
        ax2.hist(errors, bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', linewidth=2)
        ax2.axvline(errors.mean(), color='green', linestyle='-', 
                   linewidth=2, label=f'Error Medio = {errors.mean():.2f}')
        ax2.set_xlabel('Error de Predicción (días)')
        ax2.set_ylabel('Frecuencia')
        ax2.set_title('Distribución de Errores')
        ax2.legend()
        
        # 3. Métricas por rango de RUL
        ax3 = axes[1, 0]
        convergence_metrics = {k: v for k, v in metrics_dict.items() 
                             if any(label in k for label in ['0-2d', '2-5d', '5-10d', '>10d'])}
        if convergence_metrics:
            conv_df = pd.DataFrame(convergence_metrics, index=[0]).melt()
            sns.barplot(data=conv_df, x='variable', y='value', ax=ax3)
            ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
            ax3.set_title('Error por Rango de RUL')
        
        # 4. Otras métricas RUL
        ax4 = axes[1, 1]
        rul_metrics = {k: v for k, v in metrics_dict.items() 
                      if k in ['prognostic_horizon', 'alpha_accuracy', 
                             'lambda_accuracy', 'directional_accuracy']}
        if rul_metrics:
            metrics_df = pd.DataFrame(rul_metrics, index=[0]).melt()
            sns.barplot(data=metrics_df, x='variable', y='value', ax=ax4)
            ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
            ax4.set_title('Métricas RUL Específicas')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
     
    def _calculate_basic_metrics(self, y_true, y_pred):
        """Métricas básicas, ahora con manejo de división por cero."""
        return {
            'accuracy': np.mean(y_true == y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0)
        }
    
    def _calculate_critical_class_metrics(self, y_true, y_pred, critical_classes):
        """Métricas para clases críticas, con manejo de división por cero."""
        metrics = {}
        for c_class in critical_classes:
            y_true_binary = (y_true == c_class)
            y_pred_binary = (y_pred == c_class)
            # Solo calcular si la clase crítica existe en los datos reales
            if y_true_binary.sum() > 0:
                metrics[f'precision_{c_class}'] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                metrics[f'recall_{c_class}'] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
                metrics[f'f1_{c_class}'] = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        return metrics
    
    def _calculate_precision_at_k(self, y_true, y_proba, unique_classes, critical_classes, k_values=[5, 10]):
        """Calcular Precision@K para clases críticas. VERSIÓN FINAL CORREGIDA."""
        metrics = {}
        if critical_classes is None or y_proba is None or len(y_proba) == 0:
            return metrics
            
        if len(unique_classes) != y_proba.shape[1]:
            self.logger.log_warning("Inconsistencia en el número de clases y columnas de probabilidad. Saltando Precision@K.")
            return metrics

        y_true_series = pd.Series(y_true)
        y_true_critical_mask = y_true_series.isin(critical_classes)
        
        if not y_true_critical_mask.any():
            for k in k_values:
                metrics[f'precision_at_{k}_critical'] = 0.0
            return metrics

        top_k_indices = np.argsort(y_proba, axis=1)[:, ::-1]

        for k in k_values:
            # Asegurarnos de que k no es mayor que el número de clases disponibles
            effective_k = min(k, len(unique_classes))
            
            top_k_pred_classes = unique_classes[top_k_indices[:, :effective_k]]
            
            # Comparamos el array de NumPy con los .values de la Serie de Pandas
            comparison_matrix = (top_k_pred_classes.T == y_true.values).T
            correct_predictions_mask = np.any(comparison_matrix, axis=1)
            
            # El resto de la lógica es la misma
            precision = np.sum(correct_predictions_mask[y_true_critical_mask]) / np.sum(y_true_critical_mask)
            metrics[f'precision_at_{k}_critical'] = precision

        return metrics
        
    def _calculate_distribution_similarity(self, dist1, dist2):
        """Calcula similitud entre distribuciones usando Jensen-Shannon"""
        all_classes = set(dist1.index) | set(dist2.index)
        dist1_aligned = pd.Series([dist1.get(cls, 0) for cls in all_classes])
        dist2_aligned = pd.Series([dist2.get(cls, 0) for cls in all_classes])
        
        js_distance = jensenshannon(dist1_aligned, dist2_aligned)
        return 1 - (js_distance if not np.isnan(js_distance) else 0)
    
    def _calculate_cost_metrics(self, y_true, y_pred, unique_classes):
        """MÉTODO AÑADIDO: Calcular métricas basadas en costo operacional."""
        cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
        
        fp_cost, fn_cost = 0, 0
        
        for i, true_class in enumerate(unique_classes):
            for j, pred_class in enumerate(unique_classes):
                if i == j: continue
                error_count = cm[i, j]
                
                is_true_critical = self._is_critical_class(true_class)
                is_pred_critical = self._is_critical_class(pred_class)

                # Falso Negativo: Era crítico, pero se predijo como no crítico
                if is_true_critical and not is_pred_critical:
                    fn_cost += error_count * self.operational_costs.get('false_negative_cost', 10000)
                # Falso Positivo: Era no crítico, pero se predijo como crítico
                elif not is_true_critical and is_pred_critical:
                    fp_cost += error_count * self.operational_costs.get('false_positive_cost', 500)
        
        total_cost = fp_cost + fn_cost
        return {
            'total_operational_cost': total_cost,
            'cost_per_prediction': total_cost / len(y_true) if len(y_true) > 0 else 0,
            'total_fp_cost': fp_cost,
            'total_fn_cost': fn_cost
        }
        
    def _is_critical_class(self, class_name):
        """
        Determina si una clase es crítica LEYENDO DESDE LA CONFIGURACIÓN.
        VERSIÓN FINAL, CORRECTA Y ROBUSTA.
        """
        # Lee la lista de palabras clave directamente desde el objeto de configuración
        critical_keywords = self.config.CLASSIFICATION_CONFIG.get('critical_fault_classes', [])
        
        # Comprueba si el nombre de la clase es exactamente una de las clases críticas
        return str(class_name) in critical_keywords
    
    def _analyze_convergence_near_failure(self, y_true, y_pred):
        """Analiza convergencia de predicciones cerca del fallo"""
        bins = [0, 2, 5, 10, np.inf]
        bin_labels = ['0-2d', '2-5d', '5-10d', '>10d']
        
        convergence_metrics = {}
        for i, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
            mask = (y_true >= bin_start) & (y_true < bin_end)
            if np.any(mask):
                convergence_metrics[f'MAE_{bin_labels[i]}'] = (
                    mean_absolute_error(y_true[mask], y_pred[mask])
                )
        
        return convergence_metrics
    
    def _simulate_time_to_detection(self, y_true, y_pred):
        """
        Simula el tiempo de detección para anomalías.
        VERSIÓN FINAL COMPLETAMENTE REESCRITA Y CORRECTA.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Identificar dónde empiezan y terminan los bloques de anomalías verdaderas
        is_true_anomaly = np.array([self._is_critical_class(c) for c in y_true])
        true_anomaly_blocks = np.where(np.diff(is_true_anomaly.astype(int)))[0] + 1
        
        # Si no hay cambios, puede ser todo normal o todo anomalía
        if len(true_anomaly_blocks) == 0:
            if is_true_anomaly.all(): # Todo el dataset es una anomalía
                true_anomaly_events = [(0, len(y_true))]
            else: # No hay anomalías
                return {'avg_time_to_detection': 0, 'median_time_to_detection': 0, 'detection_rate': 1.0}
        else:
            # Creamos los pares (inicio, fin) de cada bloque de anomalía
            starts = np.concatenate(([0], true_anomaly_blocks))
            ends = np.concatenate((true_anomaly_blocks, [len(y_true)]))
            
            true_anomaly_events = []
            for s, e in zip(starts, ends):
                # Nos quedamos solo con los bloques que son realmente anomalías
                if self._is_critical_class(y_true[s]):
                    true_anomaly_events.append((s, e))

        if not true_anomaly_events:
            return {'avg_time_to_detection': 0, 'median_time_to_detection': 0, 'detection_rate': 1.0}

        detection_times = []
        total_anomaly_events = len(true_anomaly_events)

        # Para cada evento de anomalía, ver si fue detectado
        for start, end in true_anomaly_events:
            # Extraer el segmento de predicciones correspondiente al bloque de anomalía
            prediction_segment = y_pred[start:end]
            is_pred_anomaly = np.array([self._is_critical_class(c) for c in prediction_segment])
            
            # Comprobar si hay alguna detección en este bloque
            if np.any(is_pred_anomaly):
                # Encontrar el índice de la *primera* detección dentro del bloque
                first_detection_index_in_segment = np.where(is_pred_anomaly)[0][0]
                detection_time = first_detection_index_in_segment  # El tiempo es el offset desde el inicio
                detection_times.append(detection_time)

        # Calcular las métricas finales
        num_detected_events = len(detection_times)
        detection_rate = num_detected_events / total_anomaly_events if total_anomaly_events > 0 else 1.0
        
        if detection_times:
            avg_time = np.mean(detection_times)
            median_time = np.median(detection_times)
        else:
            avg_time = float('inf')
            median_time = float('inf')

        return {
            'avg_time_to_detection': avg_time,
            'median_time_to_detection': median_time,
            'detection_rate': detection_rate
        }
    
    def _calculate_prognostic_horizon(self, y_true, y_pred):
        """Calcular horizonte pronóstico"""
        threshold = self.config.RUL_CONFIG['prognostic_horizon_threshold']
        
        valid_samples = y_true > threshold
        
        if not np.any(valid_samples):
            return {'prognostic_horizon': 0.0, 'prognostic_horizon_samples': 0}
        
        horizon_predictions = y_pred[valid_samples]
        horizon_true = y_true[valid_samples]
        
        useful_predictions = np.abs(horizon_predictions - horizon_true) <= threshold
        prognostic_horizon = np.mean(useful_predictions)
        
        return {
            'prognostic_horizon': prognostic_horizon,
            'prognostic_horizon_samples': np.sum(valid_samples)
        }
    
    def _calculate_alpha_lambda_metrics(self, y_true, y_pred, alpha=0.5, lambda_val=5):
        """Alpha-Lambda metrics para evaluación RUL"""
        relative_error = np.abs(y_true - y_pred) / np.maximum(y_true, 0.1)
        alpha_accuracy = np.mean(relative_error <= alpha)
        
        lambda_accuracy = np.mean(np.abs(y_true - y_pred) <= lambda_val)
        
        return {
            'alpha_accuracy': alpha_accuracy,
            'lambda_accuracy': lambda_accuracy,
            'alpha_parameter': alpha,
            'lambda_parameter': lambda_val
        }
    
    def _calculate_directional_accuracy(self, y_true, y_pred):
        """Calcular precisión direccional (tendencia)"""
        if len(y_true) < 2:
            return {'directional_accuracy': 0.0}
        
        true_direction = np.diff(y_true) < 0
        pred_direction = np.diff(y_pred) < 0
        
        directional_accuracy = np.mean(true_direction == pred_direction)
        
        return {'directional_accuracy': directional_accuracy}
    
    def _is_critical_class(self, class_name):
        """Determinar si una clase es crítica"""
        critical_keywords = ['Fuga', 'Obstruccion', 'Sobrepresion', 'Degradacion']
        return any(keyword in str(class_name) for keyword in critical_keywords)
    
    @staticmethod
    def format_prediction_summary(predictions: pd.Series, model_name: str, show_examples: bool = False, top_n: int = 3) -> str:
        """
        Genera un string de resumen estadístico legible para una serie de predicciones.
        
        Args:
            predictions (pd.Series): La serie de pandas con las predicciones.
            model_name (str): El nombre del modelo para el título del resumen.
            show_examples (bool): Si es True, muestra ejemplos del inicio y fin. Por defecto es False.
            top_n (int): Cuántos ejemplos mostrar si show_examples es True.

        Returns:
            str: Un string formateado listo para ser logueado.
        """
        if not isinstance(predictions, pd.Series) or predictions.empty:
            return f"  - [!] No se encontraron predicciones válidas para el modelo '{model_name}'."

        # Iniciar la construcción del string de resumen
        summary_lines = [
            f"  - Resumen Estadístico de Predicciones para: '{model_name}'",
            f"    - Total de Predicciones: {len(predictions):,}"
        ]

        # Calcular distribución de clases (conteo y porcentaje)
        dist_counts = predictions.value_counts()
        dist_pct = predictions.value_counts(normalize=True).mul(100).round(2)

        summary_lines.append("    - Distribución de Clases Predichas:")
        for class_name, count in dist_counts.items():
            summary_lines.append(f"      - {str(class_name):<12}: {count:>8,} registros ({dist_pct[class_name]:>5.2f}%)")

        # Mostrar ejemplos solo si se solicita explícitamente
        if show_examples and len(predictions) > top_n * 2:
            summary_lines.append(f"\n    - Primeras {top_n} predicciones de ejemplo:")
            summary_lines.append(predictions.head(top_n).to_string())
            
            summary_lines.append(f"\n    - Últimas {top_n} predicciones de ejemplo:")
            summary_lines.append(predictions.tail(top_n).to_string())

        return "\n".join(summary_lines)
