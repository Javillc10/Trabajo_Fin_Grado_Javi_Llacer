# ==============================================================================
# ARCHIVO COMPLETO Y CORREGIDO: _04_Model_Interpretation_and_Business_Impact.py
# VERSIÓN FINAL CON CORRECCIÓN DEL VALUEERROR EN CONFUSION_MATRIX
# ==============================================================================

import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import os
from src.utils.visualization_tools import IndustrialVisualizer
from src.utils.unified_metrics import UnifiedMetrics

class ModelInterpretability:

    def __init__(self, config, logger, session_id):
        self.config = config
        self.logger = logger
        self.session_id = session_id
        self.results_dir = os.path.join('results', self.session_id)
        self.session_results_dir = os.path.join('results', self.session_id)
        self.interp_dir = os.path.join(self.session_results_dir, 'interpretability_analysis')
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.interp_dir, exist_ok=True) 
    
    def analyze_model(self, model_name, model_object, feature_names, X_sample):
        self.logger.log_subsection(f"Análisis de Interpretabilidad - {model_name.upper()}")
        
        # --- Lógica de Feature Importance (sin cambios) ---
        if hasattr(model_object, 'feature_importances_'):
            importance_df = pd.DataFrame({'feature': feature_names, 'importance': model_object.feature_importances_}).sort_values('importance', ascending=False)
            self.logger.log_step("Top 10 características más importantes (del modelo):")
            for _, row in importance_df.head(10).iterrows(): self.logger.log_step(f"  - {row['feature']}: {row['importance']:.4f}")
            self._plot_feature_importance(importance_df, model_name)
        else: 
            self.logger.log_step(f"El modelo '{model_name}' no tiene el atributo 'feature_importances_'.")
            
        # --- Lógica de SHAP (mejorada) ---
        try:
            self._analyze_and_save_shap_objects(model_object, X_sample, model_name)
        except Exception as e:
            self.logger.log_error(f"Falló el análisis SHAP para '{model_name}': {e}", exc_info=True)
    
    def _plot_feature_importance(self, importance_df, model_name):
        plt.figure(figsize=(12, 8)); sns.barplot(data=importance_df.head(20), y='feature', x='importance', palette='viridis'); plt.title(f'Importancia de Características - {model_name}'); plt.tight_layout()
        save_path = os.path.join(self.results_dir, f'feature_importance_{model_name}.png'); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close(); self.logger.log_artifact(f"Gráfico de Importancia de Features ({model_name})", save_path)
    
    def _analyze_shap_values(self, model_object, X_sample, model_name, feature_names):
        self.logger.log_step(f"Iniciando análisis SHAP para {model_name}...")
        X_shap = X_sample.sample(n=min(1000, len(X_sample)), random_state=self.config.RANDOM_SEED)
        explainer = shap.Explainer(model_object, X_shap) 
        shap_values = explainer(X_shap, check_additivity=False) 
        if isinstance(shap_values.values, list): shap_values_for_plot = np.abs(shap_values.values).mean(axis=0)
        else: shap_values_for_plot = shap_values.values
        shap.summary_plot(shap_values_for_plot, X_shap, feature_names=feature_names, show=False); plt.title(f'Resumen de Valores SHAP - {model_name}'); plt.tight_layout()
        save_path = os.path.join(self.results_dir, f'shap_summary_{model_name}.png'); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close(); self.logger.log_artifact(f"Gráfico de Resumen SHAP ({model_name})", save_path); self.logger.log_success(f"Análisis SHAP completado para {model_name}.")

    def _analyze_and_save_shap_objects(self, model_object, X_sample, model_name):
        """
        Calcula los valores SHAP, genera el gráfico de resumen estático Y
        GUARDA LOS OBJETOS PARA LA WEB INTERACTIVA.
        """
        self.logger.log_step(f"Iniciando análisis SHAP para {model_name}...")
        
        # Usamos una muestra más pequeña para el cálculo SHAP para agilizar
        X_shap_sample = X_sample.sample(n=min(1000, len(X_sample)), random_state=self.config.RANDOM_SEED)
        
        # 1. Calcular explainer y shap_values (como antes)
        explainer = shap.Explainer(model_object, X_shap_sample) 
        shap_values = explainer(X_shap_sample, check_additivity=False) 
        
        # 2. Generar el gráfico estático (como antes)
        if isinstance(shap_values.values, list): 
            shap_values_for_plot = np.abs(shap_values.values).mean(axis=0)
        else: 
            shap_values_for_plot = shap_values.values
            
        shap.summary_plot(shap_values_for_plot, X_shap_sample, show=False)
        plt.title(f'Resumen de Valores SHAP - {model_name}')
        plt.tight_layout()
        save_path_png = os.path.join(self.interp_dir, f'shap_summary_{model_name}.png')
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.log_artifact(f"Gráfico de Resumen SHAP ({model_name})", save_path_png)
        
        # --- ¡NUEVO! GUARDAR LOS INGREDIENTES PARA LA WEB ---
        self.logger.log_step(f"Guardando artefactos SHAP interactivos para {model_name}...")
        
        # Guardamos los objetos usando joblib
        joblib.dump(explainer, os.path.join(self.interp_dir, f'shap_explainer_{model_name}.joblib'))
        joblib.dump(shap_values, os.path.join(self.interp_dir, f'shap_values_{model_name}.joblib'))        
        # Guardamos la misma muestra de datos usada para el cálculo
        X_shap_sample.to_parquet(os.path.join(self.interp_dir, f'X_shap_sample_{model_name}.parquet'))

        self.logger.log_success(f"Análisis SHAP y guardado de artefactos interactivos completado para {model_name}.")
        
class BusinessImpactAnalysis:

    def __init__(self, config, logger):
        self.config = config; self.logger = logger; self.operational_costs = self.config.BUSINESS_CONFIG['operational_costs']; self.critical_classes = self.config.CLASSIFICATION_CONFIG['critical_fault_classes']
    def analyze(self, classification_results, rul_results):
        self.logger.log_subsection("Análisis de Impacto de Negocio"); classification_summary = self._calculate_classification_costs(classification_results); rul_summary = self._analyze_rul_cost_benefit(rul_results); recommendations = self._generate_recommendations(classification_summary, rul_summary)
        self.logger.log_step("Recomendaciones de Negocio Generadas:"); 
        for rec in recommendations: self.logger.log_finding(rec)
        return {'classification_summary': classification_summary, 'rul_summary': rul_summary, 'recommendations': recommendations}
    def _calculate_classification_costs(self, classification_results):
        if not (classification_results and 'y_true' in classification_results and 'y_pred' in classification_results): self.logger.log_warning("No se proporcionaron resultados de clasificación válidos para el análisis de costos."); return {}
        self.logger.log_step("Calculando costos de clasificación..."); y_true = classification_results['y_true']; y_pred = classification_results['y_pred']; labels = sorted(list(set(y_true) | set(y_pred))); cm = confusion_matrix(y_true, y_pred, labels=labels); fp_cost = 0; fn_cost = 0
        for critical_class_name in self.critical_classes:
            if critical_class_name not in labels: continue
            class_index = labels.index(critical_class_name); fp_count = cm[:, class_index].sum() - cm[class_index, class_index]; fn_count = cm[class_index, :].sum() - cm[class_index, class_index]; fp_cost += fp_count * self.operational_costs['false_positive_cost']; fn_cost += fn_count * self.operational_costs['false_negative_cost']
        monitoring_cost = 30 * self.operational_costs['sensor_monitoring_cost_per_day']; total_cost = fp_cost + fn_cost + monitoring_cost; summary = {'false_positive_cost': fp_cost, 'false_negative_cost': fn_cost, 'monitoring_cost': monitoring_cost, 'total_cost': total_cost}
        self.logger.log_step(f"Costo total de clasificación estimado: {total_cost:,.2f} (FP: {fp_cost:,.2f}, FN: {fn_cost:,.2f})"); return summary
    def _analyze_rul_cost_benefit(self, rul_results):
        if not rul_results: self.logger.log_warning("No se proporcionaron resultados de RUL para el análisis."); return {}
        self.logger.log_step("Simulando costo-beneficio de la implementación de RUL..."); maintenance_opt = self.config.BUSINESS_CONFIG['maintenance_optimization']; annual_maintenance_cost = 100000; annual_downtime_cost = 50000
        benefits = {'maintenance_savings': annual_maintenance_cost * maintenance_opt['planned_maintenance_savings'], 'inventory_savings': (annual_maintenance_cost * 0.2) * maintenance_opt['inventory_optimization'], 'downtime_savings': annual_downtime_cost * maintenance_opt['downtime_reduction']}; total_benefits = sum(benefits.values())
        costs = {'model_development': 25000, 'system_integration': 15000, 'training': 5000, 'annual_maintenance': 8000}; total_initial_cost = costs['model_development'] + costs['system_integration'] + costs['training']; net_annual_benefit = total_benefits - costs['annual_maintenance']; payback_period_years = total_initial_cost / net_annual_benefit if net_annual_benefit > 0 else float('inf')
        summary = {'total_annual_benefits': total_benefits, 'net_annual_benefit': net_annual_benefit, 'payback_period_years': payback_period_years}; self.logger.log_step(f"Beneficio neto anual estimado de RUL: {net_annual_benefit:,.2f}"); return summary
    def _generate_recommendations(self, clf_summary, rul_summary):
        recommendations = []
        if clf_summary:
            if clf_summary['false_negative_cost'] > clf_summary['false_positive_cost'] * 5: recommendations.append("Acción Crítica: El costo de fallos no detectados (FN) es muy alto. Priorizar el aumento de la sensibilidad (Recall) del modelo para clases críticas, incluso si aumenta ligeramente los falsos positivos.")
            else: recommendations.append("Recomendación: Optimizar el balance entre Precisión y Recall para minimizar el costo total.")
        if rul_summary:
            if rul_summary['payback_period_years'] < 2: recommendations.append(f"Recomendación Estratégica: La implementación del sistema RUL es altamente rentable, con un período de recuperación de solo {rul_summary['payback_period_years']:.1f} años. Proceder con la integración.")
            else: recommendations.append("Recomendación: Evaluar la mejora del modelo RUL para aumentar los ahorros y reducir el período de recuperación de la inversión.")
        recommendations.append("Siguiente Paso: Validar el rendimiento del modelo con datos de producción reales para confirmar las estimaciones de impacto de negocio."); return recommendations

class FinalReporter:
    def __init__(self, config, logger, session_id):
        self.config = config
        self.logger = logger
        self.session_id = session_id
        self.results_dir = os.path.join('results', self.session_id)
        self.visualizer = IndustrialVisualizer(config)
        self.metrics_calculator = UnifiedMetrics(config, self.logger)

    def generate_full_report(self, df_processed, classification_results, rul_results):
        self.logger.log_section("INICIO DE LA FASE DE REPORTE Y VISUALIZACIÓN FINAL")
        self.logger.log_subsection("Generando Gráficos del Análisis Exploratorio")
        self._generate_eda_visualizations(df_processed)
        self.logger.log_subsection("Generando Reportes de Modelos de Clasificación")
        self._report_classification_models(classification_results)
        self.logger.log_subsection("Generando Reportes de Modelos RUL")
        self._report_rul_models(rul_results)
        self.logger.log_subsection("Generando Análisis de Interpretabilidad de Modelos")
        self._report_interpretability(df_processed, classification_results, rul_results)
        self.logger.log_subsection("Generando Análisis de Impacto de Negocio")
        self._report_business_impact(classification_results, rul_results)
        self.logger.log_success("Todos los reportes y visualizaciones han sido generados.")

    def _generate_eda_visualizations(self, df_processed):
        eda_path = os.path.join(self.results_dir, 'eda_sensor_evolution.png')
        self.visualizer.plot_sensor_evolution(df_processed, save_path=eda_path)
        self.logger.log_artifact("Gráfico de Evolución de Sensores (EDA)", eda_path)
        degradation_path = os.path.join(self.results_dir, 'eda_degradation_analysis.png')
        self.visualizer.plot_degradation_analysis(df_processed, save_path=degradation_path)
        self.logger.log_artifact("Gráfico de Análisis de Degradación (EDA)", degradation_path)

    def _report_classification_models(self, classification_results):
        self.logger.log_subsection("Generando Reportes de Modelos de Clasificación")
        if not (classification_results and 'models' in classification_results):
            self.logger.log_warning("No se encontraron resultados de clasificación para reportar.")
            return

        for model_name, model_data in classification_results['models'].items():
            self.logger.log_step(f"Generando reporte para el modelo: {model_name}")
            
            predictions = model_data.get('predictions')
            y_test = model_data.get('y_test')

            if predictions is not None:
                # Aseguramos que predictions sea un objeto Serie de Pandas para el resumen
                if not isinstance(predictions, pd.Series):
                    predictions = pd.Series(predictions)
                prediction_summary = UnifiedMetrics.format_prediction_summary(predictions, model_name)
                self.logger.log_step(prediction_summary)

            if model_data.get('metrics') and y_test is not None and predictions is not None:
                
                # ### INICIO DE LA CORRECCIÓN FINAL ###
                # En lugar de usar las clases del modelo, usamos las clases que REALMENTE
                # están presentes en los datos de este fold para evitar el error.
                labels_present = sorted(list(set(y_test) | set(predictions)))
                # ### FIN DE LA CORRECCIÓN FINAL ###

                dashboard_data = {
                    'y_true': pd.Series(y_test),
                    'y_pred': predictions,
                    # Usamos la lista de etiquetas robusta que acabamos de crear
                    'confusion_matrix': confusion_matrix(y_test, predictions, labels=labels_present)
                }
                dashboard_data.update(model_data['metrics'])
                
                save_path = os.path.join(self.results_dir, f'dashboard_classification_{model_name}.png')
                self.metrics_calculator.plot_metrics_dashboard(dashboard_data, 'classification', save_path)
                self.logger.log_artifact(f"Dashboard Clasificación ({model_name})", save_path)
            else:
                self.logger.log_warning(f"No se pudieron generar gráficos para '{model_name}' por falta de datos.")

    def _report_rul_models(self, rul_results):
        """Genera los dashboards y gráficos específicos para los modelos RUL."""
        if not rul_results:
            self.logger.log_warning("No se encontraron resultados de RUL para reportar.")
            return

        for component, component_data in rul_results.items():
            if 'models' not in component_data:
                continue
                
            for model_name, model_info in component_data['models'].items():
                predictions_df = model_info.get('predictions_df') # Buscamos el DataFrame con y_true y y_pred
                
                if predictions_df is not None and not predictions_df.empty:
                    self.logger.log_step(f"  - Generando gráfico de convergencia RUL para '{component} - {model_name}'...")
                    save_path = os.path.join(self.results_dir, f"rul_convergence_{component}_{model_name}.png")
                    
                    # Llamada a la nueva función de visualización
                    self.visualizer.plot_rul_convergence(
                        df_preds=predictions_df,
                        component_name=f"{component} ({model_name})",
                        save_path=save_path
                    )
                    self.logger.log_artifact(f"Gráfico de Convergencia RUL ({component} - {model_name})", save_path)
                else:
                    self.logger.log_warning(f"No se encontraron datos de predicción para el gráfico RUL de '{component} - {model_name}'.")

    def _report_interpretability(self, df, clf_results, rul_results):
        interpreter = ModelInterpretability(self.config, self.logger, self.session_id)
        if clf_results and 'models' in clf_results:
            best_model_name = max(clf_results['models'], key=lambda k: clf_results['models'][k].get('metrics', {}).get('f1_macro', 0))
            model_data = clf_results['models'][best_model_name]
            feature_names = clf_results['feature_columns']
            # Asegurarse de que X_sample tenga las columnas correctas
            if all(col in df.columns for col in feature_names):
                X_sample = df[feature_names]
                interpreter.analyze_model(best_model_name, model_data['model'], feature_names, X_sample)
            else:
                self.logger.log_warning("No se pudo generar análisis de interpretabilidad por falta de columnas en el dataframe procesado.")

    def _report_business_impact(self, clf_results, rul_results):
        analyzer = BusinessImpactAnalysis(self.config, self.logger)
        clf_impact_data = None
        if clf_results and 'models' in clf_results:
            best_model_name = max(clf_results['models'], key=lambda k: clf_results['models'][k].get('metrics', {}).get('f1_macro', 0))
            model_data = clf_results['models'][best_model_name]
            clf_impact_data = {'y_true': model_data.get('y_test'), 'y_pred': model_data.get('predictions')}
        analyzer.analyze(clf_impact_data, rul_results)

def run_final_reporting_step(df_processed, classification_results, rul_results, config, logger, session_id):
    try:
        reporter = FinalReporter(config, logger, session_id)
        reporter.generate_full_report(df_processed, classification_results, rul_results)
    except Exception as e:
        logger.log_critical(f"Error fatal durante la fase de reporte final: {e}", exc_info=True)
        # No relanzamos el error para permitir que el pipeline continúe si es posible
