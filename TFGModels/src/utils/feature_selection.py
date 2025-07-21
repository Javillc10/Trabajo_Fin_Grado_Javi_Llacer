"""
Módulo de selección inteligente de features para mantenimiento predictivo
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from src.utils.training_logger import TrainingLogger
import warnings

warnings.filterwarnings('ignore')

class FeatureSelector:
    """
    Clase para selección inteligente y automática de features.
    VERSIÓN MEJORADA: Ahora soporta tanto 'classification' como 'regression'.
    """
    

    def __init__(self, config, logger, task_type='classification'):
        self.config = config
        self.logger = logger
        self.task_type = task_type  
        self.selected_features_ = None
        self.feature_importance_ = {}
        self.removed_features_ = {}
        self.imputer_ = None
        self.variance_selector_ = None
        self.correlated_to_drop_ = []
        self.kbest_selector_ = None
            
    def fit(self, X, y=None):
        """
        Aprende qué features seleccionar. Ahora es consciente de la tarea.
        """
        self.logger.log_step(f"Ajustando el selector de features (fit) para tarea: '{self.task_type}'...")
        
        sample_size = self.config.FEATURE_SELECTION.get('importance_sample_size', 200000)
        X_fit, y_fit = (X, y)
        if len(X) > sample_size:
            self.logger.log_step(f"Dataset grande detectado. Usando muestra de {sample_size:,} para el ajuste del selector.")
            X_fit = X.sample(n=sample_size, random_state=self.config.RANDOM_SEED)
            y_fit = y.loc[X_fit.index] if y is not None else None

        self.imputer_ = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(self.imputer_.fit_transform(X_fit), columns=X_fit.columns, index=X_fit.index)

        # ... (La lógica de varianza y correlación no cambia) ...
        self.logger.log_step("   - Analizando varianza...")
        self.variance_selector_ = VarianceThreshold(threshold=self.config.FEATURE_SELECTION.get('variance_threshold', 0.0))
        self.variance_selector_.fit(X_imputed)
        low_variance_cols = X_imputed.columns[~self.variance_selector_.get_support()]
        self.removed_features_['low_variance'] = low_variance_cols.tolist()
        X_variance = X_imputed.drop(columns=low_variance_cols)
        self.logger.log_step(f"     - {len(low_variance_cols)} features con baja varianza identificadas.")

        self.logger.log_step("   - Analizando correlaciones...")
        correlation_threshold = self.config.FEATURE_SELECTION.get('correlation_threshold', 1.0)
        corr_matrix = X_variance.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.correlated_to_drop_ = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
        self.removed_features_['correlated'] = self.correlated_to_drop_
        X_correlated = X_variance.drop(columns=self.correlated_to_drop_)
        self.logger.log_step(f"     - {len(self.correlated_to_drop_)} features altamente correlacionadas identificadas.")


        if y_fit is not None:
            # ### MODIFICADO ###: Lógica para elegir la función de puntuación correcta
            self.logger.log_step(f"   - Analizando importancia de features (SelectKBest for {self.task_type})...")
            
            if self.task_type == 'regression':
                score_func = f_regression
                self.logger.log_step("     - Usando 'f_regression' para la puntuación.")
            else: # Por defecto, usamos clasificación
                score_func = f_classif
                self.logger.log_step("     - Usando 'f_classif' para la puntuación.")

            k = self.config.FEATURE_SELECTION.get('k_best_features', 30)
            
            if k < X_correlated.shape[1]:
                # Ahora usamos la 'score_func' que hemos elegido dinámicamente
                self.kbest_selector_ = SelectKBest(score_func=score_func, k=k)
                self.kbest_selector_.fit(X_correlated, y_fit)
                kbest_support = self.kbest_selector_.get_support()
                kbest_removed = X_correlated.columns[~kbest_support]
                self.removed_features_['importance'] = kbest_removed.tolist()
                self.logger.log_step(f"     - {len(kbest_removed)} features de baja importancia identificadas.")
            else:
                self.kbest_selector_ = None
                self.removed_features_['importance'] = []
        else:
            self.kbest_selector_ = None
            self.removed_features_['importance'] = []

        # --- PASO FINAL CORREGIDO Y SIMPLIFICADO ---
        final_cols_candidate = X_imputed.columns[self.variance_selector_.get_support()].tolist()
        final_cols_candidate = [col for col in final_cols_candidate if col not in self.correlated_to_drop_]
        
        if self.kbest_selector_ is not None:
            kbest_mask = self.kbest_selector_.get_support()
            temp_df = pd.DataFrame(columns=final_cols_candidate)
            selected_cols_after_kbest = temp_df.columns[kbest_mask]
            self.selected_features_ = selected_cols_after_kbest.tolist()
        else:
            self.selected_features_ = final_cols_candidate

        self.logger.log_success(f"Selector ajustado. {len(self.selected_features_)} features serán seleccionadas.")
        return self

    def transform(self, X):
        """
        Transforma los datos. Asume que los datos ya han sido imputados.
        """
        if self.selected_features_ is None:
            raise RuntimeError("Debes llamar a 'fit' antes de 'transform'.")
            
        self.logger.log_step("Aplicando transformación con selector ajustado...")
        
        # Ya no necesitamos imputar aquí, directamente devolvemos las columnas
        return X[self.selected_features_]

    def fit_transform(self, X, y=None):
        """
        Ajusta el selector y transforma los datos.
        """
        self.fit(X, y)
        return self.transform(X)
    
    def _remove_low_variance_features(self, X):
        """Eliminar features con varianza muy baja"""
        print("\n📊 Analizando varianza de features...")
        
        threshold = self.config.FEATURE_SELECTION.get('variance_threshold', 0.01)
        # Aplicar VarianceThreshold
        selector = VarianceThreshold(threshold)
        selector.fit(X)
        
        # Identificar features eliminadas
        features_to_keep = X.columns[selector.get_support()].tolist()
        removed_features = X.columns[~selector.get_support()].tolist()
        
        self.removed_features_['low_variance'] = removed_features
        print(f"   ❌ Features eliminadas por baja varianza: {len(removed_features)}")
        
        return X[features_to_keep]
    
    def _remove_correlated_features(self, X):
        """Eliminar features altamente correlacionadas"""
        print("\n🔗 Analizando correlaciones...")
        
        threshold = self.config.FEATURE_SELECTION.get('correlation_threshold', 0.95) # Usar 0.95 como default si no está en config
        # Calcular matriz de correlación
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Identificar features a eliminar
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        self.removed_features_['correlated'] = to_drop
        print(f"   ❌ Features eliminadas por alta correlación: {len(to_drop)}")
        
        return X.drop(columns=to_drop)
    
    def _select_by_importance(self, X, y):
        """
        Selecciona las 'k' mejores features usando la estrategia "Sample, Fit, Transform"
        para manejar datasets grandes de forma eficiente.
        """
        self.logger.log_step("🎯 Seleccionando features por importancia (modo eficiente)...")
        
        if X.shape[0] != y.shape[0]:
            error_msg = f"Inconsistencia de datos: X tiene {X.shape[0]} muestras y y tiene {y.shape[0]}."
            self.logger.log_critical(error_msg)
            raise ValueError(error_msg)

        k = self.config.FEATURE_SELECTION.get('k_best_features', 15)
        if k >= X.shape[1]:
            self.logger.log_step(f"WARN: 'k' ({k}) es >= al nº de features ({X.shape[1]}). Omitiendo selección.")
            return X
            
        from sklearn.feature_selection import SelectKBest, f_classif
        
        # 1. SAMPLE: Tomar una muestra si el dataset es grande
        sample_size = self.config.FEATURE_SELECTION.get('importance_sample_size', 200000)
        
        if X.shape[0] > sample_size:
            self.logger.log_step(f"   Dataset grande detectado. Usando muestra de {sample_size:,} registros para el cálculo.")
            X_sample = X.sample(n=sample_size, random_state=self.config.RANDOM_SEED)
            y_sample = y.loc[X_sample.index]
        else:
            X_sample = X
            y_sample = y

        # 2. FIT: Ajustar el selector SOBRE LA MUESTRA (rápido)
        self.logger.log_step("   Ajustando el selector sobre la muestra...")
        k_best = SelectKBest(score_func=f_classif, k=k)
        try:
            k_best.fit(X_sample, y_sample)
        except Exception as e:
            self.logger.log_critical(f"Error durante SelectKBest.fit en la muestra: {e}")
            raise e

        # 3. TRANSFORM: Transformar el dataset COMPLETO (muy rápido)
        self.logger.log_step("   Transformando el dataset completo con el selector ajustado...")
        X_new = k_best.transform(X)
        
        selected_features = X.columns[k_best.get_support()]
        removed_features = X.columns[~k_best.get_support()]
        
        # Guardar la importancia de las features (calculada sobre la muestra)
        scores = {feat: score for feat, score in zip(X.columns[k_best.get_support()], k_best.scores_[k_best.get_support()])}
        self.feature_importance_.update(scores)

        self.logger.log_success(f"   ✅ Features seleccionadas por importancia: {len(selected_features)}")
        if len(removed_features) > 0:
            self.removed_features_['importance'] = removed_features.tolist() # Guardar features eliminadas
            self.logger.log_step(f"   ❌ Features descartadas por baja importancia: {len(removed_features)}")
            
        return pd.DataFrame(X_new, columns=selected_features, index=X.index)
    
    def _prioritize_by_domain(self, X):
        """Priorizar features por relevancia de dominio"""
        print("\n🏭 Priorizando por relevancia industrial...")
        
        # Definir categorías de prioridad
        priority_patterns = {
            'high': ['_deg_', 'degradation', 'trend', 'failure'],
            'medium': ['operational', 'stress', 'cycle', 'efficiency'],
            'low': ['spectral', 'fft', 'variability']
        }
        
        # Clasificar features
        categorized_features = {
            'high': [],
            'medium': [],
            'low': [],
            'other': []
        }
        
        for feature in X.columns:
            categorized = False
            for priority, patterns in priority_patterns.items():
                if any(pattern in feature.lower() for pattern in patterns):
                    categorized_features[priority].append(feature)
                    categorized = True
                    break
            if not categorized:
                categorized_features['other'].append(feature)
        
        # Si hay demasiadas features, mantener proporción por prioridad
        total_features = X.shape[1]
        if total_features > 80:  # Límite objetivo
            target_high = 40     # 50% para high
            target_medium = 25   # ~31% para medium
            target_low = 15      # ~19% para low y other
            
            # Seleccionar features manteniendo prioridades
            selected_features = (
                categorized_features['high'][:target_high] +
                categorized_features['medium'][:target_medium] +
                categorized_features['low'][:target_low] +
                categorized_features['other'][:target_low]
            )
            
            removed_features = [f for f in X.columns if f not in selected_features]
            self.removed_features_['priority'] = removed_features
            print(f"   ❌ Features eliminadas por prioridad: {len(removed_features)}")
            
            return X[selected_features]
        
        return X
    
    def _generate_selection_report(self, X_original, X_selected):
        """Generar reporte detallado de la selección"""
        total_original = len(X_original.columns)
        total_selected = len(X_selected.columns)
        total_removed = total_original - total_selected
        
        print("\n📊 REPORTE DE SELECCIÓN DE FEATURES")
        print("=" * 50)
        print(f"Features originales: {total_original}")
        print(f"Features seleccionadas: {total_selected}")
        print(f"Features eliminadas: {total_removed}")
        print(f"Reducción: {(total_removed/total_original)*100:.1f}%")
        
        print("\n🔍 Features eliminadas por criterio:")
        for criterion, features in self.removed_features_.items():
            print(f"   • {criterion}: {len(features)}")
        
        if self.feature_importance_:
            print("\n🏆 Top 10 features más importantes:")
            top_features = sorted(self.feature_importance_.items(), 
                                key=lambda x: x[1], reverse=True)[:10]
            for feature, importance in top_features:
                print(f"   • {feature}: {importance:.3f}")
