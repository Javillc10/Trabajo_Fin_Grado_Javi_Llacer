# src/utils/visualization_tools.py
"""
Herramientas de visualización especializadas para mantenimiento predictivo
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.dates import DateFormatter
import warnings
warnings.filterwarnings('ignore')
import matplotlib


matplotlib.rcParams['agg.path.chunksize'] = 20000
matplotlib.rcParams['path.simplify_threshold'] = 1.0


class IndustrialVisualizer:
    """Visualizador especializado para datos industriales"""
    
    def __init__(self, config):
        self.config = config
        self.viz_config = config.VISUALIZATION_CONFIG
        
        # Configurar estilo
        plt.style.use('default')  # Usar estilo por defecto más compatible
        sns.set_palette(self.viz_config['color_palette'])
        
    def plot_sensor_evolution(self, df, save_path=None):
        """Visualizar evolución temporal de sensores de forma eficiente."""
        
        sensor_columns = [col for col in self.config.SENSOR_CONFIG['sensor_columns'] 
                        if col in df.columns]
        
        if not sensor_columns:
            print("No se encontraron columnas de sensores para visualizar")
            return
        
        # --- NUEVO: Muestreamos los datos si son demasiados para visualizar ---
        # Esto previene el error 'Exceeded cell block limit' y hace el gráfico más rápido
        max_points_to_plot = 20000  # Un límite razonable para una visualización clara
        if len(df) > max_points_to_plot:
            print(f"INFO: El dataset es muy grande ({len(df)} puntos). Muestreando a {max_points_to_plot} puntos para la visualización de sensores.")
            df_sample = df.sample(n=max_points_to_plot, random_state=42).sort_values(by='timestamp')
        else:
            df_sample = df
        # --- FIN DEL CÓDIGO NUEVO ---

        n_sensors = len(sensor_columns)
        fig, axes = plt.subplots(n_sensors, 1, figsize=(15, 4 * n_sensors), sharex=True)
        
        if n_sensors == 1:
            axes = [axes]
        
        for i, sensor in enumerate(sensor_columns):
            ax = axes[i]
            
            # --- MODIFICADO: Usamos df_sample en lugar de df ---
            ax.plot(df_sample['timestamp'], df_sample[sensor], linewidth=0.5, alpha=0.7, label='Datos Muestreados')
            
            # La media móvil se calcula sobre el dataframe original para ser precisa
            window_size = min(len(df) // 100, 2000) # Ajuste de ventana para que sea representativa
            if window_size > 1:
                moving_avg = df[sensor].rolling(window_size, center=True).mean()
                # Ahora filtramos la media móvil para que coincida con los puntos muestreados
                ax.plot(df['timestamp'], moving_avg, color='red', linewidth=2, 
                    label=f'Media Móvil (calculada sobre datos completos)')
            
            ax.set_ylabel(f'{sensor.replace("sensor_presion_", "").replace("_bar", "")} (bar)')
            ax.set_title(f'Evolución Temporal - {sensor}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            if 'timestamp' in df_sample.columns:
                ax.xaxis.set_major_formatter(DateFormatter('%m/%d %H:%M'))
        
        plt.xlabel('Tiempo')
        plt.suptitle('Evolución Temporal de Sensores', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Ajuste para que el suptitle no se solape
        
        if save_path:
            plt.savefig(save_path, dpi=self.viz_config['dpi'], bbox_inches='tight')
        else:
            # Asegurarse de que el directorio de resultados existe
            import os
            results_dir = self.config.RESULTS_DIR
            os.makedirs(results_dir, exist_ok=True)
            plt.savefig(os.path.join(results_dir, 'sensor_evolution.png'), 
                    dpi=self.viz_config['dpi'], bbox_inches='tight')
        plt.close(fig) # Cerrar la figura para liberar memoria
        
     
    def plot_degradation_analysis(self, df, save_path=None):
        """Visualizar análisis de degradación"""
        
        # Buscamos la única columna target correcta desde la configuración.
        target_col = self.config.COLUMN_MAPPING.get('target_classification')
                
        if target_col is None or target_col not in df.columns:
            print(f"No se encontró la columna objetivo '{target_col}' para la visualización.")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análisis de Degradación del Sistema', fontsize=16, fontweight='bold')
            
        # 1. Distribución de estados
        ax1 = axes[0, 0]
        estado_counts = df[target_col].value_counts()
        
        # Rotar labels si son muy largos
        estado_counts.plot(kind='bar', ax=ax1)
        ax1.set_title('Distribución de Estados del Sistema')
        ax1.set_xlabel('Estado')
        ax1.set_ylabel('Frecuencia')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 2. Evolución de estados por semana
        if 'timestamp' in df.columns:
            ax2 = axes[0, 1]
            
            # Agregar semana
            df_temp = df.copy()
            df_temp['semana'] = df_temp['timestamp'].dt.isocalendar().week
            df_temp['semana_relativa'] = df_temp['semana'] - df_temp['semana'].min() + 1
            
            # Porcentaje de operación normal por semana
            normal_operation = 'Normal_Operacion'
            if normal_operation in df_temp[target_col].unique():
                weekly_normal = df_temp.groupby('semana_relativa')[target_col].apply(
                    lambda x: (x == normal_operation).mean() * 100
                )
                
                ax2.plot(weekly_normal.index, weekly_normal.values, 
                        marker='o', linewidth=3, markersize=8, color='green')
                ax2.set_title('Degradación Temporal (% Operación Normal)')
                ax2.set_xlabel('Semana')
                ax2.set_ylabel('% Operación Normal')
                ax2.grid(True, alpha=0.3)
                
                # Línea de tendencia
                z = np.polyfit(weekly_normal.index, weekly_normal.values, 1)
                p = np.poly1d(z)
                ax2.plot(weekly_normal.index, p(weekly_normal.index), 
                        "r--", alpha=0.8, label=f'Tendencia: {z[0]:.1f}%/semana')
                ax2.legend()
        
        # 3. Mapa de calor de estados por hora del día
        ax3 = axes[1, 0]
        
        if 'timestamp' in df.columns:
            df_temp = df.copy()
            df_temp['hora'] = df_temp['timestamp'].dt.hour
            df_temp['dia'] = df_temp['timestamp'].dt.day
            
            # Crear tabla de contingencia
            estados_principales = df_temp[target_col].value_counts().head(8).index
            df_filtered = df_temp[df_temp[target_col].isin(estados_principales)]
            
            if len(df_filtered) > 0:
                contingency = pd.crosstab(df_filtered['hora'], df_filtered[target_col],
                                        normalize='index') * 100
                
                sns.heatmap(contingency.T, annot=False, cmap='YlOrRd', ax=ax3, cbar_kws={'label': '% Tiempo'})
                ax3.set_title('Estados por Hora del Día')
                ax3.set_xlabel('Hora del Día')
                ax3.set_ylabel('Estado del Sistema')
        
        # 4. Boxplot de sensores por estado crítico vs normal
        ax4 = axes[1, 1]
        
        sensor_columns = [col for col in self.config.SENSOR_CONFIG['sensor_columns'] 
                         if col in df.columns]
        
        if sensor_columns and len(sensor_columns) > 0:
            # Usar primer sensor disponible
            sensor = sensor_columns[0]
            
            # Clasificar estados en críticos vs normales
            df_temp = df.copy()
            critical_keywords = ['Fuga', 'Obstruccion', 'Sobrepresion', 'Degradacion']
            df_temp['es_critico'] = df_temp[target_col].apply(
                lambda x: any(keyword in str(x) for keyword in critical_keywords)
            )
            
            # Boxplot
            sns.boxplot(data=df_temp, x='es_critico', y=sensor, ax=ax4)
            ax4.set_title(f'Distribución de {sensor} por Criticidad')
            ax4.set_xlabel('Estado Crítico')
            ax4.set_xticklabels(['Normal', 'Crítico'])
            ax4.set_ylabel(f'{sensor} (bar)')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.viz_config['dpi'], bbox_inches='tight')
        else:
            plt.savefig(f'{self.config.RESULTS_DIR}/degradation_analysis.png', 
                       dpi=self.viz_config['dpi'], bbox_inches='tight')
        
     
    def plot_feature_correlation_matrix(self, df, features=None, save_path=None):
        """Visualizar matriz de correlación de features"""
        
        if features is None:
            # Seleccionar features numéricas
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            features = [col for col in numeric_columns if col not in ['timestamp']]
        
        if len(features) == 0:
            print("No se encontraron features numéricas para correlación")
            return
        
        # Limitar número de features para visualización
        if len(features) > 30:
            features = features[:30]
            print(f"Limitando a top 30 features para visualización")
        
        # Calcular matriz de correlación
        corr_matrix = df[features].corr()
        
        # Crear figura
        plt.figure(figsize=(12, 10))
        
        # Heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Máscara triangular superior
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('Matriz de Correlación de Features', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.viz_config['dpi'], bbox_inches='tight')
        else:
            plt.savefig(f'{self.config.RESULTS_DIR}/feature_correlation_matrix.png', 
                       dpi=self.viz_config['dpi'], bbox_inches='tight')
        
         
        # Imprimir correlaciones más altas
        print("\nTop 10 Correlaciones Más Altas:")
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
        
        corr_pairs_df = pd.DataFrame(corr_pairs)
        corr_pairs_df['abs_correlation'] = np.abs(corr_pairs_df['correlation'])
        top_corr = corr_pairs_df.nlargest(10, 'abs_correlation')
        
        for _, row in top_corr.iterrows():
            print(f"  {row['feature1']} <-> {row['feature2']}: {row['correlation']:.3f}")
    
    def plot_model_comparison(self, results_dict, metric_name='f1_macro', save_path=None):
        """Comparar rendimiento de modelos"""
        
        models = list(results_dict.keys())
        scores = [results_dict[model].get(metric_name, 0) for model in models]
        
        # Crear figura
        plt.figure(figsize=(12, 8))
        
        # Gráfico de barras
        bars = plt.bar(models, scores, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Colorear barras según rendimiento
        colors = ['green' if score > 0.7 else 'orange' if score > 0.5 else 'red' for score in scores]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Añadir valores en las barras
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Configuración
        plt.title(f'Comparación de Modelos - {metric_name.upper()}', fontsize=14, fontweight='bold')
        plt.xlabel('Modelo')
        plt.ylabel(metric_name.replace('_', ' ').title())
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Línea objetivo (si está definida)
        target_metrics = self.config.METRICS_CONFIG
        if 'classification' in target_metrics and metric_name in ['f1_macro']:
            target_value = target_metrics['classification'].get('target_f1_macro', 0.7)
            plt.axhline(y=target_value, color='red', linestyle='--', linewidth=2, 
                       label=f'Objetivo: {target_value}')
            plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.viz_config['dpi'], bbox_inches='tight')
        else:
            plt.savefig(f'{self.config.RESULTS_DIR}/model_comparison_{metric_name}.png', 
                       dpi=self.viz_config['dpi'], bbox_inches='tight')
        
    def plot_rul_convergence(self, df_preds, component_name, model_name=None, save_path=None):
        """Visualiza la convergencia de las predicciones de RUL frente al valor real.
        
        Args:
            df_preds (pd.DataFrame): DataFrame con columnas 'y_true' y 'y_pred'
            component_name (str): Nombre del componente (e.g. 'aceite', 'frenos')
            model_name (str, optional): Nombre del modelo para mensajes de error
            save_path (str, optional): Ruta para guardar el gráfico
        """
        import os
        
        if df_preds is None or df_preds.empty:
            model_info = f" para modelo {model_name}" if model_name else ""
            print(f"[WARNING] No se encontraron datos de predicción para el gráfico RUL de '{component_name}{model_info}'")
            return
            
        if 'y_true' not in df_preds.columns or 'y_pred' not in df_preds.columns:
            print(f"[ERROR] DataFrame de predicciones no contiene las columnas requeridas ('y_true', 'y_pred')")
            return

        plt.figure(figsize=(12, 8))

        # Muestreamos si hay demasiados puntos para una visualización clara
        max_points = 5000
        if len(df_preds) > max_points:
            df_sample = df_preds.sample(n=max_points, random_state=42)
        else:
            df_sample = df_preds

        # Graficamos el RUL real vs el predicho
        plt.scatter(df_sample['y_true'], df_sample['y_pred'], alpha=0.4, s=20, label='Predicciones')
        
        # Línea de predicción perfecta (y=x)
        min_val = min(df_sample['y_true'].min(), df_sample['y_pred'].min())
        max_val = max(df_sample['y_true'].max(), df_sample['y_pred'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, label='Predicción Perfecta')

        plt.title(f'Análisis de Predicción RUL - {component_name.replace("_", " ").title()}', fontsize=16, fontweight='bold')
        plt.xlabel('RUL Real (días)')
        plt.ylabel('RUL Predicho (días)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axis('equal') # Ejes con la misma escala para una correcta interpretación
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=self.viz_config['dpi'])
        
        plt.close()
     
     
    def plot_rul_predictions_timeline(self, df_rul, component_name=None, model_name=None, save_path=None):
        """Visualizar predicciones RUL en línea temporal
        
        Args:
            df_rul (pd.DataFrame): DataFrame con columnas 'RUL_days' y 'timestamp'
            component_name (str, optional): Nombre del componente para mensajes de error
            model_name (str, optional): Nombre del modelo para mensajes de error
            save_path (str, optional): Ruta para guardar el gráfico
        """
        if df_rul is None or df_rul.empty:
            model_info = f" para modelo {model_name}" if model_name else ""
            comp_info = f" de '{component_name}'" if component_name else ""
            print(f"[WARNING] No se encontraron datos de predicción para la línea temporal RUL{comp_info}{model_info}")
            return
            
        if 'RUL_days' not in df_rul.columns or 'timestamp' not in df_rul.columns:
            print("[ERROR] DataFrame de RUL no contiene las columnas requeridas ('RUL_days', 'timestamp')")
            return
        
        # Muestrear datos si son demasiados
        if len(df_rul) > 10000:
            df_sample = df_rul.sample(n=10000).sort_values('timestamp')
        else:
            df_sample = df_rul.sort_values('timestamp')
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        fig.suptitle('Análisis Temporal de RUL', fontsize=16, fontweight='bold')
        
        # 1. Evolución del RUL real
        ax1 = axes[0]
        ax1.plot(df_sample['timestamp'], df_sample['RUL_days'], 
                linewidth=1, alpha=0.8, color='blue', label='RUL Real')
        
        # Destacar períodos críticos (RUL < 5 días)
        critical_mask = df_sample['RUL_days'] < 5
        if np.any(critical_mask):
            ax1.scatter(df_sample.loc[critical_mask, 'timestamp'], 
                       df_sample.loc[critical_mask, 'RUL_days'],
                       color='red', s=10, alpha=0.7, label='RUL Crítico (<5 días)')
        
        ax1.set_ylabel('RUL (días)')
        ax1.set_title('Evolución del RUL Real')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Líneas de referencia
        ax1.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Umbral Crítico')
        ax1.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='Umbral Advertencia')
        
        # 2. Distribución de RUL por período
        ax2 = axes[1]
        
        # Agregar información de semana
        df_sample_temp = df_sample.copy()
        df_sample_temp['semana'] = df_sample_temp['timestamp'].dt.isocalendar().week
        df_sample_temp['semana_relativa'] = df_sample_temp['semana'] - df_sample_temp['semana'].min() + 1
        
        # Boxplot por semana
        weeks = sorted(df_sample_temp['semana_relativa'].unique())
        rul_by_week = [df_sample_temp[df_sample_temp['semana_relativa'] == week]['RUL_days'].values 
                      for week in weeks]
        
        bp = ax2.boxplot(rul_by_week, labels=[f'S{w}' for w in weeks], patch_artist=True)
        
        # Colorear boxplots
        colors = ['lightblue', 'lightgreen', 'orange', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
        
        ax2.set_xlabel('Semana')
        ax2.set_ylabel('RUL (días)')
        ax2.set_title('Distribución de RUL por Semana')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.viz_config['dpi'], bbox_inches='tight')
        else:
            plt.savefig(f'{self.config.RESULTS_DIR}/rul_timeline_analysis.png', 
                       dpi=self.viz_config['dpi'], bbox_inches='tight')
