import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- CONFIGURACIÓN ---
# La ruta a tu dataset "dorado" de 1 millón de filas
INPUT_DATA_PATH = r'C:\Users\Pilar\Desktop\TFGFinal\TFGSintetico\dataset_rul_variable_ENRIQUECIDO\train_data.parquet'

# Dónde guardar los nuevos datasets de entrenamiento y test
OUTPUT_DIR = 'final_split_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(OUTPUT_DIR, 'final_train.parquet')
TEST_PATH = os.path.join(OUTPUT_DIR, 'final_test.parquet')

# --- SCRIPT ---
print(f"🚀 Iniciando el split del dataset final...")
print(f"Cargando datos desde: {INPUT_DATA_PATH}")

df = pd.read_parquet(INPUT_DATA_PATH)
print(f"✅ Datos cargados. Shape: {df.shape}")

# Agrupamos los estados si no existe 'estado_agrupado'
if 'estado_agrupado' not in df.columns:
    print("   - Creando 'estado_agrupado' desde 'estado_sistema'...")
    def map_state(estado):
        if pd.isna(estado): return estado
        estado_str = str(estado)
        if 'Normal' in estado_str: return 'Normal'
        if any(crit in estado_str for crit in ['Inminente', 'Critico', 'Severa', 'Falla', 'Error', 'Perdida']): return 'Critico'
        return 'Advertencia'
    df['estado_agrupado'] = df['estado_sistema'].apply(map_state)

print(f"\n📊 Distribución de clases en el dataset completo:")
print(df['estado_agrupado'].value_counts(normalize=True) * 100)

# El paso más importante: dividir los datos
# Usamos 'stratify' para asegurar que ambos splits tienen la misma distribución de clases
train_df, test_df = train_test_split(
    df,
    test_size=0.3,  # 30% para test, 70% para train
    random_state=42,
    stratify=df['estado_agrupado']
)

print(f"\n✂️  Split realizado:")
print(f"   - Tamaño del Train set: {train_df.shape}")
print(f"   - Tamaño del Test set:  {test_df.shape}")

print(f"\n📊 Distribución de clases en el NUEVO train set:")
print(train_df['estado_agrupado'].value_counts(normalize=True) * 100)

print(f"\n📊 Distribución de clases en el NUEVO test set:")
print(test_df['estado_agrupado'].value_counts(normalize=True) * 100)

# Guardar los nuevos ficheros
print(f"\n💾 Guardando los nuevos datasets...")
train_df.to_parquet(TRAIN_PATH, index=False)
test_df.to_parquet(TEST_PATH, index=False)

print(f"✅ Proceso completado. Nuevos datasets en la carpeta: '{OUTPUT_DIR}'")
