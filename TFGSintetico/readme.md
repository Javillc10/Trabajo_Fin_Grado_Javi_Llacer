# TFGSintetico - Industrial Synthetic Data Generator

## Overview

TFGSintetico is a specialized synthetic data generator for industrial predictive maintenance, designed to create realistic time-series data with variable Remaining Useful Life (RUL) patterns. The framework produces high-quality datasets that simulate an automotive fluid filling station with multiple components and realistic degradation patterns.

## Key Features

1. **Realistic RUL Generation**
   - Variable RUL range: 0.5-25 days
   - Non-linear degradation patterns
   - Component-specific degradation rates
   - Accumulated stress effects

2. **Component Simulation**
   ```python
   Components:
   - Bomba Aceite (Oil Pump)
   - Filtro Frenos (Brake Filter)
   - Válvula Refrigerante (Coolant Valve)
   ```

3. **Sensor Data Generation**
   - Pressure sensors with realistic noise
   - Temperature variations
   - Operational condition effects
   - Maintenance impact simulation

## Data Characteristics

### 1. Time Series Structure
```sql
- timestamp components (fecha, hora, minuto, segundo, milisegundo)
- sensor readings (presión aceite, frenos, refrigerante)
- system states
- operational hours
- RUL values
```

### 2. System States
- Normal Operation
- Gradual Degradation
- Advanced Degradation
- Critical RUL
- Component-specific Failures

### 3. Data Volume
- 8,640,000 samples total
- 20Hz sampling frequency
- 120 productive hours
- SQLite database storage

## Technical Specifications

1. **Physical Models**
```python
def calculate_realistic_rul_definitivo(
    component_hours,
    stress_factors,
    base_life=2000,
    component_type='bomba_aceite',
    health_factor=1.0
)
```

2. **Sensor Generation**
```python
def generate_realistic_sensor_reading_definitivo(
    base_value,
    time_hours,
    component_health,
    operational_conditions,
    degradation_severity=1.0
)
```

3. **Data Storage**
```sql
Database Schema:
- datos_estacion (main data)
- metadatos_generacion (metadata)
With optimized indexes for efficient querying
```

## Usage

### 1. Basic Generation
```python
from TFGSintetico.Generador_database import EstacionLlenadoDatasetGeneradorDefinitivo

generator = EstacionLlenadoDatasetGeneradorDefinitivo()
db_path, queries_path = generator.generar_dataset_completo()
```

### 2. Custom Configuration
```python
config = {
    'chunk_size': 500000,
    'max_memory_gb': 8,
    'enable_ml_features': True,
    'database_name': 'estacion_custom.db',
    'seed': 42
}
generator = EstacionLlenadoDatasetGeneradorDefinitivo(config)
```

### 3. Data Verification
```python
generator.verificar_base_datos()
```

## Quality Guarantees

1. **RUL Variability**
   - Minimum: 0.5 days
   - Maximum: 25 days
   - Non-linear degradation
   - Component-specific patterns

2. **Sensor Realism**
   - Coefficient of Variation: 0.02-0.08
   - Realistic noise patterns
   - Operation-dependent variations
   - Maintenance effect modeling

3. **Physical Consistency**
   - Stress accumulation
   - Health degradation
   - Maintenance impacts
   - Environmental effects

## Memory Management

1. **Batch Processing**
   - Configurable chunk sizes
   - Memory monitoring
   - Dynamic batch adjustment
   - Efficient SQLite integration

2. **Performance Optimization**
   ```python
   Processing Modes:
   - development: Fast iterations (≤200K samples)
   - balanced: Speed-accuracy trade-off
   - optimized: High-speed for large datasets
   - complete: Maximum accuracy with chunking
   ```

## Integration Examples

### 1. Reading Generated Data
```python
import sqlite3
import pandas as pd

def load_data(db_path, limit=None):
    query = """
    SELECT *
    FROM datos_estacion
    ORDER BY timestamp_unix
    """
    if limit:
        query += f" LIMIT {limit}"
        
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df
```

### 2. Analyzing RUL Patterns
```python
def analyze_rul_patterns(df):
    rul_stats = {
        'aceite': df['rul_dias_aceite'].describe(),
        'frenos': df['rul_dias_frenos'].describe(),
        'refrigerante': df['rul_dias_refrigerante'].describe()
    }
    return rul_stats
```

### 3. Validating Data Quality
```python
def validate_data_quality(df):
    # Sensor variation coefficients
    cv_sensors = {
        col: df[col].std() / df[col].mean()
        for col in ['sensor_presion_aceite_bar', 
                   'sensor_presion_frenos_bar',
                   'sensor_presion_refrigerante_bar']
    }
    return cv_sensors
```

## Best Practices

1. **Data Generation**
   - Use provided configuration presets
   - Monitor system resources
   - Validate outputs regularly
   - Enable checkpoints for large generations

2. **Data Usage**
   - Use batch processing for large datasets
   - Implement proper indexing
   - Monitor query performance
   - Validate physical consistency

3. **Integration**
   - Use provided SQL queries
   - Implement proper error handling
   - Monitor memory usage
   - Validate data consistency

## Documentation Structure

For AI understanding, focus on:

1. **Generation Logic**
   - Component degradation models
   - Sensor data generation
   - RUL calculation
   - State determination

2. **Data Structure**
   - Time series format
   - Feature relationships
   - Physical constraints
   - Quality metrics

3. **Integration Patterns**
   - Data loading methods
   - Analysis techniques
   - Validation approaches
   - Performance optimization
