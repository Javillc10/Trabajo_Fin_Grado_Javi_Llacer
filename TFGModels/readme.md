# TFGModels - Industrial Predictive Maintenance Framework

## Project Overview

TFGModels is a comprehensive industrial predictive maintenance framework that combines:
- Multi-class fault classification
- Remaining Useful Life (RUL) estimation
- Model interpretability
- Business impact analysis

## Core Components

### 1. Main Pipeline (main.py)

The central orchestration script that coordinates:
- Configuration validation
- Data loading & preprocessing
- Model training & evaluation
- Results generation
- Resource management

Key phases:
```python
Phase 0: Configuration validation
Phase 1: Data loading & EDA
Phase 2: Feature engineering
Phase 3: Temporal validation
Phase 4: Fault classification
Phase 5: RUL estimation
Phase 6: Model interpretation
Phase 7: Visualization
Phase 8: Final report generation
```

### 2. Model Training (src/)

#### EDA and Preprocessing (_01_EDA_and_Preprocessing.py)
- Data quality assessment
- Feature engineering pipeline
- Data validation strategies

#### Fault Classification (_02_Fault_Classification_Models.py)
- Multi-class fault detection
- Class imbalance handling with SMOTE
- Baseline & advanced models (LogReg, RF, XGBoost)
- Memory-efficient batch processing
- Critical class weighting

#### RUL Estimation (_03_RUL_Estimation_Models.py)
- Remaining useful life prediction
- Time-series features
- Prognostics modeling

#### Model Interpretation (_04_Model_Interpretation_and_Business_Impact.py)
- Feature importance analysis
- Business impact metrics
- Cost-benefit analysis

### 3. Utilities (src/utils/)

#### Feature Engineering (feature_engineering.py)
- Time-based feature creation
- Domain-specific transformations
- Feature selection

#### Industrial Metrics (metrics_industriales.py)
- Custom evaluation metrics
- Critical class handling
- Cost-weighted metrics

#### Model Persistence (model_persistence.py)
- Model saving/loading
- Metadata management
- Session tracking

#### Validation Strategies (validation_strategies.py)
- Temporal validation
- Anti-leakage mechanisms
- Drift detection

## Data Requirements

The framework expects data in one of two formats:

1. SQLite Database:
```sql
- estado_sistema (target)
- sensor_presion_aceite_bar
- timestamp/fecha/hora components
Additional sensor readings
```

2. CSV Format:
```
timestamp, estado_sistema, sensor1, sensor2, ...
```

## Code Example

```python
# Training pipeline
python main.py --mode train --save-models

# Inference
python main.py --mode infer --model-session SESSION_ID --data-path new_data.csv
```

## Output Structure

```
results/
├── session_{timestamp}/
│   ├── metrics/
│   │   ├── classification_metrics.json
│   │   └── rul_metrics.json
│   ├── visualizations/
│   │   ├── sensor_evolution.png
│   │   └── model_comparison.png
│   ├── comprehensive_report.md
│   └── final_results.json
```

## Key Features

1. **Robust Data Processing**
   - Memory-efficient batch processing
   - Automated data quality checks
   - Missing value handling

2. **Advanced Modeling**
   - Class imbalance management
   - Critical state detection
   - Temporal validation

3. **Industrial Focus**
   - Domain-specific metrics
   - Cost-sensitive evaluation
   - Business impact analysis

4. **Production Ready**
   - Model persistence
   - Session management
   - Resource optimization

## Requirements

See requirements.txt for detailed dependencies. Key libraries:
- pandas>=1.5.0
- scikit-learn>=1.1.0
- xgboost>=1.6.0
- imbalanced-learn>=0.9.0

## Best Practices

1. **Data Preparation**
   - Use SQLite for large datasets
   - Include all relevant sensor data
   - Maintain temporal order

2. **Model Training**
   - Enable save_models for persistence
   - Monitor memory usage
   - Review temporal validation

3. **Inference**
   - Provide model_session ID
   - Match feature columns
   - Check resource availability

## Documentation Structure

## Configuration Guide

### Core Settings (config.py)

1. **Project Configuration**
   - Version: 1.0.0
   - Random Seed: 42
   - Development Mode: Configurable

2. **Processing Modes**
   - Development: Fast iterations (<=200K samples)
   - Balanced: Speed-accuracy trade-off
   - Optimized: High-speed for large datasets
   - Complete: Maximum accuracy with chunking

3. **Data Processing**
   ```python
   SENSOR_CONFIG = {
       'sampling_frequency_hz': 20,
       'temporal_windows_hours': [1, 6, 24],
       'disable_spectral_analysis': False
   }
   ```

4. **Validation Strategy**
   ```python
   VALIDATION_CONFIG = {
       'gap_hours': 12,
       'n_splits': 5,
       'test_size': 0.2,
       'shuffle': False
   }
   ```

5. **Model Configuration**
   - Random Forest: 200 estimators, max_depth 15
   - XGBoost: 200 estimators, learning rate 0.1
   - Support for both classification and regression

6. **Business Metrics**
   ```python
   BUSINESS_CONFIG = {
       'false_positive_cost': 500,
       'false_negative_cost': 15000,
       'maintenance_cost': 2000,
       'downtime_cost_per_hour': 5000
   }
   ```

7. **Performance Targets**
   ```python
   METRICS_CONFIG = {
       'classification': {
           'target_f1_macro': 0.70,
           'target_precision_critical': 0.75
       },
       'regression': {
           'target_mae_days': 3.5,
           'target_r2': 0.60
       }
   }
   ```

## For AI Understanding

Focus on these key aspects:

1. **Core Logic**
   - main.py: Pipeline orchestration
   - config.py: Extensive configuration system
   - inference.py: Production deployment

2. **Model Development**
   - Classification with class imbalance handling
   - RUL estimation with temporal features
   - Memory-efficient feature engineering

3. **Evaluation System**
   - Industry-specific metrics
   - Cost-sensitive business impact
   - Robust temporal validation
