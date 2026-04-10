# 📡 NavIC / GNSS ML Error Correction System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Enabled-brightgreen)
![Status](https://img.shields.io/badge/Status-Complete-success)
![License](https://img.shields.io/badge/License-MIT-yellow)
![NavIC](https://img.shields.io/badge/NavIC-PNT-orange)
![GNSS](https://img.shields.io/badge/GNSS-Positioning-purple)

---

## Overview

This project is an **end-to-end machine learning system for NavIC / GNSS error correction**. It addresses positioning inaccuracy caused by signal degradation, multipath, poor satellite geometry, and device orientation effects in real-world GNSS environments.

The pipeline converts raw GNSS logs into structured features, trains multiple regression models, evaluates them quantitatively, and visualizes corrected versus raw positioning through an interactive Streamlit dashboard.

---

## Problem Statement

GNSS/NavIC positioning in real-world environments often suffers from:

- Non-Line-of-Sight (NLOS) signal loss
- Multipath propagation
- Poor satellite geometry
- Orientation-dependent signal variation
- Noise in SNR, elevation, and azimuth measurements

These issues cause inaccurate position estimates, which are especially problematic in:

- Urban navigation
- Logistics and last-mile delivery
- Surveying and infrastructure monitoring
- Smart mobility and transport analytics
- Precision positioning experiments

This project learns GNSS error patterns from signal-quality and orientation features and uses ML models to predict and correct the position error.

---

## Solution Summary

```text
Raw GNSS Logs
    ↓
Parsing
    ↓
Feature Engineering
    ↓
Label Creation
    ↓
Model Training
    ↓
Error Prediction
    ↓
Position Correction
    ↓
Interactive Visualization
```

The system predicts the error in **X** and **Y** directions and uses those predictions to correct the raw GNSS position.

---

## Key Features

### 1. Raw GNSS Log Processing
- Parses Android GNSS logger files
- Extracts meaningful position and signal fields
- Separates data by:
  - No inclination
  - Inclination-based scenarios

### 2. Feature Engineering
- Mean SNR
- Satellite count
- Mean elevation
- Mean azimuth
- Roll, pitch, yaw
- Inclination angle
- Position jump

### 3. Supervised Learning Dataset
- Generates ML-ready labeled data
- Computes:
  - `x`, `y`
  - `gt_x`, `gt_y`
  - `error_x`, `error_y`
  - `original_error`

### 4. Multi-Model Training
- Random Forest
- XGBoost
- SVM
- LSTM
- TCN

### 5. Evaluation and Comparison
- RMSE
- MAE
- R²
- Accuracy
- Precision
- Recall
- Confusion matrix
- Error distribution comparison

### 6. Interactive Dashboard
- Model selection
- Dynamic model details
- Performance tables
- Bar charts
- Confusion matrices
- Error distribution plots
- Raw vs corrected trajectory
- Dynamic insights

---

## Dataset Organization

```text
dataset/
├── raw/
│   ├── data with no inclination/
│   └── data with inclination/
│
├── processed/
│   ├── parsed/
│   ├── merged.csv
│   ├── merged_with_angle.csv
│   ├── cleaned_dataset.csv
│   ├── final_features.csv
│   ├── ml_dataset.csv
│   ├── corrected_dataset.csv
│   └── model_comparison_results.csv
│
└── ground_truth/
    └── ground_truth.txt
```

---

## Model Comparison

The project compares five models:

| Model | Type | Purpose |
|------|------|---------|
| Random Forest | Ensemble Regression | Strong baseline for tabular GNSS features |
| XGBoost | Boosted Trees | High-performance tabular regression |
| SVM | Kernel Regression | Classical non-linear benchmark |
| LSTM | Deep Learning | Sequence-based temporal learning |
| TCN | Deep Learning | Temporal convolutional modeling |

### Observed Behavior
- Tree-based models perform best on this dataset
- Random Forest and XGBoost are the strongest models
- LSTM and TCN underperform because the dataset behaves more like structured tabular data than a strongly sequential time series

---

## Evaluation Metrics

The system uses both regression and classification-style evaluation.

### Regression Metrics
- RMSE_X
- RMSE_Y
- MAE_X
- MAE_Y
- R²_X
- R²_Y

### Classification Metrics
For practical usability analysis, a threshold-based error classification is also displayed:
- Accuracy
- Precision
- Recall
- Confusion Matrix

---

## Dashboard Features

The Streamlit dashboard provides:

- Dynamic model selection
- Model-specific algorithm explanation
- Premium tabular performance view
- RMSE / MAE / R² comparison charts
- Confusion matrix for classical regression models
- Raw vs corrected trajectory comparison
- Error distribution before and after correction
- Dynamic model insights in bullet format
- Sample data preview

---

## Workflow

### 1. Parse Raw Logs
```bash
python dataset_parser.py
```

### 2. Merge Parsed Files
```bash
python merge_datasets.py
```

### 3. Add Inclination Feature
```bash
python add_angle_feature.py
```

### 4. Add Realistic Noise
```bash
python add_noise.py
```

### 5. Clean and Sort Dataset
```bash
python clean_dataset.py
```

### 6. Add Position Jump
```bash
python add_position_jump.py
```

### 7. Create Labels
```bash
python create_labels.py
```

### 8. Train Models
```bash
python train_rf.py
python train_xgb.py
python train_svm.py
python train_lstm.py
python train_tcn.py
```

### 9. Evaluate All Models
```bash
python evaluate_all_models.py
```

### 10. Launch Dashboard
```bash
streamlit run app1.py
```

---

## Final Output Files

The pipeline generates the following key outputs:

- `parsed/*.csv`
- `merged.csv`
- `merged_with_angle.csv`
- `merged_dataset_noisy.csv`
- `cleaned_dataset.csv`
- `final_features.csv`
- `ml_dataset.csv`
- `corrected_dataset.csv`
- `model_comparison_results.csv`

---

## Project Architecture

```text
Raw GNSS TXT Logs
        ↓
Parser
        ↓
Merged CSV
        ↓
Angle + Noise + Cleaning
        ↓
Feature Engineering
        ↓
Label Creation
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Streamlit Dashboard
```

---

## Technical Stack

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **XGBoost**
- **TensorFlow / Keras**
- **Streamlit**
- **Plotly**
- **Matplotlib**
- **Seaborn**
- **Joblib**

---

## Results Interpretation

The final evaluation shows that:

- Tree-based models are more effective for this GNSS dataset
- Random Forest produces the strongest correction performance
- XGBoost remains highly competitive
- Sequence models are weaker because the dataset lacks strong temporal continuity
- ML-based correction improves positioning estimates visually and numerically

---

## Repository Structure

```text
navic-ml-error-correction/
├── dataset/
├── models/
├── app1.py
├── dataset_parser.py
├── merge_datasets.py
├── add_angle_feature.py
├── add_noise.py
├── clean_dataset.py
├── add_position_jump.py
├── create_labels.py
├── train_rf.py
├── train_xgb.py
├── train_svm.py
├── train_lstm.py
├── train_tcn.py
├── evaluate_all_models.py
├── apply_correction.py
├── model_comparison_results.csv
└── README.md
```

---

## How the Correction Works

For each selected model:

1. The model predicts `error_x` and `error_y`
2. The predicted error is subtracted from the raw position
3. The corrected position is obtained
4. The error reduction is visualized in the dashboard

```text
Corrected X = Raw X - Predicted Error X
Corrected Y = Raw Y - Predicted Error Y
```

---

## Project Highlights

- End-to-end ML pipeline
- NavIC-aware feature engineering
- Multi-model benchmarking
- Professional dashboard visualization
- Dynamic insights per selected model
- Real-world GNSS correction logic

---

## Author

**Hema Sagar Koppusetti**

---

## License

This project is licensed under the **MIT License**.
