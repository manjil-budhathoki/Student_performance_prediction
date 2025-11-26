# Anuj's Tasks - Quick Start

## Overview
Anuj's implementation adds hyperparameter tuning, explainability, and error analysis to the Student Performance Prediction project.

## Quick Run (All Tasks)

### Option 1: PowerShell Script (Recommended)
```powershell
.\run_member_b.ps1
```

### Option 2: Manual Execution
```powershell
# 1. Hyperparameter Tuning (~5-15 minutes)
python src/optimization/tune_model.py

# 2. Explainability Analysis (~2-5 minutes)
python src/optimization/explainability.py

# 3. Error Analysis (<1 minute)
python src/optimization/error_analysis.py
```

## Prerequisites

1. **Install Dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

2. **Required Input:**
   - `data/processed/student_performance_cleaned.csv` (created by Manjil)

## Output Files

After running all scripts, you'll have:

```
models/
├── tuned_model.pkl                     # Best performing model

reports/
├── tuned_results.json                  # Performance metrics
├── explainability/
│   ├── shap_summary_plot.png          # Feature importance visualization
│   ├── shap_force_plot.png            # Individual prediction explanation
│   ├── partial_dependence_plots.png   # Feature effect curves
│   └── feature_importance.png         # Feature ranking
└── error_analysis/
    └── error_report.md                # Comprehensive error analysis
```

## Scripts Description

### 1. `tune_model.py`
- Tunes RandomForestRegressor and GradientBoostingRegressor
- Uses GridSearchCV with 5-fold cross-validation
- Evaluates: R², MSE, RMSE, MAE
- Selects and saves best model

### 2. `explainability.py`
- Generates SHAP plots for model interpretation
- Creates partial dependence plots
- Shows feature importance rankings
- Explains prediction patterns

### 3. `error_analysis.py`
- Analyzes worst predictions
- Calculates residual statistics
- Provides improvement recommendations
- Generates detailed markdown report

## Key Features

✅ **Consistent with Manjil:** Same train/test split (test_size=0.2, random_state=42)  
✅ **Modular:** Each script runs independently  
✅ **Professional:** Comprehensive logging and error handling  
✅ **Reproducible:** Fixed random seeds throughout  

## Troubleshooting

**Issue:** `FileNotFoundError: tuned_model.pkl`  
**Fix:** Run `tune_model.py` first

**Issue:** `ModuleNotFoundError: shap`  
**Fix:** `pip install shap`

**Issue:** GridSearchCV too slow  
**Fix:** Reduce param_grid size in `tune_model.py` lines 56-62 and 89-97

## Performance Expectations

- **Model Improvement:** R² typically improves by 0.05-0.15 vs baseline
- **Best Model:** Usually GradientBoosting or RandomForest
- **Runtime:** Total ~10-20 minutes on typical systems

---

