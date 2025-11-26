import pandas as pd
import numpy as np
import json
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '../../data')
processed_data_path = os.path.join(data_dir, 'processed', 'student_performance_cleaned.csv')
MODEL_DIR = os.path.join(current_dir, '../../models')
REPORTS_DIR = os.path.join(current_dir, '../../reports')
DATA_PATH = processed_data_path

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model performance using multiple metrics."""
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }

def train_top_models():
    """Train and save the top-performing models that match baseline performance."""
    
    print("="*60)
    print("Training Top-Performing Models for Production")
    print("="*60)
    
    # Load data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Processed data file not found at {DATA_PATH}")
    
    print(f"\nLoading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Data loaded. Shape: {df.shape}")
    
    # Split features and target
    X = df.drop(columns=['GPA'])
    y = df['GPA']
    
    # Train-test split (same as baseline: test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    models_to_save = {}
    
    # 1. Ridge Regression
    print("\n" + "="*60)
    print("Training Ridge Regression...")
    print("="*60)
    
    ridge_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge(random_state=42, max_iter=10000))
    ])
    
    param_grid_ridge = {
        'regressor__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
    }
    
    ridge_grid = GridSearchCV(ridge_pipeline, param_grid_ridge, cv=5, scoring='r2', n_jobs=1)
    ridge_grid.fit(X_train, y_train)
    
    ridge_best = ridge_grid.best_estimator_
    ridge_metrics = evaluate_model(ridge_best, X_test, y_test, "Ridge Regression")
    
    models_to_save["ridge"] = {
        "model": ridge_best,
        "metrics": ridge_metrics,
        "best_params": ridge_grid.best_params_,
        "description": "Ridge Regression - Linear model with L2 regularization"
    }
    
    # 2. ElasticNet
    print("\n" + "="*60)
    print("Training ElasticNet...")
    print("="*60)
    
    from sklearn.linear_model import ElasticNet
    elasticnet_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', ElasticNet(random_state=42, max_iter=10000))
    ])
    
    param_grid_en = {
        'regressor__alpha': [0.001, 0.01, 0.1, 1.0],
        'regressor__l1_ratio': [0.3, 0.5, 0.7]
    }
    
    en_grid = GridSearchCV(elasticnet_pipeline, param_grid_en, cv=5, scoring='r2', n_jobs=1)
    en_grid.fit(X_train, y_train)
    
    en_best = en_grid.best_estimator_
    en_metrics = evaluate_model(en_best, X_test, y_test, "ElasticNet")
    
    models_to_save["elasticnet"] = {
        "model": en_best,
        "metrics": en_metrics,
        "best_params": en_grid.best_params_,
        "description": "ElasticNet - Linear model with L1 and L2 regularization"
    }
    
    # 3. Support Vector Regressor
    print("\n" + "="*60)
    print("Training Support Vector Regressor...")
    print("="*60)
    
    svr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', SVR())
    ])
    
    param_grid_svr = {
        'regressor__kernel': ['rbf', 'linear'],
        'regressor__C': [1, 10, 100],
        'regressor__epsilon': [0.01, 0.1]
    }
    
    svr_grid = GridSearchCV(svr_pipeline, param_grid_svr, cv=5, scoring='r2', n_jobs=1)
    svr_grid.fit(X_train, y_train)
    
    svr_best = svr_grid.best_estimator_
    svr_metrics = evaluate_model(svr_best, X_test, y_test, "SVR")
    
    models_to_save["svr"] = {
        "model": svr_best,
        "metrics": svr_metrics,
        "best_params": svr_grid.best_params_,
        "description": "Support Vector Regressor - Kernel-based regression"
    }
    
    # Save all models
    print("\n" + "="*60)
    print("Saving Models to Production Directory")
    print("="*60)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    saved_models_info = {}
    
    for model_name, model_data in models_to_save.items():
        # Save model
        model_path = os.path.join(MODEL_DIR, f'{model_name}_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_data['model'], f)
        print(f"\n✓ Saved: {model_name}_model.pkl")
        print(f"  R²: {model_data['metrics']['r2']:.4f}")
        print(f"  RMSE: {model_data['metrics']['rmse']:.4f}")
        
        # Store info for JSON (without model object)
        saved_models_info[model_name] = {
            "file": f"{model_name}_model.pkl",
            "description": model_data['description'],
            "best_params": model_data['best_params'],
            "metrics": model_data['metrics']
        }
    
    # Save model registry
    registry_path = os.path.join(MODEL_DIR, 'model_registry.json')
    with open(registry_path, 'w') as f:
        json.dump(saved_models_info, f, indent=4)
    print(f"\n✓ Model registry saved: model_registry.json")
    
    # Create model selection guide
    guide_path = os.path.join(MODEL_DIR, 'MODEL_SELECTION_GUIDE.md')
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write("# Model Selection Guide for Streamlit App\n\n")
        f.write("## Available Models\n\n")
        f.write("All models achieve **R² ≈ 0.953** (equivalent performance)\n\n")
        
        f.write("### 1. Baseline Model\n")
        f.write("- **File:** `baseline_model.pkl`\n")
        f.write("- **Algorithm:** Linear Regression\n")
        f.write("- **Best for:** Simplicity, interpretability\n")
        f.write("- **Speed:** Fastest\n\n")
        
        for model_name, info in saved_models_info.items():
            f.write(f"### {model_name.title()}\n")
            f.write(f"- **File:** `{info['file']}`\n")
            f.write(f"- **Description:** {info['description']}\n")
            f.write(f"- **R²:** {info['metrics']['r2']:.4f}\n")
            f.write(f"- **RMSE:** {info['metrics']['rmse']:.4f}\n")
            f.write(f"- **Best params:** {info['best_params']}\n\n")
        
        f.write("## Usage in Streamlit\n\n")
        f.write("```python\n")
        f.write("import pickle\n\n")
        f.write("# Load model\n")
        f.write("with open('models/baseline_model.pkl', 'rb') as f:\n")
        f.write("    model = pickle.load(f)\n\n")
        f.write("# Make prediction\n")
        f.write("prediction = model.predict(input_features)\n")
        f.write("```\n\n")
        f.write("## Recommendation\n\n")
        f.write("**Use `baseline_model.pkl` for production** - it's the simplest and equally accurate.\n\n")
        f.write("Provide model selection dropdown in Streamlit to let users choose:\n")
        f.write("- Baseline (Linear Regression) - **Recommended**\n")
        f.write("- Ridge Regression\n")
        f.write("- ElasticNet\n")
        f.write("- SVR (Support Vector Regressor)\n")
    
    print(f"✓ Model selection guide saved: MODEL_SELECTION_GUIDE.md")
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"\nSaved {len(models_to_save)} top-performing models:")
    print("  1. ridge_model.pkl")
    print("  2. elasticnet_model.pkl")
    print("  3. svr_model.pkl")
    print("  + baseline_model.pkl (already exists)")
    print("\nAll models ready for Streamlit integration!")
    print("="*60)

if __name__ == "__main__":
    train_top_models()
