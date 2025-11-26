import pandas as pd
import numpy as np
import json
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Try to import XGBoost and LightGBM (optional)
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not installed. Install with: pip install lightgbm")

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '../../data')
processed_data_path = os.path.join(data_dir, 'processed', 'student_performance_cleaned.csv')
MODEL_DIR = os.path.join(current_dir, '../../models')
REPORTS_DIR = os.path.join(current_dir, '../../reports')
DATA_PATH = processed_data_path

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model performance using multiple metrics.
    
    Args:
        model: Trained model pipeline
        X_test: Test features
        y_test: Test target
        model_name: Name of the model for display
        
    Returns:
        dict: Dictionary containing all metrics
    """
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

def tune_random_forest(X_train, y_train, X_test, y_test):
    """
    Tune RandomForestRegressor using GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        
    Returns:
        tuple: (best_pipeline, metrics_dict, best_params)
    """
    print("\n" + "="*60)
    print("Tuning RandomForestRegressor...")
    print("="*60)
    
    # Define pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    
    # Define hyperparameter grid (optimized for performance)
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [10, 20, None],
        'regressor__min_samples_split': [2, 5],
        'regressor__min_samples_leaf': [1, 2],
        'regressor__max_features': ['sqrt', 'log2']
    }
    
    # GridSearchCV (n_jobs=1 to avoid Windows multiprocessing issues)
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=1,
        verbose=1
    )
    
    print("Starting grid search (this may take a few minutes)...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV R² score: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    best_pipeline = grid_search.best_estimator_
    metrics = evaluate_model(best_pipeline, X_test, y_test, "RandomForest (Tuned)")
    
    return best_pipeline, metrics, grid_search.best_params_

def tune_gradient_boosting(X_train, y_train, X_test, y_test):
    """
    Tune GradientBoostingRegressor using GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        
    Returns:
        tuple: (best_pipeline, metrics_dict, best_params)
    """
    print("\n" + "="*60)
    print("Tuning GradientBoostingRegressor...")
    print("="*60)
    
    # Define pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ])
    
    # Define hyperparameter grid (optimized for performance)
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__learning_rate': [0.05, 0.1],
        'regressor__max_depth': [3, 5],
        'regressor__min_samples_split': [2, 5],
        'regressor__min_samples_leaf': [1, 2],
        'regressor__subsample': [0.8, 1.0]
    }
    
    # GridSearchCV (n_jobs=1 to avoid Windows multiprocessing issues)
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=1,
        verbose=1
    )
    
    print("Starting grid search (this may take a few minutes)...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV R² score: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    best_pipeline = grid_search.best_estimator_
    metrics = evaluate_model(best_pipeline, X_test, y_test, "GradientBoosting (Tuned)")
    
    return best_pipeline, metrics, grid_search.best_params_

def tune_xgboost(X_train, y_train, X_test, y_test):
    """
    Tune XGBRegressor using GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        
    Returns:
        tuple: (best_pipeline, metrics_dict, best_params)
    """
    print("\n" + "="*60)
    print("Tuning XGBRegressor...")
    print("="*60)
    
    # Define pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', XGBRegressor(random_state=42, verbosity=0))
    ])
    
    # Define hyperparameter grid (optimized for performance)
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__learning_rate': [0.05, 0.1],
        'regressor__max_depth': [3, 5],
        'regressor__min_child_weight': [1, 3],
        'regressor__subsample': [0.8, 1.0],
        'regressor__colsample_bytree': [0.8, 1.0]
    }
    
    # GridSearchCV (n_jobs=1 to avoid Windows multiprocessing issues)
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=1,
        verbose=1
    )
    
    print("Starting grid search (this may take a few minutes)...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV R² score: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    best_pipeline = grid_search.best_estimator_
    metrics = evaluate_model(best_pipeline, X_test, y_test, "XGBoost (Tuned)")
    
    return best_pipeline, metrics, grid_search.best_params_

def tune_lightgbm(X_train, y_train, X_test, y_test):
    """
    Tune LGBMRegressor using GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        
    Returns:
        tuple: (best_pipeline, metrics_dict, best_params)
    """
    print("\n" + "="*60)
    print("Tuning LGBMRegressor...")
    print("="*60)
    
    # Define pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LGBMRegressor(random_state=42, verbosity=-1))
    ])
    
    # Define hyperparameter grid (optimized for performance)
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__learning_rate': [0.05, 0.1],
        'regressor__max_depth': [5, -1],
        'regressor__num_leaves': [31, 50],
        'regressor__min_child_samples': [20, 30],
        'regressor__subsample': [0.8, 1.0]
    }
    
    # GridSearchCV (n_jobs=1 to avoid Windows multiprocessing issues)
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=1,
        verbose=1
    )
    
    print("Starting grid search (this may take a few minutes)...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV R² score: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    best_pipeline = grid_search.best_estimator_
    metrics = evaluate_model(best_pipeline, X_test, y_test, "LightGBM (Tuned)")
    
    return best_pipeline, metrics, grid_search.best_params_

def tune_svr(X_train, y_train, X_test, y_test):
    """
    Tune Support Vector Regressor using GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        
    Returns:
        tuple: (best_pipeline, metrics_dict, best_params)
    """
    print("\n" + "="*60)
    print("Tuning Support Vector Regressor...")
    print("="*60)
    
    # Define pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', SVR())
    ])
    
    # Define hyperparameter grid (optimized for performance)
    param_grid = {
        'regressor__kernel': ['rbf', 'linear'],
        'regressor__C': [1, 10, 100],
        'regressor__epsilon': [0.01, 0.1],
        'regressor__gamma': ['scale']
    }
    
    # GridSearchCV (n_jobs=1 to avoid Windows multiprocessing issues)
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=1,
        verbose=1
    )
    
    print("Starting grid search (this may take a few minutes)...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV R² score: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    best_pipeline = grid_search.best_estimator_
    metrics = evaluate_model(best_pipeline, X_test, y_test, "SVR (Tuned)")
    
    return best_pipeline, metrics, grid_search.best_params_

def tune_elasticnet(X_train, y_train, X_test, y_test):
    """
    Tune ElasticNet using GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        
    Returns:
        tuple: (best_pipeline, metrics_dict, best_params)
    """
    print("\n" + "="*60)
    print("Tuning ElasticNet...")
    print("="*60)
    
    # Define pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', ElasticNet(random_state=42, max_iter=10000))
    ])
    
    # Define hyperparameter grid (optimized for performance)
    param_grid = {
        'regressor__alpha': [0.001, 0.01, 0.1, 1.0],
        'regressor__l1_ratio': [0.3, 0.5, 0.7],
        'regressor__selection': ['cyclic']
    }
    
    # GridSearchCV (n_jobs=1 to avoid Windows multiprocessing issues)
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=1,
        verbose=1
    )
    
    print("Starting grid search (this may take a few minutes)...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV R² score: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    best_pipeline = grid_search.best_estimator_
    metrics = evaluate_model(best_pipeline, X_test, y_test, "ElasticNet (Tuned)")
    
    return best_pipeline, metrics, grid_search.best_params_

def tune_ridge(X_train, y_train, X_test, y_test):
    """
    Tune Ridge Regression using GridSearchCV.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        
    Returns:
        tuple: (best_pipeline, metrics_dict, best_params)
    """
    print("\n" + "="*60)
    print("Tuning Ridge Regression...")
    print("="*60)
    
    # Define pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge(random_state=42, max_iter=10000))
    ])
    
    # Define hyperparameter grid (optimized for performance)
    param_grid = {
        'regressor__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
        'regressor__solver': ['auto', 'cholesky']
    }
    
    # GridSearchCV (n_jobs=1 to avoid Windows multiprocessing issues)
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=1,
        verbose=1
    )
    
    print("Starting grid search (this may take a few minutes)...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV R² score: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    best_pipeline = grid_search.best_estimator_
    metrics = evaluate_model(best_pipeline, X_test, y_test, "Ridge (Tuned)")
    
    return best_pipeline, metrics, grid_search.best_params_

def save_results(models_dict, report_path, model_path, best_model_name):
    """
    Save the best model and performance metrics.
    
    Args:
        models_dict: Dictionary containing model results
        report_path: Path to save JSON report
        model_path: Path to save the best model
        best_model_name: Name of the best performing model
    """
    # Create copy without model objects for JSON serialization
    json_results = {}
    for model_name, model_data in models_dict.items():
        json_results[model_name] = {
            "description": model_data["description"],
            "best_params": model_data["best_params"],
            "metrics": model_data["metrics"]
        }
    
    # Save performance report
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(json_results, f, indent=4)
    print(f"\nPerformance report saved to {report_path}")
    
    # Save the best model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    best_model = models_dict[best_model_name]['model']
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Best model ({best_model_name}) saved to {model_path}")

def train_tuned_models():
    """
    Main function to train and tune regression models.
    """
    print("="*60)
    print("Starting Hyperparameter Tuning for Regression Models")
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
    
    print(f"Features: {X.columns.tolist()}")
    print(f"Target: GPA")
    
    # Train-test split (same as baseline: test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Store all results
    results = {}
    
    # Tune RandomForestRegressor
    rf_model, rf_metrics, rf_params = tune_random_forest(
        X_train, y_train, X_test, y_test
    )
    results["RandomForest"] = {
        "description": "Tuned RandomForestRegressor with GridSearchCV",
        "best_params": rf_params,
        "metrics": rf_metrics,
        "model": rf_model
    }
    
    # Tune GradientBoostingRegressor
    gb_model, gb_metrics, gb_params = tune_gradient_boosting(
        X_train, y_train, X_test, y_test
    )
    results["GradientBoosting"] = {
        "description": "Tuned GradientBoostingRegressor with GridSearchCV",
        "best_params": gb_params,
        "metrics": gb_metrics,
        "model": gb_model
    }
    
    # Tune XGBoost (if available)
    if XGBOOST_AVAILABLE:
        xgb_model, xgb_metrics, xgb_params = tune_xgboost(
            X_train, y_train, X_test, y_test
        )
        results["XGBoost"] = {
            "description": "Tuned XGBRegressor with GridSearchCV",
            "best_params": xgb_params,
            "metrics": xgb_metrics,
            "model": xgb_model
        }
    
    # Tune LightGBM (if available)
    if LIGHTGBM_AVAILABLE:
        lgbm_model, lgbm_metrics, lgbm_params = tune_lightgbm(
            X_train, y_train, X_test, y_test
        )
        results["LightGBM"] = {
            "description": "Tuned LGBMRegressor with GridSearchCV",
            "best_params": lgbm_params,
            "metrics": lgbm_metrics,
            "model": lgbm_model
        }
    
    # Tune SVR
    svr_model, svr_metrics, svr_params = tune_svr(
        X_train, y_train, X_test, y_test
    )
    results["SVR"] = {
        "description": "Tuned Support Vector Regressor with GridSearchCV",
        "best_params": svr_params,
        "metrics": svr_metrics,
        "model": svr_model
    }
    
    # Tune ElasticNet
    en_model, en_metrics, en_params = tune_elasticnet(
        X_train, y_train, X_test, y_test
    )
    results["ElasticNet"] = {
        "description": "Tuned ElasticNet with GridSearchCV",
        "best_params": en_params,
        "metrics": en_metrics,
        "model": en_model
    }
    
    # Tune Ridge
    ridge_model, ridge_metrics, ridge_params = tune_ridge(
        X_train, y_train, X_test, y_test
    )
    results["Ridge"] = {
        "description": "Tuned Ridge Regression with GridSearchCV",
        "best_params": ridge_params,
        "metrics": ridge_metrics,
        "model": ridge_model
    }
    
    # Determine best model based on R² score
    best_model_name = max(
        results.keys(),
        key=lambda k: results[k]['metrics']['r2']
    )
    
    print("\n" + "="*60)
    print("FINAL RESULTS - MODEL COMPARISON")
    print("="*60)
    
    # Display all models sorted by R² score
    print("\nAll Models Ranked by R² Score:")
    print("-" * 60)
    sorted_models = sorted(results.items(), key=lambda x: x[1]['metrics']['r2'], reverse=True)
    
    for rank, (model_name, model_data) in enumerate(sorted_models, 1):
        m = model_data['metrics']
        print(f"\n{rank}. {model_name}")
        print(f"   R²:   {m['r2']:.4f}")
        print(f"   RMSE: {m['rmse']:.4f}")
        print(f"   MAE:  {m['mae']:.4f}")
    
    print("\n" + "="*60)
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   R² Score: {results[best_model_name]['metrics']['r2']:.4f}")
    print(f"   RMSE: {results[best_model_name]['metrics']['rmse']:.4f}")
    print(f"   MAE: {results[best_model_name]['metrics']['mae']:.4f}")
    
    # Save results
    report_path = os.path.join(REPORTS_DIR, 'tuned_results.json')
    model_path = os.path.join(MODEL_DIR, 'tuned_model.pkl')
    
    save_results(results, report_path, model_path, best_model_name)
    
    print("\n" + "="*60)
    print("Hyperparameter tuning completed successfully!")
    print("="*60)

if __name__ == "__main__":
    train_tuned_models()
