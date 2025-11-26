import pandas as pd
import numpy as np
import json
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '../../data')
processed_data_path = os.path.join(data_dir, 'processed', 'student_performance_cleaned.csv')
MODEL_DIR = os.path.join(current_dir, '../../models')
REPORTS_DIR = os.path.join(current_dir, '../../reports')
DATA_PATH = processed_data_path

def train_baseline():
    print("Starting baseline model training...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Processed data file not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    
    # Split features and target
    X = df.drop(columns=['GPA'])
    y = df['GPA']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Predictions
    y_test_pred = pipeline.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    print("Model evaluation completed.")
    print(f"Test MSE:  {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE:  {mae:.4f}")
    print(f"Test R2:   {r2:.4f}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'baseline_model.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Trained model saved to {model_path}")

    # Save performance report in structured JSON format
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    report = {
        "Linear Regression": {
            "description": "Baseline model using standard scaling and linear regression",
            "metrics": {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }
        }
    }
    
    report_path = os.path.join(REPORTS_DIR, 'baseline_results.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Performance report saved to {report_path}")

if __name__ == "__main__":
    train_baseline()