import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '../../data')
processed_data_path = os.path.join(data_dir, 'processed', 'student_performance_cleaned.csv')
MODEL_DIR = os.path.join(current_dir, '../../models')
REPORTS_DIR = os.path.join(current_dir, '../../reports')
EXPLAINABILITY_DIR = os.path.join(REPORTS_DIR, 'explainability')
DATA_PATH = processed_data_path

def load_model_and_data():
    """
    Load the tuned model and test data.
    
    Returns:
        tuple: (model, X_train, X_test, y_train, y_test, feature_names)
    """
    print("Loading data and model...")
    
    # Load data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Processed data file not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    
    # Split features and target
    X = df.drop(columns=['GPA'])
    y = df['GPA']
    feature_names = X.columns.tolist()
    
    # Train-test split (same as baseline and tuning: test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Load tuned model
    model_path = os.path.join(MODEL_DIR, 'tuned_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Tuned model not found at {model_path}. Please run tune_model.py first.")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model loaded from {model_path}")
    print(f"Data shape: {X.shape}")
    print(f"Test set size: {X_test.shape[0]}")
    
    return model, X_train, X_test, y_train, y_test, feature_names

def generate_shap_summary_plot(model, X_test, feature_names):
    """
    Generate SHAP summary plot showing feature importance.
    
    Args:
        model: Trained model pipeline
        X_test: Test features
        feature_names: List of feature names
    """
    print("\n" + "="*60)
    print("Generating SHAP Summary Plot...")
    print("="*60)
    
    # Get the regressor from the pipeline
    regressor = model.named_steps['regressor']
    
    # Transform test data using the scaler
    X_test_scaled = model.named_steps['scaler'].transform(X_test)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    # Create SHAP explainer
    # Use a subset of data for faster computation
    background_size = min(100, X_test_scaled_df.shape[0])
    explainer = shap.Explainer(regressor, X_test_scaled_df.iloc[:background_size])
    
    # Calculate SHAP values for test set (using subset for efficiency)
    shap_sample_size = min(200, X_test_scaled_df.shape[0])
    shap_values = explainer(X_test_scaled_df.iloc[:shap_sample_size])
    
    # Create summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_scaled_df.iloc[:shap_sample_size], show=False)
    plt.title("SHAP Summary Plot - Feature Impact on Predictions", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    os.makedirs(EXPLAINABILITY_DIR, exist_ok=True)
    summary_plot_path = os.path.join(EXPLAINABILITY_DIR, 'shap_summary_plot.png')
    plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
    print(f"SHAP summary plot saved to {summary_plot_path}")
    plt.close()

def generate_shap_force_plot(model, X_test, feature_names):
    """
    Generate SHAP force plot for individual predictions.
    
    Args:
        model: Trained model pipeline
        X_test: Test features
        feature_names: List of feature names
    """
    print("\n" + "="*60)
    print("Generating SHAP Force Plot...")
    print("="*60)
    
    # Get the regressor from the pipeline
    regressor = model.named_steps['regressor']
    
    # Transform test data using the scaler
    X_test_scaled = model.named_steps['scaler'].transform(X_test)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    # Create SHAP explainer
    background_size = min(100, X_test_scaled_df.shape[0])
    explainer = shap.Explainer(regressor, X_test_scaled_df.iloc[:background_size])
    
    # Calculate SHAP values for first instance
    shap_values = explainer(X_test_scaled_df.iloc[[0]])
    
    # Create force plot
    shap.force_plot(
        explainer.expected_value,
        shap_values.values[0],
        X_test_scaled_df.iloc[0],
        matplotlib=True,
        show=False
    )
    
    plt.title("SHAP Force Plot - Individual Prediction Explanation", fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    force_plot_path = os.path.join(EXPLAINABILITY_DIR, 'shap_force_plot.png')
    plt.savefig(force_plot_path, dpi=300, bbox_inches='tight')
    print(f"SHAP force plot saved to {force_plot_path}")
    plt.close()

def generate_partial_dependence_plots(model, X_test, feature_names):
    """
    Generate Partial Dependence Plots (PDP) for top features.
    
    Args:
        model: Trained model pipeline
        X_test: Test features
        feature_names: List of feature names
    """
    print("\n" + "="*60)
    print("Generating Partial Dependence Plots...")
    print("="*60)
    
    # Transform test data
    X_test_scaled = model.named_steps['scaler'].transform(X_test)
    
    # Select top features for PDP (based on common important features)
    # You can adjust these based on your dataset
    top_features = feature_names[:min(6, len(feature_names))]
    feature_indices = [i for i, name in enumerate(feature_names) if name in top_features]
    
    print(f"Generating PDPs for features: {top_features}")
    
    # Create PDP
    fig, ax = plt.subplots(figsize=(14, 10))
    
    display = PartialDependenceDisplay.from_estimator(
        model,
        X_test,
        features=feature_indices,
        feature_names=feature_names,
        n_cols=3,
        grid_resolution=50,
        ax=ax
    )
    
    plt.suptitle("Partial Dependence Plots - Feature Effects on GPA", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    pdp_plot_path = os.path.join(EXPLAINABILITY_DIR, 'partial_dependence_plots.png')
    plt.savefig(pdp_plot_path, dpi=300, bbox_inches='tight')
    print(f"Partial Dependence Plots saved to {pdp_plot_path}")
    plt.close()

def generate_feature_importance_plot(model, feature_names):
    """
    Generate feature importance plot from the model.
    
    Args:
        model: Trained model pipeline
        feature_names: List of feature names
    """
    print("\n" + "="*60)
    print("Generating Feature Importance Plot...")
    print("="*60)
    
    # Get the regressor from the pipeline
    regressor = model.named_steps['regressor']
    
    # Check if model has feature_importances_ attribute
    if hasattr(regressor, 'feature_importances_'):
        importances = regressor.feature_importances_
        
        # Create DataFrame for better visualization
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title('Feature Importance from Tuned Model', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # Save plot
        importance_plot_path = os.path.join(EXPLAINABILITY_DIR, 'feature_importance.png')
        plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {importance_plot_path}")
        plt.close()
        
        return feature_importance_df
    else:
        print("Model does not have feature_importances_ attribute. Skipping feature importance plot.")
        return None

def generate_explainability_report():
    """
    Main function to generate all explainability visualizations.
    """
    print("="*60)
    print("Starting Explainability Analysis")
    print("="*60)
    
    # Load model and data
    model, X_train, X_test, y_train, y_test, feature_names = load_model_and_data()
    
    # Create explainability directory
    os.makedirs(EXPLAINABILITY_DIR, exist_ok=True)
    
    try:
        # Generate feature importance plot
        feature_importance_df = generate_feature_importance_plot(model, feature_names)
        
        # Generate SHAP summary plot
        generate_shap_summary_plot(model, X_test, feature_names)
        
        # Generate SHAP force plot
        generate_shap_force_plot(model, X_test, feature_names)
        
        # Generate Partial Dependence Plots
        generate_partial_dependence_plots(model, X_test, feature_names)
        
        print("\n" + "="*60)
        print("Explainability analysis completed successfully!")
        print(f"All plots saved to: {EXPLAINABILITY_DIR}")
        print("="*60)
        
        if feature_importance_df is not None:
            print("\nTop 5 Most Important Features:")
            print(feature_importance_df.head(5).to_string(index=False))
        
    except Exception as e:
        print(f"\nError during explainability analysis: {str(e)}")
        print("Please ensure you have installed all required packages:")
        print("  pip install shap matplotlib")
        raise

if __name__ == "__main__":
    generate_explainability_report()
