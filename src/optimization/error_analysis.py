import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime
from sklearn.model_selection import train_test_split

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '../../data')
processed_data_path = os.path.join(data_dir, 'processed', 'student_performance_cleaned.csv')
MODEL_DIR = os.path.join(current_dir, '../../models')
REPORTS_DIR = os.path.join(current_dir, '../../reports')
ERROR_ANALYSIS_DIR = os.path.join(REPORTS_DIR, 'error_analysis')
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
    print(f"Test set size: {X_test.shape[0]}")
    
    return model, X_train, X_test, y_train, y_test, feature_names

def analyze_predictions(model, X_test, y_test, feature_names):
    """
    Analyze model predictions and calculate residuals.
    
    Args:
        model: Trained model pipeline
        X_test: Test features
        y_test: Test target
        feature_names: List of feature names
        
    Returns:
        DataFrame: Analysis results with predictions and residuals
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate residuals
    residuals = y_test.values - y_pred
    abs_residuals = np.abs(residuals)
    
    # Create analysis DataFrame
    analysis_df = pd.DataFrame({
        'Actual_GPA': y_test.values,
        'Predicted_GPA': y_pred,
        'Residual': residuals,
        'Abs_Residual': abs_residuals
    })
    
    # Add features
    X_test_reset = X_test.reset_index(drop=True)
    for col in feature_names:
        analysis_df[col] = X_test_reset[col].values
    
    # Sort by absolute residual (worst predictions first)
    analysis_df = analysis_df.sort_values('Abs_Residual', ascending=False).reset_index(drop=True)
    
    return analysis_df

def get_feature_importance(model, feature_names):
    """
    Extract feature importance from the model.
    
    Args:
        model: Trained model pipeline
        feature_names: List of feature names
        
    Returns:
        DataFrame: Feature importance sorted by importance
    """
    # Get the regressor from the pipeline
    regressor = model.named_steps['regressor']
    
    # Check if model has feature_importances_ attribute
    if hasattr(regressor, 'feature_importances_'):
        importances = regressor.feature_importances_
        
        # Create DataFrame
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).reset_index(drop=True)
        
        return feature_importance_df
    else:
        return None

def load_metrics():
    """
    Load performance metrics from tuned_results.json.
    
    Returns:
        dict: Performance metrics
    """
    results_path = os.path.join(REPORTS_DIR, 'tuned_results.json')
    
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
        return results
    else:
        return None

def generate_error_report(analysis_df, feature_importance_df, metrics, feature_names):
    """
    Generate comprehensive error analysis markdown report.
    
    Args:
        analysis_df: DataFrame with predictions and residuals
        feature_importance_df: DataFrame with feature importance
        metrics: Dictionary with performance metrics
        feature_names: List of feature names
    """
    print("\n" + "="*60)
    print("Generating Error Analysis Report...")
    print("="*60)
    
    # Create directory
    os.makedirs(ERROR_ANALYSIS_DIR, exist_ok=True)
    
    report_path = os.path.join(ERROR_ANALYSIS_DIR, 'error_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("# Error Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write("This report provides a comprehensive error analysis of the tuned regression model ")
        f.write("for student performance prediction. It includes worst predictions, residual analysis, ")
        f.write("feature importance insights, and recommendations for model improvement.\n\n")
        
        # Model Performance Overview
        f.write("## Model Performance Overview\n\n")
        
        if metrics:
            # Find best model
            best_model = max(metrics.keys(), key=lambda k: metrics[k]['metrics']['r2'])
            best_metrics = metrics[best_model]['metrics']
            
            f.write(f"**Best Model:** {best_model}\n\n")
            f.write("### Performance Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| R² Score | {best_metrics['r2']:.4f} |\n")
            f.write(f"| Mean Squared Error (MSE) | {best_metrics['mse']:.4f} |\n")
            f.write(f"| Root Mean Squared Error (RMSE) | {best_metrics['rmse']:.4f} |\n")
            f.write(f"| Mean Absolute Error (MAE) | {best_metrics['mae']:.4f} |\n\n")
            
            # Model comparison
            f.write("### Model Comparison\n\n")
            f.write("| Model | R² | RMSE | MAE |\n")
            f.write("|-------|-----|------|-----|\n")
            for model_name, model_data in metrics.items():
                m = model_data['metrics']
                f.write(f"| {model_name} | {m['r2']:.4f} | {m['rmse']:.4f} | {m['mae']:.4f} |\n")
            f.write("\n")
        
        # Residual Analysis
        f.write("## Residual Analysis\n\n")
        
        residuals = analysis_df['Residual']
        
        f.write("### Residual Statistics\n\n")
        f.write("| Statistic | Value |\n")
        f.write("|-----------|-------|\n")
        f.write(f"| Mean Residual | {residuals.mean():.4f} |\n")
        f.write(f"| Std Dev of Residuals | {residuals.std():.4f} |\n")
        f.write(f"| Min Residual (Over-prediction) | {residuals.min():.4f} |\n")
        f.write(f"| Max Residual (Under-prediction) | {residuals.max():.4f} |\n")
        f.write(f"| Median Absolute Residual | {analysis_df['Abs_Residual'].median():.4f} |\n\n")
        
        # Interpretation
        f.write("### Interpretation\n\n")
        if abs(residuals.mean()) < 0.1:
            f.write("✅ The mean residual is close to zero, indicating minimal systematic bias.\n\n")
        else:
            bias_direction = "over-predicting" if residuals.mean() < 0 else "under-predicting"
            f.write(f"⚠️ The model has a slight tendency towards {bias_direction} GPA values.\n\n")
        
        # Worst Predictions
        f.write("## Worst Predictions Analysis\n\n")
        f.write("The following table shows the 10 predictions with the largest errors:\n\n")
        
        worst_10 = analysis_df.head(10)
        
        f.write("| Rank | Actual GPA | Predicted GPA | Residual | Error Type |\n")
        f.write("|------|------------|---------------|----------|------------|\n")
        
        for idx, row in worst_10.iterrows():
            error_type = "Under-prediction" if row['Residual'] > 0 else "Over-prediction"
            f.write(f"| {idx+1} | {row['Actual_GPA']:.3f} | {row['Predicted_GPA']:.3f} | "
                   f"{row['Residual']:.3f} | {error_type} |\n")
        f.write("\n")
        
        # Feature values for worst predictions
        f.write("### Feature Values for Top 5 Worst Predictions\n\n")
        f.write("Understanding the feature values of the worst predictions can help identify patterns:\n\n")
        
        worst_5 = analysis_df.head(5)
        feature_cols = [col for col in feature_names if col in analysis_df.columns]
        
        f.write("| Rank | " + " | ".join(feature_cols[:8]) + " |\n")
        f.write("|------|" + "|".join(["------"] * min(8, len(feature_cols))) + "|\n")
        
        for idx, row in worst_5.iterrows():
            values = [f"{row[col]:.2f}" if isinstance(row[col], float) else str(row[col]) 
                     for col in feature_cols[:8]]
            f.write(f"| {idx+1} | " + " | ".join(values) + " |\n")
        f.write("\n")
        
        if len(feature_cols) > 8:
            f.write(f"*Note: Showing first 8 of {len(feature_cols)} features. ")
            f.write("Remaining features: " + ", ".join(feature_cols[8:]) + "*\n\n")
        
        # Feature Importance
        f.write("## Feature Importance Analysis\n\n")
        
        if feature_importance_df is not None:
            f.write("The following features have the most significant impact on GPA predictions:\n\n")
            
            f.write("| Rank | Feature | Importance | Cumulative % |\n")
            f.write("|------|---------|------------|-------------|\n")
            
            total_importance = feature_importance_df['Importance'].sum()
            cumulative = 0
            
            for idx, row in feature_importance_df.head(10).iterrows():
                cumulative += row['Importance']
                cumulative_pct = (cumulative / total_importance) * 100
                f.write(f"| {idx+1} | {row['Feature']} | {row['Importance']:.4f} | {cumulative_pct:.1f}% |\n")
            f.write("\n")
            
            # Key insights
            f.write("### Key Insights\n\n")
            top_feature = feature_importance_df.iloc[0]
            f.write(f"- **Most Important Feature:** {top_feature['Feature']} "
                   f"(importance: {top_feature['Importance']:.4f})\n")
            
            top_3 = feature_importance_df.head(3)
            top_3_importance = top_3['Importance'].sum() / total_importance * 100
            f.write(f"- The top 3 features account for {top_3_importance:.1f}% of total importance\n")
            
            top_5 = feature_importance_df.head(5)
            top_5_importance = top_5['Importance'].sum() / total_importance * 100
            f.write(f"- The top 5 features account for {top_5_importance:.1f}% of total importance\n\n")
        else:
            f.write("Feature importance analysis not available for this model type.\n\n")
        
        # Recommendations
        f.write("## Recommendations for Model Improvement\n\n")
        
        f.write("### 1. Feature Engineering\n\n")
        if feature_importance_df is not None:
            top_features = feature_importance_df.head(3)['Feature'].tolist()
            f.write(f"- Consider creating interaction features between top predictors: {', '.join(top_features)}\n")
        f.write("- Explore polynomial features for non-linear relationships\n")
        f.write("- Create domain-specific features (e.g., study-to-absence ratio, activity involvement index)\n\n")
        
        f.write("### 2. Data Quality Improvements\n\n")
        f.write("- Investigate outliers in the worst predictions for potential data quality issues\n")
        f.write("- Consider collecting additional features that might explain high-error cases\n")
        f.write("- Examine if certain student segments have consistently higher prediction errors\n\n")
        
        f.write("### 3. Model Enhancements\n\n")
        f.write("- Experiment with ensemble methods (stacking, blending)\n")
        f.write("- Try advanced algorithms: XGBoost, LightGBM, CatBoost\n")
        f.write("- Implement cross-validation for more robust hyperparameter tuning\n")
        f.write("- Consider neural networks for capturing complex patterns\n\n")
        
        f.write("### 4. Error Pattern Analysis\n\n")
        over_predictions = (analysis_df['Residual'] < 0).sum()
        under_predictions = (analysis_df['Residual'] > 0).sum()
        total = len(analysis_df)
        
        f.write(f"- Over-predictions: {over_predictions} ({over_predictions/total*100:.1f}%)\n")
        f.write(f"- Under-predictions: {under_predictions} ({under_predictions/total*100:.1f}%)\n")
        
        if abs(over_predictions - under_predictions) / total > 0.1:
            f.write("- ⚠️ Significant imbalance in error direction - consider bias correction techniques\n\n")
        else:
            f.write("- ✅ Relatively balanced error distribution\n\n")
        
        f.write("### 5. Specialized Models\n\n")
        f.write("- Consider building separate models for different GPA ranges (low, medium, high)\n")
        f.write("- Implement quantile regression to predict GPA ranges instead of point estimates\n")
        f.write("- Use confidence intervals to communicate prediction uncertainty\n\n")
        
        # Error Distribution
        f.write("## Error Distribution Summary\n\n")
        
        # Categorize errors
        small_error = (analysis_df['Abs_Residual'] <= 0.25).sum()
        medium_error = ((analysis_df['Abs_Residual'] > 0.25) & 
                       (analysis_df['Abs_Residual'] <= 0.5)).sum()
        large_error = (analysis_df['Abs_Residual'] > 0.5).sum()
        
        f.write("| Error Category | Count | Percentage |\n")
        f.write("|----------------|-------|------------|\n")
        f.write(f"| Small (≤ 0.25) | {small_error} | {small_error/total*100:.1f}% |\n")
        f.write(f"| Medium (0.25-0.5) | {medium_error} | {medium_error/total*100:.1f}% |\n")
        f.write(f"| Large (> 0.5) | {large_error} | {large_error/total*100:.1f}% |\n\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        
        if metrics and best_metrics['r2'] > 0.7:
            f.write("The tuned model shows strong predictive performance with an R² score of ")
            f.write(f"{best_metrics['r2']:.4f}. ")
        elif metrics and best_metrics['r2'] > 0.5:
            f.write("The tuned model shows moderate predictive performance with an R² score of ")
            f.write(f"{best_metrics['r2']:.4f}. ")
        else:
            f.write("The tuned model shows room for improvement with an R² score of ")
            f.write(f"{best_metrics['r2']:.4f}. ")
        
        f.write("The error analysis reveals specific areas for enhancement, particularly in:\n\n")
        f.write("1. Addressing the worst predictions through targeted feature engineering\n")
        f.write("2. Leveraging feature importance insights for model refinement\n")
        f.write("3. Exploring advanced modeling techniques to capture complex relationships\n\n")
        
        f.write("Implementing the recommendations outlined in this report should lead to ")
        f.write("improved model accuracy and more reliable GPA predictions.\n\n")
        
        f.write("---\n\n")
        f.write("*This report was automatically generated by the error analysis module.*\n")
    
    print(f"Error analysis report saved to {report_path}")
    return report_path

def run_error_analysis():
    """
    Main function to run error analysis and generate report.
    """
    print("="*60)
    print("Starting Error Analysis")
    print("="*60)
    
    # Load model and data
    model, X_train, X_test, y_train, y_test, feature_names = load_model_and_data()
    
    # Analyze predictions
    print("\nAnalyzing predictions...")
    analysis_df = analyze_predictions(model, X_test, y_test, feature_names)
    
    # Get feature importance
    print("Extracting feature importance...")
    feature_importance_df = get_feature_importance(model, feature_names)
    
    # Load metrics
    print("Loading performance metrics...")
    metrics = load_metrics()
    
    # Generate report
    report_path = generate_error_report(analysis_df, feature_importance_df, metrics, feature_names)
    
    print("\n" + "="*60)
    print("Error analysis completed successfully!")
    print(f"Report saved to: {report_path}")
    print("="*60)
    
    # Print summary
    print("\nQuick Summary:")
    print(f"  Total predictions analyzed: {len(analysis_df)}")
    print(f"  Mean absolute error: {analysis_df['Abs_Residual'].mean():.4f}")
    print(f"  Worst prediction error: {analysis_df['Abs_Residual'].max():.4f}")
    
    if feature_importance_df is not None:
        print(f"\nTop 3 Most Important Features:")
        for idx, row in feature_importance_df.head(3).iterrows():
            print(f"  {idx+1}. {row['Feature']}: {row['Importance']:.4f}")

if __name__ == "__main__":
    run_error_analysis()
