# Run Anuj's Complete Pipeline
# This script executes all Anuj's tasks in the correct order

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Student Performance Prediction - Anuj" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get the script directory
$projectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "Project Root: $projectRoot" -ForegroundColor Yellow
Write-Host ""

# Step 1: Hyperparameter Tuning
Write-Host "STEP 1: Running Hyperparameter Tuning..." -ForegroundColor Green
Write-Host "----------------------------------------" -ForegroundColor Green
python "$projectRoot\src\optimization\tune_model.py"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error in hyperparameter tuning. Exiting..." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✓ Hyperparameter tuning completed successfully!" -ForegroundColor Green
Write-Host ""

# Step 2: Explainability Analysis
Write-Host "STEP 2: Generating Explainability Visualizations..." -ForegroundColor Green
Write-Host "----------------------------------------------------" -ForegroundColor Green
python "$projectRoot\src\optimization\explainability.py"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error in explainability analysis. Exiting..." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✓ Explainability visualizations completed successfully!" -ForegroundColor Green
Write-Host ""

# Step 3: Error Analysis
Write-Host "STEP 3: Running Error Analysis..." -ForegroundColor Green
Write-Host "----------------------------------" -ForegroundColor Green
python "$projectRoot\src\optimization\error_analysis.py"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error in error analysis. Exiting..." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✓ Error analysis completed successfully!" -ForegroundColor Green
Write-Host ""

# Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ALL TASKS COMPLETED SUCCESSFULLY!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Generated Files:" -ForegroundColor Yellow
Write-Host "  - models/tuned_model.pkl" -ForegroundColor White
Write-Host "  - reports/tuned_results.json" -ForegroundColor White
Write-Host "  - reports/explainability/shap_summary_plot.png" -ForegroundColor White
Write-Host "  - reports/explainability/shap_force_plot.png" -ForegroundColor White
Write-Host "  - reports/explainability/partial_dependence_plots.png" -ForegroundColor White
Write-Host "  - reports/explainability/feature_importance.png" -ForegroundColor White
Write-Host "  - reports/error_analysis/error_report.md" -ForegroundColor White
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Review tuned_results.json for model performance" -ForegroundColor White
Write-Host "  2. Examine explainability plots in reports/explainability/" -ForegroundColor White
Write-Host "  3. Read error_report.md for improvement suggestions" -ForegroundColor White
Write-Host ""
