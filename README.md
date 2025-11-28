# Student Performance Prediction

A machine learning project to predict student GPA based on demographic, behavioral, and academic factors. Features an interactive Streamlit web application for real-time predictions.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)

## 🎯 Project Overview

This project predicts student academic performance (GPA) using machine learning models trained on student data including study habits, attendance, parental support, and extracurricular activities.

**Key Achievement:** The baseline Linear Regression model achieves **R² = 0.953**, demonstrating that simpler models can be highly effective when data exhibits strong linear relationships.

## 📊 Results Summary

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| **Linear Regression (Baseline)** | 0.9532 | 0.1966 | 0.1553 |
| Ridge Regression | 0.9532 | 0.1967 | 0.1553 |
| ElasticNet | 0.9532 | 0.1967 | 0.1553 |
| SVR | 0.9529 | 0.1974 | 0.1559 |
| XGBoost | 0.9484 | 0.2066 | 0.1612 |
| GradientBoosting | 0.9477 | 0.2080 | 0.1623 |

**Key Finding:** Absences has the strongest correlation (-0.92) with GPA, making it the dominant predictor.

## 🗂️ Project Structure

```
Student_performance_prediction/
├── app.py                          # Streamlit web application
├── data/
│   ├── raw/                        # Original dataset
│   └── processed/                  # Cleaned dataset
├── models/
│   ├── baseline_model.pkl          # Linear Regression (Manjil)
│   ├── tuned_model.pkl             # Best tuned model (Anuj)
│   ├── ridge_model.pkl             # Ridge Regression
│   ├── elasticnet_model.pkl        # ElasticNet
│   ├── svr_model.pkl               # Support Vector Regressor
│   └── model_registry.json         # Model metadata
├── src/
│   ├── eda/                        # Exploratory Data Analysis
│   │   └── eda_notebook.ipynb
│   ├── modeling/                   # Data preprocessing & baseline
│   │   ├── preprocess.py
│   │   └── model_baseline.py
│   └── optimization/               # Hyperparameter tuning & analysis
│       ├── tune_model.py
│       ├── explainability.py
│       ├── error_analysis.py
│       └── compare_all_models.py
├── reports/
│   ├── explainability/             # SHAP plots, PDPs
│   ├── error_analysis/             # Error analysis report
│   └── tuned_results.json          # Model comparison results
├── .streamlit/                     # Streamlit configuration
├── .devcontainer/                  # Dev container setup
└── requirements.txt
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/manjil-budhathoki/Student_performance_prediction.git
cd Student_performance_prediction

# Install dependencies
pip install -r requirements.txt
```

### Run the Web App

```bash
streamlit run app.py
```

### Run Model Training (Optional)

```bash
# Preprocessing
python src/modeling/preprocess.py

# Baseline model
python src/modeling/model_baseline.py

# Hyperparameter tuning (7 models)
python src/optimization/tune_model.py

# Generate explainability plots
python src/optimization/explainability.py

# Error analysis
python src/optimization/error_analysis.py
```

## 🔧 Features

### Input Features (12)
| Feature | Description | Type |
|---------|-------------|------|
| Age | Student age (15-18) | Numeric |
| Gender | Male/Female | Categorical |
| Ethnicity | Caucasian/African American/Asian/Other | Categorical |
| ParentalEducation | High School/Some College/Bachelor/Higher | Categorical |
| StudyTimeWeekly | Hours of study per week | Numeric |
| Absences | Number of absences | Numeric |
| Tutoring | Yes/No | Binary |
| ParentalSupport | Low/Medium/High | Categorical |
| Extracurricular | Participation in activities | Binary |
| Sports | Sports participation | Binary |
| Music | Music participation | Binary |
| Volunteering | Volunteering activities | Binary |

### Target Variable
- **GPA**: Grade Point Average (0.0 - 4.0)

## 📈 Model Explainability

The project includes comprehensive explainability analysis:
- **SHAP Summary Plot**: Feature importance and impact direction
- **SHAP Force Plot**: Individual prediction explanations
- **Partial Dependence Plots**: Feature effect curves

Key insights:
- **Absences** is the strongest predictor (negative correlation)
- **Study time** has positive impact on GPA
- Linear relationships dominate, explaining why simple models excel

## 👥 Team Contributions

| Member | Role | Contributions |
|--------|------|---------------|
| **Manjil** | Data Engineering & Baseline | Preprocessing, EDA, baseline Linear Regression model |
| **Anuj** | ML Engineering & Analysis | Hyperparameter tuning (7 models), explainability (SHAP/PDP), error analysis, model comparison |
| **Samyak** | Application Development | Streamlit web app, deployment configuration |

## 📁 Key Files

- `app.py` - Interactive Streamlit application
- `src/optimization/tune_model.py` - Multi-model hyperparameter tuning
- `src/optimization/explainability.py` - SHAP and PDP visualizations
- `reports/error_analysis/error_report.md` - Detailed error analysis
- `MODEL_COMPARISON_INSIGHTS.md` - Why baseline outperforms complex models

## 🛠️ Technologies Used

- **Python 3.12**
- **scikit-learn** - Machine learning models
- **XGBoost / LightGBM** - Gradient boosting
- **SHAP** - Model explainability
- **Streamlit** - Web application
- **Plotly** - Interactive visualizations
- **Pandas / NumPy** - Data processing

## 📄 License

This project is for educational purposes.

## 🔗 Links

- [GitHub Repository](https://github.com/manjil-budhathoki/Student_performance_prediction)
- [Live Demo](https://studentperformanceprediction-4emwdtwcrznxgzbqp6pcjs.streamlit.app/) *(if deployed)*