# Baseline Model Performance Report

## Model Details
- **Model:** Linear Regression

## Key Metrics

### R2 Score: 0.953
This represents the accuracy of the model relative to the data variance. Our model explains 95.3% of the reasons why grades differ between students. This is a very high score.

### MAE (Mean Absolute Error): 0.155
This is the average margin of error. On average, the model's prediction is off by only 0.15 GPA points. For example, if a student has a 3.0 GPA, the model typically predicts between 2.85 and 3.15.

### RMSE (Root Mean Squared Error): 0.197
This metric is similar to MAE but penalizes large errors more heavily. A score of 0.197 is low, indicating the model rarely makes massive mistakes (like predicting a 4.0 for a failing student).

## Conclusion
The Linear Regression model is highly effective for this dataset. This is primarily because the relationship between the most important feature (Absences) and the target (GPA) is linear and consistent.