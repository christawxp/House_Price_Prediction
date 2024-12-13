# House Price Prediction

## Overview
This project aimed to predict house prices by leveraging structured and text-based features with advanced feature engineering and machine learning techniques. A combination of **XGBoost Random Forest Regressor** and **Gradient Boosted Random Forest** models was used, achieving competitive R-squared scores with ensemble predictions.

## Features
### Feature Engineering
- Created over 200 features, including:
  - `bathrooms_per_room`: Ratio of bathrooms to rooms.
  - `area_per_room`: Average room size.
  - Encoded presence of keywords like "balcón" and "reformada" from descriptions.
### Text Processing
- Applied **TF-IDF** and **Bag of Words** for text features such as title, subtitle, and description.

## Techniques
- One-hot encoding for categorical variables.
- Gradient Boosting and XGBoost models with hyperparameter tuning via grid search.
- 10-fold cross-validation to prevent overfitting.

## Results
### Final Models
- **Gradient Boosted Random Forest Regressor**: R² = 0.591.
- **XGBoost Random Forest**: R² = 0.517.
- Predictions averaged across all models, balancing bias and variance effectively.

## Limitations
- Small dataset (~800 records) limited generalizability.
- Spanish text data required additional preprocessing compared to English models.
- Time constraints restricted exploration of deep learning models.
