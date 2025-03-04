# Ecommerce Customer Reordering Prediction

## Overview
This project focuses on predicting whether a customer will reorder a product using historical ecommerce data. The aim is to apply machine learning models, including **LSTM (Long Short-Term Memory)**, **XGBoost**, and an **Ensemble model**, to predict customer reordering behavior based on various features such as order day, hour, product categories, and prior order information.

## Dataset
The dataset consists of transactional data from an ecommerce platform, with the following key features:
- **order_dow**: Day of the week the order was placed.
- **order_hour_of_day**: Hour of the day when the order was made.
- **days_since_prior_order**: Number of days since the customer's last order.
- **department_id**: The department the product belongs to.
- **product_id**: The unique identifier for the product.
- **reordered**: A binary target variable indicating whether a product was reordered.

### Data Preprocessing
Key preprocessing steps included:
- **Missing Values Handling**: Imputed missing values using median values for numerical columns.
- **Categorical Encoding**: Encoded categorical variables like **department_id** and **product_id** using **LabelEncoder**.
- **Feature Scaling**: Applied **MinMaxScaler** to scale the features.
- **Outlier Removal**: Detected and removed outliers using **Z-score** method.
- **SMOTE**: Applied Synthetic Minority Over-sampling Technique (SMOTE) to handle class imbalance.
- **Time-Series Preprocessing**: Created sequences for LSTM model to capture time-dependent relationships in the data.

## Models
The following models were trained and evaluated:
1. **LSTM (Long Short-Term Memory)**: Used for sequential data prediction, capturing temporal dependencies in customer ordering behavior.
2. **XGBoost**: A gradient boosting classifier known for high performance in classification tasks.
3. **Ensemble Model**: A combination of predictions from both LSTM and XGBoost for better accuracy.

## Evaluation Metrics
The models were evaluated based on **Mean Absolute Error (MAE)**. After tuning, the final results were:
- **LSTM MAE**: 0.4806
- **XGBoost MAE**: 0.4469
- **Ensemble MAE**: 0.4469

Despite the low R-squared values, which indicate a lack of a strong linear relationship, all models were able to effectively capture the underlying nonlinear patterns in the data. The **LSTM model** provided the best performance after tuning.

## Future Work
- **Feature Engineering**: Further refinement of features to enhance model prediction.
- **Hyperparameter Tuning**: Experiment with more advanced tuning techniques like **GridSearchCV** or **RandomizedSearchCV**.
- **Model Interpretability**: Implement tools like **SHAP** to explain model predictions for better transparency.

## Getting Started

### Prerequisites
Ensure you have the following Python libraries installed:
- TensorFlow
- XGBoost
- Pandas
- Numpy
- Scikit-learn
- Imbalanced-learn (for SMOTE)
- Matplotlib (for plotting)

Install required libraries via pip:
```bash
pip install -r requirements.txt
# time_Series_pred
