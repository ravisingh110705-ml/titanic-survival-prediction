# Titanic Survival Prediction

A binary classification project predicting passenger survival on the Titanic using multiple ML models, with hyperparameter tuning via GridSearchCV.

## Live Demo
Try it yourself: https://titanic-survival-prediction-gj8muwnsgjwfkrs3bqvm79.streamlit.app

## Problem

Given passenger information (class, sex, age, fare, etc.), predict whether a passenger survived (binary: 0 = No, 1 = Yes).

## Data Cleaning

- Dropped `Cabin` (too many missing values), `Name`, `Ticket`, and `PassengerId` (not predictive).
- Filled missing `Embarked` values with the mode, and missing `Age` values with the median.
- Encoded `Sex` as binary and one-hot encoded `Embarked` with `drop_first=True`.
- Scaled features using `StandardScaler`, fit only on the training set.

## Models Compared

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.810 | 0.786 | 0.743 | 0.764 |
| Random Forest | 0.799 | 0.764 | 0.743 | 0.753 |
| Decision Tree (max_depth=4) | 0.799 | 0.839 | 0.635 | 0.723 |
| K-Nearest Neighbors | 0.804 | 0.783 | 0.730 | 0.755 |
| **SVM (tuned via GridSearchCV)** | **0.821** | **0.839** | **0.703** | **0.765** |
| Naive Bayes | 0.771 | 0.720 | 0.730 | 0.725 |

SVM achieved the best accuracy after tuning hyperparameters (`C`, `kernel`, `gamma`) with 5-fold cross-validation via GridSearchCV, and was selected as the final model. Decision Tree and KNN were also tested at multiple depths/neighbor counts to check for overfitting, comparing train vs. test accuracy.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn

## Files

- `main.ipynb` — full notebook: data cleaning, encoding, model training, tuning, and evaluation
- `data/` — dataset used for training and testing

## Author

Ravi Singh Bungla — [GitHub](https://github.com/ravisingh110705-ml) | [LinkedIn](your-linkedin-url-here)

## Other Projects
- [Heart Disease Prediction](https://github.com/ravisingh110705-ml/heart-disease-prediction)
- [Medical Insurance Cost Prediction](https://github.com/ravisingh110705-ml/medical-insurance-cost-prediction)
- [Telco Customer Churn Prediction](https://github.com/ravisingh110705-ml/telco-churn-prediction)
