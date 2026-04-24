# Gym Member Churn Prediction

## Overview
This project applies data mining techniques to predict whether a gym member will cancel their membership (churn). The goal is to identify patterns in member behavior that indicate a higher likelihood of cancellation.

## Objectives
- Build classification models to predict churn
- Compare multiple machine learning algorithms
- Identify key factors influencing churn

## Dataset
The dataset contains gym member information including demographic and behavioral features such as:
- Age
- Gender
- Membership type
- Visit frequency
- Workout duration

The dataset includes a labeled churn variable indicating whether a member has canceled their membership.

## Methods
The following machine learning models were implemented:
- Logistic Regression
- Decision Tree
- Random Forest

## Results Summary
The models were evaluated using accuracy, precision, recall, and F1-score.  
Random Forest achieved the best overall performance.

## Key Findings
- **Random Forest** performed best overall with 97.2% cross-validation accuracy and 90% test accuracy
- **Member activity levels** strongly influence churn, with visit frequency being the most important predictor (73% feature importance)
- **Behavioral features** are more predictive than demographic features, suggesting engagement-focused retention strategies
- **Class imbalance** was addressed using SMOTE, improving model performance on the minority churn class

## Project Structure
- `data/` – dataset file
- `notebooks/` – Jupyter Notebook with full workflow
- `results/figures/` – generated visualizations
- `results/tables/` – model comparison outputs

## How to Run
1. Install dependencies:
```bash pip install -r requirements.txt```
2. Launch Jupyter Notebook:
```jupyter notebook```
3. Open and Run
```notebooks/churn_analysis.ipynb``