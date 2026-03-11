# Resampling-Based Bias Mitigation for Diabetes Diagnosis in Imbalanced Clinical Populations

This project investigates methods for improving predictive fairness in clinical machine learning models trained on imbalanced demographic data.

## Motivation
Medical datasets often contain demographic imbalance, which can lead to biased predictive models. 
This project examines several strategies for handling demographic imbalance when predicting diabetes using the NHANES 2017-2018 dataset.

## Data and Preprocessing
Dataset: National Health and Nutrition Examination Survey (NHANES) 2017-2018 (https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017)

Label: Diabetes status (binary diabetic/non-diabetic) based on glycohemoglobin levels
Features:
- Demographics (age, sex, ethnicity, education)
- Biomarkers (cholesterol, triglycerides, creatinine, HDL)
- Blood pressure
- BMI and anthropometrics

The data were preprocessed as follows: MICE imputation for missing numeric values, one-hot encoding of categorical features, and feature standardization.

## Experiments
For each of the following model families:
- Logistic Regression
- Decision Trees
- Neural Networks

We ran the following imbalance mitigation techniques:
- Naive upsampling of minority group
- Cost-sensitive learning
- K-means cluster-based resampling
- Gaussian mixture-based resampling

Hyperparameters were tuned using 5-fold cross-validation. Each model-experiment pair was evaluated on a hold-out test set using F1 score, accuracy, AUPRC, precision, and recall.

## Authors
Charlotte Imbert - MS Statistics, Stanford University
Katie Liu - MS Computer Science, Stanford University
Benjia Zhang - MS Electrical Engineering, Stanford University

Every time you pull from the repo:
- Run uv sync
- Run source .venv/bin/activate




