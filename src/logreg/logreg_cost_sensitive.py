"""
cost_sensitive.py trains a logistic regression model on the dataset with cost-sensitive learning.
It outputs the final loss of the model without regularization and with regularization.
Regularization is done using L2 regularization and the factor can be altered in the main function.
Cost-sensitive learning is done by upweighting the minority class misclassification.
"""

import util
import numpy as np
import pandas as pd
import argparse
from logreg_src import *
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from util import *

def main(test: bool = False):
    #Threshold
    threshold_w_cs = 0.35
    
    #Load data
    X_original, Y_original = load_csv('src/data/model_ready/train_processed.csv', label_col='diabetes', add_intercept=True)
    
    #Regularized with cost-sensitive learning
    original_df = pd.read_csv('src/data/model_ready/train_processed.csv')
    sample_weight = calculate_sample_weight(original_df)
    theta_w_cs = logistic_regression(X_original, Y_original, max_iter=5000, lambda_reg=0.001, sample_weight=sample_weight)
    train_probs_w_cs = 1 / (1 + np.exp(-(X_original @ theta_w_cs)))
    _, train_pred_w_cs = f1_from_probs(Y_original, train_probs_w_cs, threshold_w_cs)
    train_accuracy_w_cs = accuracy_score(Y_original, train_pred_w_cs)
    print("Train Accuracy With Cost Sensitive Learning:", train_accuracy_w_cs)
    
    if test:
        X_test, Y_test = load_csv('src/data/model_ready/test_processed.csv', label_col='diabetes', add_intercept=True)
        test_data_df = pd.read_csv('src/data/model_ready/test_processed.csv')
        X_test_df, Y_test_df = test_data_df.drop(columns=["diabetes"]), test_data_df["diabetes"]
        output_model_path = 'src/results/logreg'
        prob_w_cs = 1 / (1 + np.exp(-(X_test @ theta_w_cs)))
        
        evaluate_by_ethnicity(X_test_df, Y_test_df, prob_w_cs, threshold_w_cs, output_model_path=output_model_path, experiment_type='cost_sensitive')
        import pdb; pdb.set_trace()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test or Train")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    main(test = args.test)
    
    

