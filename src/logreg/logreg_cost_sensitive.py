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
    #Load data
    X_original, Y_original = load_csv('src/data/model_ready/train_processed.csv', label_col='diabetes', add_intercept=True)
    
    #Regularized without cost-sensitive learning
    theta_wo_cs = logistic_regression(X_original, Y_original, max_iter=100000, lambda_reg=10) #TODO: add hyperparameter from Charlotte's CV results
    
    #Regularized with cost-sensitive learning
    original_df = pd.read_csv('src/data/model_ready/train_processed.csv')
    sample_weight = calculate_sample_weight(original_df)
    theta_w_cs = logistic_regression(X_original, Y_original, max_iter=100000, lambda_reg=10, sample_weight=sample_weight) #TODO: add hyperparameter from Charlotte's CV results
    
    
    if test:
        X_test, Y_test = load_csv('src/data/model_ready/test_processed.csv', label_col='diabetes', add_intercept=True)
        threshold_wo_cs = 0 #TODO
        threshold_w_cs = 0 #TODO
        
        prob_wo_cs = 1 / (1 + np.exp(-(X_test @ theta_wo_cs)))
        
        prob_w_cs = 1 / (1 + np.exp(-(X_test @ theta_w_cs)))
        
        print_results(Y_test, prob_wo_cs, threshold_wo_cs)
        print_results(Y_test, prob_w_cs, threshold_w_cs)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test or Train")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    main(test = args.test)
    
    
if __name__ == '__main__':
    main()
    

