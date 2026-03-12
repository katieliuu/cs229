"""
baseline.py trains a logistic regression model on the baseline dataset.
It outputs the final loss and weights of the model without regularization and with regularization.
Regularization is done using L2 regularization and the factor can be altered in the main function.
"""
import util
import numpy as np
import argparse
from logreg_src import *
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from util import *


def main(test: bool = False):
    X_original, Y_original = load_csv('src/data/model_ready/train_processed.csv', label_col='diabetes', add_intercept=True)
    
    #Baseline without regularization
    theta_wo_reg = logistic_regression(X_original, Y_original, max_iter=5000, lambda_reg=0)
    
    #Baseline with regularization
    theta_w_reg = logistic_regression(X_original, Y_original, max_iter=5000, lambda_reg=0.0001) #TODO: add hyperparameter from Charlotte's CV results
    
    
    if test:
        X_test, Y_test = load_csv('src/data/model_ready/test_processed.csv', label_col='diabetes', add_intercept=True)
        threshold_wo_reg = 0.35 #TODO
        threshold_w_reg = 0.35 #TODO
        
        prob_wo_reg = 1 / (1 + np.exp(-(X_test @ theta_wo_reg)))
        print_results(Y_test, prob_wo_reg, threshold_wo_reg)
        
        prob_w_reg = 1 / (1 + np.exp(-(X_test @ theta_w_reg)))
        print_results(Y_test, prob_w_reg, threshold_w_reg)
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test or Train")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    main(test = args.test)
    

