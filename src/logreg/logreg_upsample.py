"""
logreg_upsample.py trains a logistic regression model on the dataset with naive upsampling.
Regularization is done using L2 regularization and the factor can be altered in the training function.
Upsampling is done by upsampling the minority class to match the majority class.
- naive repetition: repeat the minority class examples to match the majority class
"""

import util
import numpy as np
import pandas as pd
import argparse
from logreg_src import *
from upsample import *
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from util import *

def main(test: bool = False):
    #Load data
    training_data = pd.read_csv('src/data/model_ready/train_processed.csv')
    kappa_1, kappa_4, kappa_6 = get_natural_kappas(training_data)
    kappa_mult_1, kappa_mult_4, kappa_mult_6 = 1.5, 1.0, 1.0 #TODO: add kappa multipliers from cv
    upsampled_training = naive_upsample(training_data, kappa_mult_1 * kappa_1, kappa_mult_4 * kappa_4, kappa_mult_6 * kappa_6) #TODO: add kappa values from cv
    X_upsampled, Y_upsampled = upsampled_training.drop(columns=["diabetes"]).to_numpy(), upsampled_training["diabetes"].to_numpy()
    
    #Logreg with naive-upsampled data
    theta_up_naive = logistic_regression(X_upsampled, Y_upsampled, max_iter=5000, lambda_reg=10) #TODO: add hyperparameter from Charlotte's CV results
    
    if test:
        X_test, Y_test = load_csv('src/data/model_ready/test_processed.csv', label_col='diabetes', add_intercept=True)
        threshold_up_naive = 0 #TODO
        
        prob_up_naive = 1 / (1 + np.exp(-(X_test @ theta_up_naive)))
        
        print_results(Y_test, prob_up_naive, threshold_up_naive)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test or Train")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    main(test = args.test)
    
    

