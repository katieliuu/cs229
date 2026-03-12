"""
logreg_upsample.py trains a logistic regression model on the dataset with naive upsampling.
Regularization is done using L2 regularization and the factor can be altered in the training function.
Upsampling is done by upsampling the minority class to match the majority class.
- naive repetition: repeat the minority class examples to match the majority class
"""

import numpy as np
import pandas as pd
import argparse
from logreg_src import *
import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
from src.upsample import *
from util import *

def main(test: bool = False):
    #Load data
    training_data = pd.read_csv('src/data/model_ready/train_processed.csv')
    print(training_data.shape)
    target_count = training_data[training_data["RIDRETH3_3.0"] == 1.0].shape[0]
    kappa_1 = len(training_data[training_data["RIDRETH3_1.0"] == 1.0]) / target_count
    kappa_4 = len(training_data[training_data["RIDRETH3_4.0"] == 1.0]) / target_count
    kappa_6 = len(training_data[training_data["RIDRETH3_6.0"] == 1.0]) / target_count
    
    kappa_mult_1, kappa_mult_4, kappa_mult_6 = 1.5, 1.0, 1.5 #TODO: add kappa multipliers from cv
    upsampled_training = naive_upsample(training_data, kappa_mult_1 * kappa_1, kappa_mult_4 * kappa_4, kappa_mult_6 * kappa_6) #TODO: add kappa values from cv
    X_upsampled, Y_upsampled = upsampled_training.drop(columns=["diabetes"]).to_numpy(), upsampled_training["diabetes"].to_numpy()
    X_upsampled = add_intercept_fn(X_upsampled)
    #Logreg with naive-upsampled data
    theta_up_naive = logistic_regression(X_upsampled, Y_upsampled, max_iter=5000, lambda_reg=0.0001) #TODO: add hyperparameter from Charlotte's CV results
    
    if test:
        X_test, Y_test = load_csv('src/data/model_ready/test_processed.csv', label_col='diabetes', add_intercept=True)
        threshold_up_naive = 0.35 #TODO
        print(theta_up_naive.shape)
        print(X_test.shape)
        prob_up_naive = 1 / (1 + np.exp(-(X_test @ theta_up_naive)))
        
        print_results(Y_test, prob_up_naive, threshold_up_naive)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test or Train")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    main(test = args.test)
    
    

