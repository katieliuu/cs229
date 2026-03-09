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


def main():
    
    #Naive upsampling
    X_original, Y_original = util.load_csv('src/data/model_ready/train_raw.csv', label_col='diabetes', add_intercept=True)
    print("X_original shape:", X_original.shape)
    print("Y_original shape:", Y_original.shape)
    
    training_data = np.concatenate((X_original, Y_original), axis=1)
    upsampled_training = naive_upsample(training_data, kappa_1, kappa_4, kappa_6) #TODO: add kappa values from cv
    X_upsampled, Y_upsampled = upsampled_training.drop(columns=["diabetes"]), upsampled_training["diabetes"]
    theta_up_naive = logistic_regression(X_upsampled, Y_upsampled, max_iter=5000, lambda_reg=10, penalty_weight=1) #TODO: add hyperparameter from Charlotte's CV results
    
    
    
if __name__ == '__main__':
    main()
    

