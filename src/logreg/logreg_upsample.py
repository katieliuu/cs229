"""
logreg_upsample.py trains a logistic regression model on the dataset with different upsampling methods.
This includes bootstraaping, random top-off, and naive repetition.
It outputs the final loss of the model with regularization.
Regularization is done using L2 regularization and the factor can be altered in the main function.
Upsampling is done by upsampling the minority class to match the majority class.
The upsampling methods are:
- bootstrap: bootstrap the minority class to match the majority class
- random top-off: randomly select a certain number of minority class examples to match the majority class
- naive repetition: repeat the minority class examples to match the majority class
"""

import util
import numpy as np
import argparse
from logreg_src import *


def main():
    
    #Bootstrap upsampling
    X_up_bootstrap, Y_up_bootstrap = util.load_csv('src/data/model_ready/train_processed_upsampled_bootstrap.csv', label_col='diabetes', add_intercept=True)
    print("X_up_bootstrap shape:", X_up_bootstrap.shape)
    print("Y_up_bootstrap shape:", Y_up_bootstrap.shape)
    minority_feature_index = util.return_minority_feature_index('src/data/model_ready/train_processed.csv')
    final_loss_up_bootstrap, theta_up_bootstrap = logistic_regression(X_up_bootstrap, Y_up_bootstrap, max_iter=100000, lambda_reg=10, penalty_weight=1)
    
    #Naive upsampling
    X_up_naive, Y_up_naive = util.load_csv('src/data/model_ready/train_processed_upsampled_naive.csv', label_col='diabetes', add_intercept=True)
    print("X_up_naive shape:", X_up_naive.shape)
    print("Y_up_naive shape:", Y_up_naive.shape)
    minority_feature_index = util.return_minority_feature_index('src/data/model_ready/train_processed.csv')
    final_loss_up_naive, theta_up_naive = logistic_regression(X_up_naive, Y_up_naive, max_iter=100000, lambda_reg=10, penalty_weight=1)
    
    #Random top-off upsampling
    X_up_random_topoff, Y_up_random_topoff = util.load_csv('src/data/model_ready/train_processed_upsampled_random_topoff.csv', label_col='diabetes', add_intercept=True)
    print("X_up_random_topoff shape:", X_up_random_topoff.shape)
    print("Y_up_random_topoff shape:", Y_up_random_topoff.shape)
    minority_feature_index = util.return_minority_feature_index('src/data/model_ready/train_processed.csv')
    final_loss_up_random_topoff, theta_up_random_topoff = logistic_regression(X_up_random_topoff, Y_up_random_topoff, max_iter=100000, lambda_reg=10, penalty_weight=1)
    
    #Print final losses
    print(f'Bootstrap upsampling final loss: {final_loss_up_bootstrap}')
    print(f'Naive upsampling final loss: {final_loss_up_naive}')
    print(f'Random top-off upsampling final loss: {final_loss_up_random_topoff}')
    
if __name__ == '__main__':
    main()
    

