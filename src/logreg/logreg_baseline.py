"""
baseline.py trains a logistic regression model on the baseline dataset.
It outputs the final loss and weights of the model without regularization and with regularization.
Regularization is done using L2 regularization and the factor can be altered in the main function.
"""
import util
import numpy as np
import argparse
from logreg_src import *


def main():

    X_original, Y_original = util.load_csv('src/data/model_ready/train_raw.csv', label_col='diabetes', add_intercept=True)
    print("X_original shape:", X_original.shape)
    print("Y_original shape:", Y_original.shape)
    
    #Baseline without regularization
    train_loss_wo_reg, theta_wo_reg = logistic_regression(X_original, Y_original, max_iter=100000, lambda_reg=0)
    
    #Baseline with regularization
    train_loss_w_reg, theta_w_reg = logistic_regression(X_original, Y_original, max_iter=100000, lambda_reg=10) #TODO: add hyperparameter from Charlotte's CV results
    
    print(f'Baseline without regularization train loss: {train_loss_wo_reg}')
    print(f'Baseline with regularization train loss: {train_loss_w_reg}')
    
    
    
if __name__ == '__main__':
    main()
    

