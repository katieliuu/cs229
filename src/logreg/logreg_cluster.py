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

    X_cluster, Y_cluster = util.load_csv('src/data/model_ready/train_upsampled_scratch_kprototypes.csv', label_col='diabetes', add_intercept=True)
    print("X_cluster shape:", X_cluster.shape)
    print("Y_cluster shape:", Y_cluster.shape)
    
    #Logistic regression with cluster data
    train_loss_cluster, theta_w_cluster = logistic_regression(X_cluster, Y_cluster, max_iter=5000, lambda_reg=10) #TODO: add hyperparameter from Charlotte's CV results
    
    print(f'Logistic regression with cluster data train loss: {train_loss_cluster}')
    
    
    
if __name__ == '__main__':
    main()
    



