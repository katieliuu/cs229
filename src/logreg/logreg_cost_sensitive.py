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


def main():
    
    X_original, Y_original = util.load_csv('src/data/model_ready/train_raw.csv', label_col='diabetes', add_intercept=True)
    print("X_original shape:", X_original.shape)
    print("Y_original shape:", Y_original.shape)
    minority_feature_index = util.return_minority_feature_index('src/data/model_ready/train_processed.csv')
    
    
    #Regularized without cost-sensitive learning
    train_loss_w_reg_wo_cs, theta_w_reg_wo_cs = logistic_regression(X_original, Y_original, max_iter=100000, lambda_reg=10, penalty_weight=1)
    
    #Regularized with cost-sensitive learning
    X_original_df = pd.read_csv('src/data/model_ready/train_raw.csv')
    sample_weight = util.calculate_sample_weight(X_original_df)
    train_loss_w_reg_w_cs, theta_w_reg_w_cs = logistic_regression(X_original, Y_original, max_iter=100000, lambda_reg=10, sample_weight=sample_weight) #TODO: add hyperparameter from Charlotte's CV results
    
    #Print final losses
    print(f'Regularized without cost-sensitive learning train loss: {train_loss_w_reg_wo_cs}')
    print(f'Regularized with cost-sensitive learning train loss: {train_loss_w_reg_w_cs}')
    
    
if __name__ == '__main__':
    main()
    

