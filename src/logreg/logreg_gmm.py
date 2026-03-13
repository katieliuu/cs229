"""
baseline.py trains a logistic regression model on the baseline dataset.
It outputs the final loss and weights of the model without regularization and with regularization.
Regularization is done using L2 regularization and the factor can be altered in the main function.
"""
from util import *
import numpy as np
import argparse
from logreg_src import *
import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
from src.gmm import gmm_cluster_upsample
from util import *

def main(test: bool = False):
    # Load data
    training_data = pd.read_csv('src/data/model_ready/train_processed.csv')
    gmm_data = gmm_cluster_upsample(training_data, max_iter=150, n_components=2)
    X_gmm, Y_gmm = gmm_data.drop(columns=["diabetes"]).to_numpy(), gmm_data["diabetes"].to_numpy()
    X_gmm = add_intercept_fn(X_gmm)
    #Logistic regression with cluster data
    theta_w_gmm = logistic_regression(X_gmm, Y_gmm, max_iter=5000, lambda_reg=0.0) 
    
    
    if test:
        X_test, Y_test = load_csv('src/data/model_ready/test_processed.csv', label_col='diabetes', add_intercept=True)
        threshold_w_gmm = 0.35
        
        prob_w_gmm = 1 / (1 + np.exp(-(X_test @ theta_w_gmm)))
        
        print_results(Y_test, prob_w_gmm, threshold_w_gmm)
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test or Train")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    main(test = args.test)
    



