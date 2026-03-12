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
from src.kprototypes import run_k_prototypes
from util import *

def main(test: bool = False):
    # Load data
    training_data = pd.read_csv('src/data/model_ready/train_processed.csv')
    cluster_data = run_k_prototypes(training_data, max_iter=150, n_clusters=3, print_every=10, gamma=1.0)#TODO: add hyperparameter from Charlotte's CV results
    X_cluster, Y_cluster = cluster_data.drop(columns=["diabetes"]).to_numpy(), cluster_data["diabetes"].to_numpy()
    X_cluster = add_intercept_fn(X_cluster)
    #Logistic regression with cluster data
    theta_w_cluster = logistic_regression(X_cluster, Y_cluster, max_iter=5000, lambda_reg=10) #TODO: add hyperparameter from Charlotte's CV results
    
    
    if test:
        X_test, Y_test = load_csv('src/data/model_ready/test_processed.csv', label_col='diabetes', add_intercept=True)
        threshold_w_cluster = 0 #TODO
        
        prob_w_cluster = 1 / (1 + np.exp(-(X_test @ theta_w_cluster)))
        
        print_results(Y_test, prob_w_cluster, threshold_w_cluster)
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test or Train")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    main(test = args.test)
    



