"""
This script trains and tests a logistic regression model with GMM clustered and naive
upsample training data.
Args:
    test: bool, whether to test the model
Returns:
    None
To train:
    python logreg_gmm.py
To test:
    python logreg_gmm.py --test
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

def main(test: bool = False):
    #Threshold
    threshold_w_gmm = 0.35
    
    # Load data
    training_data = pd.read_csv('src/data/model_ready/train_processed.csv')
    gmm_data = gmm_cluster_upsample(training_data, max_iter=150, n_components=2)
    X_gmm, Y_gmm = gmm_data.drop(columns=["diabetes"]).to_numpy(), gmm_data["diabetes"].to_numpy()
    X_gmm = add_intercept_fn(X_gmm)
    #Logistic regression with cluster data
    theta_w_gmm = logistic_regression(X_gmm, Y_gmm, max_iter=5000, lambda_reg=0.0) 
    train_probs_w_gmm = 1 / (1 + np.exp(-(X_gmm @ theta_w_gmm)))
    _, train_pred_w_gmm = f1_from_probs(Y_gmm, train_probs_w_gmm, threshold_w_gmm)
    train_accuracy_w_gmm = accuracy_score(Y_gmm, train_pred_w_gmm)
    print("Train Accuracy With GMM Data:", train_accuracy_w_gmm)
    
    if test:
        X_test, Y_test = load_csv('src/data/model_ready/test_processed.csv', label_col='diabetes', add_intercept=True)
        test_data_df = pd.read_csv('src/data/model_ready/test_processed.csv')
        X_test_df, Y_test_df = test_data_df.drop(columns=["diabetes"]), test_data_df["diabetes"]
        output_model_path = 'src/results/logreg'
        prob_w_gmm = 1 / (1 + np.exp(-(X_test @ theta_w_gmm)))
        evaluate_by_ethnicity(X_test_df, Y_test_df, prob_w_gmm, threshold_w_gmm, output_model_path=output_model_path, experiment_type='gmm')
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test or Train")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    main(test = args.test)
    



