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
    #Threshold
    threshold_w_cluster = 0.35
    
    # Load data
    training_data = pd.read_csv('src/data/model_ready/train_processed.csv')
    cluster_data = run_k_prototypes(training_data, max_iter=150, n_clusters=6, print_every=10, gamma=1.0)
    X_cluster, Y_cluster = cluster_data.drop(columns=["diabetes"]).to_numpy(), cluster_data["diabetes"].to_numpy()
    X_cluster = add_intercept_fn(X_cluster)
    #Logistic regression with cluster data
    theta_w_cluster = logistic_regression(X_cluster, Y_cluster, max_iter=5000, lambda_reg=0.001)
    train_probs_w_cluster = 1 / (1 + np.exp(-(X_cluster @ theta_w_cluster)))
    _, train_pred_w_cluster = f1_from_probs(Y_cluster, train_probs_w_cluster, threshold_w_cluster)
    train_accuracy_w_cluster = accuracy_score(Y_cluster, train_pred_w_cluster)
    print("Train Accuracy With Cluster Data:", train_accuracy_w_cluster)
    
    if test:
        X_test, Y_test = load_csv('src/data/model_ready/test_processed.csv', label_col='diabetes', add_intercept=True)
        test_data_df = pd.read_csv('src/data/model_ready/test_processed.csv')
        X_test_df, Y_test_df = test_data_df.drop(columns=["diabetes"]), test_data_df["diabetes"]
        output_model_path = 'src/results/logreg'
        prob_w_cluster = 1 / (1 + np.exp(-(X_test @ theta_w_cluster)))
        
        evaluate_by_ethnicity(X_test_df, Y_test_df, prob_w_cluster, threshold_w_cluster, output_model_path=output_model_path, experiment_type='cluster')
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test or Train")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    main(test = args.test)
    



