"""
nn_cluster.py trains a Neural Network model on the clustered/upsampled dataset.
It outputs the final training accuracy and evaluates on the test set.

AI Use: Google Gemini was used as a collaborator to help brainstorm
code structure and debug implementation issues.
"""

import pandas as pd
import numpy as np
import argparse
import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
from src.kprototypes import run_k_prototypes
from util import evaluate_by_ethnicity, predict_probs
from neuralnetwork import nn_train, forward_prop, backward_prop, get_initial_params, one_hot_labels


def main(test: bool = False):
    # load data and cluster-upsample
    print("Loading data and running k-prototypes...")
    training_data = pd.read_csv('src/data/model_ready/train_processed.csv')

    # k prototypes parameters
    n_clusters = 4
    gamma = 1.5

    # run k prototypes with params
    cluster_data = run_k_prototypes(training_data, 
                                    max_iter=150, 
                                    n_clusters=n_clusters, 
                                    print_every=10, 
                                    gamma=gamma)
    
    X_cluster = cluster_data.drop(columns=["diabetes"]).to_numpy()
    Y_cluster = cluster_data["diabetes"].to_numpy()
    
    # convert labels to one-hot for the neural network
    Y_train_oh = one_hot_labels(Y_cluster, num_classes=2)
    
    # network hyperparameters
    hidden_width = 64
    learning_rate = 0.05
    num_epochs = 50
    batch_size = 64
    activation_func = 'relu'
    weight_decay = 0.001
    dropout = 0.0

    print("Training Neural Network with Cluster Data...")
    params_cluster, _, _, acc_train_cluster, _ = nn_train(
        train_data=X_cluster, train_labels=Y_train_oh,
        dev_data=X_cluster, dev_labels=Y_train_oh, # using train as dev just to satisfy signature
        get_initial_params_func=get_initial_params,
        forward_prop_func=forward_prop,
        backward_prop_func=backward_prop,
        num_hidden=hidden_width, learning_rate=learning_rate, 
        num_epochs=num_epochs, batch_size=batch_size, num_classes=2,
        activation=activation_func, dropout_rate=dropout, reg=weight_decay
    )
    
    # print the accuracy from the final epoch
    print("Train Accuracy With Cluster Data:", acc_train_cluster[-1])
    
    if test:
        print("\nEvaluating on Test Set...")
        testing_data = pd.read_csv('src/data/model_ready/test_processed.csv')
        
        # kept dataframe to pass to evaluate_by_ethnicity
        X_test_df = testing_data.drop(columns=["diabetes"])
        X_test = X_test_df.to_numpy()
        Y_test = testing_data["diabetes"].to_numpy()
        
        # standardized output paths for util func
        output_dir = "src/results/nn"
        os.makedirs(output_dir, exist_ok=True)
    
        threshold_cluster = 0.35
        probs_cluster = predict_probs(X_test, params_cluster, activation=activation_func)
        
        # use evaluate_by_ethnicity 
        evaluate_by_ethnicity(
            X_test_df, 
            Y_test, 
            probs_cluster, 
            threshold_cluster,
            output_model_path=output_dir, 
            experiment_type="cluster"
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test or Train Neural Network Cluster Setup")
    parser.add_argument("--test", action="store_true", help="Evaluate the model on the test set")
    args = parser.parse_args()
    
    main(test=args.test)