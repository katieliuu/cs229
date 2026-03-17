"""
nn_cost_sensitive.py trains a Neural Network model on the dataset with cost-sensitive learning.
It outputs the final training accuracy and evaluates on the test set.
Cost-sensitive learning is done by upweighting the minority class misclassification using sample weights.

AI Use: Google Gemini was used as a collaborator to help brainstorm
code structure and debug implementation issues.
"""

import pandas as pd
import numpy as np
import argparse
import os
from util import calculate_sample_weight, evaluate_by_ethnicity, predict_probs
from neuralnetwork import nn_train, forward_prop, backward_prop, get_initial_params, one_hot_labels


def main(test: bool = False):
    # load data
    training_data = pd.read_csv('src/data/model_ready/train_processed.csv')
    X_train = training_data.drop(columns=["diabetes"]).to_numpy()
    Y_train = training_data["diabetes"].to_numpy()
    
    # convert labels to one-hot for the neural network
    Y_train_oh = one_hot_labels(Y_train, num_classes=2)
    
    # cost sensitive parameters / weights
    sample_weights = calculate_sample_weight(training_data)
    
    # ensure it's a numpy array for the neural network
    if isinstance(sample_weights, pd.Series):
        sample_weights = sample_weights.to_numpy()
    else:
        sample_weights = np.array(sample_weights)
    
    # network hyperparameters
    hidden_width = 64
    learning_rate = 0.05
    num_epochs = 50
    batch_size = 64
    activation_func = 'relu'
    weight_decay = 0.001
    dropout = 0.0

    print("Training Neural Network without Cost-Sensitive Learning...")
    params_wo_cs, _, _, acc_train_wo, _ = nn_train(
        train_data=X_train, train_labels=Y_train_oh,
        dev_data=X_train, dev_labels=Y_train_oh, # using train as dev just to satisfy signature
        get_initial_params_func=get_initial_params,
        forward_prop_func=forward_prop,
        backward_prop_func=backward_prop,
        num_hidden=hidden_width, learning_rate=learning_rate, 
        num_epochs=num_epochs, batch_size=batch_size, num_classes=2,
        activation=activation_func, dropout_rate=dropout, reg=weight_decay,
        sample_weights=None
    )

    print("\nTraining Neural Network with Cost-Sensitive Learning...")
    params_w_cs, _, _, acc_train_w, _ = nn_train(
        train_data=X_train, train_labels=Y_train_oh,
        dev_data=X_train, dev_labels=Y_train_oh,
        get_initial_params_func=get_initial_params,
        forward_prop_func=forward_prop,
        backward_prop_func=backward_prop,
        num_hidden=hidden_width, learning_rate=learning_rate, 
        num_epochs=num_epochs, batch_size=batch_size, num_classes=2,
        activation=activation_func, dropout_rate=dropout, reg=weight_decay,
        sample_weights=sample_weights
    )
    
    # print the accuracy from the final epoch
    print("\nFinal Train Accuracy Without Cost-Sensitive:", acc_train_wo[-1])
    print("Final Train Accuracy With Cost-Sensitive:", acc_train_w[-1])
    
    if test:
        print("\nEvaluating on Test Set...")
        testing_data = pd.read_csv('src/data/model_ready/test_processed.csv')
        
        # kept dataframe to pass to evaluate_by_ethnicity
        X_test_df = testing_data.drop(columns=["diabetes"])
        X_test = X_test_df.to_numpy()
        Y_test = testing_data["diabetes"].to_numpy()
        
        # added output directories
        output_dir = "src/results/nn"
        os.makedirs(output_dir, exist_ok=True)
        
        # threshold_wo_cs = 0.35
        threshold_w_cs = 0.45
        
        # probs_wo_cs = predict_probs(X_test, params_wo_cs, activation=activation_func)
        probs_w_cs = predict_probs(X_test, params_w_cs, activation=activation_func)
        
        # print("\n--- Results Without Cost-Sensitive Learning ---")
        # use evaluate_by_ethnicity
        # evaluate_by_ethnicity(X_test_df, Y_test, probs_wo_cs, threshold_wo_cs, output_model_path=output_dir, experiment_type="baseline_wo_cs")
        
        print("\n--- Results With Cost-Sensitive Learning ---")
        # use evaluate_by_ethnicity
        evaluate_by_ethnicity(X_test_df, Y_test, probs_w_cs, threshold_w_cs, output_model_path=output_dir, experiment_type="cost_sensitive")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test or Train Neural Network Cost Sensitive Setup")
    parser.add_argument("--test", action="store_true", help="Evaluate the model on the test set")
    args = parser.parse_args()
    
    main(test=args.test)