"""
nn_baseline.py trains a Neural Network model on the baseline dataset.
It outputs the final training accuracy and evaluates on the test set
with and without regularization aka L2 weight decay and dropout.

AI Use: Google Gemini was used as a collaborator to help brainstorm
code structure and debug implementation issues.
"""

import pandas as pd
import numpy as np
import argparse
import os
from util import evaluate_by_ethnicity, predict_probs
from neuralnetwork import nn_train, forward_prop, backward_prop, get_initial_params, one_hot_labels


def main(test: bool = False):
    # load data
    training_data = pd.read_csv('src/data/model_ready/train_processed.csv')
    X_train = training_data.drop(columns=["diabetes"]).to_numpy()
    Y_train = training_data["diabetes"].to_numpy()
    
    # convert labels to one-hot for the neural network
    Y_train_oh = one_hot_labels(Y_train, num_classes=2)
    
    # network hyperparameters
    hidden_width = 64
    learning_rate = 0.05
    num_epochs = 50
    batch_size = 64
    dropout = 0.0
    activation_func = 'relu'

    print("Training Baseline Neural Network (Without Regularization)...")
    # baseline without regularization
    params_wo_reg, _, _, acc_train_wo, _ = nn_train(
        train_data=X_train, train_labels=Y_train_oh,
        dev_data=X_train, dev_labels=Y_train_oh, # using train as dev
        get_initial_params_func=get_initial_params,
        forward_prop_func=forward_prop,
        backward_prop_func=backward_prop,
        num_hidden=hidden_width, learning_rate=learning_rate, 
        num_epochs=num_epochs, batch_size=batch_size, num_classes=2,
        activation=activation_func, dropout_rate=dropout, reg=0.0
    )
    
    print("Training Regularized Neural Network...")
    # baseline with regularization
    params_w_reg, _, _, acc_train_w, _ = nn_train(
        train_data=X_train, train_labels=Y_train_oh,
        dev_data=X_train, dev_labels=Y_train_oh,
        get_initial_params_func=get_initial_params,
        forward_prop_func=forward_prop,
        backward_prop_func=backward_prop,
        num_hidden=hidden_width, learning_rate=learning_rate, 
        num_epochs=num_epochs, batch_size=batch_size, num_classes=2,
        activation=activation_func, dropout_rate=dropout, reg=0.001
    )

    # nn_train returns a list of accuracies per epoch; we want the final one
    print("\nFinal Train Accuracy Without Regularization:", acc_train_wo[-1])
    print("Final Train Accuracy With Regularization:", acc_train_w[-1])
    
    if test:
        print("\nEvaluating on Test Set...")
        testing_data = pd.read_csv('src/data/model_ready/test_processed.csv')
        
        # pass dataframe to evaluate_by_ethnicity
        X_test_df = testing_data.drop(columns=["diabetes"])
        X_test = X_test_df.to_numpy()
        Y_test = testing_data["diabetes"].to_numpy()
        
        # make sure output directory exists before util.py tries to save to it
        output_dir = "src/results/nn"
        os.makedirs(output_dir, exist_ok=True)
        
        # test unregularized model
        threshold_wo_reg = 0.5
        probs_wo_reg = predict_probs(X_test, params_wo_reg, activation=activation_func)
        print("\n--- Results Without Regularization ---")
        
        # evaluate_by_ethnicity w X_test_df
        evaluate_by_ethnicity(
            X_test_df,
            Y_test, 
            probs_wo_reg, 
            threshold_wo_reg, 
            output_model_path=output_dir, 
            experiment_type="baseline_wo_reg"
        )
        
        # test regularized model
        threshold_w_reg = 0.35
        probs_w_reg = predict_probs(X_test, params_w_reg, activation=activation_func)
        print("\n--- Results With Regularization ---")
        
        # evaluate_by_ethnicity w X_test_df
        evaluate_by_ethnicity(
            X_test_df,
            Y_test, 
            probs_w_reg, 
            threshold_w_reg, 
            output_model_path=output_dir, 
            experiment_type="baseline_w_reg"
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test or Train Neural Network Baseline")
    parser.add_argument("--test", action="store_true", help="Evaluate the model on the test set")
    args = parser.parse_args()
    
    main(test=args.test)