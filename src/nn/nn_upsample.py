"""
nn_upsample.py trains a Neural Network model on the dataset with naive upsampling.
It outputs the final training accuracy and evaluates on the test set.
Upsampling is done by duplicating minority class examples based on natural kappas.

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
from src.upsample import naive_upsample
from util import evaluate_by_ethnicity, predict_probs
from neuralnetwork import nn_train, forward_prop, backward_prop, get_initial_params, one_hot_labels


def main(test: bool = False):
    # load data
    print("Loading data and applying naive upsampling...")
    training_data = pd.read_csv('src/data/model_ready/train_processed.csv')
    
    # calculate kappas for upsampling
    target_count = training_data[training_data["RIDRETH3_3.0"] == 1.0].shape[0]
    kappa_1 = len(training_data[training_data["RIDRETH3_1.0"] == 1.0]) / target_count
    kappa_4 = len(training_data[training_data["RIDRETH3_4.0"] == 1.0]) / target_count
    kappa_6 = len(training_data[training_data["RIDRETH3_6.0"] == 1.0]) / target_count
    
    # upsampling parameters
    kappa_mult_1 = 1.0
    kappa_mult_4 = 1.5
    kappa_mult_6 = 1.5
    
    # apply naive upsampling
    upsampled_training = naive_upsample(
        training_data, 
        kappa_mult_1 * kappa_1, 
        kappa_mult_4 * kappa_4, 
        kappa_mult_6 * kappa_6
    )
    
    X_upsampled = upsampled_training.drop(columns=["diabetes"]).to_numpy()
    Y_upsampled = upsampled_training["diabetes"].to_numpy()
    
    # convert labels to one-hot for the neural network
    Y_train_oh = one_hot_labels(Y_upsampled, num_classes=2)
    
    # network hyperparameters
    hidden_width = 64
    learning_rate = 0.05
    num_epochs = 50
    batch_size = 64
    activation_func = 'relu'
    weight_decay = 0.001
    dropout = 0.0

    print("Training Neural Network with Naively Upsampled Data...")
    params_up, _, _, acc_train_up, _ = nn_train(
        train_data=X_upsampled, train_labels=Y_train_oh,
        dev_data=X_upsampled, dev_labels=Y_train_oh, # using train as dev just to satisfy signature
        get_initial_params_func=get_initial_params,
        forward_prop_func=forward_prop,
        backward_prop_func=backward_prop,
        num_hidden=hidden_width, learning_rate=learning_rate, 
        num_epochs=num_epochs, batch_size=batch_size, num_classes=2,
        activation=activation_func, dropout_rate=dropout, reg=weight_decay
    )
    
    # print the accuracy from the final epoch
    print("Train Accuracy With Upsampled Data:", acc_train_up[-1])
    
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
        
        threshold_upsampled = 0.35
        probs_upsampled = predict_probs(X_test, params_up, activation=activation_func)
        
        # switched to evaluate_by_ethnicity
        evaluate_by_ethnicity(X_test_df, Y_test, probs_upsampled, threshold_upsampled, output_model_path=output_dir, experiment_type="upsample")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test or Train Neural Network Upsampled Setup")
    parser.add_argument("--test", action="store_true", help="Evaluate the model on the test set")
    args = parser.parse_args()
    
    main(test=args.test)