'''
This script trains and tests a logistic regression model with naive upsampled training data.
Args:
    test: bool, whether to test the model
Returns:
    None
To train:
    python logreg_upsample.py
To test:
    python logreg_upsample.py --test
'''
import numpy as np
import pandas as pd
import argparse
from logreg_src import *
import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
from src.upsample import *
from util import *

def main(test: bool = False):
    #Threshold
    threshold_up_naive = 0.35
    
    #Load data
    training_data = pd.read_csv('src/data/model_ready/train_processed.csv')
    target_count = training_data[training_data["RIDRETH3_3.0"] == 1.0].shape[0]
    kappa_1 = len(training_data[training_data["RIDRETH3_1.0"] == 1.0]) / target_count
    kappa_4 = len(training_data[training_data["RIDRETH3_4.0"] == 1.0]) / target_count
    kappa_6 = len(training_data[training_data["RIDRETH3_6.0"] == 1.0]) / target_count
    kappa_mult_1, kappa_mult_4, kappa_mult_6 = 1.5, 1.0, 1.5
    upsampled_training = naive_upsample(training_data, kappa_mult_1 * kappa_1, kappa_mult_4 * kappa_4, kappa_mult_6 * kappa_6)
    X_upsampled, Y_upsampled = upsampled_training.drop(columns=["diabetes"]).to_numpy(), upsampled_training["diabetes"].to_numpy()
    X_upsampled = add_intercept_fn(X_upsampled)
    #Logreg with naive-upsampled data
    theta_up_naive = logistic_regression(X_upsampled, Y_upsampled, max_iter=5000, lambda_reg=0.0001)
    train_probs_up_naive = 1 / (1 + np.exp(-(X_upsampled @ theta_up_naive)))
    _, train_pred_up_naive = f1_from_probs(Y_upsampled, train_probs_up_naive, threshold_up_naive)
    train_accuracy_up_naive = accuracy_score(Y_upsampled, train_pred_up_naive)
    print("Train Accuracy With Naive Upsampling:", train_accuracy_up_naive)
    
    if test:
        X_test, Y_test = load_csv('src/data/model_ready/test_processed.csv', label_col='diabetes', add_intercept=True)
        test_data_df = pd.read_csv('src/data/model_ready/test_processed.csv')
        X_test_df, Y_test_df = test_data_df.drop(columns=["diabetes"]), test_data_df["diabetes"]
        output_model_path = 'src/results/logreg'
        prob_up_naive = 1 / (1 + np.exp(-(X_test @ theta_up_naive)))
        
        evaluate_by_ethnicity(X_test_df, Y_test_df, prob_up_naive, threshold_up_naive, output_model_path=output_model_path, experiment_type='upsample')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test or Train")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    main(test = args.test)
    
    

