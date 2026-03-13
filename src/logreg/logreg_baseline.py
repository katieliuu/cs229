import numpy as np
import argparse
from logreg_src import *
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from util import *


def main(test: bool = False):
    #Threshold
    threshold_wo_reg = 0.5
    threshold_w_reg = 0.35
    
    X_original, Y_original = load_csv('src/data/model_ready/train_processed.csv', label_col='diabetes', add_intercept=True)
    
    #Baseline without regularization
    theta_wo_reg = logistic_regression(X_original, Y_original, max_iter=5000, lambda_reg=0)
    train_probs_wo_reg = 1 / (1 + np.exp(-(X_original @ theta_wo_reg)))
    _, train_pred_wo_reg = f1_from_probs(Y_original, train_probs_wo_reg, threshold_wo_reg)
    train_accuracy_wo_reg = accuracy_score(Y_original, train_pred_wo_reg)
    print("Train Accuracy Without Regularization:", train_accuracy_wo_reg)
    
    #Baseline with regularization
    theta_w_reg = logistic_regression(X_original, Y_original, max_iter=5000, lambda_reg=0.0001)
    train_probs_w_reg = 1 / (1 + np.exp(-(X_original @ theta_w_reg)))
    _, train_pred_w_reg = f1_from_probs(Y_original, train_probs_w_reg, threshold_w_reg)
    train_accuracy_w_reg = accuracy_score(Y_original, train_pred_w_reg)
    print("Train Accuracy With Regularization:", train_accuracy_w_reg)
    
    if test:
        X_test, Y_test = load_csv('src/data/model_ready/test_processed.csv', label_col='diabetes', add_intercept=True)
        test_data_df = pd.read_csv('src/data/model_ready/test_processed.csv')
        X_test_df, Y_test_df = test_data_df.drop(columns=["diabetes"]), test_data_df["diabetes"]
        
        output_model_path = 'src/results/logreg'
        
        prob_wo_reg = 1 / (1 + np.exp(-(X_test @ theta_wo_reg)))
        evaluate_by_ethnicity(X_test_df, Y_test_df, prob_wo_reg, threshold_wo_reg, output_model_path=output_model_path, experiment_type='baseline_wo_reg')
        
        prob_w_reg = 1 / (1 + np.exp(-(X_test @ theta_w_reg)))
        evaluate_by_ethnicity(X_test_df, Y_test_df, prob_w_reg, threshold_w_reg, output_model_path=output_model_path, experiment_type='baseline_w_reg')
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test or Train")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    main(test = args.test)
    

