import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
from tree_src import *
import util
import numpy as np
import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
from src.upsample import *
import argparse
import matplotlib.pyplot as plt

def main(test: bool = False):
    #Load data
    training_data = pd.read_csv('src/data/model_ready/train_processed.csv')
    kappa_1, kappa_4, kappa_6 = get_natural_kappas(training_data)
    kappa_mult_1, kappa_mult_4, kappa_mult_6 = 1.5, 1.0, 1.0
    upsampled_training = naive_upsample(training_data, kappa_mult_1 * kappa_1, kappa_mult_4 * kappa_4, kappa_mult_6 * kappa_6)
    
    X_upsampled, Y_upsampled = upsampled_training.drop(columns=["diabetes"]), upsampled_training["diabetes"]
    
    #Decision tree with upsampled data
    model_upsampled, train_accuracy_upsampled = decision_tree(X_upsampled, Y_upsampled, max_depth=None, min_samples_split=2, min_samples_leaf=1, sample_weight=None)
    
    print("Train Accuracy With Upsampled Data:", train_accuracy_upsampled)
    if test:
        testing_data = pd.read_csv('src/data/model_ready/test_processed.csv')
        X_test, Y_test = testing_data.drop(columns=["diabetes"]), testing_data["diabetes"]
        
        threshold_upsampled = 0.25
        probs_upsampled = test_tree(model_upsampled, X_test, Y_test)
        
        print_results(Y_test, probs_upsampled, threshold_upsampled)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test or Train")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    main(test = args.test)
    
    
    