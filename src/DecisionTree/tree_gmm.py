import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
from tree_src import *
import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
from src.gmm import gmm_cluster_upsample
import argparse
from util import *
import matplotlib.pyplot as plt

def main(test: bool = False):
    #Threshold
    threshold_gmm = 0.25
    
    #Load data and cluster-upsample
    training_data = pd.read_csv('src/data/model_ready/train_processed.csv')
    gmm_data = gmm_cluster_upsample(training_data, max_iter=150, n_components=5)
    X_gmm, Y_gmm = gmm_data.drop(columns=["diabetes"]), gmm_data["diabetes"]
    
    #Decision tree with cluster data
    model_gmm, train_accuracy_gmm = decision_tree(X_gmm, Y_gmm, max_depth=30, min_samples_split=2, min_samples_leaf=1, sample_weight=None, threshold=threshold_gmm)
    print("Train Accuracy With GMM Data:", train_accuracy_gmm)
    
    if test:
        testing_data = pd.read_csv('src/data/model_ready/test_processed.csv')
        X_test, Y_test = testing_data.drop(columns=["diabetes"]), testing_data["diabetes"]
        
        probs_gmm = test_tree(model_gmm, X_test, Y_test)
    
        print_results(Y_test, probs_gmm, threshold_gmm)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test or Train")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    main(test = args.test)