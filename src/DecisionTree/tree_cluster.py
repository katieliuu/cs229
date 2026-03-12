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
from src.kprototypes import run_k_prototypes
import argparse
import util
import matplotlib.pyplot as plt

def main(test: bool = False):
    #Load data and cluster-upsample
    training_data = pd.read_csv('src/data/model_ready/train_processed.csv')
    cluster_data = run_k_prototypes(training_data, max_iter=150, n_clusters=5, print_every=10, gamma=1.0) #TODO: hyperparameters FILLED
    X_cluster, Y_cluster = cluster_data.drop(columns=["diabetes"]), cluster_data["diabetes"]
    
    #Decision tree with cluster data
    model_cluster, train_accuracy_cluster = decision_tree(X_cluster, Y_cluster, max_depth=20, min_samples_split=2, min_samples_leaf=1, sample_weight=None) #TODO: hyperparameters FILLED
    print(model_cluster.get_depth())
    print("Train Accuracy With Cluster Data:", train_accuracy_cluster)
    
    if test:
        testing_data = pd.read_csv('src/data/model_ready/test_processed.csv')
        X_test, Y_test = testing_data.drop(columns=["diabetes"]), testing_data["diabetes"]
        
        threshold_cluster = 0.25
        probs_cluster = test_tree(model_cluster, X_test, Y_test)
    
        print_results(Y_test, probs_cluster, threshold_cluster)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test or Train")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    main(test = args.test)
    
    