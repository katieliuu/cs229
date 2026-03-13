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
from util import *
import matplotlib.pyplot as plt

def main(test: bool = False):
    #Threshold
    threshold_cluster = 0.45
    #Load data and cluster-upsample
    training_data = pd.read_csv('src/data/model_ready/train_processed.csv')
    cluster_data = run_k_prototypes(training_data, max_iter=150, n_clusters=5, print_every=10, gamma=1.0)
    X_cluster, Y_cluster = cluster_data.drop(columns=["diabetes"]), cluster_data["diabetes"]
    
    #Decision tree with cluster data
    model_cluster, train_accuracy_cluster = decision_tree(X_cluster, Y_cluster, max_depth=30, min_samples_split=2, min_samples_leaf=1, sample_weight=None, threshold=threshold_cluster)
    print("Train Accuracy With Cluster Data:", train_accuracy_cluster)
    
    if test:
        testing_data = pd.read_csv('src/data/model_ready/test_processed.csv')
        X_test, Y_test = testing_data.drop(columns=["diabetes"]), testing_data["diabetes"]
        output_model_path = 'src/results/DecisionTree'
        probs_cluster = test_tree(model_cluster, X_test, Y_test)
        evaluate_by_ethnicity(X_test, Y_test, probs_cluster, threshold_cluster, output_model_path=output_model_path, experiment_type='cluster')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test or Train")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    main(test = args.test)
    
    