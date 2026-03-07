import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tree_src import *
import util

def main():
    #Train Model
    X_cluster, Y_cluster = util.load_csv('src/data/model_ready/train_upsampled_scratch_kprototypes.csv', label_col='diabetes', add_intercept=True)
    print("X_cluster shape:", X_cluster.shape)
    print("Y_cluster shape:", Y_cluster.shape)
    
    #Decision tree with cluster data
    model_cluster, train_accuracy_cluster = decision_tree(X_cluster, Y_cluster, max_depth=None, min_samples_split=2, min_samples_leaf=1, sample_weight=None) #TODO: add hyperparameter from Charlotte's CV results
    
    print(f'Decision tree with cluster data train accuracy: {train_accuracy_cluster}')
    
    