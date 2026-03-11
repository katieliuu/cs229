import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
from tree_src import *
from kprototypes import run_k_prototypes
from cv_logreg import f1_from_probs
import argparse
import util
import matplotlib.pyplot as plt

def main(test: bool = False):
    #Train Model
    X_original, Y_original = util.load_csv('src/data/model_ready/train_processed.csv', label_col='diabetes', add_intercept=True)
    print("X_original shape:", X_original.shape)
    print("Y_original shape:", Y_original.shape)
    
    training_data = np.concatenate((X_original, Y_original), axis=1)
    cluster_data = run_k_prototypes(training_data, max_iter=150, n_clusters=3, print_every=10, gamma=1.0)#TODO: add hyperparameter from Charlotte's CV results
    X_cluster, Y_cluster = cluster_data.drop(columns=["diabetes"]), cluster_data["diabetes"]
    
    #Decision tree with cluster data
    model_cluster, train_accuracy_cluster = decision_tree(X_cluster, Y_cluster, max_depth=None, min_samples_split=2, min_samples_leaf=1, sample_weight=None) #TODO: add hyperparameter from Charlotte's CV results
    
    if test:
        threshold_cluster = 0 #TODO
        X_test, Y_test = util.load_csv('src/data/model_ready/test_processed.csv', label_col='diabetes', add_intercept=True)
        Y_pred_cluster, test_accuracy_cluster, probs_cluster = test_tree(model_cluster, X_test, Y_test)
        f1_cluster, precision_cluster, recall_cluster, tp_cluster, fp_cluster, tn_cluster, fn_cluster, preds_cluster = f1_from_probs(Y_test, probs_cluster, threshold_cluster)
        #TODO: implement other metrics (confusion matrix, accuracy, f1)
    
        #Confusion Matrix
        cm = confusion_matrix(Y_test, preds_cluster, labels=['Diabetic', 'Non-Diabetic'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Diabetic', 'Non-Diabetic'])
        disp.plot()
        plt.show()
        #Accuracy
        accuracy = accuracy_score(Y_test, preds_cluster)
        print(f'Cluster Data Accuracy: {accuracy}')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test or Train")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    main(test = args.test)
    
    