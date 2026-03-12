import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
from tree_src import *
from util import *
import argparse
import matplotlib.pyplot as plt

def main(test: bool = False):
    #Load data
    training_data = pd.read_csv('src/data/model_ready/train_processed.csv')
    X_original, Y_original = training_data.drop(columns=["diabetes"]), training_data["diabetes"]
    
    #Baseline with regularization without cost sensitive learning
    model_wo_cs, train_accuracy_wo_cs = decision_tree(X_original, Y_original, max_depth=None, min_samples_split=2, min_samples_leaf=1)
    
    #baseline with regularizationwith cost sensitive learning
    sample_weight = calculate_sample_weight(training_data)
    model_w_cs, train_accuracy_w_cs = decision_tree(X_original, Y_original, max_depth=None, min_samples_split=2, min_samples_leaf=1, sample_weight=sample_weight)
    
    if test: 
        testing_data = pd.read_csv('src/data/model_ready/test_processed.csv')
        X_test, Y_test = testing_data.drop(columns=["diabetes"]), testing_data["diabetes"]
        
        threshold_wo_cs = 0.25
        probs_wo_cs = test_tree(model_wo_cs, X_test, Y_test)
        
        threshold_w_cs = 0.25 
        probs_w_cs = test_tree(model_w_cs, X_test, Y_test)
    
        print_results(Y_test, probs_w_cs, threshold_w_cs)
        print_results(Y_test, probs_wo_cs, threshold_wo_cs)
        
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test or Train")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    main(test = args.test)
    
    
    