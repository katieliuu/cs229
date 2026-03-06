import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tree_src import *
import util

def main():
    #Train Model
    X_original, Y_original = util.load_csv('src/data/model_ready/train_raw.csv', label_col='diabetes', add_intercept=True)
    print("X_original shape:", X_original.shape)
    print("Y_original shape:", Y_original.shape)
    
    #baseline without regularization
    model, train_accuracy = decision_tree(X_original, Y_original) #TODO: add hyperparameter from Charlotte's CV results
    
    #baseline with regularization
    model_w_reg, train_accuracy_w_reg = decision_tree(X_original, Y_original, max_depth=None, min_samples_split=2, min_samples_leaf=1) #TODO: add hyperparameter from Charlotte's CV results
    
    print(f'Baseline without regularization train accuracy: {train_accuracy}')
    print(f'Baseline with regularization train accuracy: {train_accuracy_w_reg}')
    
    
    
    #Test model
    '''
    y_pred, test_accuracy = test_tree(model, X_test, y_test)
    y_pred_w_reg, test_accuracy_w_reg = test_tree(model_w_reg, X_test, y_test)
    print(f'Baseline without regularization test accuracy: {test_accuracy}')
    print(f'Baseline with regularization test accuracy: {test_accuracy_w_reg}')
    
    #TODO: add other metrics
    '''
    
    