import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tree_src import *
import util
import numpy as np
from upsample import *

def main():
    #Train Model
    X_original, Y_original = util.load_csv('src/data/model_ready/train_raw.csv', label_col='diabetes', add_intercept=True)
    print("X_original shape:", X_original.shape)
    print("Y_original shape:", Y_original.shape)
    
    training_data = np.concatenate((X_original, Y_original), axis=1)
    upsampled_training = naive_upsample(training_data, kappa_1, kappa_4, kappa_6) #TODO: add kappa values from cv
    X_upsampled, Y_upsampled = upsampled_training.drop(columns=["diabetes"]), upsampled_training["diabetes"]
    
    #decision tree with upsampled data
    model_upsampled, train_accuracy_upsampled = decision_tree(X_upsampled, Y_upsampled, max_depth=None, min_samples_split=2, min_samples_leaf=1, sample_weight=None) #TODO: add hyperparameter from Charlotte's CV results
    
    print(f'Decision tree with upsampled data train accuracy: {train_accuracy_upsampled}')
    
    
    