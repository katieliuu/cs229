'''
This script contains the source code for the decision tree model. It contains functions
used to create DecisionTreeClassifier and predict probabilities.
'''
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from util import f1_from_probs
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay, precision_recall_curve
import matplotlib.pyplot as plt


def decision_tree(X_train, y_train, max_depth=None, min_samples_split=2, min_samples_leaf=1, sample_weight=None, threshold=0.5):
    '''
    This function trains a decision tree model on the training data and returns the model
    and the accuracy of the model on the training data.
    Args:
        X_train: pandas dataframe, training data features
        y_train: pandas series, training labels
        max_depth: int, max depth of the tree
        min_samples_split: int, min samples split
        min_samples_leaf: int, min samples leaf
        sample_weight: pandas series, sample weights
        threshold: float, threshold for f1 score
    Returns:
        model: DecisionTreeClassifier, trained model
        train_accuracy: float, accuracy of the model on the training data
    '''
    model = DecisionTreeClassifier(random_state=3, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    #train model
    model.fit(X_train, y_train, sample_weight=sample_weight)
    probs = model.predict_proba(X_train)[:, 1]
    _, train_pred = f1_from_probs(y_train, probs, threshold)
    train_accuracy = accuracy_score(y_train, train_pred)
    return model, train_accuracy
    
def test_tree(model, X_test, y_test):
    '''
    This function tests the decision tree model on the test data and returns the probabilities
    of the model on the test data.
    Args:
        model: DecisionTreeClassifier, trained model
        X_test: pandas dataframe, test data features
        y_test: pandas series, test labels
    Returns:
        probs: pandas series, probabilities of test data of being diabetic
    '''
    probs = model.predict_proba(X_test)[:, 1] #probability of test data of being diabetic
    return probs
