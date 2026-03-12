import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay, precision_recall_curve
import matplotlib.pyplot as plt


def decision_tree(X_train, y_train, max_depth=None, min_samples_split=2, min_samples_leaf=1, sample_weight=None):
    model = DecisionTreeClassifier(random_state=3, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    #train model
    model.fit(X_train, y_train, sample_weight=sample_weight)
    train_accuracy = model.score(X_train, y_train)
    return model, train_accuracy
    
def test_tree(model, X_test, y_test):
    #y_pred = model.predict(X_test)
    #test_accuracy = model.score(X_test, y_test)
    probs = model.predict_proba(X_test)[:, 1] #probability of being diabetic column
    return probs
