import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
from tree_src import *
import util
import numpy as np
from upsample import *
import argparse
from cv_logreg import f1_from_probs
import matplotlib.pyplot as plt

def main(test: bool = False):
    #Train Model
    X_original, Y_original = util.load_csv('src/data/model_ready/train_raw.csv', label_col='diabetes', add_intercept=True)
    print("X_original shape:", X_original.shape)
    print("Y_original shape:", Y_original.shape)
    
    training_data = np.concatenate((X_original, Y_original), axis=1)
    upsampled_training = naive_upsample(training_data, kappa_1, kappa_4, kappa_6) #TODO: add kappa values from cv
    X_upsampled, Y_upsampled = upsampled_training.drop(columns=["diabetes"]), upsampled_training["diabetes"]
    
    #decision tree with upsampled data
    model_upsampled, train_accuracy_upsampled = decision_tree(X_upsampled, Y_upsampled, max_depth=None, min_samples_split=2, min_samples_leaf=1, sample_weight=None) #TODO: add hyperparameter from Charlotte's CV results
    
    if test:
        threshold_upsampled = 0 #TODO
        X_test, Y_test = util.load_csv('src/data/model_ready/test_processed.csv', label_col='diabetes', add_intercept=True)
        Y_pred_upsampled, test_accuracy_upsampled, probs_upsampled = test_tree(model_upsampled, X_test, Y_test)
        f1_upsampled, precision_upsampled, recall_upsampled, tp_upsampled, fp_upsampled, tn_upsampled, fn_upsampled, preds_upsampled = f1_from_probs(Y_test, probs_upsampled, threshold_upsampled)
        #TODO: implement other metrics (confusion matrix, accuracy, f1)
        
        #Confusion Matrix
        cm = confusion_matrix(Y_test, preds_upsampled, labels=['Diabetic', 'Non-Diabetic'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Diabetic', 'Non-Diabetic'])
        disp.plot()
        plt.show()
        #Accuracy
        accuracy = accuracy_score(Y_test, preds_upsampled)
        print(f'Accuracy: {accuracy}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test or Train")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    main(test = args.test)
    
    
    