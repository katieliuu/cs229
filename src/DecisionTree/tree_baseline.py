import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
from tree_src import *
import util
import argparse
from cv_logreg import f1_from_probs
import matplotlib.pyplot as plt

def main(test: bool = False):
    #Train Model
    X_original, Y_original = util.load_csv('src/data/model_ready/train_raw.csv', label_col='diabetes', add_intercept=True)
    print("X_original shape:", X_original.shape)
    print("Y_original shape:", Y_original.shape)
    
    #baseline without regularization
    model_wo_reg, train_accuracy_wo_reg = decision_tree(X_original, Y_original) #TODO: add hyperparameter from Charlotte's CV results
    
    #baseline with regularization
    model_w_reg, train_accuracy_w_reg = decision_tree(X_original, Y_original, max_depth=None, min_samples_split=2, min_samples_leaf=1, sample_weight=None) #TODO: add hyperparameter from Charlotte's CV results

    if test:
        threshold_wo_reg = 0 #TODO
        X_test, Y_test = util.load_csv('src/data/model_ready/test_processed.csv', label_col='diabetes', add_intercept=True)
        Y_pred_wo_reg, test_accuracy_wo_reg, probs_wo_reg = test_tree(model_wo_reg, X_test, Y_test)
        f1_wo_reg, precision_wo_reg, recall_wo_reg, tp_wo_reg, fp_wo_reg, tn_wo_reg, fn_wo_reg, preds_wo_reg = f1_from_probs(Y_test, probs_wo_reg, threshold_wo_reg)
        
        threshold_w_reg = 0 #TODO
        Y_pred_w_reg, test_accuracy_w_reg, probs_w_reg = test_tree(model_w_reg, X_test, Y_test)
        f1_w_reg, precision_w_reg, recall_w_reg, tp_w_reg, fp_w_reg, tn_w_reg, fn_w_reg, preds_w_reg = f1_from_probs(Y_test, probs_w_reg, threshold_w_reg)
    
        #Confusion Matrix
        cm = confusion_matrix(Y_test, preds_wo_reg, labels=['Diabetic', 'Non-Diabetic'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Diabetic', 'Non-Diabetic'])
        disp.plot()
        plt.show()
        #Accuracy
        accuracy = accuracy_score(Y_test, preds_wo_reg)
        print(f'Without Regularization Accuracy: {accuracy}')
        #Confusion Matrix
        cm = confusion_matrix(Y_test, preds_w_reg, labels=['Diabetic', 'Non-Diabetic'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Diabetic', 'Non-Diabetic'])
        disp.plot()
        plt.show()
        #Accuracy
        accuracy = accuracy_score(Y_test, preds_w_reg)
        print(f'With Regularization Accuracy: {accuracy}')
    
    if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="Test or Train")
        parser.add_argument("--test", action="store_true")
        args = parser.parse_args()
        
        main(test = args.test)
        
        
        