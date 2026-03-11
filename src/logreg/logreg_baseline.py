"""
baseline.py trains a logistic regression model on the baseline dataset.
It outputs the final loss and weights of the model without regularization and with regularization.
Regularization is done using L2 regularization and the factor can be altered in the main function.
"""
import util
import numpy as np
import argparse
from logreg_src import *
from cv_logreg import f1_from_probs
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def main(test: bool = False):

    X_original, Y_original = util.load_csv('src/data/model_ready/train_processed.csv', label_col='diabetes', add_intercept=True)
    print("X_original shape:", X_original.shape)
    print("Y_original shape:", Y_original.shape)
    
    #Baseline without regularization
    theta_wo_reg = logistic_regression(X_original, Y_original, max_iter=5000, lambda_reg=0)
    
    #Baseline with regularization
    theta_w_reg = logistic_regression(X_original, Y_original, max_iter=5000, lambda_reg=10) #TODO: add hyperparameter from Charlotte's CV results
    
    
    
    if test:
        X_test, Y_test = util.load_csv('src/data/model_ready/test_processed.csv', label_col='diabetes', add_intercept=True)
        threshold_wo_reg = 0 #TODO
        threshold_w_reg = 0 #TODO
        
        prob_wo_reg = 1 / (1 + np.exp(-(X_test @ theta_wo_reg)))
        
        f1_wo_reg, precision_wo_reg, recall_wo_reg, tp_wo_reg, fp_wo_reg, tn_wo_reg, fn_wo_reg, preds_wo_reg = f1_from_probs(Y_test, prob_wo_reg, threshold_wo_reg)
        #TODO: implement other metrics (confusion matrix, accuracy, f1, )
        
        prob_w_reg = 1 / (1 + np.exp(-(X_test @ theta_w_reg)))
        f1_w_reg, precision_w_reg, recall_w_reg, tp_w_reg, fp_w_reg, tn_w_reg, fn_w_reg, preds_w_reg = f1_from_probs(Y_test, prob_w_reg, threshold_w_reg)
        #TODO: implement other metrics
        
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
    

