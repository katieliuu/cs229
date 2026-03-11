"""
cost_sensitive.py trains a logistic regression model on the dataset with cost-sensitive learning.
It outputs the final loss of the model without regularization and with regularization.
Regularization is done using L2 regularization and the factor can be altered in the main function.
Cost-sensitive learning is done by upweighting the minority class misclassification.
"""

import util
import numpy as np
import pandas as pd
import argparse
from logreg_src import *
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def main(test: bool = False):
    
    X_original, Y_original = util.load_csv('src/data/model_ready/train_processed.csv', label_col='diabetes', add_intercept=True)
    print("X_original shape:", X_original.shape)
    print("Y_original shape:", Y_original.shape)
    minority_feature_index = util.return_minority_feature_index('src/data/model_ready/train_processed.csv')
    
    
    #Regularized without cost-sensitive learning
    theta_wo_cs = logistic_regression(X_original, Y_original, max_iter=100000, lambda_reg=10, penalty_weight=1)
    
    #Regularized with cost-sensitive learning
    original_df = pd.read_csv('src/data/model_ready/train_processed.csv')
    sample_weight = util.calculate_sample_weight(original_df)
    theta_w_cs = logistic_regression(X_original, Y_original, max_iter=100000, lambda_reg=10, sample_weight=sample_weight) #TODO: add hyperparameter from Charlotte's CV results
    
    
    if test:
        X_test, Y_test = util.load_csv('src/data/model_ready/test_processed.csv', label_col='diabetes', add_intercept=True)
        threshold_wo_cs = 0 #TODO
        threshold_w_cs = 0 #TODO
        
        prob_wo_cs = 1 / (1 + np.exp(-(X_test @ theta_wo_cs)))
        f1_wo_cs, precision_wo_cs, recall_wo_cs, tp_wo_cs, fp_wo_cs, tn_wo_cs, fn_wo_cs, preds_wo_cs = f1_from_probs(Y_test, prob_wo_cs, threshold_wo_cs)
        #TODO: implement other metrics (confusion matrix, accuracy, f1)
        
        prob_w_cs = 1 / (1 + np.exp(-(X_test @ theta_w_cs)))
        f1_w_cs, precision_w_cs, recall_w_cs, tp_w_cs, fp_w_cs, tn_w_cs, fn_w_cs, preds_w_cs = f1_from_probs(Y_test, prob_w_cs, threshold_w_cs)
        #TODO: implement other metrics
        
        #Confusion Matrix
        cm = confusion_matrix(Y_test, preds_wo_cs, labels=['Diabetic', 'Non-Diabetic'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Diabetic', 'Non-Diabetic'])
        disp.plot()
        plt.show()
        #Accuracy
        accuracy = accuracy_score(Y_test, preds_wo_cs)
        print(f'Without Cost Sensitive Learning Accuracy: {accuracy}')
        #Confusion Matrix
        cm = confusion_matrix(Y_test, preds_w_cs, labels=['Diabetic', 'Non-Diabetic'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Diabetic', 'Non-Diabetic'])
        disp.plot()
        plt.show()
        #Accuracy
        accuracy = accuracy_score(Y_test, preds_w_cs)
        print(f'With Cost Sensitive Learning Accuracy: {accuracy}')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test or Train")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    main(test = args.test)
    
    
if __name__ == '__main__':
    main()
    

