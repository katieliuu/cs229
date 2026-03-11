"""
logreg_upsample.py trains a logistic regression model on the dataset with naive upsampling.
Regularization is done using L2 regularization and the factor can be altered in the training function.
Upsampling is done by upsampling the minority class to match the majority class.
- naive repetition: repeat the minority class examples to match the majority class
"""

import util
import numpy as np
import pandas as pd
import argparse
from logreg_src import *
from upsample import *
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def main(test: bool = False):
    
    #Naive upsampling
    X_original, Y_original = util.load_csv('src/data/model_ready/train_processed.csv', label_col='diabetes', add_intercept=True)
    print("X_original shape:", X_original.shape)
    print("Y_original shape:", Y_original.shape)
    
    training_data = np.concatenate((X_original, Y_original), axis=1)
    upsampled_training = naive_upsample(training_data, kappa_1, kappa_4, kappa_6) #TODO: add kappa values from cv
    X_upsampled, Y_upsampled = upsampled_training.drop(columns=["diabetes"]), upsampled_training["diabetes"]
    theta_up_naive = logistic_regression(X_upsampled, Y_upsampled, max_iter=5000, lambda_reg=10, penalty_weight=1) #TODO: add hyperparameter from Charlotte's CV results
    
    if test:
        X_test, Y_test = util.load_csv('src/data/model_ready/test_processed.csv', label_col='diabetes', add_intercept=True)
        threshold_up_naive = 0 #TODO
        
        prob_up_naive = 1 / (1 + np.exp(-(X_test @ theta_up_naive)))
        
        f1_up_naive, precision_up_naive, recall_up_naive, tp_up_naive, fp_up_naive, tn_up_naive, fn_up_naive, preds_up_naive = f1_from_probs(Y_test, prob_up_naive, threshold_up_naive)
        #TODO: implement other metrics (confusion matrix, accuracy, f1)
        
        #Confusion Matrix
        cm = confusion_matrix(Y_test, preds_up_naive, labels=['Diabetic', 'Non-Diabetic'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Diabetic', 'Non-Diabetic'])
        disp.plot()
        plt.show()
        #Accuracy
        accuracy = accuracy_score(Y_test, preds_up_naive)
        print(f'Naive Upsampling Accuracy: {accuracy}')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test or Train")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    main(test = args.test)
    
    

