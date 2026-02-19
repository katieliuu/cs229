
# %%
import numpy as np
import pandas as pd
import util
from random import random
import os
from logreg import LogisticRegression
from pathlib import Path


matches = list(Path(".").rglob("X_train_imputed.csv"))
print("Matches found:", len(matches))
for m in matches[:10]:
    print(m.resolve())
# %%
df = pd.read_csv(matches[0])


# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
# Ratio of class 0 to class 1
kappa = 0.1 # what if we tried different values of kappa?

def confusion_matrix(y_val, probs):
    y_val = np.array(y_val)
    probs = np.array(probs)
    preds = (probs >= 0.5).astype(int)
    tp = int(np.sum((y_val == 1) & (preds == 1)))
    tn = int(np.sum((y_val == 0) & (preds == 0)))
    fp = int(np.sum((y_val == 0) & (preds == 1)))
    fn = int(np.sum((y_val == 1) & (preds == 0)))
    return tp, tn, fp, fn

# don't need this naive implementation if we already have a baseline
def run_naive(x_train, y_train, x_val, y_val, out_path):
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    probs = clf.predict(x_val)
    np.savetxt(out_path, probs)
    plot_path = out_path.replace('.txt', '.png')
    util.plot(x_val, y_val, clf.theta, plot_path)
    tp, tn, fp, fn = confusion_matrix(y_val, probs)
    results = np.array([tp, tn, fp, fn])
    np.savetxt(out_path.replace("_pred.txt", "_confusion_matrix.txt"), results)

def run_upsampling(x_train, y_train, x_val, y_val, out_path, imbalance_col):
    x_train_array = np.array(x_train)
    y_train_array = np.array(y_train)
    minority_condition = (x_train_array[imbalance_col == 1.0 or imbalance_col == 6.0]) #nh asian, mexican american
    majority_condition = (x_train_array[imbalance_col == 3.0 or imbalance_col == 4.0]) #nh white and nh black
    minority_y = y_train_array[minority_condition]
    majority_y = y_train_array[majority_condition]
    minority_x = x_train_array[minority_condition]
    majority_x = x_train_array[majority_condition]
    kappa_inv = int(round(1.0 / kappa))
    minority_x_rep = np.tile(minority_x, (kappa_inv, 1))
    minority_y_rep = np.tile(minority_y, kappa_inv) 
    x_train_upsampled = np.vstack([majority_x, minority_x_rep])
    y_train_upsampled = np.concatenate([majority_y, minority_y_rep])
    clf = LogisticRegression()
    clf.fit(x_train_upsampled, y_train_upsampled)
    probs = clf.predict(x_val)
    np.savetxt(out_path, probs)
    plot_path = out_path.replace('.txt', '.png')
    util.plot(x_val, y_val, clf.theta, plot_path)
    tp, tn, fp, fn = confusion_matrix(y_val, probs)
    results = np.array([tp, tn, fp, fn])
    np.savetxt(out_path.replace("_pred.txt", "_confusion_matrix.txt"), results)
    
    # is validation set the test set? i'm a bit confused about how we train this
def main(train_path, validation_path, save_path):
    """Problem: Logistic regression for imbalanced labels.

    Run under the following conditions:
        1. naive logistic regression
        2. upsampling minority class

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.
        save_path: Path to save predictions.
    """
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_upsampling = save_path.replace(WILDCARD, 'upsampling')

    # *** START CODE HERE ***
    # Part (b): Vanilla logistic regression
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt() as a 1D numpy array
    # Part (d): Upsampling minority class
    # Make sure to save predicted probabilities to output_path_upsampling using np.savetxt() as a 1D numpy array
    # Repeat minority examples 1 / kappa times
    
    x_train, y_train = util.load_csv(train_path, label_col = "diabetes" add_intercept=True)
    x_val, y_val = util.load_csv(validation_path, label_col = "diabetes", add_intercept=True)
    
    print('==== Training naive logistic regression model ====')
    run_naive(x_train, y_train, x_val, y_val, output_path_naive)
    print('==== Training logistic regression model with upsampling minority features ====')
    run_upsampling(x_train, y_train, x_val, y_val, output_path_upsampling)
    

if __name__ == '__main__':
    os.makedirs("src/data/models", exist_ok=True)
    main(train_path="src/data/model_ready/X_train_imputed.csv",
         validation_path="INSERT VAL PATH",
         save_path="src/data/models/imbalanced_X_pred_logreg.txt")