import csv
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import os
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, auc

def print_results(y_test, y_probs, threshold, output_model_path: str=None, experiment_type: str=None, ethnicity: str=None):
    '''
    Args:
        y_test: True labels
        y_probs: Predicted probabilities
        threshold: Threshold for classification
        output_model_path: Path to save the model(should be ../results/logreg/ or ../results/decision_tree/ or ../results/gmm/ or ../results/cost_sensitive/)
        experiment_type: Type of experiment(should be baseline_wo_reg, baseline_w_reg, upsample, cluster, gmm, cost_sensitive)
        ethnicity: Ethnicity of the patient(should be 1.0, 3.0, 4.0, 6.0)
    '''
    
    print("First 50 probabilities:", y_probs[0:49])
    #f1 score
    f1, y_pred = f1_from_probs(y_test, y_probs, threshold)
    print(f'F1 Score: {f1}')
    
    #Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    

    #Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Diabetic', 'Diabetic'])
    # disp.plot()
    # plt.show()
    if ethnicity is not None:
        confusion_matrix_path = os.path.join(output_model_path, f"{experiment_type}_{ethnicity}_confusion_matrix.png")
    else:
        confusion_matrix_path = os.path.join(output_model_path, f"{experiment_type}_confusion_matrix.png")
    disp.figure_.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    
    #Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    plt.plot(recall, precision, marker='o')
    for i, t in enumerate(thresholds):
        plt.annotate(f"{t:.2f}", (recall[i+1], precision[i+1]))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve with Thresholds')
    if ethnicity is not None:
        precision_recall_curve_path = os.path.join(output_model_path, f"{experiment_type}_{ethnicity}_precision_recall_curve.png")
    else:
        precision_recall_curve_path = os.path.join(output_model_path, f"{experiment_type}_precision_recall_curve.png")
    plt.savefig(precision_recall_curve_path, dpi=300, bbox_inches='tight')
    
    metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    }

    if ethnicity is not None:
        csv_file_path = os.path.join(output_model_path, f"{experiment_type}_{ethnicity}_metrics.csv")
    else:
        csv_file_path = os.path.join(output_model_path, f"{experiment_type}_metrics.csv")
    with open(csv_file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)
        print(f"Metrics saved to {csv_file_path}")
        
    


def f1_from_probs(y_true, probs, threshold):
    y_true = np.asarray(y_true)
    preds = (probs >= threshold).astype(int)
    tp = np.sum((preds == 1) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))   
    tn = np.sum((preds == 0) & (y_true == 0))

    # precision
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    # recall
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    # f1
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return f1, preds

def calculate_sample_weight(dframe):
    """Calculate sample weight for each class"""

    # ethnicity column names
    ethnicity_cols = [
        "RIDRETH3_1.0",
        "RIDRETH3_3.0",
        "RIDRETH3_4.0",
        "RIDRETH3_6.0"
    ]

    # total samples
    N = len(dframe)

    # number of ethnicity groups
    K = len(ethnicity_cols)

    # count samples in each ethnicity group
    group_counts = dframe[ethnicity_cols].sum()

    # compute weights using w_g = N / (K * n_g)
    weights_map = {col: N / (K * group_counts[col]) for col in ethnicity_cols}

    # assign weight to each patient
    sample_weight = dframe[ethnicity_cols].dot(pd.Series(weights_map))
    sample_weight = sample_weight.to_numpy()
        
    return sample_weight

def return_minority_feature_index(csv_path):
    with open(csv_path, 'r', newline='') as csv_fh:
        headers = csv_fh.readline().strip().split(',')
    minority_feature_index = [i for i in range(len(headers)) if headers[i] == "RIDRETH3_1.0" or headers[i] == "RIDRETH3_4.0" or headers[i] == "RIDRETH3_6.0"]
    return minority_feature_index

def add_intercept_fn(x):
    """Add intercept to matrix x.

    Args:
        x: 2D NumPy array.

    Returns:
        New matrix same as x with 1's in the 0th column.
    """
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x

    return new_x


def load_xy_csv(x_path, y_path, add_intercept=False):
    with open(x_path, 'r', newline='') as csv_fh_x:
        headers = csv_fh_x.readline().strip().split(',')
    x_cols = [i for i in range(len(headers)) if headers[i] != "SEQN"]
    inputs = np.loadtxt(x_path, delimiter=',', skiprows=1, usecols=x_cols)
    
    with open(y_path, 'r', newline='') as csv_fh_y:
        headers = csv_fh_y.readline().strip().split(',')
    y_cols = [i for i in range(len(headers)) if headers[i] != "SEQN"]
    labels = np.loadtxt(y_path, delimiter=',', skiprows=1, usecols=y_cols)
    
    if add_intercept:
        inputs = add_intercept_fn(inputs)
    
    return inputs, labels

def load_csv(csv_path, label_col=None, add_intercept=False):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.
         label_col: Name of column to use as labels (should be 'y' or 'l').
         add_intercept: Add an intercept entry to x-values.
         header_included: headers of features we want to use for this 

    Returns:
        xs: Numpy array of x-values (inputs).
        ys: Numpy array of y-values (labels).
    """

    # Load headers
    with open(csv_path, 'r', newline='') as csv_fh:
        headers = csv_fh.readline().strip().split(',')
    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i] != label_col and headers[i] != 'SEQN']
    l_cols = [i for i in range(len(headers)) if headers[i] == label_col]
    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols)
    labels = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=l_cols)

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)

    if add_intercept:
        inputs = add_intercept_fn(inputs)

    return inputs, labels

def load_spam_dataset(tsv_path):
    """Load the spam dataset from a TSV file

    Args:
         csv_path: Path to TSV file containing dataset.

    Returns:
        messages: A list of string values containing the text of each message.
        labels: The binary labels (0 or 1) for each message. A 1 indicates spam.
    """

    messages = []
    labels = []

    with open(tsv_path, 'r', newline='', encoding='utf8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')

        for label, message in reader:
            messages.append(message)
            labels.append(1 if label == 'spam' else 0)

    return messages, np.array(labels)

def plot(x, y, theta, save_path, correction=1.0):
    """Plot dataset and fitted logistic regression parameters.

    Args:
        x: Matrix of training examples, one per row.
        y: Vector of labels in {0, 1}.
        theta: Vector of parameters for logistic regression model.
        save_path: Path to save the plot.
        correction: Correction factor to apply (Problem 2(e) only).
    """
    # Plot dataset
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)

    # Plot decision boundary (found by solving for theta^T x = 0)
    x1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
    x2 = -(theta[0] / theta[2] * correction + theta[1] / theta[2] * x1)
    plt.plot(x1, x2, c='red', linewidth=2)

    # Add labels and save to disk
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(save_path)


def plot_contour(predict_fn):
    """Plot a contour given the provided prediction function"""
    x, y = np.meshgrid(np.linspace(-10, 10, num=20), np.linspace(-10, 10, num=20))
    z = np.zeros(x.shape)

    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            z[i, j] = predict_fn([x[i, j], y[i, j]])

    plt.contourf(x, y, z, levels=[-float('inf'), 0, float('inf')], colors=['orange', 'cyan'])

def plot_points(x, y):
    """Plot some points where x are the coordinates and y is the label"""
    x_one = x[y == 0, :]
    x_two = x[y == 1, :]

    plt.scatter(x_one[:,0], x_one[:,1], marker='x', color='red')
    plt.scatter(x_two[:,0], x_two[:,1], marker='o', color='blue')

def write_json(filename, value):
    """Write the provided value as JSON to the given filename"""
    with open(filename, 'w') as f:
        json.dump(value, f)
