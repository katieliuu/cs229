"""
05_transform_and_cv carries out 5-fold CV on the training set, fits and applies
the full preprocessing pipeline: MICE (iterative imputation) on the numeric columns,
one-hot encoding on the categorical columns, and standardization onf the numeric features.
Within the CV loop, both processing and evaluation are carried out. The cv_pipeline function
takes in the experiment type as one of its arguments.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from baseline import logistic_regression
from cost_sensitive import logistic_regression as cost_logreg
from upsample import upsample_minority_class
from cluster import # fill later
from gmm import # fill later

# identify numeric vs. cat cols
numeric_cols = ["RIDAGEYR", "LBXTC", "LBDHDD", "LBXSTR", "LBXSCR", "LBXHSCRP", "DBP_mean", "SBP_mean", "BMXBMI", "BMXHIP", "SMQ020"]
cat_cols = ["DMDEDUC2", "RIDRETH3", "RIAGENDR"]

def preprocess_fit_transform(X_train):
    mice = IterativeImputer(random_state=3, max_iter=20)
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    X_train_imputed = pd.DataFrame(mice.fit_transform(X_train[numeric_cols]), columns=numeric_cols, index=X_train.index) # MICE imputation on numeric features
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=numeric_cols, index=X_train.index) # scale numeric features
    X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[cat_cols]), columns=encoder.get_feature_names_out(cat_cols), index=X_train.index) # one-hot encode categorical features
    X_train_processed = pd.concat([X_train_scaled, X_train_encoded], axis=1)

    return X_train_processed, mice, scaler, encoder

def preprocess_transform(X_val, mice, scaler, encoder):
    X_numeric = pd.DataFrame(mice.transform(X_val[numeric_cols]), columns=numeric_cols, index=X_val.index)
    X_scaled = pd.DataFrame(scaler.transform(X_numeric), columns=numeric_cols, index=X_val.index)
    X_cat = pd.DataFrame(encoder.transform(X_val[cat_cols]), columns=encoder.get_feature_names_out(cat_cols), index=X_val.index)

    return pd.concat([X_scaled, X_cat], axis=1)

def cv_pipeline(experiment_type="baseline", kappa=1, lambda_reg=0.0, alpha=1.0, threshold = 0.5, n_splits = 5, random_state = 3):

    df = pd.read_csv("src/data/model_ready/train_raw.csv")
    y = df["diabetes"].astype(int)
    X = df.drop(columns=["diabetes", "SEQN"])


    # gives train/val indices to split data in a stratified way (preserves the percentage of samples for each of diabetes/no diabetes)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    metrics = []

    for f, (train_idx, val_idx) in enumerate(skf.split(X, y)): # generates indices to split into train/val, iterate through pairs of fold index + tuple of (train, val) indices for that split
        X_train_f = X.iloc[train_idx]
        y_train_f = y.iloc[train_idx]
        X_val_f = X.iloc[val_idx]
        y_val_f = y.iloc[val_idx]

        # fit imputer, scaler, encoder to train folds
        X_train_preprocessed, mice, scaler, encoder = preprocess_fit_transform(X_train_f)
        # apply to validation fold
        X_val_preprocessed = preprocess_transform(X_val_f, mice, scaler, encoder)

        # experiment-specific logic
        if experiment_type == "upsample":
            train_set = pd.concat([X_train_preprocessed, y_train_f], axis=1)
            train_set = upsample_minority_class(train_set, kappa)
            y_train_f = train_set["diabetes"]
            X_train_preprocessed = train_set.drop(columns=["diabetes"])
        # INSERT UPSAMPLING LOGIC FOR GMM AND CLUSTERING
        if experiment_type in ["baseline", "upsample", "clustering", "gmm"]:
            theta = logistic_regression(X_train_preprocessed.to_numpy(), y_train_f.to_numpy(), regularize=True, lambda_reg=lambda_reg)

        elif experiment_type == "cost_sensitive":
            theta = cost_logreg(X_train_preprocessed.to_numpy(), y_train_f.to_numpy(), alpha=alpha, regularize=True, lambda_reg=lambda_reg)

        # evaluate performance on the held out fold
        probs = 1 / (1 + np.exp(-(X_val_preprocessed.to_numpy() @ theta)))  # sigmoid
        preds = (probs >= threshold).astype(int) # can alter threshold here
        y_true = y_val_f.to_numpy()
        # confusion-matrix type summary
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

        # accuracy (safe already, but for symmetry you could guard it too)
        total = tp + tn + fp + fn
        acc = (tp + tn) / total

        # dict of metrics for each fold
        metrics.append({"fold": f, "precision": precision,
                        "recall": recall, "f1": f1,
                        "acc": acc, "tp": tp,
                        "fp": fp, "tn": tn,
                        "fn": fn,})
        
    fold_metrics = pd.DataFrame(metrics)

    summary = {
        "mean_precision": fold_metrics["precision"].mean(),
        "mean_recall": fold_metrics["recall"].mean(),
        "f1_mean": fold_metrics["f1"].mean(),
        "acc_mean": fold_metrics["acc"].mean(),
    }

    return {
        "config": {
            "experiment_type": experiment_type,
            "kappa": kappa,
            "lambda_reg": lambda_reg,
            "alpha": alpha,
            "threshold": threshold,
            "n_splits": n_splits,
            "random_state": random_state
        },
        "fold_metrics": fold_metrics,
        "summary": summary
    }