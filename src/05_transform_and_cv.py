"""
05_transform_and_cv carries out 5-fold CV with hyperparameter tuning on the training set. It fits and applies
the full preprocessing pipeline: MICE (iterative imputation) on the numeric columns,
one-hot encoding on the categorical columns, and standardization onf the numeric features.
Within both the inner and outer CV loop, both processing and evaluation are carried out. The cv_pipeline function
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
from itertools import product

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

def cv_tune_pipeline(experiment_type = "baseline", n_splits = 5, inner_splits = 3, random_state = 3):

    df = pd.read_csv("src/data/model_ready/train_raw.csv")
    y = df["diabetes"].astype(int)
    X = df.drop(columns=["diabetes", "SEQN"])

    lambda_grid = [0.0, 1e-4, 1e-3, 1e-2]
    threshold_grid = [0.3, 0.4, 0.5]
    kappa_grid = [1, 2, 3]
    alpha_grid = [0.5, 1.0, 2.0]

    # INSERT GMM AND CLUSTERING PARAM GRIDS
    if experiment_type == "baseline":
        param_grid = list(product(lambda_grid, threshold_grid))
    elif experiment_type == "upsample":
        param_grid = list(product(kappa_grid, lambda_grid, threshold_grid))
    elif experiment_type == "cost_sensitive":
        param_grid = list(product(alpha_grid, lambda_grid, threshold_grid))
    
    def f1_from_probs(y_true, probs, threshold):
        y_true = np.asarray(y_true)
        preds = (probs >= threshold).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))

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
        return f1

    # outer cv starts here: evaluation using tuned params
    # gives train/val indices to split data in a stratified way (preserves the percentage of samples for each of diabetes/no diabetes)
    skf_outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metrics = []

    # loop through each outer fold
    for f, (train_idx, val_idx) in enumerate(skf_outer.split(X, y)): # generates indices to split into train/val, iterate through pairs of fold index + tuple of (train, val) indices for that split
        X_train_f = X.iloc[train_idx]
        y_train_f = y.iloc[train_idx]
        X_val_f = X.iloc[val_idx]
        y_val_f = y.iloc[val_idx]

        skf_inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
        score_star = -np.inf
        params_star = None

        # loop through all possible hyperparam combinations, do inner 3-fold cv on them
        for params in param_grid:
            scores_inner = []
            for (train_idx_inner, val_idx_inner) in skf_inner.split(X_train_f, y_train_f):
                X_train_f_inner = X_train_f.iloc[train_idx_inner]
                y_train_f_inner = y_train_f.iloc[train_idx_inner]
                X_val_f_inner = X_train_f.iloc[val_idx_inner]
                y_val_f_inner = y_train_f.iloc[val_idx_inner]

                # experiment-specific params: ADD GMM and CLUSTERING params
                if experiment_type == "baseline":
                    lambda_reg, threshold = params
                    alpha = None
                    kappa = None
                elif experiment_type == "upsample":
                    kappa, lambda_reg, threshold = params
                    alpha = None
                elif experiment_type == "cost_sensitive":
                    alpha, lambda_reg, threshold = params
                    kappa = None

                X_train_inner_preprocessed, mice, scaler, encoder = preprocess_fit_transform(X_train_f_inner)
                X_val_inner_preprocessed = preprocess_transform(X_val_f_inner, mice, scaler, encoder)

                # manipulate fold for specific experiment
                if experiment_type == "upsample":
                    train_set_inner = pd.concat([X_train_inner_preprocessed, y_train_f_inner], axis=1)
                    train_set_inner = upsample_minority_class(train_set_inner, kappa)
                    y_train_f_inner = train_set_inner["diabetes"]
                    X_train_inner_preprocessed = train_set_inner.drop(columns=["diabetes"])

                # inner model fit ADD K MEANS AND GMM
                if experiment_type in ["baseline", "upsample"]:
                    theta = logistic_regression(X_train_inner_preprocessed.to_numpy(), y_train_f_inner.to_numpy(),
                                                regularize = True,
                                                lambda_reg = lambda_reg)
                elif experiment_type == "cost_sensitive":
                    theta = cost_logreg(X_train_inner_preprocessed.to_numpy(), y_train_f_inner.to_numpy(),
                                        alpha = alpha, regularize = True, lambda_reg = lambda_reg)
                    
                # prediction and score for inner loop
                probs = 1 / (1 + np.exp(-(X_val_inner_preprocessed.to_numpy() @ theta)))
                scores_inner.append(f1_from_probs(y_val_f_inner, probs, threshold))

            # running update of optimal parameters
            mean_inner = float(np.mean(scores_inner))
            if mean_inner > score_star:
                score_star = mean_inner
                params_star = params

        # now that we have the best parameters, refit the model on outer train fold with those params
        # experiment-specific params: ADD GMM and CLUSTERING params
        if experiment_type == "baseline":
            lambda_reg, threshold = params_star
            alpha = None
            kappa = None
        elif experiment_type == "upsample":
            kappa, lambda_reg, threshold = params_star
            alpha = None
        elif experiment_type == "cost_sensitive":
            alpha, lambda_reg, threshold = params_star
            kappa = None

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

        # outer fold evaluation
        probs = 1 / (1 + np.exp(-(X_val_preprocessed.to_numpy() @ theta)))
        f1 = f1_from_probs(y_val_f, probs, threshold)

        # dict of metrics for each fold
        metrics.append({"fold": f, "f1": f1,
                        "inner_f1_star": score_star,
                        "lambda_reg": lambda_reg,
                        "threshold": threshold, "kappa": kappa,
                        "alpha": alpha})
        
    fold_metrics = pd.DataFrame(metrics)

    return {
        "experiment_type": experiment_type,
        "fold_metrics": fold_metrics,
        "summary" : {"f1_mean": fold_metrics["f1"].mean(),
                     "f1_std": fold_metrics["f1"].std()}}