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
from logreg.logreg_src import logistic_regression
from upsample import naive_upsample, get_natural_kappas
from logreg.util import calculate_sample_weight
from kprototypes import run_k_prototypes
#from gmm import # fill later w any preprocessing/experiment specific functions
from itertools import product
import json

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

def cv_tune_pipeline_logreg(experiment_type = "baseline", n_splits = 5, inner_splits = 3, random_state = 3):

    df = pd.read_csv("src/data/model_ready/train_raw.csv")
    y = df["diabetes"].astype(int)
    X = df.drop(columns=["diabetes"])

    # compute natural kappas for upsampling
    nat_kap_1, nat_kap_4, nat_kap_6 = get_natural_kappas(X)

    lambda_grid = [0, 0.01, 0.1, 1, 10]
    threshold_grid = [0.3, 0.4, 0.5]
    gamma_grid = [0.25, 0.5, 1, 2]
    n_clusters_grid = [3, 4, 5, 6]
    kappa_1_grid = [nat_kap_1, nat_kap_1 * 0.5, nat_kap_1 * 1.5]
    kappa_4_grid = [nat_kap_4, nat_kap_4 * 0.5, nat_kap_4 * 1.5]
    kappa_6_grid =  [nat_kap_6, nat_kap_6 * 0.5, nat_kap_6 * 1.5]

    # INSERT GMM AND CLUSTERING PARAM GRIDS
    if experiment_type == "baseline":
        param_grid = list(product(lambda_grid, threshold_grid))
    elif experiment_type == "upsample":
        param_grid = list(product(lambda_grid, threshold_grid, kappa_1_grid, kappa_4_grid, kappa_6_grid))
    elif experiment_type == "cluster":
        param_grid = list(product(gamma_grid, n_clusters_grid, lambda_grid, threshold_grid))
    elif experiment_type == "cost_sensitive":
        param_grid = list(product(lambda_grid, threshold_grid))

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

        #print(f"param_grid is {param_grid}")
        # loop through all possible hyperparam combinations, do inner 3-fold cv on them
        for params in param_grid:
            #print(f"param is {params}")
            scores_inner = []
            for (train_idx_inner, val_idx_inner) in skf_inner.split(X_train_f, y_train_f):
                X_train_f_inner = X_train_f.iloc[train_idx_inner]
                y_train_f_inner = y_train_f.iloc[train_idx_inner]
                X_val_f_inner = X_train_f.iloc[val_idx_inner]
                y_val_f_inner = y_train_f.iloc[val_idx_inner]

                # experiment-specific params: ADD GMM and CLUSTERING params
                if experiment_type == "baseline":
                    lambda_reg, threshold = params
                    gamma = None
                    n_clusters = None
                    kappa_1 = None
                    kappa_4 = None
                    kappa_6 = None
                    sample_weight = None
                elif experiment_type == "upsample":
                    lambda_reg, threshold, kappa_1, kappa_4, kappa_6 = params
                    gamma = None
                    n_clusters = None
                    sample_weight = None
                elif experiment_type == "cluster":
                    gamma, n_clusters, lambda_reg, threshold = params
                    kappa_1 = None
                    kappa_4 = None
                    kappa_6 = None
                    sample_weight = None
                elif experiment_type == "cost_sensitive":
                    lambda_reg, threshold = params
                    gamma = None
                    n_clusters = None
                    kappa_1 = None
                    kappa_4 = None
                    kappa_6 = None
                    sample_weight = None

                X_train_inner_preprocessed, mice, scaler, encoder = preprocess_fit_transform(X_train_f_inner)
                X_val_inner_preprocessed = preprocess_transform(X_val_f_inner, mice, scaler, encoder)

                # manipulate fold for specific experiment
                if experiment_type == "upsample":
                    train_set_inner = pd.concat([X_train_inner_preprocessed, y_train_f_inner.rename("diabetes")], axis=1)
                    train_set_inner = naive_upsample(train_set_inner, kappa_1=kappa_1, kappa_4=kappa_4, kappa_6=kappa_6)
                    y_train_f_inner = train_set_inner["diabetes"]
                    X_train_inner_preprocessed = train_set_inner.drop(columns=["diabetes"])
                
                elif experiment_type == "cluster":
                    # clustering code. do this before one hot encoding, after scaling/imputation?
                    train_set_inner = pd.concat([X_train_inner_preprocessed, y_train_f_inner.rename("diabetes")], axis=1)
                    train_set_inner = run_k_prototypes(train_set_inner, gamma = gamma, n_clusters = n_clusters)
                    y_train_f_inner = train_set_inner["diabetes"]
                    X_train_inner_preprocessed = train_set_inner.drop(columns=["diabetes"])

                elif experiment_type == "cost_sensitive":
                    train_set_inner = pd.concat([X_train_inner_preprocessed, y_train_f_inner.rename("diabetes")], axis=1)
                    sample_weight = calculate_sample_weight(train_set_inner)
                    y_train_f_inner = train_set_inner["diabetes"]
                    X_train_inner_preprocessed = train_set_inner.drop(columns=["diabetes"])

                # inner model fit ADD GMM
                if experiment_type in ["baseline", "upsample", "cluster"]:
                    theta = logistic_regression(X_train_inner_preprocessed.to_numpy(), y_train_f_inner.to_numpy(),
                                                lambda_reg = lambda_reg)
                # change to account for penalty weight calculations
                elif experiment_type == "cost_sensitive":
                    theta = logistic_regression(X_train_inner_preprocessed.to_numpy(), y_train_f_inner.to_numpy(),
                                        sample_weight = sample_weight, lambda_reg = lambda_reg)
                    
                # prediction and score for inner loop
                probs = 1 / (1 + np.exp(-(X_val_inner_preprocessed.to_numpy() @ theta)))
                scores_inner.append(f1_from_probs(y_val_f_inner, probs, threshold))

            # running update of optimal parameters
            mean_inner = float(np.mean(scores_inner))
            if mean_inner > score_star:
                score_star = mean_inner
                params_star = params

        #print(f"params_star = {params_star}")
        # now that we have the best parameters, refit the model on outer train fold with those params
        # experiment-specific params: ADD GMM params
        if experiment_type == "baseline":
            lambda_reg, threshold = params_star
            gamma = None
            n_clusters = None
            kappa_1 = None
            kappa_4 = None
            kappa_6 = None
            sample_weight = None
        elif experiment_type == "upsample":
            lambda_reg, threshold, kappa_1, kappa_4, kappa_6 = params_star
            gamma = None
            n_clusters = None
            sample_weight = None
        elif experiment_type == "cluster":
            gamma, n_clusters, lambda_reg, threshold = params_star
            kappa_1 = None
            kappa_4 = None
            kappa_6 = None
            sample_weight = None
        elif experiment_type == "cost_sensitive":
            lambda_reg, threshold = params_star
            gamma = None
            n_clusters = None
            kappa_1 = None
            kappa_4 = None
            kappa_6 = None
            sample_weight = None

        # fit imputer, scaler, encoder to train folds
        X_train_preprocessed, mice, scaler, encoder = preprocess_fit_transform(X_train_f)
        # apply to validation fold
        X_val_preprocessed = preprocess_transform(X_val_f, mice, scaler, encoder)

        # experiment-specific logic
        if experiment_type == "upsample":
            train_set = pd.concat([X_train_preprocessed, y_train_f.rename("diabetes")], axis=1)
            train_set = naive_upsample(train_set, kappa_1=kappa_1, kappa_4=kappa_4, kappa_6=kappa_6)
            y_train_f = train_set["diabetes"]
            X_train_preprocessed = train_set.drop(columns=["diabetes"])
        
        elif experiment_type == "cluster":
            train_set = pd.concat([X_train_preprocessed, y_train_f.rename("diabetes")], axis=1)
            train_set = run_k_prototypes(train_set, gamma = gamma, n_clusters = n_clusters)
            y_train_f = train_set["diabetes"]
            X_train_preprocessed = train_set.drop(columns=["diabetes"])

        elif experiment_type == "cost_sensitive":
            train_set = pd.concat([X_train_preprocessed, y_train_f.rename("diabetes")], axis=1)
            sample_weight = calculate_sample_weight(train_set)
            y_train_f = train_set["diabetes"]
            X_train_preprocessed = train_set.drop(columns=["diabetes"])

        # INSERT UPSAMPLING LOGIC FOR GMM
        if experiment_type in ["baseline", "upsample", "cluster"]:
            theta = logistic_regression(X_train_preprocessed.to_numpy(), y_train_f.to_numpy(), lambda_reg=lambda_reg)

        elif experiment_type == "cost_sensitive":
            theta = logistic_regression(X_train_preprocessed.to_numpy(), y_train_f.to_numpy(),
                                        sample_weight = sample_weight, lambda_reg = lambda_reg)

        # outer fold evaluation
        probs = 1 / (1 + np.exp(-(X_val_preprocessed.to_numpy() @ theta)))
        f1 = f1_from_probs(y_val_f, probs, threshold)

        # dict of metrics for each fold
        metrics.append({"fold": f, "f1": f1,
                        "inner_f1_star": score_star,
                        "lambda_reg": lambda_reg,
                        "threshold": threshold,
                        "gamma": gamma,
                        "n_clusters": n_clusters})
        #"sample_weight": sample_weight,
        
    fold_metrics = pd.DataFrame(metrics)

    return {
        "experiment_type": experiment_type,
        "fold_metrics": fold_metrics,
        "summary" : {"f1_mean": fold_metrics["f1"].mean(),
                     "f1_std": fold_metrics["f1"].std()}}

def main():
    baseline_metrics_dict = cv_tune_pipeline_logreg()
    baseline_metrics_dict["fold_metrics"] = baseline_metrics_dict["fold_metrics"].to_dict(orient="records")

    baseline_save_path = 'src/metrics/baseline_log_reg_parameters.json'
    with open(baseline_save_path, mode = 'w') as file:
        json.dump(baseline_metrics_dict, file, indent = 4)

    print(f"JSON file '{baseline_save_path}' created successfully")

    upsample_metrics_dict = cv_tune_pipeline_logreg(experiment_type="upsample")
    upsample_metrics_dict["fold_metrics"] = upsample_metrics_dict["fold_metrics"].to_dict(orient="records")

    upsample_save_path = 'src/metrics/upsample_log_reg_parameters.json'
    with open(upsample_save_path, mode = 'w') as file:
        json.dump(upsample_metrics_dict, file, indent = 4)

    print(f"JSON file '{upsample_save_path}' created successfully")

    cluster_metrics_dict = cv_tune_pipeline_logreg(experiment_type="cluster")
    cluster_metrics_dict["fold_metrics"] = cluster_metrics_dict["fold_metrics"].to_dict(orient="records")

    cluster_save_path = 'src/metrics/cluster_log_reg_parameters.json'
    with open(cluster_save_path, mode = 'w') as file:
        json.dump(cluster_metrics_dict, file, indent = 4)

    print(f"JSON file '{cluster_save_path}' created successfully")

    cost_metrics_dict = cv_tune_pipeline_logreg(experiment_type="cost_sensitive")
    cost_metrics_dict["fold_metrics"] = cost_metrics_dict["fold_metrics"].to_dict(orient="records")

    cost_save_path = 'src/metrics/cost_log_reg_parameters.json'
    with open(cost_save_path, mode = 'w') as file:
        json.dump(cost_metrics_dict, file, indent = 4)

    print(f"JSON file '{cost_save_path}' created successfully")

if __name__ == '__main__':
    main()