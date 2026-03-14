"""
cv_neuralnet does nested stratified cross-validation on the training
data to tune and select hyperparameters for the Neural Network experiments.
The inner cross-validation is 3-fold and selects the best performing hyperparameters
according to F1 score. The outer cross-validation is 5-fold and provides an 
estimate of the model's performance using those best hyperparameters fitted on
the outer training fold. Results (hyperparameters selected and F1 score achieved) are
saved to JSON files.
AI Use: GPT-5 was used as a collaborator for conceptual understanding of the steps involved in
nesed cross-validation, and to ensure no data leakage occured. It was also used for help with
how to save the parameters and metrics to a JSON file.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from upsample import naive_upsample, get_natural_kappas
from logreg.util import calculate_sample_weight
from kprototypes import run_k_prototypes
from gmm import gmm_cluster_upsample
from itertools import product
import json
from src.nn.neuralnetwork import nn_train, forward_prop, backward_prop, get_initial_params, one_hot_labels
from joblib import Parallel, delayed

# identify numeric vs. cat cols
numeric_cols = ["RIDAGEYR", "LBXTC", "LBDHDD", "LBXSTR", "LBXSCR", "LBXHSCRP", "DBP_mean", "SBP_mean", "BMXBMI", "BMXHIP", "SMQ020"]
cat_cols = ["DMDEDUC2", "RIDRETH3", "RIAGENDR"]

# the same preprocessing functions from 05_process_before_test
def preprocess_fit_transform(X_train):
    mice = IterativeImputer(random_state=3, max_iter=20)
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    X_train_imputed = pd.DataFrame(mice.fit_transform(X_train[numeric_cols]), columns=numeric_cols, index=X_train.index) # MICE imputation on numeric features
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=numeric_cols, index=X_train.index) # scale numeric features
    X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[cat_cols]), columns=encoder.get_feature_names_out(cat_cols), index=X_train.index) # one-hot encode categorical features
    X_train_processed = pd.concat([X_train_scaled, X_train_encoded], axis=1)

    return X_train_processed, mice, scaler, encoder

# the same preprocessing functions from 05_process_before_test
def preprocess_transform(X_val, mice, scaler, encoder):
    X_numeric = pd.DataFrame(mice.transform(X_val[numeric_cols]), columns=numeric_cols, index=X_val.index)
    X_scaled = pd.DataFrame(scaler.transform(X_numeric), columns=numeric_cols, index=X_val.index)
    X_cat = pd.DataFrame(encoder.transform(X_val[cat_cols]), columns=encoder.get_feature_names_out(cat_cols), index=X_val.index)

    return pd.concat([X_scaled, X_cat], axis=1)

# define function to calculate F1 score from the model's predicted probablities and true y values
def f1_from_probs(y_true, probs, threshold):
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
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
    return f1

# define function to fit an MLP using NumPy
def fit_numpy_mlp(X_train, y_train, X_dev, y_dev, hidden_width, lr, batch_size, activation, dropout, weight_decay, num_epochs, sample_weights):
    y_train_oh = one_hot_labels(y_train.to_numpy(), num_classes=2)
    y_dev_oh = one_hot_labels(y_dev.to_numpy(), num_classes=2)

    params, _, _, _, _ = nn_train(
        train_data=X_train.to_numpy(),
        train_labels=y_train_oh,
        dev_data=X_dev.to_numpy(),
        dev_labels=y_dev_oh,
        get_initial_params_func=get_initial_params,
        forward_prop_func=forward_prop,
        backward_prop_func=backward_prop,
        num_hidden=hidden_width,
        learning_rate=lr,
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_classes=2,
        activation=activation,
        dropout_rate=dropout,
        reg=weight_decay,
        sample_weights=sample_weights
    )
    return params

# define function to obtain class probabilities from NumPy MLP
def predict_probs_numpy_mlp(X, params, activation):
    dummy_labels = np.zeros((X.shape[0], 2))
    _, output, _ = forward_prop(
        X.to_numpy(), dummy_labels, params,
        is_training=False, activation=activation, dropout_rate=0.0)
    return output[:, 1]

# encapsulate cv code into its own function
def _eval_params_inner(params, X_train_f, y_train_f, experiment_type, nat_kap_1, nat_kap_4, nat_kap_6, inner_splits, random_state):
    # evaluate a single hyperparameter combo via inner CV; returns (mean_f1, params)
    scores_inner = []
    # define stratified folds to keep consistent diabetes prevalence across folds
    skf_inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)

    # experiment-specific params
    if experiment_type == "baseline":
        lr, hidden_width, weight_decay, dropout, batch_size, activation, num_epochs, threshold = params
        gamma = None
        n_clusters = None
        kappa_1 = None
        kappa_4 = None
        kappa_6 = None
        sample_weight = None
        n_comps = None
        kappa_mult_1 = None
        kappa_mult_4 = None
        kappa_mult_6 = None
    elif experiment_type == "upsample":
        lr, hidden_width, weight_decay, dropout, batch_size, activation, num_epochs, threshold, kappa_mult_1, kappa_mult_4, kappa_mult_6 = params
        kappa_1 = nat_kap_1 * kappa_mult_1
        kappa_4 = nat_kap_4 * kappa_mult_4
        kappa_6 = nat_kap_6 * kappa_mult_6
        gamma = None
        n_clusters = None
        sample_weight = None
        n_comps = None
    elif experiment_type == "cluster":
        lr, hidden_width, weight_decay, dropout, batch_size, activation, num_epochs, gamma, n_clusters, threshold = params
        kappa_1 = None
        kappa_4 = None
        kappa_6 = None
        sample_weight = None
        n_comps = None
        kappa_mult_1 = None
        kappa_mult_4 = None
        kappa_mult_6 = None
    elif experiment_type == "cost_sensitive":
        lr, hidden_width, weight_decay, dropout, batch_size, activation, num_epochs, threshold = params
        gamma = None
        n_clusters = None
        kappa_1 = None
        kappa_4 = None
        kappa_6 = None
        sample_weight = None
        n_comps = None
        kappa_mult_1 = None
        kappa_mult_4 = None
        kappa_mult_6 = None
    elif experiment_type == "gmm":
        lr, hidden_width, weight_decay, dropout, batch_size, activation, num_epochs, threshold, n_comps = params
        gamma = None
        n_clusters = None
        kappa_1 = None
        kappa_4 = None
        kappa_6 = None
        sample_weight = None
        kappa_mult_1 = None
        kappa_mult_4 = None
        kappa_mult_6 = None

    for (train_idx_inner, val_idx_inner) in skf_inner.split(X_train_f, y_train_f):
        X_train_f_inner = X_train_f.iloc[train_idx_inner]
        y_train_f_inner = y_train_f.iloc[train_idx_inner]
        X_val_f_inner = X_train_f.iloc[val_idx_inner]
        y_val_f_inner = y_train_f.iloc[val_idx_inner]

        # apply preprocessing pipeline to inner train fold
        X_train_inner_preprocessed, mice, scaler, encoder = preprocess_fit_transform(X_train_f_inner)
        X_val_inner_preprocessed = preprocess_transform(X_val_f_inner, mice, scaler, encoder)

        # manipulate fold for specific experiment
        # upsampling
        if experiment_type == "upsample":
            train_set_inner = pd.concat([X_train_inner_preprocessed, y_train_f_inner.rename("diabetes")], axis=1)
            train_set_inner = naive_upsample(train_set_inner, kappa_1=kappa_1, kappa_4=kappa_4, kappa_6=kappa_6)
            y_train_f_inner = train_set_inner["diabetes"]
            X_train_inner_preprocessed = train_set_inner.drop(columns=["diabetes"])
        
        # clustering
        elif experiment_type == "cluster":
            # clustering code. do this before one hot encoding, after scaling/imputation?
            train_set_inner = pd.concat([X_train_inner_preprocessed, y_train_f_inner.rename("diabetes")], axis=1)
            train_set_inner = run_k_prototypes(train_set_inner, gamma = gamma, n_clusters = n_clusters)
            y_train_f_inner = train_set_inner["diabetes"]
            X_train_inner_preprocessed = train_set_inner.drop(columns=["diabetes"])

        # cost-sensitive
        elif experiment_type == "cost_sensitive":
            train_set_inner = pd.concat([X_train_inner_preprocessed, y_train_f_inner.rename("diabetes")], axis=1)
            sample_weight = calculate_sample_weight(train_set_inner)
            y_train_f_inner = train_set_inner["diabetes"]
            X_train_inner_preprocessed = train_set_inner.drop(columns=["diabetes"])
        
        # gmm
        elif experiment_type == "gmm":
            train_set_inner = pd.concat([X_train_inner_preprocessed, y_train_f_inner.rename("diabetes")], axis=1)
            train_set_inner = gmm_cluster_upsample(train_set_inner, n_components=n_comps)
            y_train_f_inner = train_set_inner["diabetes"]
            X_train_inner_preprocessed = train_set_inner.drop(columns=["diabetes"])

        # fit NumPy MLP using specific hyperparameter combination
        params_nn = fit_numpy_mlp(
            X_train=X_train_inner_preprocessed,
            y_train=y_train_f_inner,
            X_dev=X_val_inner_preprocessed,
            y_dev=y_val_f_inner,
            hidden_width=hidden_width,
            lr=lr,
            batch_size=batch_size,
            activation=activation,
            dropout=dropout,
            weight_decay=weight_decay,
            num_epochs=num_epochs,
            sample_weights=sample_weight
        )
        
        # predict class probabilities
        probs = predict_probs_numpy_mlp(
            X_val_inner_preprocessed, params_nn, activation=activation)
        
        scores_inner.append(f1_from_probs(y_val_f_inner, probs, threshold))
        # inner model fit

    return float(np.mean(scores_inner)), params

def cv_tune_pipeline_nn(experiment_type = "baseline", n_splits = 5, inner_splits = 3, random_state = 3):

    np.random.seed(random_state)
    df = pd.read_csv("src/data/model_ready/train_raw.csv")
    y = df["diabetes"].astype(int)
    X = df.drop(columns=["diabetes"])

    # specify grid of parameters to search through
    lr_grid = [0.01, 0.03, 0.05]
    hidden_width_grid = [16, 32, 64]
    weight_decay_grid = [0.0, 1e-4, 1e-3]
    dropout_grid = [0.0, 0.2]
    batch_size_grid = [64, 128]
    activation_grid = ['relu', 'sigmoid']
    num_epochs_grid = [30, 50]
    threshold_grid = [0.35, 0.45, 0.55]
    n_comps_grid = [2, 3, 4, 5, 6]
    gamma_grid = [0.5, 1.0, 1.5]
    n_clusters_grid = [3, 4, 5]

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
        # compute natural kappas for upsampling
        nat_kap_1, nat_kap_4, nat_kap_6 = get_natural_kappas(X_train_f)
        # kappa multiplier grid for upsampling experiment
        kappa_mult_grid = [1.0, 1.5]

        # define parameter grids (list(product() creates all possible combinations of the specified hyperparameters)
        if experiment_type == "baseline":
            param_grid = list(product(lr_grid, hidden_width_grid, weight_decay_grid, dropout_grid, batch_size_grid, activation_grid, num_epochs_grid, threshold_grid))
        elif experiment_type == "upsample":
            param_grid = list(product(lr_grid, hidden_width_grid, weight_decay_grid, dropout_grid, batch_size_grid, activation_grid, num_epochs_grid, threshold_grid, kappa_mult_grid, kappa_mult_grid, kappa_mult_grid))
        elif experiment_type == "cluster":
            param_grid = list(product(lr_grid, hidden_width_grid, weight_decay_grid, dropout_grid, batch_size_grid, activation_grid, num_epochs_grid, gamma_grid, n_clusters_grid, threshold_grid))
        elif experiment_type == "cost_sensitive":
            param_grid = list(product(lr_grid, hidden_width_grid, weight_decay_grid, dropout_grid, batch_size_grid, activation_grid, num_epochs_grid, threshold_grid))
        elif experiment_type == "gmm":
            param_grid = list(product(lr_grid, hidden_width_grid, weight_decay_grid, dropout_grid, batch_size_grid, activation_grid, num_epochs_grid, threshold_grid, n_comps_grid))

        score_star = -np.inf
        params_star = None

        # loop through all possible hyperparam combinations, do inner 3-fold cv on them. Run in parallel.
        results = Parallel(n_jobs=-1)(
            delayed(_eval_params_inner)(params, X_train_f, y_train_f, experiment_type, nat_kap_1, nat_kap_4, nat_kap_6, inner_splits, random_state)
            for params in param_grid
        )

        # running update of optimal parameters
        for mean_inner, params in results:
            if mean_inner > score_star:
                score_star = mean_inner
                params_star = params

        # now that we have the best parameters, refit the model on outer train fold with those params
        # experiment-specific params
        if experiment_type == "baseline":
            lr, hidden_width, weight_decay, dropout, batch_size, activation, num_epochs, threshold = params_star
            gamma = None
            n_clusters = None
            kappa_1 = None
            kappa_4 = None
            kappa_6 = None
            sample_weight = None
            n_comps = None
            kappa_mult_1 = None
            kappa_mult_4 = None
            kappa_mult_6 = None
        elif experiment_type == "upsample":
            lr, hidden_width, weight_decay, dropout, batch_size, activation, num_epochs, threshold, kappa_mult_1, kappa_mult_4, kappa_mult_6 = params_star
            kappa_1 = nat_kap_1 * kappa_mult_1
            kappa_4 = nat_kap_4 * kappa_mult_4
            kappa_6 = nat_kap_6 * kappa_mult_6
            gamma = None
            n_clusters = None
            sample_weight = None
            n_comps = None
        elif experiment_type == "cluster":
            lr, hidden_width, weight_decay, dropout, batch_size, activation, num_epochs, gamma, n_clusters, threshold = params_star
            kappa_1 = None
            kappa_4 = None
            kappa_6 = None
            sample_weight = None
            n_comps = None
            kappa_mult_1 = None
            kappa_mult_4 = None
            kappa_mult_6 = None
        elif experiment_type == "cost_sensitive":
            lr, hidden_width, weight_decay, dropout, batch_size, activation, num_epochs, threshold = params_star
            gamma = None
            n_clusters = None
            kappa_1 = None
            kappa_4 = None
            kappa_6 = None
            sample_weight = None
            n_comps = None
            kappa_mult_1 = None
            kappa_mult_4 = None
            kappa_mult_6 = None
        elif experiment_type == "gmm":
            lr, hidden_width, weight_decay, dropout, batch_size, activation, num_epochs, threshold, n_comps = params_star
            gamma = None
            n_clusters = None
            kappa_1 = None
            kappa_4 = None
            kappa_6 = None
            sample_weight = None
            kappa_mult_1 = None
            kappa_mult_4 = None
            kappa_mult_6 = None
        
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
        
        elif experiment_type == "gmm":
            train_set = pd.concat([X_train_preprocessed, y_train_f.rename("diabetes")], axis=1)
            train_set = gmm_cluster_upsample(train_set, n_components=n_comps)
            y_train_f = train_set["diabetes"]
            X_train_preprocessed = train_set.drop(columns=["diabetes"])

        params_nn = fit_numpy_mlp(
                    X_train=X_train_preprocessed,
                    y_train=y_train_f,
                    X_dev=X_val_preprocessed,
                    y_dev=y_val_f,
                    hidden_width=hidden_width,
                    lr=lr,
                    batch_size=batch_size,
                    activation=activation,
                    dropout=dropout,
                    weight_decay=weight_decay,
                    num_epochs=num_epochs,
                    sample_weights=sample_weight
                )
        
        # outer fold evaluation
        probs = predict_probs_numpy_mlp(X_val_preprocessed, params_nn, activation=activation)
        f1 = f1_from_probs(y_val_f, probs, threshold)

        # dict of metrics for each fold
        metrics.append({"fold": f, "f1": f1,
                        "inner_f1_star": score_star,
                        "lr": lr,
                        "hidden_width": hidden_width,
                        "weight_decay": weight_decay,
                        "dropout": dropout,
                        "batch_size": batch_size,
                        "activation": activation,
                        "num_epochs": num_epochs,
                        "threshold": threshold,
                        "gamma": gamma,
                        "n_clusters": n_clusters,
                        "n_comps": n_comps,
                        "kappa_1": kappa_1,
                        "kappa_4": kappa_4,
                        "kappa_6": kappa_6,
                        "kappa_mult_1": kappa_mult_1,
                        "kappa_mult_4": kappa_mult_4,
                        "kappa_mult_6": kappa_mult_6,
                        "natural_kappa_1": nat_kap_1,
                        "natural_kappa_4": nat_kap_4,
                        "natural_kappa_6": nat_kap_6})
        #"sample_weight": sample_weight,
    
    # convert the metrics to a pd df
    fold_metrics = pd.DataFrame(metrics)

    return {
        "experiment_type": experiment_type,
        "fold_metrics": fold_metrics,
        "summary" : {"f1_mean": fold_metrics["f1"].mean(),
                     "f1_std": fold_metrics["f1"].std()}}

# run neural network cv for each experiment type and save the results as a JSON
def main():
    baseline_metrics_dict = cv_tune_pipeline_nn()
    baseline_metrics_dict["fold_metrics"] = baseline_metrics_dict["fold_metrics"].to_dict(orient="records")

    baseline_save_path = 'src/metrics/baseline_nn_parameters.json'
    with open(baseline_save_path, mode = 'w') as file:
        json.dump(baseline_metrics_dict, file, indent = 4)

    print(f"JSON file '{baseline_save_path}' created successfully")

    upsample_metrics_dict = cv_tune_pipeline_nn(experiment_type="upsample")
    upsample_metrics_dict["fold_metrics"] = upsample_metrics_dict["fold_metrics"].to_dict(orient="records")

    upsample_save_path = 'src/metrics/upsample_nn_parameters.json'
    with open(upsample_save_path, mode = 'w') as file:
        json.dump(upsample_metrics_dict, file, indent = 4)

    print(f"JSON file '{upsample_save_path}' created successfully")

    cluster_metrics_dict = cv_tune_pipeline_nn(experiment_type="cluster")
    cluster_metrics_dict["fold_metrics"] = cluster_metrics_dict["fold_metrics"].to_dict(orient="records")

    cluster_save_path = 'src/metrics/cluster_nn_parameters.json'
    with open(cluster_save_path, mode = 'w') as file:
        json.dump(cluster_metrics_dict, file, indent = 4)

    print(f"JSON file '{cluster_save_path}' created successfully")

    cost_metrics_dict = cv_tune_pipeline_nn(experiment_type="cost_sensitive")
    cost_metrics_dict["fold_metrics"] = cost_metrics_dict["fold_metrics"].to_dict(orient="records")

    cost_save_path = 'src/metrics/cost_nn_parameters.json'
    with open(cost_save_path, mode = 'w') as file:
        json.dump(cost_metrics_dict, file, indent = 4)

    print(f"JSON file '{cost_save_path}' created successfully")

    gmm_metrics_dict = cv_tune_pipeline_nn(experiment_type="gmm")
    gmm_metrics_dict["fold_metrics"] = gmm_metrics_dict["fold_metrics"].to_dict(orient="records")

    gmm_save_path = 'src/metrics/gmm_nn_parameters.json'
    with open(gmm_save_path, mode = 'w') as file:
        json.dump(gmm_metrics_dict, file, indent = 4)

    print(f"JSON file '{gmm_save_path}' created successfully")

if __name__ == '__main__':
    main()