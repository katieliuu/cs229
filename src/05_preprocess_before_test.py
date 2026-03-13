"""
05_preprocess_before_test is used to preprocess the full training set,
after cross-validation and before final evaluation on the held out test set.
It performs MICE imputation and standardization on numeric features, and one-hot
encoding on the categorical features.
AI Use: GPT-5 was used for conceptual understanding of the sequential ordering
of preprocessing steps and for help with concatenating the preprocessed columns.
"""
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# specify numerical and categorical functions
numeric_cols = ["RIDAGEYR", "LBXTC", "LBDHDD", "LBXSTR", "LBXSCR", "LBXHSCRP", "DBP_mean", "SBP_mean", "BMXBMI", "BMXHIP", "SMQ020"]
cat_cols = ["DMDEDUC2", "RIDRETH3", "RIAGENDR"]

# encapsulate the preprocessing pipeline into one function
def preprocess_fit_transform(X_train):
    mice = IterativeImputer(random_state=3, max_iter=20)
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# fit each on the training set:
    # impute numeric columns
    X_train_imputed = pd.DataFrame(mice.fit_transform(X_train[numeric_cols]), columns=numeric_cols, index=X_train.index) # MICE imputation on numeric features
    # standardiza numeric columns
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=numeric_cols, index=X_train.index) # scale numeric features
    # one-hot encode categorical features
    X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[cat_cols]), columns=encoder.get_feature_names_out(cat_cols), index=X_train.index) # one-hot encode categorical features
    # join the processed train columns together
    X_train_processed = pd.concat([X_train_scaled, X_train_encoded], axis=1)

    return X_train_processed, mice, scaler, encoder

# transform the test data using the fitted preprocessing objects (mice, scaler, encoder)
def preprocess_transform(X_val, mice, scaler, encoder):
    X_numeric = pd.DataFrame(mice.transform(X_val[numeric_cols]), columns=numeric_cols, index=X_val.index)
    X_scaled = pd.DataFrame(scaler.transform(X_numeric), columns=numeric_cols, index=X_val.index)
    X_cat = pd.DataFrame(encoder.transform(X_val[cat_cols]), columns=encoder.get_feature_names_out(cat_cols), index=X_val.index)

    return pd.concat([X_scaled, X_cat], axis=1)

train_df = pd.read_csv("src/data/model_ready/train_raw.csv")
test_df = pd.read_csv("src/data/model_ready/test_raw.csv")

# split train and test datasets into X and y
y_train = train_df["diabetes"].astype(int)
X_train = train_df.drop(columns=["diabetes"])
y_test = test_df["diabetes"].astype(int)
X_test = test_df.drop(columns=["diabetes"])

# call the above helpers on the full training set
X_train_preprocessed, mice, scaler, encoder = preprocess_fit_transform(X_train)
X_test_preprocessed = preprocess_transform(X_test, mice, scaler, encoder)
train_processed = X_train_preprocessed.copy()
train_processed["diabetes"] = y_train.values
test_processed = X_test_preprocessed.copy()
test_processed["diabetes"] = y_test.values

# save prepared data for evaluation on test set
train_processed.to_csv("src/data/model_ready/train_processed.csv", index=False)
test_processed.to_csv("src/data/model_ready/test_processed.csv", index=False)