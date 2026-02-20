"""
04_train_test_split_impute splits the dataset into training, validation
and testing sets, performs MICE (iterative imputation) on the numeric columns,
one-hot encodes the categorical columns, standardizes the numeric features,
and outputs 3 csv files corresponding to the train, validation and test sets.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# load preprocessed dataset
df = pd.read_csv("src/data/analysis_ready/nhanes_joined_2017_2018.csv", na_values=["."])

patient_id = df["SEQN"]
X = df.drop(columns=["diabetes", "SEQN"])
y = df["diabetes"].astype(int)

# create training set
X_train_full, X_test, y_train_full, y_test, id_train_full, id_test = train_test_split(X, y, patient_id, test_size=0.20, random_state=3, stratify=y)
# create val set inside train set
X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(X_train_full, y_train_full, id_train_full, test_size=0.20, random_state=3, stratify=y_train_full)

# deal with numeric columns
numeric_cols = ["RIDAGEYR", "LBXTC", "LBDHDD", "LBXSTR", "LBXSCR", "LBXHSCRP", "DBP_mean", "SBP_mean", "BMXBMI", "BMXHIP", "SMQ020"]
mice = IterativeImputer(random_state=3, max_iter=20)
X_train_imputed = pd.DataFrame(mice.fit_transform(X_train[numeric_cols]), columns=numeric_cols, index=X_train.index)
X_val_imputed = pd.DataFrame(mice.transform(X_val[numeric_cols]), columns=numeric_cols, index=X_val.index)
X_test_imputed = pd.DataFrame(mice.transform(X_test[numeric_cols]), columns=numeric_cols, index=X_test.index)

# scale numeric features
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=numeric_cols, index=X_train.index)
X_val_scaled = pd.DataFrame(scaler.transform(X_val_imputed), columns=numeric_cols, index=X_val.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=numeric_cols, index=X_test.index)

# deal with categorical columns
cat_cols = ["DMDEDUC2", "RIDRETH3", "RIAGENDR"]
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[cat_cols]), columns=encoder.get_feature_names_out(cat_cols), index=X_train.index)
X_val_encoded = pd.DataFrame(encoder.transform(X_val[cat_cols]), columns=encoder.get_feature_names_out(cat_cols), index=X_val.index)
X_test_encoded = pd.DataFrame(encoder.transform(X_test[cat_cols]), columns=encoder.get_feature_names_out(cat_cols), index=X_test.index)

train_cols = [X_train_scaled, X_train_encoded, y_train, id_train]
val_cols = [X_val_scaled, X_val_encoded, y_val, id_val]
test_cols = [X_test_scaled, X_test_encoded, y_test, id_test]

train = pd.concat(train_cols, axis=1)
val = pd.concat(val_cols, axis=1)
test = pd.concat(test_cols, axis=1)

train.to_csv("src/data/model_ready/train_processed.csv", index=False)
val.to_csv("src/data/model_ready/val_processed.csv", index=False)
test.to_csv("src/data/model_ready/test_processed.csv", index=False)