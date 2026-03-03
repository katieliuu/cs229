"""
04_train_test_split creates the master train and test set for this
project, before any imputation/feature scaling/one-hot encoding.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

# load preprocessed dataset (pre-imputation, scaling, OHE)
df = pd.read_csv("src/data/analysis_ready/nhanes_diabetes_base.csv", na_values=["."])

patient_id = df["SEQN"]
X = df.drop(columns=["diabetes", "SEQN"])
y = df["diabetes"].astype(int)

# create training set
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(X, y, patient_id, test_size=0.20, random_state=3, stratify=y)

train = pd.concat([X_train, y_train, id_train], axis=1)
test = pd.concat([X_test, y_test, id_test], axis=1)

train.to_csv("src/data/model_ready/train_raw.csv", index=False)
test.to_csv("src/data/model_ready/test_raw.csv", index=False)