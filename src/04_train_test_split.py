"""
04_train_test_split creates the master train and test set for this
project, before any imputation/feature scaling/one-hot encoding.
It rebalances the diabetes label per ethnicity, on the training set only.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# load preprocessed dataset (pre-imputation, scaling, OHE)
df = pd.read_csv("src/data/analysis_ready/nhanes_diabetes_base.csv", na_values=["."])

# create X and y
patient_id = df["SEQN"]
X = df.drop(columns=["diabetes", "SEQN"])
y = df["diabetes"].astype(int)

# create training set
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(X, y, patient_id, test_size=0.20, random_state=3, stratify=y)

train = pd.concat([X_train, y_train, id_train], axis=1)
test = pd.concat([X_test, y_test, id_test], axis=1)

ethnicities = [1.0, 3.0, 4.0, 6.0]
# list of dfs
ethnicities_balanced = []
    # find exact count of the majority class (no diabetes) in training set, for that ethnicity
for ethn in ethnicities:
    # isolate diabetes / no diabetes
    majority = train[(train["RIDRETH3"] == ethn) & (train["diabetes"] == 0)] # patients for that ethn w/o diabetes
    minority = train[(train["RIDRETH3"] == ethn) & (train["diabetes"] == 1)] # patients for that ethn w/ diabetes

    # calculate how many new samples are needed to "top off" each class
    n_needed = len(majority) - len(minority)
    # sample the difference with replacement
    minority_extra = minority.sample(n=n_needed, replace=True, random_state=3)
    # combine the majority class, original minority class, and the extra samples
    train_ethn_bal = pd.concat([majority, minority, minority_extra], ignore_index=True)
    ethnicities_balanced.append(train_ethn_bal)
    # verify
    print("--- TOP OFF LABEL REBALANCING ---")
    print(f"Diabetes: {train_ethn_bal['diabetes'].sum()}") # for that ethnicity print how many diabetes patients
    print(f"Total:  {train_ethn_bal.shape[0]}") # for that ethnicity print total patients. should be the above x 2

train_balanced = pd.concat(ethnicities_balanced, ignore_index=True)
# shuffle so that duplicates aren't grouped
train_balanced = train_balanced.sample(frac=1.0, random_state=3).reset_index(drop=True)
print(train_balanced.groupby("RIDRETH3")["diabetes"].agg(n="count", positives="sum", prevalence="mean")) # check diabetes prevalence per ethincity on balanced training set

# create directory if it doesn't exist
model_ready_dir = Path('src/data/model_ready')
model_ready_dir.mkdir(parents=True, exist_ok=True)

train_balanced_save = train_balanced.drop(columns=["SEQN"])
test_save = test.drop(columns=["SEQN"])
train_balanced_save.to_csv("src/data/model_ready/train_raw.csv", index=False)
test_save.to_csv("src/data/model_ready/test_raw.csv", index=False)