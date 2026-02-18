from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
import os

# load preprocessed dataset
df = pd.read_csv('src/data/analysis_ready/nhanes_joined_2017_2018.csv')
X = df.drop(columns=["diabetes"])
y = df["diabetes"]

# don't change the seed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3, stratify=y)

# split into numeric and non-numeric cols as mice only takes in numeric
num_vars = X_train.select_dtypes(include="number").columns
non_num_vars = X_train.select_dtypes(exclude="number").columns
X_train_non = X_train[non_num_vars].copy()
X_test_non  = X_test[non_num_vars].copy()

# imputation only on numeric
mice = IterativeImputer(random_state=3, max_iter=20, skip_complete=True)

X_train_num_imp = pd.DataFrame(mice.fit_transform(X_train[num_vars]), columns=num_vars, index=X_train.index)
X_test_num_imp = pd.DataFrame(mice.transform(X_test[num_vars]), columns=num_vars, index=X_test.index)

# rejoin and keep same column order
X_train_imp = pd.concat([X_train_num_imp, X_train_non], axis=1)[X_train.columns]
X_test_imp  = pd.concat([X_test_num_imp,  X_test_non],  axis=1)[X_test.columns]

os.makedirs("src/data/model_ready", exist_ok=True)

X_train_imp.to_csv("src/data/model_ready/X_train_imputed.csv", index=True)
y_train.to_csv("src/data/model_ready/y_train.csv", index=True)
X_test_imp.to_csv("src/data/model_ready/X_test_imputed.csv", index=True)
y_test.to_csv("src/data/model_ready/y_test.csv", index=True)
