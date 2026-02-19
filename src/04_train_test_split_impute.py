from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

RANDOM_STATE = 3
TEST_SIZE = 0.20
VAL_SIZE = 0.20

IN_PATH = Path("src/data/analysis_ready/nhanes_joined_2017_2018.csv")
OUT_DIR = Path("src/data/model_ready")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# load preprocessed dataset
df = pd.read_csv(IN_PATH)
X = df.drop(columns=["diabetes"])
y = df["diabetes"]

# create training set
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
# create val set inside train set
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train_full
)
# specify non-numeric cols
cat_cols = []
for c in ["RIAGENDR", "RIDRETH3", "DMDEDUC2"]:
    if c in X_train.columns:
        cat_cols.append(c)

num_cols = X_train.columns.difference(cat_cols)

mice = IterativeImputer(random_state=RANDOM_STATE, max_iter=20, skip_complete=True)

# mice only on numeric columns
X_train_num_imp = pd.DataFrame(mice.fit_transform(X_train[num_cols]), columns=num_cols, index=X_train.index)
X_val_num_imp = pd.DataFrame(mice.transform(X_val[num_cols]), columns=num_cols, index=X_val.index)
X_test_num_imp = pd.DataFrame(mice.transform(X_test[num_cols]), columns=num_cols, index=X_test.index)

# function to rejoin categoricals to imputed dataset
def attach_categoricals(X_num_imp: pd.DataFrame, X_orig: pd.DataFrame) -> pd.DataFrame:
    X_cat = X_orig[cat_cols].copy() if cat_cols else pd.DataFrame(index=X_orig.index)
    for c in cat_cols:
        X_cat[c] = X_cat[c].astype("category")
    X_imp = pd.concat([X_num_imp, X_cat], axis=1)
    X_imp = X_imp[X_orig.columns]
    return X_imp

X_train_imp = attach_categoricals(X_train_num_imp, X_train)
X_val_imp   = attach_categoricals(X_val_num_imp, X_val)
X_test_imp  = attach_categoricals(X_test_num_imp, X_test)

X_train_imp.to_csv(OUT_DIR / "X_train_imputed.csv", index=True)
y_train.to_csv(OUT_DIR / "y_train.csv", index=True)
X_val_imp.to_csv(OUT_DIR / "X_val_imputed.csv", index=True)
y_val.to_csv(OUT_DIR / "y_val.csv", index=True)
X_test_imp.to_csv(OUT_DIR / "X_test_imputed.csv", index=True)
y_test.to_csv(OUT_DIR / "y_test.csv", index=True)
