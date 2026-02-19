from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 3
TEST_SIZE = 0.20
VAL_SIZE = 0.20

IN_PATH = Path("src/data/analysis_ready/nhanes_joined_2017_2018.csv")
OUT_DIR = Path("src/data/model_ready")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COL = "diabetes"

# helper functions
def clean_binary_smq(s: pd.Series) -> pd.Series:
    """
    SMQ020 encodings:
      1 = Yes
      2 = No
      . = Missing
    """
    s = s.copy()
    s = s.replace(".", np.nan)
    s = s.replace({1: 1, 2: 0})
    return s


def postprocess_smq(df_imp: pd.DataFrame) -> pd.DataFrame:
    """
    After MICE, SMQ020 may be fractional. Convert back to binary.
    """
    df_imp = df_imp.copy()
    if "SMQ020" in df_imp.columns:
        df_imp["SMQ020"] = np.round(df_imp["SMQ020"]).astype(int).clip(0, 1)
    return df_imp


def save_split(X_df: pd.DataFrame, y_s: pd.Series, name: str) -> None:
    out = X_df.copy()
    out[LABEL_COL] = y_s.values
    out.to_csv(OUT_DIR / f"{name}.csv", index=False)

# load preprocessed dataset
df = pd.read_csv(IN_PATH, na_values=["."])

X = df.drop(columns=[LABEL_COL])
y = df[LABEL_COL].astype(int)

# create training set
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
# create val set inside train set
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train_full)

# one-hot encode categorical columbs
cat_cols = [c for c in ["DMDEDUC2", "RIDRETH3", "RIAGENDR", "SMQ020"] if c in X_train.columns]

# Clean SMQ020 before imputation
for split in [X_train, X_val, X_test]:
    if "SMQ020" in split.columns:
        split["SMQ020"] = clean_binary_smq(split["SMQ020"])

mice_cols = X_train.columns
mice = IterativeImputer(random_state=RANDOM_STATE, max_iter=20, skip_complete=True)

X_train_imp = pd.DataFrame(
    mice.fit_transform(X_train[mice_cols]),
    columns=mice_cols,
    index=X_train.index,)

X_val_imp = pd.DataFrame(
    mice.transform(X_val[mice_cols]),
    columns=mice_cols,
    index=X_val.index,)

X_test_imp = pd.DataFrame(
    mice.transform(X_test[mice_cols]),
    columns=mice_cols,
    index=X_test.index,)

# Post-process SMQ020 back to {0,1}
X_train_imp = postprocess_smq(X_train_imp)
X_val_imp = postprocess_smq(X_val_imp)
X_test_imp = postprocess_smq(X_test_imp)

if cat_cols:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(X_train_imp[cat_cols])

    ohe_names = list(ohe.get_feature_names_out(cat_cols))

    def apply_ohe(df_imp: pd.DataFrame) -> pd.DataFrame:
        ohe_arr = ohe.transform(df_imp[cat_cols])
        ohe_df = pd.DataFrame(ohe_arr, columns=ohe_names, index=df_imp.index)
        df_rest = df_imp.drop(columns=cat_cols)
        return pd.concat([df_rest, ohe_df], axis=1)

    X_train_enc = apply_ohe(X_train_imp)
    X_val_enc = apply_ohe(X_val_imp)
    X_test_enc = apply_ohe(X_test_imp)

else:
    ohe_names = []
    X_train_enc, X_val_enc, X_test_enc = X_train_imp, X_val_imp, X_test_imp

cont_cols = [c for c in X_train_enc.columns if c not in ohe_names]

scaler = StandardScaler()
scaler.fit(X_train_enc[cont_cols])

def apply_scaler(df_enc: pd.DataFrame) -> pd.DataFrame:
    df_enc = df_enc.copy()
    df_enc[cont_cols] = scaler.transform(df_enc[cont_cols])
    return df_enc

X_train_final = apply_scaler(X_train_enc)
X_val_final = apply_scaler(X_val_enc)
X_test_final = apply_scaler(X_test_enc)

save_split(X_train_final, y_train, "train_processed")
save_split(X_val_final, y_val, "val_processed")
save_split(X_test_final, y_test, "test_processed")