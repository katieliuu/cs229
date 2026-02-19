from pathlib import Path
import numpy as np
import pandas as pd
from nhanes_io import load_xpt_folder, left_join_many

KEY = "SEQN"
tables = load_xpt_folder("src/data/raw_xpt")
sorted(tables.keys())

base = tables["DEMO_J"]
JOIN_TABLES = [
    "BPX_J",
    "ALB_CR_J",
    "BIOPRO_J",
    "GHB_J",
    "GLU_J",
    "HDL_J",
    "HSCRP_J",
    "INS_J",
    "PAQ_J",
    "SMQ_J",
    "TCHOL_J",
    "TRIGLY_J",
    "BMX_J"
    ]

df = left_join_many(base=base, tables=tables, table_names=JOIN_TABLES, key=KEY)
df.shape

# Check tables joined correctly
print("Base rows:", base.shape[0])
print("Final rows:", df.shape[0])
df["SEQN"].is_unique

df.info()
# only keep adults
df = df[df["RIDAGEYR"] >= 20]
# remove 'other'
df = df[(df["RIDRETH3"] != 7.0) & (df["RIDRETH3"] != 2.0)]
df["LBXGLU"].isna().mean()
df["LBXGH"].isna().mean()
# only keep obs where LBXGH is not missing
df = df[df["LBXGH"].notna()].copy()
# clinical diabetes threshold
df["diabetes"] = (df["LBXGH"] >= 6.5).astype(int)
df = df.drop(columns=["LBXGH"])

# combine the multiple BP readings
df["SBP_mean"] = df[["BPXSY1", "BPXSY2", "BPXSY3"]].mean(axis=1)
df["DBP_mean"] = df[["BPXDI1", "BPXDI2", "BPXDI3"]].mean(axis=1)
# only keep columns relevant to diabetes
cols_for_analysis = ["RIAGENDR", "RIDAGEYR", "RIDRETH3", "DMDEDUC2", "LBXTC", "LBDHDD", "LBXSTR", "LBXSCR", "LBXHSCRP", "LBXGLU", "DBP_mean", "SBP_mean", "BMXBMI", "BMXHIP", "SMQ020", "diabetes"]
df = df[cols_for_analysis]

df.groupby("RIAGENDR")["diabetes"].agg(n="count", positives="sum", prevalence="mean")
df.groupby("RIDRETH3")["diabetes"].agg(n="count", positives="sum", prevalence="mean")

pct_missing = df.isna().mean()
df = df.loc[:, pct_missing <= 0.40]
df.isna().mean().sort_values(ascending=False)
cat_cols = ["RIDRETH3", "RIAGENDR", "DMDEDUC2"]
for col in cat_cols:
    df[col] = df[col].astype("category")

# Save csv to disk
OUT_DIR = Path("src/data/analysis_ready")
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "nhanes_joined_2017_2018.csv"
df.to_csv(out_path, index=False)

