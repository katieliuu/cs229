"""
03_analysis loads the partially preprocessed csv and performs some
data exploration, filtering and cleaning. It saves a new version of
the preprocessed dataset as a csv.
"""
from pathlib import Path
import numpy as np
import pandas as pd

df = pd.read_csv("src/data/analysis_ready/nhanes_joined_2017_2018.csv")

# Cleaning:
# only keep adults
df = df[df["RIDAGEYR"] >= 20]
# remove 'other'
df = df[(df["RIDRETH3"] != 7.0) & (df["RIDRETH3"] != 2.0)]
# check missingness in label candidates
df["LBXGLU"].isna().mean()
df["LBXGH"].isna().mean()
# only keep obs where LBXGH is not missing
df = df[df["LBXGH"].notna()].copy()
# clinical diabetes threshold
df["diabetes"] = (df["LBXGH"] >= 6.5).astype(int)
# remove LBXGH as column once binary flag created
df = df.drop(columns=["LBXGH"])
# combine the multiple BP readings into averages
df["SBP_mean"] = df[["BPXSY1", "BPXSY2", "BPXSY3"]].mean(axis=1)
df["DBP_mean"] = df[["BPXDI1", "BPXDI2", "BPXDI3"]].mean(axis=1)
# only keep columns we deem relevant to diabetes
cols_for_analysis = ["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "DMDEDUC2", "LBXTC", "LBDHDD", "LBXSTR", "LBXSCR", "LBXHSCRP", "LBXGLU", "DBP_mean", "SBP_mean", "BMXBMI", "BMXHIP", "SMQ020", "diabetes"]
df = df[cols_for_analysis]
# clean smoking column
df["SMQ020"].unique()
df["SMQ020"].dtype
df["SMQ020"] = df["SMQ020"].replace({1: 1, 2: 0})

# distribution of demographics and positive cases
df.groupby("RIAGENDR")["diabetes"].agg(n="count", positives="sum", prevalence="mean")
df.groupby("RIDRETH3")["diabetes"].agg(n="count", positives="sum", prevalence="mean")

# explore missingness
pct_missing = df.isna().mean()
df = df.loc[:, pct_missing <= 0.40]
df.isna().mean().sort_values(ascending=False)

df.to_csv("src/data/analysis_ready/nhanes_joined_2017_2018.csv", index=False)

