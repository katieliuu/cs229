# %%
from pathlib import Path
import pandas as pd
from nhanes_io import load_xpt_folder, left_join_many

KEY = "SEQN"
tables = load_xpt_folder("data/raw_xpt")
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

# %%
# Check tables joined correctly
print("Base rows:", base.shape[0])
print("Final rows:", df.shape[0])
df["SEQN"].is_unique

# %%
# Save csv to disk
OUT_DIR = Path("data/analysis_ready")
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "nhanes_joined_2017_2018.csv"
df.to_csv(out_path, index=False)

