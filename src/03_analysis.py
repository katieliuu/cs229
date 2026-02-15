# %%
from pathlib import Path
import pandas as pd
from nhanes_io import load_xpt_folder, left_join_many

KEY = "SEQN"
tables = load_xpt_folder("data/raw_xpt")
sorted(tables.keys())

base = tables["DEMO_J"]
JOIN_TABLES = [
    "BPX_J"]

df = left_join_many(base=base, tables=tables, table_names=JOIN_TABLES, key=KEY)
df.shape

