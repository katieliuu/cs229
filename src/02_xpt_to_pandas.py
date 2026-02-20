"""
02_xpt_to_pandas converts the xpt files extracted in the
first script to pandas DataFrames stored in a dictionary
using the pyreadstat module. Information on how to do this
was obtained from the module's documentation:
https://ofajardo.github.io/pyreadstat_documentation/_build/html/index.html#module-pyreadstat.pyreadstat
The DataFrames are then joined into one DataFrame using repeated left
joins, with the starting DataFrame being 'DEMO_J' (all patients.)
The output DataFrame is saved as a csv to src/data/analysis_ready
"""
from pathlib import Path
import os
import pyreadstat
import pandas as pd

data_dir = Path("src/data/raw_xpt")
xpt_list = os.listdir(data_dir)
print("Files and directories in '", data_dir, "' :")
print(xpt_list)

datasets_dict = {}
for file in xpt_list:
    path = data_dir / file
    df, metadata = pyreadstat.read_xport(path, output_format = 'pandas')
    df_name = path.stem
    datasets_dict[df_name] = df

joined = datasets_dict['DEMO_J']
datasets = list(dataset for name, dataset in datasets_dict.items() if name != 'DEMO_J')
print("initial rows:", joined.shape[0])
for d in datasets:
# iterate through the dataframes in the dictionary, left joining one at a time
    joined = pd.merge(joined, d, on="SEQN", how="outer")

print("Final rows:", joined.shape[0])
joined["SEQN"].is_unique

# again might have to do os makedir
joined.to_csv("src/data/analysis_ready/nhanes_joined_2017_2018.csv", index=False)

