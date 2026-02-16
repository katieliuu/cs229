# cs229
CS 229 Final Project - NHANES Disease Classification

Every time you pull from the repo:
- Run uv sync
- Run source .venv/bin/activate

The Python script 01_download_raw_data downloads xpt files from the NHANES website. The specific URLs corresponding to specific data files can be changed as needed. Run this script first.

In the nhanes_io Python script are defined some helper functions that 03_analysis.py imports. These helpers:
- Convert xpt files to pandas DataFrames
- Performs a left join on multiple DataFrames to create one combined DataFrame
The DFs are joined by 'SEQN' which is how NHANES defines unique participant ID.
You do not need to run this one since it is called in 03_analysis.py

After running the download script you should run 03_analysis.py. It executes the helper functions described above. In this script, you can specify the names of the tables that you want to join (these names can be found in data/raw_xpt). The script also saves the combined DF as a csv to disk. 
