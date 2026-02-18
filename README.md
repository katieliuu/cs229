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

After running the download script you should run 03_analysis.py. It executes the helper functions described above. In this script, you can specify the names of the tables that you want to join (these names can be found in data/raw_xpt). This script also performs some data cleaning, filtering, and EDA, and defines the column that will be used as the diabetes 'ground truth' (glycohemoglobin levels). Before any modeling, need to check the data types of each column. Since gender and ethnicity are encoded numerically, model will try to find numerical relationships between the different categories, which doesn't make sense, so these columns were converted to type categorical. Lastly, it saves the combined DF as a csv to disk.

The next script to run is 04_train_test_split_impute.py. As the name suggests this splits the combined DF into a train and test set (80-20 split). Don't change the random_seed argument as this is essential for reproducibility. Stratify mitigates against label imbalance in the splits. This script carries out MICE imputation: it is fit using the training X observations, and then applied to the test X observations. y never gets imputed. Running this script will save 4 csvs to the src/data/model_ready folder, corresponding to the imputed X train and imputed X test, and non-imputed y train and y test. (X train and y train together form the 'training set'). Don't touch the test csvs until final model evaluation. 

