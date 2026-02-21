"""
upsample.py reads the processed training set and upsamples the minority 
racial classes to perfectly match the majority class (Non-Hispanic White). 
It outputs a new csv file for the upsampled training set.
"""
import pandas as pd

# load pre-processed baseline training data
train = pd.read_csv("src/data/model_ready/train_processed.csv")

# find exact count of the majority class (NH White) in training set
target_count = train[train["RIDRETH3_3.0"] == 1].shape[0]

# isolate each class
majority_3 = train[train["RIDRETH3_3.0"] == 1]
minority_1 = train[train["RIDRETH3_1.0"] == 1]
minority_4 = train[train["RIDRETH3_4.0"] == 1]
minority_6 = train[train["RIDRETH3_6.0"] == 1]

# resample each minority class with replacement to match the target count
minority_1_upsampled = minority_1.sample(n=target_count, replace=True, random_state=3)
minority_4_upsampled = minority_4.sample(n=target_count, replace=True, random_state=3)
minority_6_upsampled = minority_6.sample(n=target_count, replace=True, random_state=3)

# combine back together
train_upsampled = pd.concat([
    majority_3, 
    minority_1_upsampled, 
    minority_4_upsampled, 
    minority_6_upsampled
], ignore_index=True)

# we could reshuffle before here. tbd.
train_upsampled.to_csv("src/data/model_ready/train_processed_upsampled.csv", index=False)

# verify
print(f"Non-Hispanic White: {train_upsampled['RIDRETH3_3.0'].sum()}")
print(f"Mexican American:  {train_upsampled['RIDRETH3_1.0'].sum()}")
print(f"Non-Hispanic Black:  {train_upsampled['RIDRETH3_4.0'].sum()}")
print(f"Non-Hispanic Asian:  {train_upsampled['RIDRETH3_6.0'].sum()}")