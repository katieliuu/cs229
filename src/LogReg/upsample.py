"""
upsample.py reads the processed training set and upsamples the minority 
racial classes to perfectly match the majority class (Non-Hispanic White). 
It outputs a new csv file for the upsampled training set.
"""
import pandas as pd
import argparse
import math

# set up command line argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=["bootstrap", "random_top_off", "naive_repeat"], required=True)
args = parser.parse_args()

# load pre-processed baseline training data
train = pd.read_csv("src/data/model_ready/train_processed.csv")

if args.method == "bootstrap":
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
    train_upsampled.to_csv("src/data/model_ready/train_processed_upsampled_bootstrap.csv", index=False)

    # verify
    print("--- BOOTSTRAPPING ---")
    print(f"Non-Hispanic White: {train_upsampled['RIDRETH3_3.0'].sum()}")
    print(f"Mexican American:  {train_upsampled['RIDRETH3_1.0'].sum()}")
    print(f"Non-Hispanic Black:  {train_upsampled['RIDRETH3_4.0'].sum()}")
    print(f"Non-Hispanic Asian:  {train_upsampled['RIDRETH3_6.0'].sum()}")

elif args.method == "random_top_off":
    # find exact count of the majority class (NH White) in training set
    target_count = train[train["RIDRETH3_3.0"] == 1].shape[0]

    # isolate each class
    majority_3 = train[train["RIDRETH3_3.0"] == 1]
    minority_1 = train[train["RIDRETH3_1.0"] == 1]
    minority_4 = train[train["RIDRETH3_4.0"] == 1]
    minority_6 = train[train["RIDRETH3_6.0"] == 1]

    # calculate how many new samples are needed to "top off" each class
    n_needed_1 = target_count - len(minority_1)
    n_needed_4 = target_count - len(minority_4)
    n_needed_6 = target_count - len(minority_6)

    # sample the difference with replacement
    minority_1_extra = minority_1.sample(n=n_needed_1, replace=True, random_state=3)
    minority_4_extra = minority_4.sample(n=n_needed_4, replace=True, random_state=3)
    minority_6_extra = minority_6.sample(n=n_needed_6, replace=True, random_state=3)

    # combine the majority class, the original minority classes, and the extra samples
    train_upsampled = pd.concat([
        majority_3, 
        minority_1, minority_1_extra, 
        minority_4, minority_4_extra, 
        minority_6, minority_6_extra
    ], ignore_index=True)

    # we could reshuffle before here. tbd.
    train_upsampled.to_csv("src/data/model_ready/train_processed_upsampled_random_top_off.csv", index=False)

    # verify
    print("--- TOP OFF ---")
    print(f"Non-Hispanic White: {train_upsampled['RIDRETH3_3.0'].sum()}")
    print(f"Mexican American:  {train_upsampled['RIDRETH3_1.0'].sum()}")
    print(f"Non-Hispanic Black:  {train_upsampled['RIDRETH3_4.0'].sum()}")
    print(f"Non-Hispanic Asian:  {train_upsampled['RIDRETH3_6.0'].sum()}")

elif args.method == "naive_repeat":
    # find exact count of the majority class (NH White) in training set
    target_count = train[train["RIDRETH3_3.0"] == 1].shape[0]

    # isolate each class
    majority_3 = train[train["RIDRETH3_3.0"] == 1]
    minority_1 = train[train["RIDRETH3_1.0"] == 1]
    minority_4 = train[train["RIDRETH3_4.0"] == 1]
    minority_6 = train[train["RIDRETH3_6.0"] == 1]

    # calculate kappa (minority / majority) using ceiling division so we round up
    # kappa calculates how many whole times the minority class fits into the majority class
    repeat_1 = math.ceil(target_count / len(minority_1))
    repeat_4 = math.ceil(target_count / len(minority_4))
    repeat_6 = math.ceil(target_count / len(minority_6))

    # duplicates the dataset that many times (1 / kappa)
    minority_1_repeated = pd.concat([minority_1] * repeat_1, ignore_index=True)
    minority_4_repeated = pd.concat([minority_4] * repeat_4, ignore_index=True)
    minority_6_repeated = pd.concat([minority_6] * repeat_6, ignore_index=True)

    # combine back together
    train_upsampled = pd.concat([
        majority_3, 
        minority_1_repeated, 
        minority_4_repeated, 
        minority_6_repeated
    ], ignore_index=True)

    # we could reshuffle before here. tbd.
    train_upsampled.to_csv("src/data/model_ready/train_processed_upsampled_naive.csv", index=False)

    # verify
    print("--- NAIVE REPEAT ---")
    print(f"Target count to approximate: {target_count}")
    print(f"Non-Hispanic White: {train_upsampled['RIDRETH3_3.0'].sum()}")
    print(f"Mexican American:  {train_upsampled['RIDRETH3_1.0'].sum()}")
    print(f"Non-Hispanic Black:  {train_upsampled['RIDRETH3_4.0'].sum()}")
    print(f"Non-Hispanic Asian:  {train_upsampled['RIDRETH3_6.0'].sum()}")
    