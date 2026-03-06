"""
upsample.py reads the processed training set and upsamples the minority 
racial classes to perfectly match the majority class (Non-Hispanic White). 
It outputs a new csv file for the upsampled training set.
"""
import pandas as pd
import argparse
import math

def naive_upsample(training_data):
    # find exact count of the majority class (NH White) in training set
    target_count = training_data[training_data["RIDRETH3_3.0"] == 1].shape[0]

    # isolate each class
    majority_3 = training_data[training_data["RIDRETH3_3.0"] == 1]
    minority_1 = training_data[training_data["RIDRETH3_1.0"] == 1]
    minority_4 = training_data[training_data["RIDRETH3_4.0"] == 1]
    minority_6 = training_data[training_data["RIDRETH3_6.0"] == 1]

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

    return train_upsampled
    