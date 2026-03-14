"""
upsample.py reads the processed training set and upsamples the minority 
racial classes based on specified kappa ratios. 
It outputs a new dataframe for the upsampled training set.

AI Use: Google Gemini was used as a collaborator to help brainstorm
code structure and debug implementation issues.
"""

import pandas as pd
import argparse
import math

def get_natural_kappas(training_data):
    """
    Helper function to calculate the natural minority/majority ratios in the dataset.
    """
    target_count = training_data[training_data["RIDRETH3"] == 3.0].shape[0]
    
    kappa_1 = len(training_data[training_data["RIDRETH3"] == 1.0]) / target_count
    kappa_4 = len(training_data[training_data["RIDRETH3"] == 4.0]) / target_count
    kappa_6 = len(training_data[training_data["RIDRETH3"] == 6.0]) / target_count

    #print(f"natural kappas are kappa_1 = {kappa_1}, kappa_4 = {kappa_4}, kappa_6 = {kappa_6}")
    
    return kappa_1, kappa_4, kappa_6

def naive_upsample(training_data, kappa_1, kappa_4, kappa_6):
    """
    Upsamples the minority classes based on the provided kappa values.
    """
    # isolate each class
    majority_3 = training_data[training_data["RIDRETH3_3.0"] == 1]
    minority_1 = training_data[training_data["RIDRETH3_1.0"] == 1]
    minority_4 = training_data[training_data["RIDRETH3_4.0"] == 1]
    minority_6 = training_data[training_data["RIDRETH3_6.0"] == 1]

    # calculate repeat count (1 / kappa), rounded up
    repeat_1 = math.ceil(1 / kappa_1)
    repeat_4 = math.ceil(1 / kappa_4)
    repeat_6 = math.ceil(1 / kappa_6)

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

    #print(f"finished upsampling for kappa_1 = {kappa_1}, kappa_4 = {kappa_4}, kappa_6 = {kappa_6}")
    return train_upsampled
    