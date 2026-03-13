'''
This script trains and tests a decision tree model with regularization and cost sensitive
learning.
Args:
    test: bool, whether to test the model
Returns:
    None
To train:
    python tree_cost_sensitive.py
To test:
    python tree_cost_sensitive.py --test
'''
import pandas as pd
from tree_src import *
from util import *
import argparse

def main(test: bool = False):
    #Threshold
    threshold_w_cs = 0.25
    
    #Load data
    training_data = pd.read_csv('src/data/model_ready/train_processed.csv')
    X_original, Y_original = training_data.drop(columns=["diabetes"]), training_data["diabetes"]
    
    #Baseline with regularizationwith cost sensitive learning
    sample_weight = calculate_sample_weight(training_data)
    model_w_cs, train_accuracy_w_cs = decision_tree(X_original, Y_original, max_depth=25, min_samples_split=2, min_samples_leaf=1, sample_weight=sample_weight, threshold=threshold_w_cs)
    
    print("Train Accuracy With Cost Sensitive Learning:", train_accuracy_w_cs)
    if test: 
        testing_data = pd.read_csv('src/data/model_ready/test_processed.csv')
        X_test, Y_test = testing_data.drop(columns=["diabetes"]), testing_data["diabetes"]
        output_model_path = 'src/results/decisiontree'
        probs_w_cs = test_tree(model_w_cs, X_test, Y_test)
        evaluate_by_ethnicity(X_test, Y_test, probs_w_cs, threshold_w_cs, output_model_path=output_model_path, experiment_type='cost_sensitive')
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test or Train")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    
    main(test = args.test)
    
    
    