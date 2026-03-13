'''
This script trains and tests a decision tree model with and without regularization (max depth,
min samples split, min samples leaf).
Args:
    test: bool, whether to test the model
Returns:
    None
To train:
    python tree_baseline.py
To test:
    python tree_baseline.py --test
'''
import pandas as pd
from tree_src import *
from util import *
import argparse

def main(test: bool = False):
    #Threshold
    threshold_wo_reg = 0.5
    threshold_w_reg = 0.25
    
    #Load data
    training_data = pd.read_csv('src/data/model_ready/train_processed.csv')
    X_original, Y_original = training_data.drop(columns=["diabetes"]), training_data["diabetes"]
    
    #Baseline without regularization
    model_wo_reg, train_accuracy_wo_reg = decision_tree(X_original, Y_original, threshold=threshold_wo_reg)
    
    #Baseline with regularization (max depth, min samples split, min samples leaf)
    model_w_reg, train_accuracy_w_reg = decision_tree(X_original, Y_original, max_depth=25, min_samples_split=2, min_samples_leaf=1, threshold=threshold_w_reg)

    print("Train Accuracy Without Regularization:", train_accuracy_wo_reg)
    print("Train Accuracy With Regularization:", train_accuracy_w_reg)
    
    if test:
        testing_data = pd.read_csv('src/data/model_ready/test_processed.csv')
        X_test, Y_test = testing_data.drop(columns=["diabetes"]), testing_data["diabetes"]
        output_model_path = 'src/results/decisiontree'
        probs_wo_reg = test_tree(model_wo_reg, X_test, Y_test)
         
        probs_w_reg = test_tree(model_w_reg, X_test, Y_test)
        
        evaluate_by_ethnicity(X_test, Y_test, probs_w_reg, threshold_w_reg, output_model_path=output_model_path, experiment_type='baseline_w_reg')
        evaluate_by_ethnicity(X_test, Y_test, probs_wo_reg, threshold_wo_reg, output_model_path=output_model_path, experiment_type='baseline_wo_reg')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test or Train")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    main(test = args.test)
        
        
        