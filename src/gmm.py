from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
import argparse
from kprototypes import naive_upsample_clusters

def gmm(training_data, max_iter=150, n_components=3):
    #taking away diabetes column so it's not used as a feature during GMM clustering
    ignore_cols = ["diabetes"]
    X_df = training_data.drop(columns=ignore_cols, errors='ignore')
    data_matrix = X_df.to_numpy()
    
    #fitting GMM to the data
    gmm = GaussianMixture(n_components=3, max_iter=max_iter)
    gmm.fit(data_matrix)
    
    #predicting clusters for the data
    X_df["Cluster"] = gmm.predict(data_matrix)
    
    #checking cluster distribution before upsampling
    print("\nCluster distribution before upsampling:")
    print(train_upsampled['Cluster'].value_counts().sort_index())
    
    #upsampling clusters to match the largest one
    train_upsampled = naive_upsample_clusters(X_df, cluster_col='Cluster')
    
    #checking cluster distribution after upsampling
    print("\nCluster distribution after upsampling:")
    print(train_upsampled['Cluster'].value_counts().sort_index())
    
    #dropping cluster column so it doesn't leak into predictive models later
    train_upsampled = train_upsampled.drop(columns=['Cluster'])
    print(f"\nCOMPLETE.")
    return train_upsampled