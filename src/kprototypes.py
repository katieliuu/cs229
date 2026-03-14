"""
kprototypes.py implements K-Prototypes clustering from scratch for mixed-type data
(continuous + categorical features). Uses Euclidean distance for
continuous features and Hamming distance for categorical features,
combined via a gamma scaling parameter. After clustering, minority clusters are 
upsampled via naive duplication to match the size of the largest cluster, and the 
resulting dataset is returned for use in downstream classification.
"""

import pandas as pd
import numpy as np
import argparse
import math

def init_centroids(num_clusters, data):
    """
    Initialize centroids by randomly picking rows from the dataset.
    """
    # randomly select num_clusters number of unique indices
    random_indices = np.random.choice(data.shape[0], size=num_clusters, replace=False)

    # extract the values at those indices
    centroids_init = data[random_indices].astype(np.float64)
    return centroids_init

def get_mode(arr):
    """
    Helper function to find the mode of categorical columns along axis 0.
    """
    modes = []
    for col in range(arr.shape[1]):
        # np.unique returns sorted values and their counts
        vals, counts = np.unique(arr[:, col], return_counts=True)
        # append the value that has the highest count
        modes.append(vals[np.argmax(counts)])
    return np.array(modes)

def update_centroids(centroids, data, cat_indices, cont_indices, gamma=1.0, max_iter=150, print_every=10):
    """
    Carry out the centroid update step `max_iter` times for K-Prototypes.
    """
    current_centroids = np.copy(centroids).astype(np.float64)
    num_clusters = current_centroids.shape[0]
    N = data.shape[0]
    
    labels = np.zeros(N)

    for i in range(max_iter):
        # 1. CALCULATE DISTANCES
        # Euclidean distance for continuous features
        dist_cont = np.linalg.norm(
            data[:, np.newaxis, cont_indices] - current_centroids[np.newaxis, :, cont_indices], axis=2
        )
        
        # Hamming distance for categorical features (sum of mismatches)
        dist_cat = np.sum(
            data[:, np.newaxis, cat_indices] != current_centroids[np.newaxis, :, cat_indices], axis=2
        )
        
        # Combine distances (gamma scales the categorical distance)
        distances = dist_cont + (gamma * dist_cat)

        # 2. ASSIGN LABELS aka assignments to centroids
        labels = np.argmin(distances, axis=1)
        new_centroids = np.zeros_like(current_centroids)

        # 3. RECOMPUTE CENTROIDS
        for k in range(num_clusters):
            cluster_points = data[labels == k]
            
            # handle edge case where a cluster might lose all points
            if len(cluster_points) > 0:
                # mean for continuous features
                new_centroids[k, cont_indices] = np.mean(cluster_points[:, cont_indices], axis=0)
                
                # mode for categorical features
                if len(cat_indices) > 0:
                    new_centroids[k, cat_indices] = get_mode(cluster_points[:, cat_indices])
            else:
                new_centroids[k] = current_centroids[k]
        
        # 4. CHECK CONVERGENCE
        if np.allclose(current_centroids, new_centroids):
            print(f"We've converged early at iter {i}.")
            break

        current_centroids = new_centroids

        # modular print statements 
        if (i + 1) % print_every == 0:
            print(f"Iteration {i + 1} complete!")
    
    return current_centroids, labels

def naive_upsample_clusters(df, cluster_col='Cluster'):
    """
    Identifies the largest cluster and upsamples all other clusters 
    to match or slightly exceed its size using naive duplication.
    """
    # determine the target size based on the largest cluster
    cluster_counts = df[cluster_col].value_counts()
    majority_cluster = cluster_counts.idxmax()
    majority_count = cluster_counts.max()
    
    upsampled_dfs = []
    
    # loop through every cluster in the dataset
    for cluster_label in cluster_counts.index:
        cluster_subset = df[df[cluster_col] == cluster_label]
        
        if cluster_label == majority_cluster:
            # add the majority cluster exactly as it is
            upsampled_dfs.append(cluster_subset)
        else:
            # calculate multiplier for minority clusters
            repeat_count = math.ceil(majority_count / len(cluster_subset))
            
            # duplicate the subset
            repeated_subset = pd.concat([cluster_subset] * repeat_count, ignore_index=True)
            upsampled_dfs.append(repeated_subset)
            
    # put back together
    train_upsampled = pd.concat(upsampled_dfs, ignore_index=True)
    return train_upsampled

def run_k_prototypes(training_data, max_iter=150, n_clusters=3, print_every=10, gamma=1.0):
    # define columns to ignore and the raw categorical columns
    # cont cols are "RIDAGEYR", "LBXTC", "LBDHDD", "LBXSTR", "LBXSCR", "LBXHSCRP", "DBP_mean", "SBP_mean", "BMXBMI", "BMXHIP", "SMQ020"
    ignore_cols = ["diabetes"]
    categorical_cols = ["DMDEDUC2_1.0", 
                        "DMDEDUC2_2.0",
                        "DMDEDUC2_3.0",
                        "DMDEDUC2_4.0",
                        "DMDEDUC2_5.0",
                        "DMDEDUC2_7.0",
                        "RIDRETH3_9.0", 
                        "RIDRETH3_1.0",
                        "RIDRETH3_3.0", 
                        "RIDRETH3_4.0", 
                        "RIDRETH3_6.0", 
                        "RIAGENDR_1.0", 
                        "RIAGENDR_2.0"]

    # prepare feature matrix
    X_df = training_data.drop(columns=ignore_cols, errors='ignore')
    data_matrix = X_df.values

    # map indices for the algorithm
    cat_indices = [X_df.columns.get_loc(col) for col in categorical_cols if col in X_df.columns]
    cont_indices = [i for i in range(data_matrix.shape[1]) if i not in cat_indices]

    print(f"--- Running from-scratch K-PROTOTYPES with {n_clusters} clusters (Gamma: {gamma}) ---")
    
    # initialize
    print('[INFO] Centroids initialized')
    centroids_init = init_centroids(n_clusters, data_matrix)

    # update
    print('[INFO] Updating centroids ...')
    final_centroids, cluster_labels = update_centroids(
        centroids_init, 
        data_matrix, 
        cat_indices=cat_indices,
        cont_indices=cont_indices,
        gamma=gamma,
        max_iter=max_iter,
        print_every=print_every
    )

    # attach labels to original dataframe
    training_data['Cluster'] = cluster_labels

    print("\nCluster distribution before upsampling:")
    print(training_data['Cluster'].value_counts().sort_index())
    
    # upsample all minority clusters to match the largest one
    print("\nUpsampling minority clusters...")
    train_upsampled = naive_upsample_clusters(training_data, cluster_col='Cluster')
    
    print("\nCluster distribution after upsampling:")
    print(train_upsampled['Cluster'].value_counts().sort_index())
    
    # drop cluster column so it doesn't leak into predictive models later
    train_upsampled = train_upsampled.drop(columns=['Cluster'])
    print(f"\nCOMPLETE.")

    return train_upsampled
