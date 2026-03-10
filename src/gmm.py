from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
from kprototypes import naive_upsample_clusters

def gmm_cluster_upsample(training_data, max_iter=150, n_components=3):
    training_data = training_data.copy()

    ignore_cols = ["diabetes"]
    X_df = training_data.drop(columns=ignore_cols, errors="ignore")
    data_matrix = X_df.to_numpy()

    model = GaussianMixture(
        n_components=n_components,
        max_iter=max_iter,
        random_state=42
    )
    model.fit(data_matrix)

    training_data["Cluster"] = model.predict(data_matrix)

    print("\nCluster distribution before upsampling:")
    print(training_data["Cluster"].value_counts().sort_index())

    train_upsampled = naive_upsample_clusters(training_data, cluster_col="Cluster")

    print("\nCluster distribution after upsampling:")
    print(train_upsampled["Cluster"].value_counts().sort_index())

    train_upsampled = train_upsampled.drop(columns=["Cluster"])

    print("\nCOMPLETE.")
    return train_upsampled