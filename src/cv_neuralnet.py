max_depth_grid = [2, 4, 6, 8]
min_samples_split_grid = [4, 6, 8, 10, 12]
min_samples_leaf_grid = [2, 4, 6]
max_leaf_nodes_grid = [15, 20, 25, 30]
threshold_grid = [0.3, 0.4, 0.5] # figure out if DTs use this
kappa_grid = [1, 2, 3]
penalty_weight_grid = [0.5, 1.0, 2.0] # what is the equivalent for DT

# INSERT GMM AND CLUSTERING PARAM GRIDS
if experiment_type == "baseline":
    param_grid = list(product(max_depth_grid, min_samples_split_grid, min_samples_leaf_grid, max_leaf_nodes_grid, threshold_grid))
elif experiment_type == "upsample":
    param_grid = list(product(kappa_grid, max_depth_grid, min_samples_split_grid, min_samples_leaf_grid, max_leaf_nodes_grid, threshold_grid))
elif experiment_type == "cost_sensitive":
    param_grid = list(product(penalty_weight_grid, max_depth_grid, min_samples_split_grid, min_samples_leaf_grid, max_leaf_nodes_grid, threshold_grid))