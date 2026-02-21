# Important note: you do not have to modify this file for your homework.

import util
import numpy as np
import argparse

def parse_arguments():
    """ Args parser for methods and hyperparameters """
    parser = argparse.ArgumentParser(description='Logistic Regression')
    parser.add_argument('--lambda_reg', type=float, default=10)
    return parser.parse_args()


def calc_grad_and_loss(X, Y, theta, cost_sensitive, penalty_weight, minority_feature_index):
    """Compute gradient (ascent) and loss for logistic regression."""
    args = parse_arguments()
    epsilon = 1e-15
    count, _ = X.shape

    z = X.dot(theta)
    z = np.clip(z, -500, 500)
    probs = 1. / (1 + np.exp(-z))

    # error term for log-likelihood gradient ascent
    err = (Y - probs)

    if not cost_sensitive:
        grad = (err.dot(X) - 2 * args.lambda_reg * theta) / count
        grad[0] = err.dot(X[:, 0]) / count  # don't regularize bias
        loss = -np.mean(Y * np.log(probs + epsilon) + (1 - Y) * np.log(1 - probs + epsilon))

    else:
        # upweight only minority points
        alpha = penalty_weight
        w = np.ones_like(Y, dtype=float)
        # w will be alpha if any of the minority feature columns have 1
        w[(X[:, minority_feature_index] == 1).any(axis=1)] = alpha

        # weighted gradient ascent: X^T (w ⊙ (Y - p))
        weighted_err = w * err
        grad = (weighted_err.dot(X) - 2 * args.lambda_reg * theta) / count
        grad[0] = weighted_err.dot(X[:, 0]) / count  # don't regularize bias

        # (optional but usually correct) weighted loss:
        # normalize by sum of weights so scale is comparable across alpha
        loss_terms = -(Y * np.log(probs + epsilon) + (1 - Y) * np.log(1 - probs + epsilon))
        loss = np.sum(w * loss_terms) / np.sum(w)

    return grad, loss


def logistic_regression(X, Y, max_iter=100000, cost_sensitive=False, penalty_weight=10, minority_feature_index=None):
    """Train a logistic regression model."""
    theta = np.zeros(X.shape[1])
    learning_rate = 0.01

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad, loss = calc_grad_and_loss(X, Y, theta, cost_sensitive=cost_sensitive, penalty_weight=penalty_weight, minority_feature_index=minority_feature_index)
        theta = theta + learning_rate * grad
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
            print(f'Loss: {loss}')
            print(f'Weights: {theta}')
        if np.linalg.norm(prev_theta - theta) < 1e-15 or i > max_iter:
            print('Converged in %d iterations' % i-1)
            break
    return loss


def main():
    print('==== Training model on data set A ====')
    X_original, Y_original = util.load_csv('src/data/model_ready/train_processed.csv', label_col='diabetes', add_intercept=True)
    print("Xa shape:", X_original.shape)
    print("Ya shape:", Y_original.shape)
    #baseline experiment
    final_loss_baseline = logistic_regression(X_original, Y_original, max_iter=100000)
    #cost-sensitive experiment
    minority_feature_index = util.return_minority_feature_index('src/data/model_ready/train_processed.csv')
    final_loss_cost_sensitive = logistic_regression(X_original, Y_original, max_iter=100000, cost_sensitive=True, penalty_weight=10, minority_feature_index=minority_feature_index)
    
    '''
    #Upsampled dataset experiment
    #TODO: CHANGE PATH TO UPSAMPLED TRAIN DATASET
    X_upsampled, Y_upsampled = util.load_csv('src/data/model_ready/train_processed.csv', label_col='diabetes', add_intercept=True)
    final_loss_upsampled = logistic_regression(X_upsampled, Y_upsampled, max_iter=500000)
    print(f'Upsampled final loss: {final_loss}')
    
    #Cluster-Upsampled experiment
    #TODO: CHANGE PATH TO CLUSTER-UPSAMPLED TRAIN DATASET
    X_cluster_upsampled, Y_cluster_upsampled = util.load_csv('src/data/model_ready/train_processed.csv', label_col='diabetes', add_intercept=True)
    final_loss_cluster_upsampled = logistic_regression(X_cluster_upsampled, Y_cluster_upsampled, max_iter=500000)
    print(f'Cluster-Upsampled final loss: {final_loss}')
    
    '''
    print(f'Baseline final loss: {final_loss_baseline}')
    print(f'Cost-sensitive final loss: {final_loss_cost_sensitive}')
    #print(f'Upsampled final loss: {final_loss_upsampled}')
    #print(f'Cluster-Upsampled final loss: {final_loss_cluster_upsampled}')
if __name__ == '__main__':
    main()
