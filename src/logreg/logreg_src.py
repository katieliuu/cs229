"""
logistic_regression.py trains a logistic regression model on the dataset.
The model functions will be imported and used in other scripts that experiment with different hyperparameters or methods.
"""

import numpy as np
def f1_from_probs(y_true, probs, threshold):
    y_true = np.asarray(y_true)
    preds = (probs >= threshold).astype(int)
    tp = np.sum((preds == 1) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    tn = np.sum((preds == 0) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))
    

    # precision
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    # recall
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    # f1
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall, tp, fp, tn, fn

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def calc_grad_and_loss(X, Y, theta, lambda_reg=0.0, sample_weight=None):
    epsilon = 1e-6
    count, _ = X.shape

    z = X @ theta
    probs = sigmoid(z)
    err = Y - probs

    if sample_weight is None:
        w = np.ones(count, dtype=float)
    else:
        w = np.asarray(sample_weight, dtype=float).reshape(-1)
        assert len(w) == count, "Sample weight length must match number of samples"

    weight_sum = np.sum(w)
    weighted_err = w * err

    # gradient ascent on weighted log-likelihood
    grad = (X.T @ weighted_err) / weight_sum

    # L2 regularization, excluding bias
    grad[1:] -= 2 * lambda_reg * theta[1:]

    # weighted cross-entropy loss
    loss_terms = -(Y * np.log(probs + epsilon) + (1 - Y) * np.log(1 - probs + epsilon))
    data_loss = np.sum(w * loss_terms) / weight_sum

    reg_loss = lambda_reg * np.sum(theta[1:] ** 2)
    total_loss = data_loss + reg_loss

    return grad, total_loss

def logistic_regression(X_train, Y_train, max_iter=5000, lambda_reg=0.0,
                        sample_weight=None, learning_rate=0.01):
    theta = np.zeros(X_train.shape[1], dtype=float)

    for i in range(1, max_iter + 1):
        prev_theta = theta.copy()

        grad, loss = calc_grad_and_loss(
            X_train,
            Y_train,
            theta,
            lambda_reg=lambda_reg,
            sample_weight=sample_weight
        )

        theta = theta + learning_rate * grad

        if i % 10000 == 0:
            print(f"Finished {i} iterations")
            print(f"Loss: {loss}")
            print(f"Weights: {theta}")

        if np.linalg.norm(theta - prev_theta) < 1e-6:
            print(f"Converged in {i} iterations")
            break

    return theta

'''
def main():
    print('==== Training model on data set A ====')
    X_original, Y_original = util.load_csv('src/data/model_ready/train_processed.csv', label_col='diabetes', add_intercept=True)
    print("Xa shape:", X_original.shape)
    print("Ya shape:", Y_original.shape)
    minority_feature_index = util.return_minority_feature_index('src/data/model_ready/train_processed.csv')
    
    
    #Regularized without cost-sensitive learning
    final_loss_w_reg_wo_cs = logistic_regression(X_original, Y_original, max_iter=100000, lambda_reg=10)
    
    #Regularized with cost-sensitive learning only on minority class
    final_loss_w_reg_w_cs = logistic_regression(X_original, Y_original, max_iter=100000, lambda_reg=10, penalty_weight=10, minority_feature_index=minority_feature_index)
    
    #Regularized with cost-sensitive learning on all classes
    final_loss_w_reg_w_cs_all = logistic_regression(X_original, Y_original, max_iter=100000, lambda_reg=10, penalty_weight=10, minority_feature_index=None)
    
    print(f'Regularized without cost-sensitive learning final loss: {final_loss_w_reg_wo_cs}')
    print(f'Regularized with cost-sensitive learning final loss: {final_loss_w_reg_w_cs}')
    print(f'Regularized with cost-sensitive learning on all classes final loss: {final_loss_w_reg_w_cs_all}')
    
    
if __name__ == '__main__':
    main()
'''  

