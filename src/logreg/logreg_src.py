"""
This script contains the source code for the logistic regression model. It contains 
functions used to create logistic regression model and predict probabilities.
"""

import numpy as np

def sigmoid(z):
    '''
    This function applies the sigmoid function to the input.
    Args:
        z: numpy array, input to the sigmoid function
    Returns:
        sigmoid(z): numpy array, output of the sigmoid function
    '''
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def calc_grad_and_loss(X, Y, theta, lambda_reg=0.0, sample_weight=None):
    '''
    This function calculates the gradient and loss of the logistic regression model.
    Args:
        X: numpy array, features
        Y: numpy array, labels
        theta: numpy array, weights
        lambda_reg: float, regularization parameter
        sample_weight: numpy array, sample weights
    Returns:
        grad: numpy array, gradient
        total_loss: float, total loss
    '''
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
    '''
    This function trains a logistic regression model on the training data and returns the 
    model weights.
    Args:
        X_train: numpy array, training data features
        Y_train: numpy array, training labels
        max_iter: int, max number of iterations
        lambda_reg: float, regularization parameter
        sample_weight: numpy array, sample weights
        learning_rate: float, learning rate
    Returns:
        theta: numpy array, model weights
    '''
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
