# Important note: you do not have to modify this file for your homework.

import util
import numpy as np

def calc_grad_and_loss(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    epsilon = 1e-15
    count, _ = X.shape
    z = X.dot(theta)
    z = np.clip(z, -500, 500)
    probs = 1. / (1 + np.exp(-z))
    grad = (Y - probs).dot(X) / count
    loss = -np.mean(Y * np.log(probs + epsilon) + (1 - Y) * np.log(1 - probs + epsilon))

    return grad, loss

def logistic_regression(X, Y):
    """Train a logistic regression model."""
    theta = np.zeros(X.shape[1])
    learning_rate = 0.01

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad, loss = calc_grad_and_loss(X, Y, theta)
        theta = theta + learning_rate * grad
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
            print(f'Loss: {loss}')
            print(f'Weights: {theta}')
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return


def main():
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('data/analysis_ready/small_dataset.csv', label_col='LBXGLU_bin', add_intercept=True)
    print("Xa shape:", Xa.shape)
    print("Ya shape:", Ya.shape)
    print("Ya unique:", np.unique(Ya)[:10])
    logistic_regression(Xa, Ya)
    

if __name__ == '__main__':
    main()
