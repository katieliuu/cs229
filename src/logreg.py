# Important note: you do not have to modify this file for your homework.

import util
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    count, _ = X.shape

    probs = 1. / (1 + np.exp(-X.dot(theta)))
    grad = (Y - probs).dot(X)

    return grad


def logistic_regression(X, Y):
    """Train a logistic regression model."""
    theta = np.zeros(X.shape[1])
    learning_rate = 0.1

    i = 0
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta + learning_rate * grad
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print('Converged in %d iterations' % i)
            break
    return


def main():
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('data/analysis_ready/nhanes_joined_2017_2018.csv', add_intercept=True)
    logistic_regression(Xa, Ya)

    
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))

    # --- Dataset A ---
    Ya = np.asarray(Ya).ravel()
    idx1 = (Ya == 1)
    idx0 = (Ya == 0)

    ax0.scatter(Xa[idx0, 1], Xa[idx0, 2], c='blue', marker='s', label='y=0')
    ax0.scatter(Xa[idx1, 1], Xa[idx1, 2], c='red', marker='*', label='y=1')
    ax0.set_xlabel("x1")
    ax0.set_ylabel("x2")
    ax0.set_title("Dataset A")
    ax0.legend()

    # --- Dataset B ---
    Yb = np.asarray(Yb).ravel()
    idx1 = (Yb == 1)
    idx0 = (Yb == 0)

    ax1.scatter(Xb[idx0, 1], Xb[idx0, 2], c='blue', marker='s', label='y=0')
    ax1.scatter(Xb[idx1, 1], Xb[idx1, 2], c='red', marker='*', label='y=1')
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_title("Dataset B")
    ax1.legend()

    plt.show()
    # fig = plt.figure(figsize=(10, 4))

    # ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    # ax1 = fig.add_subplot(1, 2, 2, projection='3d')

    # ax0.scatter(Xa[:,1], Xa[:,2], Ya)
    # ax1.scatter(Xb[:,1], Xb[:,2], Yb)

    # ax0.set_xlabel("x1")
    # ax0.set_ylabel("x2")
    # ax0.set_zlabel("y")

    # ax1.set_xlabel("x1")
    # ax1.set_ylabel("x2")
    # ax1.set_zlabel("y")

    # plt.show()  


if __name__ == '__main__':
    main()
