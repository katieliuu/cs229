import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    """
    Compute softmax function for a batch of input values. 
    The first dimension of the input corresponds to the batch size. The second dimension
    corresponds to every class in the output. When implementing softmax, you should be careful
    to only sum over the second dimension.

    Important Note: You must be careful to avoid overflow for this function. Functions
    like softmax have a tendency to overflow when very large numbers like e^10000 are computed.
    You will know that your function is overflow resistent when it can handle input like:
    np.array([[10000, 10010, 10]]) without issues.

    Args:
        x: A 2d numpy float array of shape batch_size x number_of_classes

    Returns:
        A 2d numpy float array containing the softmax results of shape batch_size x number_of_classes
    """
    # *** START CODE HERE ***
    # x is (batch size, class size). we only sum over class size. be careful of overflow.
    # deal with overflow by subtracting a column vector of max x from x
    x_max = np.max(x, axis=1, keepdims=True) # (batch_size, 1)
    x_protected = x - x_max

    # softmax eval
    numerator = np.exp(x_protected)
    denominator = np.sum(np.exp(x_protected), axis=1, keepdims=True) # (batch_size, 1)

    return numerator / denominator
    # *** END CODE HERE ***

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Args:
        x: A numpy float array

    Returns:
        A numpy float array containing the sigmoid results
    """
    # *** START CODE HERE ***
    return 1 / (1 + np.exp(-x))
    # *** END CODE HERE ***

def get_initial_params(input_size, num_hidden, num_output):
    """
    Compute the initial parameters for the neural network.

    This function should return a dictionary mapping parameter names to numpy arrays containing
    the initial values for those parameters.

    There should be four parameters for this model:
    W1 is the weight matrix for the hidden layer of size input_size x num_hidden
    b1 is the bias vector for the hidden layer of size num_hidden
    W2 is the weight matrix for the output layers of size num_hidden x num_output
    b2 is the bias vector for the output layer of size num_output

    As specified in the PDF, weight matrices should be initialized with a random normal distribution
    centered on zero and with scale 1.
    Bias vectors should be initialized with zero.
    
    Args:
        input_size: The size of the input data
        num_hidden: The number of hidden states
        num_output: The number of output classes
    
    Returns:
        A dict mapping parameter names to numpy arrays
    """

    # *** START CODE HERE ***
    # initalize the weights of the network by sampling values from a standard normal
    # initialize the bias/intercept term to 0
    # params = load_params('regularized_params.npz')

    params = {
        # W1 is the weight matrix for the hidden layer of size input_size x num_hidden
        "W1": np.random.randn(input_size, num_hidden),
        # b1 is the bias vector for the hidden layer of size num_hidden
        "b1": np.zeros(num_hidden),
        # W2 is the weight matrix for the output layers of size num_hidden x num_output
        "W2": np.random.randn(num_hidden, num_output),
        # b2 is the bias vector for the output layer of size num_output
        "b2": np.zeros(num_output)
    }
    # return a dict mapping param names to np arrays w/ initial values for those params
    return params
    # *** END CODE HERE ***

def forward_prop(data, one_hot_labels, params, is_training=False, activation='sigmoid', dropout_rate=0.0, sample_weights=None): # <-- added sample_weights    # Apply chosen activation function
    if activation == 'sigmoid':
        a = sigmoid(np.dot(data, params["W1"]) + params["b1"])
    elif activation == 'relu':
        a = np.maximum(0, np.dot(data, params["W1"]) + params["b1"]) # ReLU math
        
    # apply inverted dropout
    if is_training and dropout_rate > 0.0:
        keep_prob = 1.0 - dropout_rate
        mask = (np.random.rand(*a.shape) < keep_prob) / keep_prob
        a = a * mask
        params["dropout_mask"] = mask # save mask for backprop
    else:
        params["dropout_mask"] = None

    h_theta_bar = np.dot(a, params["W2"]) + params["b2"]
    h_theta = softmax(h_theta_bar)
    
    # add 1e-8 for log stability to prevent NaN errors during training
    cost = -np.sum(one_hot_labels * np.log(h_theta + 1e-8)) / data.shape[0] 
    
    # apply weights to cost if they exist
    if sample_weights is not None:
        weights = sample_weights.reshape(-1, 1)
        cost = -np.sum(weights * one_hot_labels * np.log(h_theta + 1e-8)) / data.shape[0]
    else:
        cost = -np.sum(one_hot_labels * np.log(h_theta + 1e-8)) / data.shape[0] 

    return (a, h_theta, cost)

def backward_prop(data, one_hot_labels, params, forward_prop_func, activation='sigmoid', dropout_rate=0.0, reg=0.0, sample_weights=None): # <-- added sample_weights
    # force is_training=True so dropout is applied during gradient calculation
    # pass sample_weights to forward_prop
    a, h_theta, cost = forward_prop_func(data, one_hot_labels, params, is_training=True, activation=activation, dropout_rate=dropout_rate, sample_weights=sample_weights)

    grad_h_theta = (h_theta - one_hot_labels) / data.shape[0]
    
    # scale gradient by sample weights
    if sample_weights is not None:
        grad_h_theta *= sample_weights.reshape(-1, 1)
    
    dLdW2 = np.dot(a.T, grad_h_theta)
    dLdb2 = np.sum(grad_h_theta, axis=0)
    dLdX = np.dot(grad_h_theta, params["W2"].T)

    # apply dropout mask to gradients (i.e. don't update dead neurons)
    if params.get("dropout_mask") is not None:
        dLdX = dLdX * params["dropout_mask"]

    # calculate derivative based on activation
    if activation == 'sigmoid':
        grad_pre_activation = dLdX * (a * (1 - a))
    elif activation == 'relu':
        grad_pre_activation = dLdX * (a > 0).astype(float) # derivative of ReLU

    dLdW1 = np.dot(data.T, grad_pre_activation)
    dLdb1 = np.sum(grad_pre_activation, axis=0)

    # apply L2 weight decay/regularization here
    dLdW1 += 2 * reg * params["W1"]
    dLdW2 += 2 * reg * params["W2"]

    return {"W1": dLdW1, "W2": dLdW2, "b1": dLdb1, "b2": dLdb2}

def backward_prop_regularized(data, one_hot_labels, params, forward_prop_func, reg, activation='sigmoid'):
    grads = backward_prop(data, one_hot_labels, params, forward_prop_func, activation=activation)

    grads["W1"] += 2 * reg * params["W1"]
    grads["W2"] += 2 * reg * params["W2"]

    return grads

def gradient_descent_epoch(train_data, one_hot_train_labels, learning_rate, batch_size, params, forward_prop_func, backward_prop_func, activation='sigmoid', dropout_rate=0.0, reg=0.0, sample_weights=None): # <-- added sample_weights
    for i in range(0, train_data.shape[0], batch_size):
        batch_data = train_data[i : i + batch_size]
        batch_labels = one_hot_train_labels[i : i + batch_size]
        
        # slice weights for the current batch
        batch_weights = sample_weights[i : i + batch_size] if sample_weights is not None else None

        # pass batch_weights down
        grads = backward_prop_func(batch_data, batch_labels, params, forward_prop_func, activation=activation, dropout_rate=dropout_rate, reg=reg, sample_weights=batch_weights)

        params["W1"] -= learning_rate * grads["W1"]
        params["b1"] -= learning_rate * grads["b1"]
        params["W2"] -= learning_rate * grads["W2"]
        params["b2"] -= learning_rate * grads["b2"]
    return

def nn_train(
    train_data, train_labels, dev_data, dev_labels, 
    get_initial_params_func, forward_prop_func, backward_prop_func,
    num_hidden=300, learning_rate=5, num_epochs=30, batch_size=1000, num_classes=2,
    activation='sigmoid', dropout_rate=0.0, reg=0.0, sample_weights=None): # <-- added sample_weights

    (nexp, dim) = train_data.shape
    params = get_initial_params_func(dim, num_hidden, num_classes) 

    cost_train, cost_dev, accuracy_train, accuracy_dev = [], [], [], []

    for epoch in range(num_epochs):
        # pass sample_weights to gradient descent
        gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size, params, forward_prop_func, backward_prop_func, activation=activation, dropout_rate=dropout_rate, reg=reg, sample_weights=sample_weights)

        # eval on train and dev 
        h, output, cost = forward_prop_func(train_data, train_labels, params, activation=activation)
        cost_train.append(cost)
        accuracy_train.append(compute_accuracy(output,train_labels))
        
        h, output, cost = forward_prop_func(dev_data, dev_labels, params, activation=activation)
        cost_dev.append(cost)
        accuracy_dev.append(compute_accuracy(output, dev_labels))

    return params, cost_train, cost_dev, accuracy_train, accuracy_dev

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) == 
        np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def one_hot_labels(labels, num_classes=2): # num_classes=2 for diabetes classification
    one_hot_labels = np.zeros((labels.size, num_classes))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

# def forward_prop(data, one_hot_labels, params):
#     """
#     Implement the forward layer given the data, labels, and params.
    
#     Args:
#         data: A numpy array containing the input
#         one_hot_labels: A 2d numpy array containing the one-hot embeddings of the labels e_y.
#         params: A dictionary mapping parameter names to numpy arrays with the parameters.
#             This numpy array will contain W1, b1, W2 and b2
#             W1 and b1 represent the weights and bias for the hidden layer of the network
#             W2 and b2 represent the weights and bias for the output layer of the network

#     Returns:
#         A 3 element tuple containing:
#             1. A numpy array of the activations (after the sigmoid) of the hidden layer
#             2. A numpy array The output (after the softmax) of the output layer
#             3. The average loss for these data elements
#     """
#     # *** START CODE HERE ***
#     # DATA TO HIDDEN LAYER
#     # data is (batch_size, input_size (INFERRED))
#     # W1 is (input_size, num_hidden)
#     # b1 is (num_hidden,)
#     # a(i) = σ W[1]⊤x(i) + b[1] 
#     a = sigmoid(np.dot(data, params["W1"]) + params["b1"])

#     # HIDDEN TO OUTPUT LAYER
#     # a is (batch_size, num_hidden)
#     # W2 is (num_hidden, num_output)
#     # b2 is (num_output,)
#     # h_θ_bar(x(i)) = W[2]⊤a(i) + b[2]
#     # h_θ = softmax(hθ(x(i)))
#     h_theta_bar = np.dot(a, params["W2"]) + params["b2"]
#     h_theta = softmax(h_theta_bar)

#     # for n training examples, we average the cross entropy loss over the n examples
#     # J = -1/n * sum(e_y * log(h_theta))
#     cost = -np.sum(one_hot_labels * np.log(h_theta)) / data.shape[0]

#     return (a, h_theta, cost)
#     # *** END CODE HERE ***

# def backward_prop(data, one_hot_labels, params, forward_prop_func):
#     """
#     Implement the backward propagation gradient computation step for a neural network
    
#     Args:
#         data: A numpy array containing the input
#         one_hot_labels: A 2d numpy array containing the one-hot embeddings of the labels e_y.
#         params: A dictionary mapping parameter names to numpy arrays with the parameters.
#             This numpy array will contain W1, b1, W2 and b2
#             W1 and b1 represent the weights and bias for the hidden layer of the network
#             W2 and b2 represent the weights and bias for the output layer of the network
#         forward_prop_func: A function that follows the forward_prop API above

#     Returns:
#         A dictionary of strings to numpy arrays where each key represents the name of a weight
#         and the values represent the gradient of the loss with respect to that weight.
        
#         In particular, it should have 4 elements:
#             W1, W2, b1, and b2
#     """
#     # *** START CODE HERE ***
#     # get hidden activations, output predictions, and cost from forward_propogation
#     a, h_theta, cost = forward_prop_func(data, one_hot_labels, params)

#     # get gradient at the output layer, which is pred - true for softmax and cross entropy loss
#     # also, since CE loss is averaged over the batch, we also need to average this ???
#     # from 5a
#     grad_h_theta = (h_theta - one_hot_labels) / data.shape[0]

#     # get weight gradient at the affine layer for layer 2
#     # dL/dW = X^T @ dL/dZ
#     # here the "input" X to this layer is the hidden activation 'a'
#     # from 4b
#     dLdW2 = np.dot(a.T, grad_h_theta)

#     # get bias gradient at the affine layer for layer 2
#     # dL/db = sum(dL/dZ) over the batch dimension
#     # from 4b
#     dLdb2 = np.sum(grad_h_theta, axis=0)

#     # propogate error back to hidden layer
#     # dL/dX = dL/dZ @ W^T
#     # need to find dL/dA (grad w.r.t the hidden layer outputs) to continue backprop
#     # from 4b and 4c
#     dLdX = np.dot(grad_h_theta, params["W2"].T)

#     # get gradient at hidden layer pre-activation
#     # dL/dZ = dL/dA * σ'(Z) where σ'(Z) = A * (1 - A) element-wise
#     # from 4a
#     grad_pre_activation = dLdX * (a * (1 - a)) # why is it dLdX and not dLdA?

#     # get weight gradient at the affine layer for layer 1
#     # dL/dW = X^T @ dL/dZ
#     # from 4b
#     dLdW1 = np.dot(data.T, grad_pre_activation)
    
#     # get bias gradient at the affine layer for layer 1
#     # from 4b
#     dLdb1 = np.sum(grad_pre_activation, axis=0)

#     return {"W1": dLdW1, "W2": dLdW2, "b1": dLdb1, "b2": dLdb2}
#     # *** END CODE HERE ***

# def backward_prop_regularized(data, one_hot_labels, params, forward_prop_func, reg):
#     """
#     Implement the backward propagation gradient computation step for a neural network
#     """
#     # *** START CODE HERE ***
#     grads = backward_prop(data, one_hot_labels, params, forward_prop_func)

#     grads["W1"] += 2 * reg * params["W1"]
#     grads["W2"] += 2 * reg * params["W2"]

#     return grads
#     # *** END CODE HERE ***