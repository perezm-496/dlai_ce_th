import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def my_perceptron_forward(x, W, b, activation=sigmoid):
    """
    Correct solution for task1
    Computes the passforward for a feture input X
    """
    return activation(np.dot(W, x) +b)

