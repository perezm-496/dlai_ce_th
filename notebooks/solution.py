"""
    Code with solution to use in the test module.
    The separation allows to don't give out the solution
    with the test.
"""

import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def my_perceptron_forward(x, W, b, activation=sigmoid):
    """
    Correct solution for task1
    Computes the passforward for a feture input X of shape (N,) and
    weight matrix W of shape (M, N).
    """
    # ToDo: Manage the exceptional cases where the shape don't match, but keep it functional for check.
    z = np.matmul(W, x) + b
    a = activation(z)
    return a
 
