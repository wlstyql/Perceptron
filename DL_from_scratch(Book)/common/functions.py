import numpy as np

def step_function(x):
    return np.array(x > 0, dtype=np.int)

def identity_function(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c) # overflow prevention
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x

def relu(x):
    return np.maximum(0, x)

def mse(y, t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))
