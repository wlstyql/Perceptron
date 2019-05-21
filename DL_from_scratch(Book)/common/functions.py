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

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)  # f(x+h)
        
        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
        
    return grad
        
