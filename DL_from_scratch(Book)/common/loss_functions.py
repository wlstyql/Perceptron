import numpy as np

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

def batch_cross_entropy_error(y, t):
    delta = 1e-7
    if y.ndim == 1:
        batch_size = 1
    else:
        batch_size = y.shape[0]    
    return -np.sum(t * np.log(y + delta)) / batch_size   
