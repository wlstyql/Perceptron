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

def _numerical_gradient_no_batch(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성
    
    for idx in range(x.size):
        tmp_val = x[idx]
        
        x[idx] = tmp_val + h  # f(x+h) 계산
        fxh1 = f(x)
        
        x[idx] = tmp_val - h  # f(x-h) 계산
        fxh2 = f(x) 
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 값 복원
        
    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad
