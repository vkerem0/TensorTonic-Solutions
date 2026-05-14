import numpy as np



def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))


    

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    n, f = X.shape
    w = np.zeros(f)
    b = 0.0
    
    for i in range(steps):
        z = np.dot(X, w) + b
        y_pred = _sigmoid(z)
        dw = np.dot(X.T, (y_pred - y))
        db = np.mean(y_pred - y)
        
        w = w - lr * dw
        b = b - lr * db

    return w, b


