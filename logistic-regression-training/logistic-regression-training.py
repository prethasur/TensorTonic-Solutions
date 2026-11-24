import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    n_samples, n_features = X.shape
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    for _ in range(steps):
        z = X @ w + b
        
        y_pred = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

      
        error = y_pred - y
        dw = (X.T @ error) / n_samples
        db = float(np.mean(error))

        
        w -= lr * dw
        b -= lr * db

    return w, float(b)


    
    pass