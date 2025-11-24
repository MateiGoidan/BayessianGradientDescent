import numpy as np

def sigmoid(theta: float) -> float:
    """
    Compute the sigmoid function.
    Maps real-valued theta to a probability between 0 and 1.
    """
    return 1 / (1 + np.exp(-theta))

def gradient(theta: float, x_i: int) -> float:
    """
    Compute the gradient of the log-likelihood for a binary variable X_i
    with respect to its parameter theta, given a single sample value x_i (0 or 1).

    grad = x_i - sigmoid(theta)
    """
    p_hat = sigmoid(theta)
    return x_i - p_hat

def update_theta(theta: float, grad: float, lr: float) -> float:
    """
    Update rule for gradient ascent (maximize log-likelihood).

    theta_new = theta + lr * grad
    """
    return theta + lr * grad

def initialize_theta(low: float = -1.0, high: float = 1.0) -> float:
    """
    Initialize a parameter theta randomly between given bounds.
    """
    return np.random.uniform(low, high)
