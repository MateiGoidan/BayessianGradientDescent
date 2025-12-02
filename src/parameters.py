import numpy as np
import itertools

def initialize_theta(bn_structure, scale=0.1):
    theta = {}

    for var in bn_structure.variables:
        parents = bn_structure.parents[var]
        k = len(parents)
        parent_configs = list(itertools.product([0, 1], repeat=k))

        theta[var] = {}
        for cfg in parent_configs:
            theta[var][cfg] = np.random.randn() * scale

    return theta

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def model_prob(var, parent_config, theta):
    return sigmoid(theta[var][parent_config])

def node_log_prob(var, x_i, parent_config, theta):
    p = model_prob(var, parent_config, theta)
    return np.log(p) if x_i == 1 else np.log(1 - p)

def update_theta(var, x_i, parent_config, theta, lr):
    theta_val = theta[var][parent_config]
    p_hat = sigmoid(theta_val)
    grad = x_i - p_hat
    theta[var][parent_config] = theta_val + lr * grad

