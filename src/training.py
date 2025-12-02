import random
from structure import get_parent_config
from parameters import node_log_prob, update_theta

def sample_log_prob_model(bn_structure, sample, theta):
    logp = 0.0
    for var in bn_structure.variables:
        x_i = sample[var]
        parent_config = get_parent_config(bn_structure, var, sample)
        logp += node_log_prob(var, x_i, parent_config, theta)
    return logp

def cross_entropy_model(bn_structure, samples, theta):
    total = 0.0
    N = len(samples)
    for sample in samples:
        total += sample_log_prob_model(bn_structure, sample, theta)
    return -total / N

def train(bn_structure, samples, theta, lr=0.05, epochs=50, monitor_every=1):
    ce_history = []

    for epoch in range(1, epochs + 1):
        random.shuffle(samples)

        for sample in samples:
            for var in bn_structure.variables:
                x_i = sample[var]
                parent_config = get_parent_config(bn_structure, var, sample)
                update_theta(var, x_i, parent_config, theta, lr)

        if epoch % monitor_every == 0:
            ce = cross_entropy_model(bn_structure, samples, theta)
            ce_history.append((epoch, ce))
            print(f"Epoch {epoch}/{epochs} - Cross-Entropy: {ce:.4f}")

    return theta, ce_history
