import itertools
import numpy as np
from parameters import model_prob

def binary_entropy(p):
    if p in (0, 1):
        return 0.0
    return -(p*np.log(p) + (1-p)*np.log(1-p))

def estimate_parent_config_probs(true_bn, num_samples=50000):
    counts = {}
    for var in true_bn.nodes:
        parent_vars = [p.var_name for p in true_bn.nodes[var].parent_nodes]
        k = len(parent_vars)
        configs = list(itertools.product([0,1], repeat=k))
        counts[var] = {cfg: 0 for cfg in configs}

    for _ in range(num_samples):
        sam = true_bn.sample()
        for var in true_bn.nodes:
            parents = true_bn.nodes[var].parent_nodes
            cfg = tuple(sam[p.var_name] for p in parents)
            counts[var][cfg] += 1

    return {
        var: {cfg: counts[var][cfg] / num_samples for cfg in counts[var]}
        for var in counts
    }

def compute_true_entropy_sampling(true_bn, num_samples=50000):
    parent_probs = estimate_parent_config_probs(true_bn, num_samples)
    total = 0.0

    for var in true_bn.nodes:
        parents = true_bn.nodes[var].parent_nodes
        k = len(parents)
        configs = list(itertools.product([0,1], repeat=k))

        for cfg in configs:
            p_pa = parent_probs[var][cfg]
            p_true = true_bn.prob(var, list(cfg))
            total += p_pa * binary_entropy(p_true)

    return total
