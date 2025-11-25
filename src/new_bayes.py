import itertools
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from bayes_net import BayesNet

def initialize_theta(bn_structure, scale=0.1):
    """
        Create a dictionary of learnable parameters for each variable.

        Returns:
            theta: dict like theta[var][parent_tuple] = float
        """

    theta = {}

    for var in bn_structure.variables:
        parents = bn_structure.parents[var]
        k = len(parents)

        # Generate all parent configurations (as tuples)
        parent_configs = list(itertools.product([0, 1], repeat=k))

        # Initialize parameters for this variable
        theta[var] = {}

        for config in parent_configs:
            # random small initialization (Gaussian)
            theta[var][config] = np.random.randn() * scale

    return theta

def load_samples(samples_file, bn_structure):
    """
    Loads samples from a text file.

    Parameters:
        samples_file: path to samples file
        bn_structure: BNStructure instance (from Step 1)

    Returns:
        samples: list of dicts {var_name: 0/1}
    """

    samples = []
    with open(samples_file, "r") as f:
        # First line contains the variable names
        header = f.readline().strip().split()
        
        # Must match the BN structure order
        assert header == bn_structure.variables, \
            "Sample header does not match BN variable order!"

        # Read each sample row
        for line in f:
            row = line.strip().split()
            if not row:
                continue  # skip empty lines
            
            # Convert to ints
            values = list(map(int, row))

            # Create a dict: {var: value}
            sample = {var: val for var, val in zip(header, values)}

            samples.append(sample)

    return samples

def get_parent_config(bn_structure, var, sample):
    """
    Returns a tuple representing the parent configuration for the given variable
    in the given sample.

    Example:
        parents = ["A", "B"]
        sample = {"A":1, "B":0, "C":1}
        returns (1, 0)
    """

    parents = bn_structure.parents[var]
    if len(parents) == 0:
        return ()   # empty tuple for root nodes

    return tuple(sample[p] for p in parents)

class BNStructure:
    """
    Stores ONLY the strucure of a Bayesian Network:
    - variable names
    - parent relationships
    """
    
    def __init__(self, bn_file):
        self.variables = []      # ordered list of variable names
        self.parents = {}        # dict: var -> list of parents
        self._parse_structure(bn_file)

    def _parse_structure(self, bn_file):
        with open(bn_file, "r") as f:
            N = int(f.readline().strip())  # number of variables

            for _ in range(N):
                line = f.readline().strip()
                var, parents_str, _ = line.split(";")

                var = var.strip()
                parent_list = parents_str.split()
                parent_list = [p.strip() for p in parent_list]

                self.variables.append(var)
                self.parents[var] = parent_list

    def pretty_print(self):
        print("Bayesian Network structure:")
        for var in self.variables:
            print(f"  {var} | Parents: {self.parents[var]}")

def sigmoid(x):
    """Sigmoid function converting θ to probability."""
    return 1 / (1 + np.exp(-x))


def model_prob(var, parent_config, theta):
    """
    Compute Pθ(X = 1 | parent_config)
    using the current model parameters θ.
    """
    theta_val = theta[var][parent_config]
    return sigmoid(theta_val)


def node_log_prob(var, x_i, parent_config, theta):
    """
    Compute log Pθ(X_i = x_i | parents)
    Used in cross-entropy and log-likelihood.
    """
    p = model_prob(var, parent_config, theta)

    if x_i == 1:
        return np.log(p)
    else:
        return np.log(1 - p)

def update_theta(var, x_i, parent_config, theta, lr):
    """
    Perform a gradient ascent update on θ for a single variable in a single sample.

    θ[var][parent_config] += lr * (x_i - sigmoid(θ[var][parent_config]))
    """

    theta_val = theta[var][parent_config]
    p_hat = sigmoid(theta_val)

    grad = x_i - p_hat  

    theta[var][parent_config] = theta_val + lr * grad

def sample_log_prob_model(bn_structure, sample, theta):
    """
    Compute log Pθ(sample) under the current model parameters θ.

    log Pθ(x) = sum_i log Pθ(x[i] | parents[i])
    """

    logp = 0.0

    for var in bn_structure.variables:
        x_i = sample[var]
        parent_config = get_parent_config(bn_structure, var, sample)
        logp += node_log_prob(var, x_i, parent_config, theta)

    return logp

def cross_entropy_model(bn_structure, samples, theta):
    """
    Computes the empirical cross-entropy:
    H_hat = -(1/N) * sum_k log Pθ(x^(k))
    """

    total_logp = 0.0
    N = len(samples)

    for sample in samples:
        total_logp += sample_log_prob_model(bn_structure, sample, theta)

    return -total_logp / N

def train(bn_structure, samples, theta, lr=0.05, epochs=50, monitor_every=1):
    """
    Full gradient descent training loop for learning θ parameters.
    
    Returns:
        theta (updated parameters)
        ce_history (list of cross-entropy values over time)
    """
    
    ce_history = []

    for epoch in range(1, epochs + 1):
        # Shuffle samples each epoch
        random.shuffle(samples)

        # Loop through all samples
        for sample in samples:
            for var in bn_structure.variables:
                x_i = sample[var]
                parent_config = get_parent_config(bn_structure, var, sample)
                update_theta(var, x_i, parent_config, theta, lr)
        
        # Monitor cross-entropy
        if epoch % monitor_every == 0:
            ce = cross_entropy_model(bn_structure, samples, theta)
            ce_history.append((epoch, ce))
            print(f"Epoch {epoch}/{epochs} - Cross-Entropy: {ce:.4f}")

    return theta, ce_history

def binary_entropy(p):
    """Binary entropy hb(p) = -p log(p) - (1-p) log(1-p)."""
    if p == 0 or p == 1:
        return 0.0
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def compute_parent_config_prob(var, parent_config, true_bn):
    """
    Compute Ptrue(pa(i)) using the true BN (with real CPTs).

    parent_config = tuple of 0/1 for the parents of var
    true_bn = instance of BayesNet (original lab version)
    """

    parents = true_bn.nodes[var].parent_nodes
    parent_names = [p.var_name for p in parents]

    # Build a dictionary of parent assignments
    assignment = {parent_names[i]: parent_config[i] for i in range(len(parents))}

    # We must compute the joint probability of all parent values.
    # This requires walking through the parents' own parents recursively.

    def joint_prob(var_name, value, memo):
        """Compute P(var=value | its parents) * P(parents)"""
        if (var_name, value) in memo:
            return memo[(var_name, value)]

        node = true_bn.nodes[var_name]
        parent_list = [p.var_name for p in node.parent_nodes]

        if len(parent_list) == 0:
            # Root node
            p = true_bn.prob(var_name, [])
            memo[(var_name, 1)] = p
            memo[(var_name, 0)] = 1 - p
            return memo[(var_name, value)]

        # Compute parent values
        parent_vals = []
        for p in parent_list:
            if p in assignment:
                parent_vals.append(assignment[p])
            else:
                # Recursively compute parent values if not in assignment
                # But since pa(i) only includes immediate parents, we assume
                # the parent graph is evaluated for root probability.
                raise RuntimeError("Parent config not fully specified")

        # Compute conditional probability
        p_true = true_bn.prob(var_name, parent_vals)

        memo[(var_name, 1)] = p_true
        memo[(var_name, 0)] = 1 - p_true

        return memo[(var_name, value)]

    # Multiply probabilities of all parents (independent given their own parents already in CPT)
    memo = {}
    P = 1.0
    for i, parent in enumerate(parent_names):
        v = parent_config[i]
        P *= joint_prob(parent, v, memo)

    return P


def compute_true_entropy(true_bn):
    """
    Compute the true entropy H(Ptrue) using the real CPTs.
    """

    total_entropy = 0.0

    for var in true_bn.nodes:
        node = true_bn.nodes[var]
        parent_nodes = node.parent_nodes
        parent_names = [p.var_name for p in parent_nodes]
        k = len(parent_names)

        # Enumerate parent configurations
        parent_configs = list(itertools.product([0,1], repeat=k))

        for config in parent_configs:
            # Compute probability of this parent configuration
            P_pa = compute_parent_config_prob(var, config, true_bn)

            # True conditional probability P(Xi=1 | pa)
            parent_vals = list(config)
            p_true = true_bn.prob(var, parent_vals)

            # Binary entropy of variable Xi under this pa
            h = binary_entropy(p_true)

            # Add weighted contribution
            total_entropy += P_pa * h

    return total_entropy

def estimate_parent_config_probs(true_bn, num_samples=50000):
    """
    Estimate Ptrue(pa(i)) for every node using sampling.
    Returns: probs[var][parent_config] = probability
    """

    # Initialize counters
    counts = {}
    for var in true_bn.nodes:
        parent_vars = [p.var_name for p in true_bn.nodes[var].parent_nodes]
        k = len(parent_vars)
        configs = list(itertools.product([0,1], repeat=k))
        counts[var] = {cfg: 0 for cfg in configs}

    # Sampling loop
    for _ in range(num_samples):
        sam = true_bn.sample()    # true BN sample

        for var in true_bn.nodes:
            parents = true_bn.nodes[var].parent_nodes
            parent_names = [p.var_name for p in parents]
            cfg = tuple(sam[p] for p in parent_names)
            counts[var][cfg] += 1

    # Normalize to probabilities
    probs = {}
    for var in counts:
        total = float(num_samples)
        probs[var] = {cfg: counts[var][cfg] / total for cfg in counts[var]}

    return probs


def compute_true_entropy_sampling(true_bn, num_samples=50000):
    """
    Compute true entropy using sampling-based estimates for Ptrue(pa(i)).
    """
    parent_probs = estimate_parent_config_probs(true_bn, num_samples)
    total_entropy = 0.0

    for var in true_bn.nodes:
        node = true_bn.nodes[var]
        parent_names = [p.var_name for p in node.parent_nodes]
        k = len(parent_names)
        parent_configs = list(itertools.product([0,1], repeat=k))

        for cfg in parent_configs:
            P_pa = parent_probs[var][cfg]
            parent_vals = list(cfg)
            p_true = true_bn.prob(var, parent_vals)
            h = binary_entropy(p_true)
            total_entropy += P_pa * h

    return total_entropy


def plot_cross_entropy(ce_history, true_entropy=None):
    """
    Plot the cross-entropy values over training epochs.
    ce_history: list of (epoch, cross_entropy) pairs
    true_entropy: float or None (if provided, draw a horizontal line)
    """

    filename="cross_entropy_lr005.png"
    epochs = [e for (e, _) in ce_history]
    values = [ce for (_, ce) in ce_history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, values, marker='o', label="Cross-Entropy (model)")

    if true_entropy is not None:
        plt.axhline(y=true_entropy, color='red', linestyle='--',
                    label=f"True Entropy = {true_entropy:.4f}")

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy")
    plt.title("Cross-Entropy During Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)

def compare_cpts(bn_structure, theta, true_bn, threshold=0.05):
    """
    Compare learned CPTs (from theta) with true CPTs (from true_bn).

    threshold: if |true - learned| > threshold, highlight it
    """

    print("=== CPT Comparison ===\n")

    for var in bn_structure.variables:
        print(f"Variable: {var}")
        parents = bn_structure.parents[var]
        parent_names = parents
        k = len(parents)

        # All possible parent configurations
        parent_configs = list(itertools.product([0,1], repeat=k))

        for cfg in parent_configs:
            # True probability
            true_p = true_bn.prob(var, list(cfg))

            # Learned probability
            learned_theta = theta[var][cfg]
            learned_p = sigmoid(learned_theta)

            # Difference
            diff = abs(true_p - learned_p)

            # Format nicely
            cfg_str = ", ".join(f"{parent_names[i]}={cfg[i]}" for i in range(k)) if k > 0 else "∅"

            if diff > threshold:
                flag = " **BIG DIFF**"
            else:
                flag = ""

            print(f"  P({var}=1 | {cfg_str:10s})  true={true_p:.4f}   learned={learned_p:.4f}   diff={diff:.4f}{flag}")

        print()


bn = BNStructure("data/bn_learning")
# bn.pretty_print()

theta = initialize_theta(bn)

# print("θ parameters for variable C:")
# print(theta["C"])
#
samples = load_samples("data/samples_bn_learning", bn)
# s0 = samples[0]
# print(s0)
#
# print("Parents of C:", bn.parents["C"])
# print("Parent config for C:", get_parent_config(bn, "C", s0))
#
# print("Parents of E:", bn.parents["E"])
# print("Parent config for E:", get_parent_config(bn, "E", s0))
#
# pc_D = get_parent_config(bn, "D", s0)
# print("Parent config for D:", pc_D)
# print("θ[D][pc]:", theta["D"][pc_D])
# print("Pθ(D=1 | parents):", model_prob("D", pc_D, theta))
# print("log Pθ(D=0 | parents):", node_log_prob("D", 0, pc_D, theta))
#
# sample = samples[0]
# var = "D"
# pc = get_parent_config(bn, var, sample)
# x_i = sample[var]
#
# print("Before:", theta[var][pc])
# update_theta(var, x_i, pc, theta, lr=0.1)
# print("After:", theta[var][pc])
#
#
# ce = cross_entropy_model(bn, samples[:100], theta)
# print("Initial cross-entropy:", ce)

true_bn = BayesNet("data/bn_learning")
H_true = compute_true_entropy_sampling(true_bn, num_samples=50000)
print("Estimated True Entropy:", H_true)

print("-----------------------------")

theta, ce_history = train(bn, samples, theta, lr=0.001, epochs=200)

plot_cross_entropy(ce_history, true_entropy=H_true)

compare_cpts(bn, theta, true_bn)

# bn = BNStructure("data/bn_learning")
# bn.pretty_print()
#
# theta = initialize_theta(bn)
#
# for x in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"]:
#     print(f"Parents of {x}: {bn.parents[x]}")

