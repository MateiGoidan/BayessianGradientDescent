import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from bayes_net import BayesNet
from gradient_utils import sigmoid, gradient, update_theta, initialize_theta

BN_FILE = "data/bn_learning"
SAMPLES_FILE = "data/samples_bn_learning"
LEARNING_RATE = 0.1
EPOCHS = 50

bn = BayesNet(bn_file="data/bn_learning")

df = pd.read_csv(SAMPLES_FILE, sep=" ", engine="python")
df = df.dropna(axis=1, how="all") # cleanup in case of trailing spaces


# Initialize theta values for each variable-parent configuration
theta_params = {}
for var_name, node in bn.nodes.items():
    parent_vars = [p.var_name for p in node.parent_nodes]
    parent_configs = [list(p) for p in itertools.product([0, 1], repeat=len(parent_vars))] if parent_vars else [()]
    for config in parent_configs:
        theta_params[(var_name, tuple(config))] = initialize_theta()


# Training # Track cross-entropy
cross_entropy_history = []

# Training loop
for epoch in range(EPOCHS):
    df = df.sample(frac=1).reset_index(drop=True) # shuffle
    total_log_likelihood = 0
    for _, row in df.iterrows():
        sample = row.to_dict()
        for var_name, node in bn.nodes.items():
            x_i = sample[var_name]
            parent_vars = [p.var_name for p in node.parent_nodes]
            parent_vals = tuple(sample[p] for p in parent_vars) if parent_vars else ()
            theta = theta_params[(var_name, parent_vals)]
            p_hat = sigmoid(theta)
            grad = x_i - p_hat
            theta_params[(var_name, parent_vals)] = update_theta(theta, grad, LEARNING_RATE)


            # accumulate log-likelihood
            p = p_hat if x_i == 1 else (1 - p_hat)
            total_log_likelihood += np.log(p + 1e-9) # avoid log(0)

    cross_entropy = -total_log_likelihood / (len(df) * len(bn.nodes))
    cross_entropy_history.append(cross_entropy)
    print(f"Epoch {epoch+1}/{EPOCHS} - Cross-Entropy: {cross_entropy:.4f}")

# Plot cross-entropy
plt.plot(cross_entropy_history)
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy")
plt.title("Training Progress")
plt.grid(True)
plt.tight_layout()
plt.savefig("training_curve.png")

print("Training complete.")
