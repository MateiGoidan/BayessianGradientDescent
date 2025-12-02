from structure import BNStructure
from parameters import initialize_theta
from ploting import load_samples, plot_cross_entropy
from training import train
from true_entropy import compute_true_entropy_sampling
from comparison import compare_cpts
from bayes_net import BayesNet

# Load structure
bn = BNStructure("data/bn_learning")

# Initialize parameters
theta = initialize_theta(bn, scale=0.1)

# Load samples
samples = load_samples("data/samples_bn_learning", bn)

# Load ground-truth BN
true_bn = BayesNet("data/bn_learning")

# Estimate true entropy
H_true = compute_true_entropy_sampling(true_bn, num_samples=50000)
print("Estimated True Entropy:", H_true)

# Train model
theta, ce_history = train(bn, samples, theta, lr=0.001, epochs=50)

# Plot
plot_cross_entropy(ce_history, true_entropy=H_true)

# Compare CPTs
compare_cpts(bn, theta, true_bn)
