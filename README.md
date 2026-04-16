# Bayesian Network Parameter Learning via Gradient Descent

Learning the conditional probability tables (CPTs) of a Bayesian Network
with binary random variables from data, using stochastic gradient descent
on the negative log-likelihood.

The **structure** (graph) of the network is assumed known; only the
**parameters** of each conditional distribution are learned.

This was developed as Part 2 of Homework 1 for the *Knowledge
Representation & Reasoning* course (Master AI).

---

## Problem setup

Let `X = (X_1, ..., X_d)` be binary variables with parents `Pa(X_i)`
defined by a known DAG. The model factorizes as

```
P_theta(X) = prod_i  P_theta(X_i | Pa(X_i))
```

Each conditional is parameterized by one scalar `theta_{i, pa(i)}` per
parent configuration and modeled with a sigmoid:

```
P_theta(X_i = 1 | Pa(X_i) = pa(i)) = sigma(theta_{i, pa(i)}) = 1 / (1 + exp(-theta_{i, pa(i)}))
```

Training minimizes the cross-entropy between the true generating
distribution and the model:

```
L(theta) = -E_{x ~ P_true} [ log P_theta(x) ]
```

### Gradient derivation

For a single sample `x`, the per-node log-likelihood is

```
log P_theta(X_i = x_i | pa(i)) = x_i * log sigma(theta) + (1 - x_i) * log (1 - sigma(theta))
```

Using `d/dtheta sigma(theta) = sigma(theta) (1 - sigma(theta))`, the
gradient with respect to the parameter active for this sample reduces
to the classic logistic form:

```
grad_{theta_{i, pa(i)}}  log P_theta(X_i = x_i | pa(i))  =  x_i - sigma(theta_{i, pa(i)})
```

Only the parameter matching the observed parent configuration
`pa(i)^(k)` is updated for sample `x^(k)`. The stochastic update is

```
theta_{i, pa(i)}  <-  theta_{i, pa(i)}  +  eta * (x_i - sigma(theta_{i, pa(i)}))
```

### Entropy and cross-entropy

Because the ground-truth network is provided, the true entropy
`H(P_true)` is available as a lower bound that cross-entropy will
approach during training:

```
H(P_true) = sum_i sum_{pa(i)}  P_true(pa(i)) * h_b( P_true(X_i = 1 | pa(i)) )
```

where `h_b(p) = -p log p - (1-p) log(1-p)`. `P_true(pa(i))` is estimated
by ancestral sampling from the ground-truth network.

The model cross-entropy is estimated as the average negative
log-likelihood over samples drawn from `P_true`.

---

## Repository layout

```
.
├── data/
│   ├── bn_learning             # ground-truth BN (structure + CPTs)
│   └── samples_bn_learning     # samples drawn from the ground-truth BN
├── results/                    # saved cross-entropy plots
├── src/
│   ├── main.py                 # entry point — wires everything together
│   ├── structure.py            # parses the BN structure (DAG only)
│   ├── bayes_net.py            # full BN (structure + CPTs) for ground truth / sampling
│   ├── parameters.py           # theta initialization, sigmoid, per-sample gradient update
│   ├── training.py             # SGD loop + cross-entropy monitoring
│   ├── true_entropy.py         # estimates H(P_true) via ancestral sampling
│   ├── comparison.py           # prints learned CPTs next to ground-truth CPTs
│   └── ploting.py              # sample loader + matplotlib cross-entropy plot
├── requirements.txt
└── README.md
```

### BN file format (`data/bn_learning`)

```
N                                # number of variables
<var> ; <parent_1> <parent_2> ... ; <p_1> <p_2> ... <p_{2^k}>
...
```

Each line declares a node. The CPT row lists `P(var = 1 | pa)` for every
parent configuration, in the order produced by iterating parent values
`(1, 0)` for each parent (see `BayesNet._create_cpd` in `src/bayes_net.py`).

The shipped network has 15 binary variables (`A` .. `O`) with a DAG of
mixed fan-in up to 2.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+.

## Usage

Run from the project root (paths in `main.py` are relative to the repo
root, not to `src/`):

```bash
python src/main.py
```

This will:

1. Parse the ground-truth BN structure and CPTs from `data/bn_learning`.
2. Load training samples from `data/samples_bn_learning`.
3. Initialize all `theta_{i, pa(i)}` from `N(0, 0.1)`.
4. Estimate `H(P_true)` by ancestral sampling.
5. Run stochastic gradient descent for 50 epochs (`lr = 1e-3`).
6. Save a cross-entropy-vs-epoch plot to `results/cross_entropy.png`.
7. Print a per-entry comparison of learned vs. ground-truth CPTs.

Hyperparameters (learning rate, epochs, init scale, monitoring cadence)
are set at the call sites in `main.py` and in `train(...)` /
`initialize_theta(...)`.

---

## Results

The plot in `results/cross_entropy.png` shows the cross-entropy falling
toward the estimated `H(P_true)` line — the gap that remains is the KL
divergence between the learned model and the true distribution, which
is bounded below by sampling noise and by the finite training set size.

Additional runs with different hyperparameters are also saved:

- `results/cross_entropy_ep50.png` — longer training
- `results/cross_entropy_lr005.png` — higher learning rate (`lr = 0.05`)

The `compare_cpts` output confirms that the learned per-configuration
probabilities match the ground-truth CPTs within a small tolerance.

---

## Notes on scope

- Only Part 2 (programming) of the assignment is implemented here. The
  written problems (Part 1) are not part of this repository.
- The implementation is deliberately simple (pure-Python SGD over
  dicts, no autodiff). It is meant to make the gradient derivation
  directly visible in the code, not to be fast.
