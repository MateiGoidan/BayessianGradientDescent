import matplotlib.pyplot as plt

def load_samples(samples_file, bn_structure):
    samples = []
    with open(samples_file, "r") as f:
        header = f.readline().strip().split()
        assert header == bn_structure.variables

        for line in f:
            row = line.strip().split()
            if not row:
                continue
            values = list(map(int, row))
            sample = {var: val for var, val in zip(header, values)}
            samples.append(sample)
    return samples


def plot_cross_entropy(ce_history, true_entropy=None, filename="results/cross_entropy.png"):
    epochs = [e for (e, _) in ce_history]
    values = [ce for (_, ce) in ce_history]

    plt.figure(figsize=(8,5))
    plt.plot(epochs, values, label="Cross-Entropy")

    if true_entropy is not None:
        plt.axhline(y=true_entropy, linestyle='--', color='red',
                    label=f"True Entropy={true_entropy:.4f}")

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy")
    plt.title("Cross Entropy During Training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
