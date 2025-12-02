import itertools
from parameters import sigmoid

def compare_cpts(bn_structure, theta, true_bn, threshold=0.05):
    print("=== CPT Comparison ===")

    for var in bn_structure.variables:
        parents = bn_structure.parents[var]
        k = len(parents)
        configs = list(itertools.product([0,1], repeat=k))

        print(f"\nVariable: {var}")
        for cfg in configs:
            true_p = true_bn.prob(var, list(cfg))
            learned_p = sigmoid(theta[var][cfg])
            diff = abs(true_p - learned_p)

            cfg_str = ", ".join(f"{parents[i]}={cfg[i]}" for i in range(k)) if k>0 else "∅"
            flag = " **BIG DIFF**" if diff > threshold else ""
            print(f"  P({var}=1 | {cfg_str}) true={true_p:.4f}  learned={learned_p:.4f}  diff={diff:.4f}{flag}")
