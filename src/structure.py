import itertools

class BNStructure:
    def __init__(self, bn_file):
        self.variables = []
        self.parents = {}
        self._parse_structure(bn_file)

    def _parse_structure(self, bn_file):
        with open(bn_file, "r") as f:
            N = int(f.readline().strip())

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

def get_parent_config(bn_structure, var, sample):
    """Return tuple of parent values for a given variable in a sample."""
    parents = bn_structure.parents[var]
    if len(parents) == 0:
        return ()
    return tuple(sample[p] for p in parents)
