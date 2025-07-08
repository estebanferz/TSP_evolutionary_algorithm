from mutation.baseMutation import BaseMutation
import numpy as np 

class SwapMutation(BaseMutation):
    def __init__(self, max_swaps=10):
        self.max_swaps = max_swaps
        self.rng = np.random.default_rng()

    def mutate(self, parent: np.ndarray, mutation_rate: float) -> np.ndarray:
        if self.rng.random() >= mutation_rate:
            return parent.copy()

        child = parent.copy()
        size = len(child)

        # Elegir número de swaps entre 2 y max_swaps (agresivo y aleatorio)
        num_swaps = self.rng.integers(2, self.max_swaps + 1)

        for _ in range(num_swaps):
            i, j = self.rng.choice(size, 2, replace=False)
            child[i], child[j] = child[j], child[i]

        return child
