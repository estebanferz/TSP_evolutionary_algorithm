from baseMutation import BaseMutation
import numpy as np 

class SwapMutation(BaseMutation):

    def mutate(self, parent: np.ndarray, mutation_rate: float) -> np.ndarray:
        if np.random.random() < mutation_rate:
            # Create a copy to avoid modifying the original
            child = parent.copy()
            size = len(child)
            # Select two distinct indices
            idx1, idx2 = np.random.choice(size, 2, replace=False)
            # Swap the genes
            child[idx1], child[idx2] = child[idx2], child[idx1]
            return child
        return parent