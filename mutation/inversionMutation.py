from baseMutation import BaseMutation
import numpy as np

class InversionMutation(BaseMutation):

    def mutate(self, parent: np.ndarray, mutation_rate: float) -> np.ndarray:
        if np.random.random() < mutation_rate:
            # Create a copy to avoid modifying the original
            child = parent.copy()
            size = len(child)
            # Select two distinct cut points
            p1, p2 = np.random.choice(size, 2, replace=False)
            start, end = min(p1, p2), max(p1, p2)
            # Reverse the segment
            child[start:end+1] = child[start:end+1][::-1]
            return child
        return parent