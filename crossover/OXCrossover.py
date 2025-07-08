import numpy as np
from crossover.baseCrossover import BaseCrossover

class OXCrossover(BaseCrossover):
    
    def crossover(self, parents: np.ndarray, crossover_prob: float) -> np.ndarray:
        total_parents = parents.shape[0]
        offspring = np.empty_like(parents)
        childs = 0

        for i in range(0, total_parents, 2):
            a = parents[i]
            b = parents[i+1]

            if np.random.rand() < crossover_prob:
                # Crossover between two parents
                child_a = self.ox_crossover(a, b)
                child_b = self.ox_crossover(b, a)
                offspring[childs] = child_a
                offspring[childs + 1] = child_b
            else:
                # No crossover, offspring are the same as parents
                offspring[childs] = a.copy()
                offspring[childs + 1] = b.copy()

            childs += 2
        
        return offspring

    def ox_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        size = parent1.shape[0]
        child = -np.ones(size, dtype=parent1.dtype)

        #Select two random cut points, copy the segment from parent1 to child
        p1, p2 = sorted(np.random.choice(size, 2, replace=False))
        child[p1:p2+1] = parent1[p1:p2+1]

        #Fill remaining positions from parent2 (in order, skipping duplicates)
        # Start filling from position p2+1 (wrapping around if needed)
        ptr = (p2 + 1) % size  # Pointer to the next position to fill
        for val in np.roll(parent2, -p2-1):  # Roll parent2 to start after p2
            if val not in child:
                child[ptr] = val
                ptr = (ptr + 1) % size
                if -1 not in child:  # Early exit if child is complete
                    break

        # 4. Final validation
        assert -1 not in child, "Error: Incomplete child"
        assert len(np.unique(child)) == size, "Error: Duplicate elements in child"
        return child