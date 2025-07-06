import numpy as np
from baseCrossover import BaseCrossover

class PMXCrossover(BaseCrossover):

    def crossover(self, parents: np.ndarray, crossover_prob: float) -> np.ndarray:

        total_parents = parents.shape[0]
        offspring = np.empty_like(parents)
        childs = 0

        for i in range(0, total_parents, 2):
            
            a = parents[i]
            b = parents[i+1]

            if np.random.rand() < crossover_prob:
                #Crossover between two parents
                child_a = self.pmx_crossover(a, b)
                child_b = self.pmx_crossover(b, a)
                offspring[childs] = child_a
                offspring[childs + 1] = child_b
            else:
                # No crossover, offspring are the same as parents
                offspring[childs] = a.copy()
                offspring[childs + 1] = b.copy()

            childs += 2
        
        return offspring

    def pmx_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        size = parent1.shape[0]
        child = -np.ones(size, dtype=parent1.dtype)

        #Randomly chooses 2 swaping points, then swap segments
        p1, p2 = sorted(self.rng.choice(size, 2, replace=False))
        child[p1:p2+1] = parent1[p1:p2+1]

        #Create a mapping between parent1 and parent2 for the swapped segment
        mapping = {}
        for i in range(p1, p2+1):
            mapping[parent1[i]] = parent2[i]

        #Fill the remaining positions from parent2, resolving conflicts via mapping
        for i in range(size):
            if child[i] == -1:  # Position not filled yet
                val = parent2[i]
                while val in child:  # Resolve conflicts using mapping
                    val = mapping[val]
                child[i] = val

        # 5. Final validation
        assert -1 not in child, "Error: Incomplete child"
        assert len(np.unique(child)) == size, "Error: Duplicate elements in child"
        return child
