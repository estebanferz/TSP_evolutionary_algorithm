from mutation.baseMutation import BaseMutation
import numpy as np

class InversionMutation(BaseMutation):
    
    def __init__(self):
        self.rng = np.random.default_rng()

    def mutate(self, offspring: np.ndarray, mutation_rate: float) -> np.ndarray:
        mutated_offspring = offspring.copy()
        num_childs, num_cities = mutated_offspring.shape

        for i in range(num_childs):
            if self.rng.random() < mutation_rate:
                # Seleccionar dos ciudades
                idx1, idx2 = sorted(self.rng.choice(num_cities, 2, replace=False))

                # Invertir el segmento entre las dos ciudades (incluyendo a las ciudades)
                mutated_offspring[i, idx1:idx2+1] = mutated_offspring[i, idx1:idx2+1][::-1]

        return mutated_offspring