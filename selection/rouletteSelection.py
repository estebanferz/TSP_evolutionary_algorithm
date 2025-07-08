import numpy as np
from selection.baseSelection import BaseSelection

class RouletteSelection(BaseSelection):
    def select(self, population: np.ndarray, fitness: np.ndarray, parents: int) -> np.ndarray:
        
        # Initialize offspring array
        offspring = np.empty((parents, population.shape[1]), dtype=population.dtype)
        fitness_copy = np.clip(fitness, a_min=1e-10, a_max=None)        
        probabilities = fitness_copy / np.sum(fitness_copy)

        #Perform roulette wheel selection
        selected_indices = np.random.choice(len(population), size=parents, p=probabilities)
        for i, j in enumerate(selected_indices):
            offspring[i] = population[j]

        return offspring
