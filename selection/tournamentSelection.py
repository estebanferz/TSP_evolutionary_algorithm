import numpy as np
from selection.baseSelection import BaseSelection

class TournamentSelection(BaseSelection):
    def __init__(self, tournament_size=3):
        self.tournament_size = tournament_size
        self.rng = np.random.default_rng()

    def select(self, population: np.ndarray, fitness:np.ndarray, parents: int) -> np.ndarray:        
        
        ## Initialize offspring array
        population_size = population.shape[0]
        offspring = np.empty((parents, population.shape[1]), dtype=population.dtype)
        
        ## Perform tournament selection to selected parents
        for i in range(parents):
            selected_parents = self.rng.choice(population_size, self.tournament_size, replace=False)
            selected_fitness = fitness[selected_parents]

            best_index = np.argmax(selected_fitness)
            best_parent = selected_parents[best_index]

            offspring[i] = population[best_parent]
        
        return offspring

