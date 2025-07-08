import numpy as np
from survivor_selection.baseSurvivorSelection import BaseSurvivorSelection
from typing import Tuple

class ElitismSurvivorSelection(BaseSurvivorSelection):
    def __init__(self, elitism_ratio: float = 0.1):
        #elitism_ratio: Fraction of top individuals to preserve (0.1 = 10%)
        self.elitism_ratio = elitism_ratio

    def __str__(self):
        return f"Elitism Selection with {self.elitism_ratio} replacement rate"
    
    def select_survivors(self, population: np.ndarray, fitness: np.ndarray, offspring: np.ndarray, offspring_fitness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        # Combine parents and offspring
        combined_pop = np.vstack([population, offspring])
        combined_fitness = np.concatenate([fitness, offspring_fitness])
        
        # Calculate number of elites to keep
        n_elites = max(1, int(len(population) * self.elitism_ratio))
        
        # Get indices of top individuals
        elite_indices = np.argpartition(combined_fitness, -n_elites)[-n_elites:]
        
        # Select remaining individuals randomly
        remaining_indices = np.random.choice(
            len(combined_pop), 
            size=len(population) - n_elites,
            replace=False
        )
        
        # Combine elites and random individuals
        survivor_indices = np.concatenate([elite_indices, remaining_indices])
        new_population = combined_pop[survivor_indices]
        new_fitness = combined_fitness[survivor_indices]
        
        return new_population, new_fitness