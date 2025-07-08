from survivor_selection.baseSurvivorSelection import BaseSurvivorSelection
import numpy as np

class SteadyStateSurvivorSelection(BaseSurvivorSelection):
    def __init__(self, replacement_rate: float):
        if not 0.0 <= replacement_rate <= 1.0:
            raise ValueError("Replacement rate must be between 0 and 1.")
        self.replacement_rate = replacement_rate

    def __str__(self):
        return f"Steady State Selection with {self.replacement_rate} replacement rate"

    def select_survivors(self, population: np.ndarray, fitness: np.ndarray, offspring: np.ndarray, offspring_fitness: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        num_population = population.shape[0]
        to_replace = int(num_population * self.replacement_rate)

        if to_replace == 0 or offspring.shape[0] == 0:
            return population, fitness

        to_replace = min(to_replace, offspring.shape[0])

        # Indices de las peores soluciones de la population actual
        worst_ids = np.argsort(fitness)[:to_replace]

        # Indices de las mejores soluciones de los offspring
        best_off_ids = np.argsort(offspring_fitness)[::-1][:to_replace]

        # Reemplazar
        population_nueva = population.copy()
        nuevo_fitness = fitness.copy()
        population_nueva[worst_ids] = offspring[best_off_ids]
        nuevo_fitness[worst_ids] = offspring_fitness[best_off_ids]

        return population_nueva, nuevo_fitness